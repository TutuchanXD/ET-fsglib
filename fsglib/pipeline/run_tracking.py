from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from fsglib.attitude.solver import solve_attitude
from fsglib.common.io import load_dataset_batch_for_frame, load_npz_frame
from fsglib.common.types import (
    AttitudeSolution,
    AttitudeSolveInput,
    DatasetContext,
    FrameResult,
    MatchingContext,
    MatchingResult,
    SequenceResult,
    SolveMode,
    SolveStateMachine,
    TrackState,
)
from fsglib.ephemeris.pipeline import build_reference_stars
from fsglib.extract.pipeline import extract_stars
from fsglib.match.cache import get_match_cache
from fsglib.match.pipeline import match_stars, validate_match_hypothesis
from fsglib.pipeline.convert import candidates_to_observed
from fsglib.pipeline.evaluate import evaluate_frame_result, summarize_sequence_result
from fsglib.pipeline.run_init import _build_ephemeris_context, run_single_frame_init
from fsglib.preprocess.pipeline import preprocess_frame


def _copy_state(state: SolveStateMachine) -> SolveStateMachine:
    return deepcopy(state)


def normalize_solve_mode(mode: str | SolveMode) -> SolveMode:
    if isinstance(mode, SolveMode):
        return mode

    aliases = {
        "init": SolveMode.INIT_KNOWN_FIELD,
        "init_known_field": SolveMode.INIT_KNOWN_FIELD,
        "tracking": SolveMode.TRACKING,
        "reacquire": SolveMode.LOCAL_REACQUIRE,
        "local_reacquire": SolveMode.LOCAL_REACQUIRE,
        "lost": SolveMode.LOST_IN_SPACE,
        "lost_in_space": SolveMode.LOST_IN_SPACE,
        "safe_lost": SolveMode.SAFE_LOST,
    }
    try:
        return aliases[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported solve mode: {mode}") from exc


def _mode_value(mode: str | SolveMode) -> str:
    return normalize_solve_mode(mode).value


def _tracking_cfg_int(cfg: dict, key: str, default: int, aliases: tuple[str, ...] = ()) -> int:
    tracking_cfg = cfg.get("tracking", {})
    for candidate in (key, *aliases):
        if candidate in tracking_cfg and tracking_cfg[candidate] is not None:
            return int(tracking_cfg[candidate])
    return default


def _match_algorithm_from_cfg(cfg: dict, key: str, default: str) -> str:
    tracking_cfg = cfg.get("tracking", {})
    if key in tracking_cfg and tracking_cfg[key]:
        return str(tracking_cfg[key])
    return default


def _tracking_match_algorithm(cfg: dict) -> str:
    return _match_algorithm_from_cfg(
        cfg,
        "tracking_match_algorithm",
        cfg.get("match", {}).get("algorithm", "predicted_position"),
    )


def _local_reacquire_match_algorithm(cfg: dict) -> str:
    return _match_algorithm_from_cfg(
        cfg,
        "local_reacquire_match_algorithm",
        "predicted_position_with_pyramid_reacquire",
    )


def _lost_in_space_match_algorithm(cfg: dict) -> str:
    return _match_algorithm_from_cfg(cfg, "lost_in_space_match_algorithm", "lost_in_space")


def _cfg_with_match_algorithm(cfg: dict, algorithm: str) -> dict:
    effective_cfg = deepcopy(cfg)
    effective_cfg.setdefault("match", {})
    effective_cfg["match"]["algorithm"] = algorithm
    return effective_cfg


def _selected_match_strategy(frame_result: FrameResult) -> str | None:
    strategy = frame_result.meta.get("selected_match_strategy")
    if strategy is None:
        strategy = frame_result.matching.debug.get("selected_strategy")
    return strategy


def _validation_reason(frame_result: FrameResult, default: str) -> str:
    return (
        frame_result.meta.get("validation_reason")
        or frame_result.meta.get("tracking_validation", {}).get("reason")
        or frame_result.solution.quality.get("reason")
        or default
    )


def _compute_attitude_delta_arcsec(prior_q: np.ndarray | None, current_q: np.ndarray | None) -> float | None:
    if prior_q is None or current_q is None:
        return None

    prior_rot = Rotation.from_quat([prior_q[1], prior_q[2], prior_q[3], prior_q[0]])
    current_rot = Rotation.from_quat([current_q[1], current_q[2], current_q[3], current_q[0]])
    delta_rot = current_rot * prior_rot.inv()
    return float(np.degrees(delta_rot.magnitude()) * 3600.0)


def predict_catalog_positions(
    raw,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
    prior_q: np.ndarray | None,
    track_catalog_ids: list[int] | None,
    reference_mode: str | SolveMode = SolveMode.TRACKING,
) -> tuple:
    eph_ctx = _build_ephemeris_context(
        raw,
        cfg,
        dataset_ctx=dataset_ctx,
        mode=_mode_value(reference_mode),
        prior_attitude_q=prior_q,
        track_catalog_ids=[] if track_catalog_ids is None else track_catalog_ids,
    )
    ref = build_reference_stars(eph_ctx, models["catalog"], models["projector"], cfg)
    return ref, eph_ctx


def update_track_table(
    existing: dict[int, TrackState],
    matching,
    raw_time_s: float,
    cfg: dict,
    *,
    accept_matches: bool = True,
) -> dict[int, TrackState]:
    max_miss_count = int(cfg["tracking"].get("max_miss_count", 3))
    updated: dict[int, TrackState] = {}

    if accept_matches:
        for matched_star in matching.matched:
            residual_pix = matched_star.flags.get("residual_pix")
            observed_xy = matched_star.flags.get("observed_xy")
            updated[matched_star.catalog_id] = TrackState(
                catalog_id=matched_star.catalog_id,
                detector_id=matched_star.detector_id,
                last_xy=tuple(observed_xy) if observed_xy is not None else None,
                last_seen_time_s=raw_time_s,
                miss_count=0,
                quality_score=matched_star.match_score,
                active=True,
                last_match_score=matched_star.match_score,
                last_residual_pix=residual_pix,
            )

    for catalog_id, state in existing.items():
        if catalog_id in updated:
            continue
        next_state = deepcopy(state)
        next_state.miss_count += 1
        next_state.active = next_state.miss_count <= max_miss_count
        updated[catalog_id] = next_state

    return updated


def update_state_machine(
    state: SolveStateMachine,
    requested_mode: str | SolveMode,
    frame_result: FrameResult,
    cfg: dict,
    validation_reason: str,
) -> SolveStateMachine:
    requested = normalize_solve_mode(requested_mode)
    next_state = _copy_state(state)
    next_state.mode = _mode_value(next_state.mode)
    next_state.requested_mode = requested.value
    next_state.requested_match_algorithm = frame_result.meta.get("requested_match_algorithm")
    next_state.selected_match_strategy = _selected_match_strategy(frame_result)
    next_state.validation_reason = validation_reason

    if requested is SolveMode.INIT_KNOWN_FIELD:
        next_state.total_init_frames += 1
        if frame_result.solution.valid:
            next_state.total_init_successes += 1
            next_state.consecutive_init_failures = 0
            next_state.consecutive_tracking_failures = 0
            next_state.consecutive_reacquire_failures = 0
            next_state.consecutive_lost_in_space_failures = 0
            next_state.mode = SolveMode.TRACKING.value
            next_state.transition_reason = "init_known_field_success"
        else:
            next_state.consecutive_init_failures += 1
            lost_after = _tracking_cfg_int(cfg, "lost_after_init_failures", 3)
            if next_state.consecutive_init_failures >= lost_after:
                next_state.mode = SolveMode.LOST_IN_SPACE.value
                next_state.lost_count += 1
                next_state.transition_reason = "lost_in_space_after_init_failures"
            else:
                next_state.mode = SolveMode.INIT_KNOWN_FIELD.value
                next_state.transition_reason = validation_reason
    elif requested is SolveMode.TRACKING:
        next_state.total_tracking_frames += 1
        if frame_result.solution.valid:
            next_state.total_tracking_successes += 1
            next_state.consecutive_tracking_failures = 0
            next_state.consecutive_reacquire_failures = 0
            next_state.mode = SolveMode.TRACKING.value
            next_state.transition_reason = "tracking_success"
        else:
            next_state.consecutive_tracking_failures += 1
            reacquire_after = _tracking_cfg_int(
                cfg,
                "reacquire_after_tracking_failures",
                2,
                aliases=("reacquire_after_failures",),
            )
            if next_state.consecutive_tracking_failures >= reacquire_after:
                next_state.mode = SolveMode.LOCAL_REACQUIRE.value
                next_state.reacquire_count += 1
                next_state.transition_reason = "local_reacquire_after_tracking_failures"
            else:
                next_state.mode = SolveMode.TRACKING.value
                next_state.transition_reason = validation_reason
    elif requested is SolveMode.LOCAL_REACQUIRE:
        next_state.total_reacquire_frames += 1
        if frame_result.solution.valid:
            next_state.total_reacquire_successes += 1
            next_state.consecutive_reacquire_failures = 0
            next_state.consecutive_tracking_failures = 0
            next_state.consecutive_lost_in_space_failures = 0
            next_state.mode = SolveMode.TRACKING.value
            next_state.transition_reason = "local_reacquire_success"
        else:
            next_state.consecutive_reacquire_failures += 1
            lost_after = _tracking_cfg_int(
                cfg,
                "lost_in_space_after_reacquire_failures",
                3,
                aliases=("lost_after_init_failures",),
            )
            if next_state.consecutive_reacquire_failures >= lost_after:
                next_state.mode = SolveMode.LOST_IN_SPACE.value
                next_state.lost_count += 1
                next_state.transition_reason = "lost_in_space_after_reacquire_failures"
            else:
                next_state.mode = SolveMode.LOCAL_REACQUIRE.value
                next_state.transition_reason = validation_reason
    elif requested is SolveMode.LOST_IN_SPACE:
        next_state.total_lost_in_space_frames += 1
        if frame_result.solution.valid:
            next_state.total_lost_in_space_successes += 1
            next_state.consecutive_lost_in_space_failures = 0
            next_state.consecutive_reacquire_failures = 0
            next_state.consecutive_tracking_failures = 0
            next_state.mode = SolveMode.TRACKING.value
            next_state.transition_reason = "lost_in_space_success"
        else:
            next_state.consecutive_lost_in_space_failures += 1
            safe_lost_after = _tracking_cfg_int(cfg, "safe_lost_after_lis_failures", 1)
            if next_state.consecutive_lost_in_space_failures >= safe_lost_after:
                next_state.mode = SolveMode.SAFE_LOST.value
                next_state.safe_lost_count += 1
                next_state.transition_reason = "safe_lost_after_lis_failures"
            else:
                next_state.mode = SolveMode.LOST_IN_SPACE.value
                next_state.transition_reason = validation_reason
    else:
        next_state.mode = SolveMode.SAFE_LOST.value
        next_state.transition_reason = validation_reason or "safe_lost"

    return next_state


def _build_tracking_frame(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
    prior_q: np.ndarray | None,
    track_states: dict[int, TrackState],
    *,
    requested_mode: str | SolveMode = SolveMode.TRACKING,
    match_algorithm: str | None = None,
    reference_mode: str | SolveMode | None = None,
    track_catalog_ids: list[int] | None = None,
) -> FrameResult:
    requested = normalize_solve_mode(requested_mode)
    mode_value = requested.value
    effective_cfg = _cfg_with_match_algorithm(cfg, match_algorithm) if match_algorithm is not None else cfg
    selected_algorithm = effective_cfg.get("match", {}).get("algorithm", "predicted_position")
    if reference_mode is None:
        reference_mode = SolveMode.TRACKING

    timings: dict[str, float] = {}
    total_start = perf_counter()

    raw = load_npz_frame(npz_path, detector_id=int(effective_cfg["layout"].get("default_detector_id", 0)))

    t0 = perf_counter()
    pre = preprocess_frame(raw, calib=models.get("calib", {}), cfg=effective_cfg)
    timings["preprocess"] = perf_counter() - t0

    t0 = perf_counter()
    cand = extract_stars(pre, cfg=effective_cfg)
    timings["extract"] = perf_counter() - t0

    t0 = perf_counter()
    obs = candidates_to_observed(cand, models["projector"], effective_cfg)
    timings["convert"] = perf_counter() - t0

    t0 = perf_counter()
    active_catalog_ids = (
        track_catalog_ids
        if track_catalog_ids is not None
        else [catalog_id for catalog_id, state in track_states.items() if state.active]
    )
    ref, eph_ctx = predict_catalog_positions(
        raw,
        effective_cfg,
        models,
        dataset_ctx,
        prior_q=prior_q,
        track_catalog_ids=active_catalog_ids,
        reference_mode=reference_mode,
    )
    timings["ephemeris"] = perf_counter() - t0

    t0 = perf_counter()
    match_ctx = MatchingContext(
        mode=mode_value,
        time_s=raw.time_s,
        observed_stars=obs,
        prior_attitude_q=prior_q,
        detector_layout=effective_cfg.get("layout", {}),
        optical_model=effective_cfg.get("layout", {}),
        matching_cfg=effective_cfg.get("match", {}),
        boresight_inertial=eph_ctx.boresight_inertial,
        reference_stars=ref,
        match_cache=get_match_cache(models, effective_cfg),
    )
    matching = match_stars(match_ctx, ref, effective_cfg)
    matching.mode = mode_value
    matching.debug.setdefault("algorithm", selected_algorithm)
    matching.debug.setdefault("selected_strategy", selected_algorithm)
    timings["match"] = perf_counter() - t0

    t0 = perf_counter()
    solve_input = AttitudeSolveInput(
        time_s=raw.time_s,
        matched_stars=matching.matched,
        prior_q_ib=prior_q,
        mode=mode_value,
        solver_cfg=effective_cfg["attitude"],
    )
    solution = solve_attitude(solve_input, effective_cfg)
    attitude_delta_arcsec = _compute_attitude_delta_arcsec(prior_q, solution.q_ib)
    hypothesis_ok, hypothesis_debug = validate_match_hypothesis(
        matching,
        solution,
        effective_cfg,
        attitude_delta_arcsec=attitude_delta_arcsec,
    )
    timings["attitude"] = perf_counter() - t0

    if not hypothesis_ok:
        solution.valid = False
        solution.quality_flag = "INVALID"
    solution.mode = mode_value
    validation_reason = hypothesis_debug.get("reason", "ok" if solution.valid else f"{mode_value}_failed")

    t0 = perf_counter()
    evaluation = evaluate_frame_result(raw, pre, cand, matching, solution, dataset_ctx, cfg=effective_cfg)
    timings["evaluate"] = perf_counter() - t0
    timings["total"] = perf_counter() - total_start

    return FrameResult(
        raw=raw,
        preprocessed=pre,
        candidates=cand,
        observed=obs,
        reference=ref,
        matching=matching,
        solution=solution,
        evaluation=evaluation,
        meta={
            "dataset_batch_root": str(dataset_ctx.batch_root),
            "num_reference_stars": len(ref),
            "requested_mode": mode_value,
            "requested_match_algorithm": selected_algorithm,
            "selected_match_strategy": matching.debug.get("selected_strategy"),
            "validation_reason": validation_reason,
            "timings_s": timings,
            "tracking_validation": hypothesis_debug,
        },
    )


def _build_local_reacquire_frame(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
    prior_q: np.ndarray | None,
    track_states: dict[int, TrackState],
) -> FrameResult:
    return _build_tracking_frame(
        npz_path=npz_path,
        cfg=cfg,
        models=models,
        dataset_ctx=dataset_ctx,
        prior_q=prior_q,
        track_states=track_states,
        requested_mode=SolveMode.LOCAL_REACQUIRE,
        match_algorithm=_local_reacquire_match_algorithm(cfg),
        reference_mode=SolveMode.LOCAL_REACQUIRE,
        track_catalog_ids=[],
    )


def _build_invalid_mode_frame(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
    *,
    requested_mode: str | SolveMode,
    match_algorithm: str,
    reason: str,
) -> FrameResult:
    mode_value = _mode_value(requested_mode)
    timings: dict[str, float] = {}
    total_start = perf_counter()

    raw = load_npz_frame(npz_path, detector_id=int(cfg["layout"].get("default_detector_id", 0)))

    t0 = perf_counter()
    pre = preprocess_frame(raw, calib=models.get("calib", {}), cfg=cfg)
    timings["preprocess"] = perf_counter() - t0

    t0 = perf_counter()
    cand = extract_stars(pre, cfg=cfg)
    timings["extract"] = perf_counter() - t0

    t0 = perf_counter()
    obs = candidates_to_observed(cand, models["projector"], cfg)
    timings["convert"] = perf_counter() - t0

    matching = MatchingResult(
        matched=[],
        unmatched_observed_ids=[star.source_id for star in obs],
        unmatched_catalog_ids=[],
        mode=mode_value,
        success=False,
        score=0.0,
        debug={
            "algorithm": match_algorithm,
            "selected_strategy": reason,
            "rejection_reason": reason,
        },
    )
    solution = AttitudeSolution(
        q_ib=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        c_ib=np.eye(3, dtype=np.float64),
        euler_zyx=None,
        valid=False,
        mode=mode_value,
        num_matched=0,
        residual_rms_arcsec=np.inf,
        residual_max_arcsec=np.inf,
        quality={"reason": reason},
        quality_flag="INVALID",
        degraded_level="LOST",
    )

    t0 = perf_counter()
    evaluation = evaluate_frame_result(raw, pre, cand, matching, solution, dataset_ctx, cfg=cfg)
    timings["evaluate"] = perf_counter() - t0
    timings["total"] = perf_counter() - total_start

    return FrameResult(
        raw=raw,
        preprocessed=pre,
        candidates=cand,
        observed=obs,
        reference=[],
        matching=matching,
        solution=solution,
        evaluation=evaluation,
        meta={
            "dataset_batch_root": str(dataset_ctx.batch_root),
            "num_reference_stars": 0,
            "requested_mode": mode_value,
            "requested_match_algorithm": match_algorithm,
            "selected_match_strategy": reason,
            "validation_reason": reason,
            "timings_s": timings,
        },
    )


def _build_lost_in_space_frame(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
) -> FrameResult:
    return _build_invalid_mode_frame(
        npz_path,
        cfg,
        models,
        dataset_ctx,
        requested_mode=SolveMode.LOST_IN_SPACE,
        match_algorithm=_lost_in_space_match_algorithm(cfg),
        reason="lost_in_space_not_implemented",
    )


def _build_safe_lost_frame(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
) -> FrameResult:
    return _build_invalid_mode_frame(
        npz_path,
        cfg,
        models,
        dataset_ctx,
        requested_mode=SolveMode.SAFE_LOST,
        match_algorithm="safe_lost",
        reason="safe_lost",
    )


def run_sequence_tracking(
    npz_paths: list[str],
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext | None = None,
) -> SequenceResult:
    if not npz_paths:
        return SequenceResult(frame_results=[], track_states=[], mode_history=[], state_history=[], metrics={})

    if dataset_ctx is None:
        dataset_ctx = load_dataset_batch_for_frame(npz_paths[0], cfg=cfg)

    if "projector" in models and hasattr(models["projector"], "set_field_center"):
        models["projector"].set_field_center(
            dataset_ctx.batch_center_ra_deg if dataset_ctx is not None else None,
            dataset_ctx.batch_center_dec_deg if dataset_ctx is not None else None,
            dataset_ctx.field_offset_x_pix if dataset_ctx is not None else None,
            dataset_ctx.field_offset_y_pix if dataset_ctx is not None else None,
        )

    frame_results: list[FrameResult] = []
    mode_history: list[str] = []
    state_history: list[SolveStateMachine] = []
    track_states: dict[int, TrackState] = {}
    state = SolveStateMachine(mode=SolveMode.INIT_KNOWN_FIELD)
    prior_q: np.ndarray | None = None

    for index, npz_path in enumerate(npz_paths):
        requested = normalize_solve_mode(state.mode)
        requested_mode = requested.value

        if requested is SolveMode.INIT_KNOWN_FIELD:
            frame_result = run_single_frame_init(npz_path, cfg, models, dataset_ctx=dataset_ctx)
            frame_result.solution.mode = requested_mode
            frame_result.matching.mode = requested_mode
            frame_result.meta["requested_mode"] = requested_mode
            frame_result.meta.setdefault(
                "requested_match_algorithm",
                cfg.get("match", {}).get("algorithm", "predicted_position"),
            )
            frame_result.meta.setdefault("selected_match_strategy", frame_result.matching.debug.get("selected_strategy"))
            validation_reason = (
                "init_known_field_success"
                if frame_result.solution.valid
                else _validation_reason(frame_result, "init_known_field_failed")
            )
            frame_result.meta["validation_reason"] = validation_reason
        elif requested is SolveMode.TRACKING:
            frame_result = _build_tracking_frame(
                npz_path=npz_path,
                cfg=cfg,
                models=models,
                dataset_ctx=dataset_ctx,
                prior_q=prior_q,
                track_states=track_states,
                requested_mode=SolveMode.TRACKING,
                match_algorithm=_tracking_match_algorithm(cfg),
                reference_mode=SolveMode.TRACKING,
            )
            validation_reason = _validation_reason(frame_result, "tracking_failed")
        elif requested is SolveMode.LOCAL_REACQUIRE:
            frame_result = _build_local_reacquire_frame(
                npz_path=npz_path,
                cfg=cfg,
                models=models,
                dataset_ctx=dataset_ctx,
                prior_q=prior_q,
                track_states=track_states,
            )
            validation_reason = _validation_reason(frame_result, "local_reacquire_failed")
        elif requested is SolveMode.LOST_IN_SPACE:
            frame_result = _build_lost_in_space_frame(
                npz_path=npz_path,
                cfg=cfg,
                models=models,
                dataset_ctx=dataset_ctx,
            )
            validation_reason = _validation_reason(frame_result, "lost_in_space_not_implemented")
        else:
            frame_result = _build_safe_lost_frame(
                npz_path=npz_path,
                cfg=cfg,
                models=models,
                dataset_ctx=dataset_ctx,
            )
            validation_reason = _validation_reason(frame_result, "safe_lost")

        frame_results.append(frame_result)
        mode_history.append(requested_mode)

        if frame_result.solution.valid:
            reset_tracks = requested in {
                SolveMode.INIT_KNOWN_FIELD,
                SolveMode.LOCAL_REACQUIRE,
                SolveMode.LOST_IN_SPACE,
            }
            track_states = update_track_table(
                {} if reset_tracks else track_states,
                frame_result.matching,
                frame_result.raw.time_s,
                cfg,
                accept_matches=True,
            )
        else:
            track_states = update_track_table(
                track_states,
                frame_result.matching,
                frame_result.raw.time_s,
                cfg,
                accept_matches=False,
            )
        if frame_result.solution.valid:
            prior_q = frame_result.solution.q_ib

        state = update_state_machine(state, requested_mode, frame_result, cfg, validation_reason)
        state_history.append(_copy_state(state))

        if normalize_solve_mode(state.mode) in {SolveMode.LOST_IN_SPACE, SolveMode.SAFE_LOST}:
            prior_q = None
            track_states = {}

        if index == 0 and normalize_solve_mode(state.mode) is SolveMode.TRACKING and prior_q is None:
            prior_q = frame_result.solution.q_ib if frame_result.solution.valid else None

    sequence_result = SequenceResult(
        frame_results=frame_results,
        track_states=list(track_states.values()),
        mode_history=mode_history,
        state_history=state_history,
        metrics={},
    )
    sequence_result.metrics = summarize_sequence_result(sequence_result)
    return sequence_result
