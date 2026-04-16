from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from fsglib.attitude.solver import solve_attitude
from fsglib.common.io import load_dataset_batch_for_frame, load_npz_frame
from fsglib.common.types import (
    AttitudeSolveInput,
    DatasetContext,
    FrameResult,
    MatchingContext,
    SequenceResult,
    SolveStateMachine,
    TrackState,
)
from fsglib.ephemeris.pipeline import build_reference_stars
from fsglib.extract.pipeline import extract_stars
from fsglib.match.pipeline import associate_nearest, validate_match_hypothesis
from fsglib.pipeline.convert import candidates_to_observed
from fsglib.pipeline.evaluate import evaluate_frame_result, summarize_sequence_result
from fsglib.pipeline.run_init import _build_ephemeris_context, run_single_frame_init
from fsglib.preprocess.pipeline import preprocess_frame


def _copy_state(state: SolveStateMachine) -> SolveStateMachine:
    return deepcopy(state)


def _compute_attitude_delta_arcsec(prior_q: np.ndarray | None, current_q: np.ndarray | None) -> float | None:
    if prior_q is None or current_q is None:
        return None

    prior_rot = Rotation.from_quat([prior_q[1], prior_q[2], prior_q[3], prior_q[0]])
    current_rot = Rotation.from_quat([current_q[1], current_q[2], current_q[3], current_q[0]])
    delta_rot = current_rot * prior_rot.inv()
    return float(np.degrees(delta_rot.magnitude()) * 3600.0)


def predict_catalog_positions(raw, cfg: dict, models: dict, dataset_ctx: DatasetContext, prior_q: np.ndarray, track_catalog_ids: list[int]) -> tuple:
    eph_ctx = _build_ephemeris_context(
        raw,
        cfg,
        dataset_ctx=dataset_ctx,
        mode="tracking",
        prior_attitude_q=prior_q,
        track_catalog_ids=track_catalog_ids,
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
    requested_mode: str,
    frame_result: FrameResult,
    cfg: dict,
    validation_reason: str,
) -> SolveStateMachine:
    next_state = _copy_state(state)

    if requested_mode == "tracking":
        next_state.total_tracking_frames += 1
        if frame_result.solution.valid:
            next_state.total_tracking_successes += 1
            next_state.consecutive_tracking_failures = 0
            next_state.mode = "tracking"
            next_state.transition_reason = "tracking_success"
        else:
            next_state.consecutive_tracking_failures += 1
            if next_state.consecutive_tracking_failures >= int(cfg["tracking"].get("reacquire_after_failures", 2)):
                next_state.mode = "init"
                next_state.reacquire_count += 1
                next_state.transition_reason = "reacquire_init"
            else:
                next_state.mode = "tracking"
                next_state.transition_reason = validation_reason
    else:
        next_state.total_init_frames += 1
        if frame_result.solution.valid:
            next_state.total_init_successes += 1
            next_state.consecutive_init_failures = 0
            next_state.consecutive_tracking_failures = 0
            next_state.mode = "tracking"
            next_state.transition_reason = "init_success"
        else:
            next_state.consecutive_init_failures += 1
            lost_after = int(cfg["tracking"].get("lost_after_init_failures", 3))
            if next_state.consecutive_init_failures >= lost_after:
                next_state.mode = "lost"
                next_state.lost_count += 1
                next_state.transition_reason = "lost_after_init_failures"
            else:
                next_state.mode = "init"
                next_state.transition_reason = validation_reason

    return next_state


def _build_tracking_frame(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext,
    prior_q: np.ndarray,
    track_states: dict[int, TrackState],
) -> FrameResult:
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

    t0 = perf_counter()
    ref, eph_ctx = predict_catalog_positions(
        raw,
        cfg,
        models,
        dataset_ctx,
        prior_q=prior_q,
        track_catalog_ids=[catalog_id for catalog_id, state in track_states.items() if state.active],
    )
    timings["ephemeris"] = perf_counter() - t0

    t0 = perf_counter()
    match_ctx = MatchingContext(
        mode="tracking",
        time_s=raw.time_s,
        observed_stars=obs,
        prior_attitude_q=prior_q,
        detector_layout=cfg.get("layout", {}),
        optical_model=cfg.get("layout", {}),
        matching_cfg=cfg.get("match", {}),
        boresight_inertial=eph_ctx.boresight_inertial,
        reference_stars=ref,
    )
    matching = associate_nearest(obs, ref, cfg)
    matching.mode = "tracking"
    timings["match"] = perf_counter() - t0

    t0 = perf_counter()
    solve_input = AttitudeSolveInput(
        time_s=raw.time_s,
        matched_stars=matching.matched,
        prior_q_ib=prior_q,
        mode="tracking",
        solver_cfg=cfg["attitude"],
    )
    solution = solve_attitude(solve_input, cfg)
    attitude_delta_arcsec = _compute_attitude_delta_arcsec(prior_q, solution.q_ib)
    hypothesis_ok, hypothesis_debug = validate_match_hypothesis(
        matching,
        solution,
        cfg,
        attitude_delta_arcsec=attitude_delta_arcsec,
    )
    timings["attitude"] = perf_counter() - t0

    if not hypothesis_ok:
        solution.valid = False
        solution.quality_flag = "INVALID"
        solution.mode = "tracking"

    t0 = perf_counter()
    evaluation = evaluate_frame_result(raw, pre, cand, matching, solution, dataset_ctx, cfg=cfg)
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
            "requested_mode": "tracking",
            "timings_s": timings,
            "tracking_validation": hypothesis_debug,
        },
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
    state = SolveStateMachine(mode="init")
    prior_q: np.ndarray | None = None

    for index, npz_path in enumerate(npz_paths):
        requested_mode = state.mode if state.mode in {"init", "tracking"} else "init"

        if requested_mode == "init":
            frame_result = run_single_frame_init(npz_path, cfg, models, dataset_ctx=dataset_ctx)
            frame_result.meta["requested_mode"] = "init"
            validation_reason = "init_success" if frame_result.solution.valid else frame_result.solution.quality.get("reason", "init_failed")
        else:
            frame_result = _build_tracking_frame(
                npz_path=npz_path,
                cfg=cfg,
                models=models,
                dataset_ctx=dataset_ctx,
                prior_q=prior_q,
                track_states=track_states,
            )
            validation_reason = frame_result.meta.get("tracking_validation", {}).get("reason", "tracking_failed")

        frame_results.append(frame_result)
        mode_history.append(requested_mode)

        if frame_result.solution.valid:
            track_states = update_track_table(
                {} if requested_mode == "init" else track_states,
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

        if state.mode == "lost":
            prior_q = None
            track_states = {}

        if index == 0 and state.mode == "tracking" and prior_q is None:
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
