import json
from types import SimpleNamespace

import numpy as np
import pytest

from fsglib.common.types import (
    AttitudeSolution,
    FrameResult,
    MatchedStar,
    MatchingResult,
    ObservedStar,
    PreprocessedFrame,
    RawFrame,
    SequenceResult,
    SolveMode,
    SolveStateMachine,
    StarCandidate,
    TrackState,
)
from fsglib.ephemeris.types import ReferenceStar
from fsglib.match.pyramid import LocalPyramidCache
from fsglib.pipeline.evaluate import evaluate_dataset
from fsglib.pipeline.run_tracking import (
    _cfg_with_match_algorithm,
    _build_local_reacquire_frame,
    _build_lost_in_space_frame,
    _build_tracking_frame,
    normalize_solve_mode,
    run_sequence_tracking,
    update_state_machine,
    update_track_table,
)


def _dummy_frame_result(valid: bool, requested_mode: str, matched: int, rms: float, runtime_s: float, boresight_error: float | None = None) -> FrameResult:
    raw = RawFrame(detector_id=0, image=np.zeros((2, 2)), time_s=0.0)
    pre = PreprocessedFrame(detector_id=0, image=np.zeros((2, 2)), background=0.0, noise_map=np.ones((2, 2)), valid_mask=np.ones((2, 2), dtype=bool))
    candidates = [StarCandidate(0, 0, 0.0, 0.0, 1.0, 1.0, 1, 1.0, (0, 0, 0, 0))]
    matching = MatchingResult(matched=[], unmatched_observed_ids=[], unmatched_catalog_ids=[], mode=requested_mode, success=valid, score=float(matched), debug={})
    solution = AttitudeSolution(
        q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
        c_ib=np.eye(3),
        euler_zyx=None,
        valid=valid,
        mode=requested_mode,
        num_matched=matched,
        residual_rms_arcsec=rms,
        residual_max_arcsec=rms,
    )
    evaluation = None
    if boresight_error is not None:
        from fsglib.common.types import FrameEvaluation

        evaluation = FrameEvaluation(0, 0, None, None, 0, None, boresight_error, boresight_error, None, boresight_error)
    return FrameResult(
        raw=raw,
        preprocessed=pre,
        candidates=candidates,
        observed=[],
        reference=[],
        matching=matching,
        solution=solution,
        evaluation=evaluation,
        meta={"requested_mode": requested_mode, "timings_s": {"total": runtime_s}},
    )


def test_update_track_table_updates_matches_and_misses():
    existing = {
        1: TrackState(catalog_id=1, detector_id=0, last_xy=(1.0, 2.0), last_seen_time_s=0.0, miss_count=0),
        2: TrackState(catalog_id=2, detector_id=0, last_xy=(3.0, 4.0), last_seen_time_s=0.0, miss_count=2),
    }
    from fsglib.common.types import MatchedStar

    matching = MatchingResult(
        matched=[
            MatchedStar(0, 0, 1, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), match_score=0.8, flags={"observed_xy": (5.0, 6.0), "residual_pix": 0.4}),
        ],
        unmatched_observed_ids=[],
        unmatched_catalog_ids=[],
        mode="tracking",
        success=True,
        score=1.0,
    )

    updated = update_track_table(existing, matching, 10.0, {"tracking": {"max_miss_count": 2}})
    assert updated[1].last_xy == (5.0, 6.0)
    assert updated[1].miss_count == 0
    assert not updated[2].active
    assert updated[2].miss_count == 3


def test_solve_mode_normalizes_canonical_and_legacy_names():
    assert SolveMode.INIT_KNOWN_FIELD.value == "init_known_field"
    assert SolveMode.TRACKING.value == "tracking"
    assert SolveMode.LOCAL_REACQUIRE.value == "local_reacquire"
    assert SolveMode.LOST_IN_SPACE.value == "lost_in_space"
    assert SolveMode.SAFE_LOST.value == "safe_lost"

    assert normalize_solve_mode("init") is SolveMode.INIT_KNOWN_FIELD
    assert normalize_solve_mode("reacquire") is SolveMode.LOCAL_REACQUIRE
    assert normalize_solve_mode("lost") is SolveMode.LOST_IN_SPACE
    assert normalize_solve_mode(SolveMode.TRACKING) is SolveMode.TRACKING
    with pytest.raises(ValueError, match="unsupported solve mode"):
        normalize_solve_mode("not_a_mode")


def test_update_state_machine_uses_explicit_modes_and_audit_fields():
    cfg = {
        "tracking": {
            "reacquire_after_failures": 2,
            "lost_in_space_after_reacquire_failures": 2,
            "safe_lost_after_lis_failures": 1,
        }
    }
    bad_tracking = _dummy_frame_result(valid=False, requested_mode="tracking", matched=1, rms=100.0, runtime_s=0.1)
    bad_tracking.meta["requested_match_algorithm"] = "predicted_position"
    bad_tracking.matching.debug["selected_strategy"] = "predicted_position"

    state = SolveStateMachine(mode="tracking", consecutive_tracking_failures=1)
    state = update_state_machine(state, "tracking", bad_tracking, cfg, "attitude_invalid")
    assert state.mode == "local_reacquire"
    assert state.transition_reason == "local_reacquire_after_tracking_failures"
    assert state.reacquire_count == 1
    assert state.requested_mode == "tracking"
    assert state.requested_match_algorithm == "predicted_position"
    assert state.selected_match_strategy == "predicted_position"
    assert state.validation_reason == "attitude_invalid"

    good_reacquire = _dummy_frame_result(valid=True, requested_mode="local_reacquire", matched=5, rms=1.0, runtime_s=0.1)
    state = SolveStateMachine(mode="local_reacquire", consecutive_reacquire_failures=1)
    state = update_state_machine(state, "local_reacquire", good_reacquire, cfg, "reacquire_success")
    assert state.mode == "tracking"
    assert state.transition_reason == "local_reacquire_success"
    assert state.total_reacquire_successes == 1
    assert state.consecutive_reacquire_failures == 0
    assert state.consecutive_tracking_failures == 0

    bad_reacquire = _dummy_frame_result(valid=False, requested_mode="local_reacquire", matched=1, rms=100.0, runtime_s=0.1)
    state = SolveStateMachine(mode="local_reacquire", consecutive_reacquire_failures=1)
    state = update_state_machine(state, "local_reacquire", bad_reacquire, cfg, "local_reacquire_failed")
    assert state.mode == "lost_in_space"
    assert state.transition_reason == "lost_in_space_after_reacquire_failures"
    assert state.lost_count == 1

    bad_lis = _dummy_frame_result(valid=False, requested_mode="lost_in_space", matched=0, rms=np.inf, runtime_s=0.1)
    state = SolveStateMachine(mode="lost_in_space")
    state = update_state_machine(state, "lost_in_space", bad_lis, cfg, "lost_in_space_not_implemented")
    assert state.mode == "safe_lost"
    assert state.transition_reason == "safe_lost_after_lis_failures"
    assert state.safe_lost_count == 1


def test_cfg_with_match_algorithm_only_copies_match_override():
    cfg = {
        "match": {
            "algorithm": "predicted_position",
            "local_pyramid": {"seed_scopes": ["single_detector"]},
        },
        "ephemeris": {"large_table": [1, 2, 3]},
    }

    assert _cfg_with_match_algorithm(cfg, "predicted_position") is cfg

    overridden = _cfg_with_match_algorithm(cfg, "predicted_position_with_pyramid_reacquire")
    assert overridden is not cfg
    assert overridden["match"] is not cfg["match"]
    assert overridden["match"]["algorithm"] == "predicted_position_with_pyramid_reacquire"
    assert overridden["match"]["local_pyramid"] is cfg["match"]["local_pyramid"]
    assert overridden["ephemeris"] is cfg["ephemeris"]
    assert cfg["match"]["algorithm"] == "predicted_position"


def test_build_local_reacquire_frame_forces_reacquire_policy(monkeypatch):
    captured = {}

    def fake_build_tracking_frame(
        npz_path,
        cfg,
        models,
        dataset_ctx,
        prior_q,
        track_states,
        *,
        requested_mode,
        match_algorithm,
        reference_mode,
        track_catalog_ids,
    ):
        captured.update(
            {
                "npz_path": npz_path,
                "requested_mode": requested_mode,
                "match_algorithm": match_algorithm,
                "reference_mode": reference_mode,
                "track_catalog_ids": track_catalog_ids,
            }
        )
        frame = _dummy_frame_result(valid=True, requested_mode=str(requested_mode), matched=5, rms=1.0, runtime_s=0.1)
        frame.meta["requested_match_algorithm"] = match_algorithm
        return frame

    monkeypatch.setattr("fsglib.pipeline.run_tracking._build_tracking_frame", fake_build_tracking_frame)

    frame = _build_local_reacquire_frame(
        npz_path="frame.npz",
        cfg={"tracking": {}},
        models={},
        dataset_ctx=SimpleNamespace(batch_root="batch0"),
        prior_q=np.array([1.0, 0.0, 0.0, 0.0]),
        track_states={42: TrackState(42, 0, (10.0, 20.0), 0.0)},
    )

    assert captured["requested_mode"] is SolveMode.LOCAL_REACQUIRE
    assert captured["match_algorithm"] == "predicted_position_with_pyramid_reacquire"
    assert captured["reference_mode"] == "local_reacquire"
    assert captured["track_catalog_ids"] == []
    assert frame.meta["requested_match_algorithm"] == "predicted_position_with_pyramid_reacquire"


def test_build_lost_in_space_frame_returns_not_implemented_failure(monkeypatch):
    raw = RawFrame(detector_id=0, image=np.ones((3, 3)), time_s=12.0)
    pre = PreprocessedFrame(
        detector_id=0,
        image=np.ones((3, 3)),
        background=0.0,
        noise_map=np.ones((3, 3)),
        valid_mask=np.ones((3, 3), dtype=bool),
    )
    observed = [
        ObservedStar(
            detector_id=0,
            source_id=1,
            x=10.0,
            y=20.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=100.0,
            snr=20.0,
        )
    ]

    monkeypatch.setattr("fsglib.pipeline.run_tracking.load_npz_frame", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr("fsglib.pipeline.run_tracking.preprocess_frame", lambda *_args, **_kwargs: pre)
    monkeypatch.setattr("fsglib.pipeline.run_tracking.extract_stars", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("fsglib.pipeline.run_tracking.candidates_to_observed", lambda *_args, **_kwargs: observed)
    monkeypatch.setattr("fsglib.pipeline.run_tracking.evaluate_frame_result", lambda *_args, **_kwargs: None)

    frame = _build_lost_in_space_frame(
        npz_path="frame.npz",
        cfg={
            "layout": {"default_detector_id": 0},
            "tracking": {},
        },
        models={"projector": object()},
        dataset_ctx=SimpleNamespace(batch_root="batch0"),
    )

    assert not frame.solution.valid
    assert frame.solution.mode == "lost_in_space"
    assert frame.solution.quality["reason"] == "lost_in_space_not_implemented"
    assert frame.matching.mode == "lost_in_space"
    assert frame.matching.debug["selected_strategy"] == "lost_in_space_not_implemented"
    assert frame.meta["requested_mode"] == "lost_in_space"
    assert frame.meta["requested_match_algorithm"] == "lost_in_space"
    assert frame.meta["validation_reason"] == "lost_in_space_not_implemented"


def test_run_sequence_tracking_dispatches_explicit_modes_without_init_fallback(tmp_path, monkeypatch):
    calls = []
    dataset_ctx = SimpleNamespace(
        batch_root=tmp_path,
        batch_center_ra_deg=None,
        batch_center_dec_deg=None,
        field_offset_x_pix=None,
        field_offset_y_pix=None,
    )

    def fake_init(*_args, **_kwargs):
        calls.append("init_known_field")
        return _dummy_frame_result(valid=True, requested_mode="init_known_field", matched=5, rms=1.0, runtime_s=0.1)

    def fake_tracking(**_kwargs):
        calls.append("tracking")
        frame = _dummy_frame_result(valid=False, requested_mode="tracking", matched=1, rms=100.0, runtime_s=0.1)
        frame.meta["validation_reason"] = "tracking_failed"
        return frame

    def fake_reacquire(**_kwargs):
        calls.append("local_reacquire")
        frame = _dummy_frame_result(valid=False, requested_mode="local_reacquire", matched=1, rms=100.0, runtime_s=0.1)
        frame.meta["validation_reason"] = "local_reacquire_failed"
        return frame

    def fake_lis(**_kwargs):
        calls.append("lost_in_space")
        frame = _dummy_frame_result(valid=False, requested_mode="lost_in_space", matched=0, rms=999.0, runtime_s=0.1)
        frame.meta["validation_reason"] = "lost_in_space_not_implemented"
        return frame

    monkeypatch.setattr("fsglib.pipeline.run_tracking.run_single_frame_init", fake_init)
    monkeypatch.setattr("fsglib.pipeline.run_tracking._build_tracking_frame", fake_tracking)
    monkeypatch.setattr("fsglib.pipeline.run_tracking._build_local_reacquire_frame", fake_reacquire)
    monkeypatch.setattr("fsglib.pipeline.run_tracking._build_lost_in_space_frame", fake_lis)

    result = run_sequence_tracking(
        ["f0.npz", "f1.npz", "f2.npz", "f3.npz"],
        cfg={
            "tracking": {
                "max_miss_count": 3,
                "reacquire_after_tracking_failures": 1,
                "lost_in_space_after_reacquire_failures": 1,
                "safe_lost_after_lis_failures": 1,
            }
        },
        models={},
        dataset_ctx=dataset_ctx,
    )

    assert calls == ["init_known_field", "tracking", "local_reacquire", "lost_in_space"]
    assert result.mode_history == ["init_known_field", "tracking", "local_reacquire", "lost_in_space"]
    assert [state.mode for state in result.state_history] == ["tracking", "local_reacquire", "lost_in_space", "safe_lost"]


def test_build_tracking_frame_uses_configured_matching_algorithm(monkeypatch):
    raw = RawFrame(detector_id=0, image=np.ones((3, 3)), time_s=12.0)
    pre = PreprocessedFrame(
        detector_id=0,
        image=np.ones((3, 3)),
        background=0.0,
        noise_map=np.ones((3, 3)),
        valid_mask=np.ones((3, 3), dtype=bool),
    )
    observed = [
        ObservedStar(
            detector_id=0,
            source_id=1,
            x=10.0,
            y=20.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=100.0,
            snr=20.0,
        )
    ]
    reference = [
        ReferenceStar(
            catalog_id=42,
            time_s=12.0,
            los_inertial=np.array([0.0, 0.0, 1.0]),
            mag_g=10.0,
            detector_ids_visible=[0],
            predicted_xy={0: (10.0, 20.0)},
            predicted_valid={0: True},
            weight_hint=1.0,
        )
    ]
    matched = [
        MatchedStar(
            detector_id=0,
            source_id=1,
            catalog_id=42,
            los_body=observed[0].los_body,
            los_inertial=reference[0].los_inertial,
            weight=1.0,
            match_score=1.0,
            flags={"match_mode": "local_pyramid", "observed_xy": (10.0, 20.0), "residual_pix": 0.0},
        )
    ]
    cfg = {
        "layout": {"default_detector_id": 0},
        "match": {
            "algorithm": "local_pyramid",
            "validate_min_support": 1,
            "validate_max_residual_pix": 5.0,
            "enforce_unique_assignment": True,
        },
        "attitude": {},
        "tracking": {"max_attitude_jump_arcsec": 100.0},
    }
    called = {"local_pyramid": False, "cache": None, "pyramid_mode": None}

    monkeypatch.setattr("fsglib.pipeline.run_tracking.load_npz_frame", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr("fsglib.pipeline.run_tracking.preprocess_frame", lambda *_args, **_kwargs: pre)
    monkeypatch.setattr("fsglib.pipeline.run_tracking.extract_stars", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("fsglib.pipeline.run_tracking.candidates_to_observed", lambda *_args, **_kwargs: observed)
    monkeypatch.setattr(
        "fsglib.pipeline.run_tracking.predict_catalog_positions",
        lambda *_args, **_kwargs: (reference, SimpleNamespace(boresight_inertial=np.array([0.0, 0.0, 1.0]))),
    )

    def fake_match_local_pyramid(observed_stars, reference_stars, cfg_arg, cache=None, pyramid_mode=None):
        called["local_pyramid"] = True
        called["cache"] = cache
        called["pyramid_mode"] = pyramid_mode
        assert observed_stars is observed
        assert reference_stars is reference
        assert cfg_arg is cfg
        return MatchingResult(
            matched=matched,
            unmatched_observed_ids=[],
            unmatched_catalog_ids=[],
            mode="tracking",
            success=True,
            score=1.0,
            debug={"selected_strategy": "local_pyramid"},
        )

    monkeypatch.setattr("fsglib.match.pyramid.match_local_pyramid", fake_match_local_pyramid)
    monkeypatch.setattr(
        "fsglib.pipeline.run_tracking.solve_attitude",
        lambda *_args, **_kwargs: AttitudeSolution(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            c_ib=np.eye(3),
            euler_zyx=None,
            valid=True,
            mode="tracking",
            num_matched=1,
            residual_rms_arcsec=0.0,
            residual_max_arcsec=0.0,
        ),
    )
    monkeypatch.setattr(
        "fsglib.pipeline.run_tracking.validate_match_hypothesis",
        lambda *_args, **_kwargs: (True, {"reason": "ok"}),
    )
    monkeypatch.setattr("fsglib.pipeline.run_tracking.evaluate_frame_result", lambda *_args, **_kwargs: None)

    models = {"projector": object(), "catalog": object()}
    frame = _build_tracking_frame(
        npz_path="frame.npz",
        cfg=cfg,
        models=models,
        dataset_ctx=SimpleNamespace(batch_root="batch0"),
        prior_q=np.array([1.0, 0.0, 0.0, 0.0]),
        track_states={42: TrackState(42, 0, (10.0, 20.0), 0.0)},
    )

    assert called["local_pyramid"]
    assert isinstance(models["match_cache"], LocalPyramidCache)
    assert called["cache"] is models["match_cache"]
    assert called["pyramid_mode"] == "tracking"
    assert frame.matching.debug["algorithm"] == "local_pyramid"
    assert frame.matching.debug["selected_strategy"] == "local_pyramid"
    assert [star.catalog_id for star in frame.matching.matched] == [42]


def test_evaluate_dataset_aggregates_batches(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    for name in ["batch0_demo", "batch1_demo"]:
        frames_dir = root / name / "frames"
        frames_dir.mkdir(parents=True)
        np.savez(frames_dir / "frame000.npz", images=np.zeros((1, 1, 2, 2)), time_s=np.array([0.0]))
        (root / name / "run_meta.json").write_text(json.dumps({"field_center_ra_deg": 1.0, "field_center_dec_deg": 2.0, "detector_width_pix": 2}), encoding="utf-8")

    def fake_run_sequence_tracking(npz_paths, cfg, models, dataset_ctx=None):
        batch_name = dataset_ctx.batch_root.name
        frame_results = [
            _dummy_frame_result(valid=True, requested_mode="init", matched=10, rms=1.0, runtime_s=0.1, boresight_error=1.0),
            _dummy_frame_result(valid=batch_name == "batch0_demo", requested_mode="tracking", matched=8, rms=2.0, runtime_s=0.2, boresight_error=2.0),
        ]
        state_history = [
            SolveStateMachine(mode="tracking", transition_reason="init_success", total_init_frames=1, total_init_successes=1),
            SolveStateMachine(mode="tracking" if batch_name == "batch0_demo" else "init", transition_reason="reacquire_init" if batch_name != "batch0_demo" else "tracking_success", reacquire_count=1 if batch_name != "batch0_demo" else 0),
        ]
        return SequenceResult(frame_results=frame_results, track_states=[], mode_history=["init", "tracking"], state_history=state_history, metrics={})

    monkeypatch.setattr("fsglib.pipeline.run_tracking.run_sequence_tracking", fake_run_sequence_tracking)

    result = evaluate_dataset(str(root), cfg={}, models={})
    assert result["summary"]["num_batches"] == 2
    assert result["summary"]["reacquire_count"] == 1
    assert "mean_stage_runtime_s" in result["summary"]
    assert "mean_non_roll_error_arcsec" in result["summary"]
    assert "mean_total_attitude_error_arcsec" in result["summary"]
    assert "batch0_demo" in result["batches"]
    assert "batch1_demo" in result["batches"]


def test_evaluate_dataset_respects_sampling_controls(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    frames_dir = root / "batch0_demo" / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(5):
        np.savez(frames_dir / f"frame{idx:03d}.npz", images=np.zeros((1, 1, 2, 2)), time_s=np.array([float(idx)]))
    (root / "batch0_demo" / "run_meta.json").write_text(
        json.dumps({"field_center_ra_deg": 1.0, "field_center_dec_deg": 2.0, "detector_width_pix": 2}),
        encoding="utf-8",
    )

    seen_lengths = []

    def fake_run_sequence_tracking(npz_paths, cfg, models, dataset_ctx=None):
        seen_lengths.append(len(npz_paths))
        return SequenceResult(frame_results=[], track_states=[], mode_history=[], state_history=[], metrics={})

    monkeypatch.setattr("fsglib.pipeline.run_tracking.run_sequence_tracking", fake_run_sequence_tracking)

    result = evaluate_dataset(
        str(root),
        cfg={"evaluation": {"frame_stride": 2, "max_frames_per_batch": 2}},
        models={},
    )

    assert seen_lengths == [2]
    assert result["summary"]["frame_stride"] == 2
    assert result["summary"]["max_frames_per_batch"] == 2
