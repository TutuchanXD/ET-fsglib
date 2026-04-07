import json

import numpy as np

from fsglib.common.types import (
    AttitudeSolution,
    FrameResult,
    MatchingResult,
    PreprocessedFrame,
    RawFrame,
    SequenceResult,
    SolveStateMachine,
    StarCandidate,
    TrackState,
)
from fsglib.pipeline.evaluate import evaluate_dataset
from fsglib.pipeline.run_tracking import update_state_machine, update_track_table


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


def test_update_state_machine_handles_reacquire_and_lost():
    state = SolveStateMachine(mode="tracking", consecutive_tracking_failures=1)
    bad_frame = _dummy_frame_result(valid=False, requested_mode="tracking", matched=1, rms=100.0, runtime_s=0.1)

    state = update_state_machine(state, "tracking", bad_frame, {"tracking": {"reacquire_after_failures": 2, "lost_after_init_failures": 3}}, "attitude_invalid")
    assert state.mode == "init"
    assert state.reacquire_count == 1

    state.mode = "init"
    state.consecutive_init_failures = 2
    state = update_state_machine(state, "init", bad_frame, {"tracking": {"reacquire_after_failures": 2, "lost_after_init_failures": 3}}, "init_failed")
    assert state.mode == "lost"
    assert state.lost_count == 1


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
