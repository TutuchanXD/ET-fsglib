import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import fsglib.pipeline.evaluate as evaluate_module
from fsglib.common.types import (
    AttitudeSolution,
    DatasetContext,
    MatchingResult,
    PreprocessedFrame,
    RawFrame,
    StarCandidate,
    TruthStar,
)
from fsglib.pipeline.evaluate import evaluate_frame_result


def _truth_attitude_from_radec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra_rad = np.radians(float(ra_deg))
    dec_rad = np.radians(float(dec_deg))
    east = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0], dtype=np.float64)
    north = np.array(
        [
            -np.sin(dec_rad) * np.cos(ra_rad),
            -np.sin(dec_rad) * np.sin(ra_rad),
            np.cos(dec_rad),
        ],
        dtype=np.float64,
    )
    center = np.array(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ],
        dtype=np.float64,
    )
    return np.vstack([east, north, center])


def _dataset_ctx() -> DatasetContext:
    return DatasetContext(
        batch_root=Path("."),
        frame_paths=[],
        batch_center_ra_deg=10.0,
        batch_center_dec_deg=20.0,
        pixel_scale_arcsec_per_pix=4.83,
        detector_width_pix=1119,
        detector_height_pix=1119,
        truth_stars=[TruthStar(source_id=1, x_pix=559.0, y_pix=559.0, ra_deg=10.0, dec_deg=20.0)],
        run_meta={},
    )


def _solution_from_c_ib(c_ib: np.ndarray) -> AttitudeSolution:
    return AttitudeSolution(
        q_ib=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        c_ib=c_ib,
        euler_zyx=None,
        valid=True,
        mode="init",
        num_matched=10,
        residual_rms_arcsec=1.0,
        residual_max_arcsec=2.0,
    )


def _empty_matching() -> MatchingResult:
    return MatchingResult(
        matched=[],
        unmatched_observed_ids=[],
        unmatched_catalog_ids=[],
        mode="init",
        success=True,
        score=0.0,
    )


def _raw_frame_with_truth(truth_stars: list[TruthStar] | None = None) -> RawFrame:
    meta = {"npz_path": "frame.npz"}
    if truth_stars is not None:
        meta["truth_stars"] = truth_stars
        meta["truth_source"] = "npz_frame_truth"
    return RawFrame(
        detector_id=0,
        image=np.zeros((5, 5), dtype=np.float64),
        time_s=0.0,
        meta=meta,
    )


def test_evaluate_frame_result_separates_pure_roll_error():
    ctx = _dataset_ctx()
    c_truth = _truth_attitude_from_radec(ctx.batch_center_ra_deg, ctx.batch_center_dec_deg)
    roll_arcsec = 30.0
    delta = Rotation.from_rotvec(np.array([0.0, 0.0, np.radians(roll_arcsec / 3600.0)])).as_matrix()
    solution = _solution_from_c_ib(delta @ c_truth)

    evaluation = evaluate_frame_result(_raw_frame_with_truth(), None, [], _empty_matching(), solution, ctx)

    assert evaluation is not None
    assert evaluation.non_roll_error_arcsec is not None
    assert evaluation.roll_error_arcsec is not None
    assert evaluation.total_attitude_error_arcsec is not None
    assert evaluation.non_roll_error_arcsec < 1e-6
    assert np.isclose(evaluation.roll_error_arcsec, roll_arcsec, atol=1e-3)
    assert np.isclose(evaluation.total_attitude_error_arcsec, roll_arcsec, atol=1e-3)


def test_evaluate_frame_result_separates_pure_tilt_error():
    ctx = _dataset_ctx()
    c_truth = _truth_attitude_from_radec(ctx.batch_center_ra_deg, ctx.batch_center_dec_deg)
    tilt_arcsec = 18.0
    delta = Rotation.from_rotvec(np.array([0.0, np.radians(tilt_arcsec / 3600.0), 0.0])).as_matrix()
    solution = _solution_from_c_ib(delta @ c_truth)

    evaluation = evaluate_frame_result(_raw_frame_with_truth(), None, [], _empty_matching(), solution, ctx)

    assert evaluation is not None
    assert evaluation.non_roll_error_arcsec is not None
    assert evaluation.roll_error_arcsec is not None
    assert evaluation.total_attitude_error_arcsec is not None
    assert np.isclose(evaluation.non_roll_error_arcsec, tilt_arcsec, atol=1e-3)
    assert evaluation.roll_error_arcsec < 1e-6
    assert np.isclose(evaluation.total_attitude_error_arcsec, tilt_arcsec, atol=1e-3)


def test_evaluate_frame_result_prefers_frame_truth_over_static_truth():
    ctx = _dataset_ctx()
    raw_truth = [TruthStar(source_id=99, x_pix=10.0, y_pix=20.0, ra_deg=10.0, dec_deg=20.0)]
    raw = _raw_frame_with_truth(raw_truth)
    solution = _solution_from_c_ib(_truth_attitude_from_radec(ctx.batch_center_ra_deg, ctx.batch_center_dec_deg))
    candidates = [
        StarCandidate(
            detector_id=0,
            source_id=1,
            x=10.0,
            y=20.0,
            flux=1.0,
            peak=1.0,
            area=1,
            snr=10.0,
            bbox=(0, 0, 1, 1),
        )
    ]

    evaluation = evaluate_frame_result(raw, None, candidates, _empty_matching(), solution, ctx)

    assert evaluation is not None
    assert evaluation.num_truth_stars == 1
    assert np.isclose(evaluation.centroid_mae_pix, 0.0)
    assert evaluation.meta["truth_source"] == "npz_frame_truth"


def test_evaluate_frame_result_falls_back_to_static_truth():
    ctx = _dataset_ctx()
    raw = _raw_frame_with_truth(None)
    solution = _solution_from_c_ib(_truth_attitude_from_radec(ctx.batch_center_ra_deg, ctx.batch_center_dec_deg))
    candidates = [
        StarCandidate(
            detector_id=0,
            source_id=1,
            x=559.0,
            y=559.0,
            flux=1.0,
            peak=1.0,
            area=1,
            snr=10.0,
            bbox=(0, 0, 1, 1),
        )
    ]

    evaluation = evaluate_frame_result(raw, None, candidates, _empty_matching(), solution, ctx)

    assert evaluation is not None
    assert np.isclose(evaluation.centroid_mae_pix, 0.0)
    assert evaluation.meta["truth_source"] == "stars_ecsv_static"


def test_evaluate_frame_result_reports_xy_metrics_and_centroid_audit(monkeypatch):
    ctx = _dataset_ctx()
    raw_truth = [
        TruthStar(source_id=99, x_pix=10.0, y_pix=20.0, ra_deg=10.0, dec_deg=20.0),
        TruthStar(source_id=100, x_pix=30.0, y_pix=40.0, ra_deg=10.1, dec_deg=20.1),
    ]
    raw = _raw_frame_with_truth(raw_truth)
    pre = PreprocessedFrame(
        detector_id=0,
        image=np.zeros((64, 64), dtype=np.float64),
        background=np.zeros((64, 64), dtype=np.float64),
        noise_map=np.ones((64, 64), dtype=np.float64),
        valid_mask=np.ones((64, 64), dtype=bool),
        preprocess_meta={},
    )
    solution = _solution_from_c_ib(_truth_attitude_from_radec(ctx.batch_center_ra_deg, ctx.batch_center_dec_deg))
    candidates = [
        StarCandidate(
            detector_id=0,
            source_id=1,
            x=11.0,
            y=22.0,
            flux=1.0,
            peak=1.0,
            area=1,
            snr=10.0,
            bbox=(0, 0, 1, 1),
        ),
        StarCandidate(
            detector_id=0,
            source_id=2,
            x=29.0,
            y=39.0,
            flux=1.0,
            peak=1.0,
            area=1,
            snr=10.0,
            bbox=(0, 0, 1, 1),
        ),
    ]

    monkeypatch.setattr(
        evaluate_module,
        "compute_centroid_step_audit",
        lambda raw, preprocessed, candidates, cfg, truth_stars: {
            "enabled": True,
            "num_audited_stars": 2,
            "stage_stats": {"multi_pipeline_final": {"count": 2}},
            "transition_stats": {},
        },
    )
    cfg = {"evaluation": {"centroid_step_audit": {"enabled": True}}}

    evaluation = evaluate_frame_result(raw, pre, candidates, _empty_matching(), solution, ctx, cfg=cfg)

    assert evaluation is not None
    assert np.isclose(evaluation.centroid_mean_dx_pix, 0.0)
    assert np.isclose(evaluation.centroid_mean_dy_pix, 0.5)
    assert np.isclose(evaluation.centroid_mean_abs_dx_pix, 1.0)
    assert np.isclose(evaluation.centroid_mean_abs_dy_pix, 1.5)
    assert np.isclose(evaluation.centroid_rms_dx_pix, 1.0)
    assert np.isclose(evaluation.centroid_rms_dy_pix, np.sqrt((2.0**2 + 1.0**2) / 2.0))
    assert evaluation.meta["centroid_step_audit"]["enabled"] is True
