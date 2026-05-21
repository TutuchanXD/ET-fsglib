import numpy as np

from fsglib.common.types import (
    AttitudeSolution,
    ErrorBudgetLedger,
    FrameEvaluation,
    MatchedStar,
    MatchingResult,
    ObservedStar,
    PreprocessedFrame,
    RawFrame,
    StarCandidate,
)
from fsglib.pipeline.error_budget import (
    build_error_budget_ledger,
    summarize_error_budget_ledgers,
)


def _term(ledger: ErrorBudgetLedger, name: str):
    matches = [term for term in ledger.terms if term.name == name]
    assert matches, f"missing error-budget term: {name}"
    return matches[0]


def _base_inputs(variance_model: str = "empirical_robust"):
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0, 101.0], [102.0, 103.0]], dtype=np.float64),
        time_s=12.0,
        unit="adu",
        meta={"npz_path": "frame000001.npz"},
    )
    preprocess_meta = {
        "input_unit": "adu",
        "variance_model_effective": variance_model,
        "variance_unit": "adu^2",
        "background_rms": 2.0,
        "adc_clip": {
            "enabled": True,
            "min_value": 0.0,
            "max_value": 4095.0,
            "adc_bit_depth": 12,
            "num_clipped_high_pixels": 3,
            "num_clipped_low_pixels": 0,
        },
        "calibration": {
            "bias": {
                "applied": True,
                "path": "calibration/pr09_fake/2049x2049/bias_frame.npz",
            },
            "flat": {
                "applied": True,
                "path": "calibration/pr09_fake/2049x2049/flat_field.npz",
            },
        },
        "artifact_counts": {"saturated": 3},
    }
    if variance_model == "poisson_read_noise":
        preprocess_meta["variance_components"] = {
            "gain_e_per_output_unit": 2.0,
            "read_noise_e": 4.0,
            "quantization_noise_e": 1.0,
            "dark_current_source": "calib.dark",
            "flat_response_propagated": True,
            "flat_uncertainty_included": False,
        }
    preprocessed = PreprocessedFrame(
        detector_id=0,
        image=np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float64),
        background=10.0,
        noise_map=np.full((2, 2), 2.0, dtype=np.float64),
        valid_mask=np.ones((2, 2), dtype=bool),
        variance_map=np.full((2, 2), 4.0, dtype=np.float64),
        preprocess_meta=preprocess_meta,
    )
    candidate = StarCandidate(
        detector_id=0,
        source_id=1,
        x=10.0,
        y=20.0,
        flux=1200.0,
        peak=500.0,
        area=9,
        snr=30.0,
        bbox=(8, 18, 12, 22),
        centroid_cov_pix=np.array([[0.04, 0.0], [0.0, 0.09]], dtype=np.float64),
        flags={"centroid_covariance_source": "noise_propagation"},
    )
    observed = [
        ObservedStar(
            detector_id=0,
            source_id="0:1",
            x=10.0,
            y=20.0,
            los_body=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            flux=1200.0,
            snr=30.0,
            weight=20.0,
            centroid_cov_pix=candidate.centroid_cov_pix,
            sigma_angle_arcsec=1.5,
            flags={"sigma_angle_arcsec": 1.5, "weight_source": "variance_snr_hybrid"},
        )
    ]
    matching = MatchingResult(
        matched=[
            MatchedStar(
                detector_id=0,
                source_id="0:1",
                catalog_id=42,
                los_body=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                los_inertial=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                residual_arcsec=2.5,
                weight=20.0,
                match_score=0.9,
                flags={
                    "sigma_angle_arcsec": 1.5,
                    "residual_pix": 0.2,
                    "observed_xy": (10.0, 20.0),
                    "predicted_xy": (10.2, 20.0),
                },
            )
        ],
        unmatched_observed_ids=[],
        unmatched_catalog_ids=[],
        mode="init",
        success=True,
        score=0.9,
        debug={"selected_strategy": "predicted_position", "mean_residual_pix": 0.2},
    )
    solution = AttitudeSolution(
        q_ib=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        c_ib=np.eye(3, dtype=np.float64),
        euler_zyx=None,
        valid=True,
        mode="init",
        num_matched=1,
        residual_rms_arcsec=2.5,
        residual_max_arcsec=2.5,
        quality={
            "meta": {
                "attitude_covariance": {"available": True, "source": "sigma_angle_arcsec"},
                "robust_rejection": {
                    "enabled": True,
                    "num_rejected": 1,
                    "rejected_stars": [{"source_id": "0:9", "reasons": ["sigma_clip"]}],
                },
            }
        },
        num_rejected=1,
        sigma_non_roll_arcsec=0.8,
        sigma_roll_arcsec=1.2,
        attitude_condition_number=3.0,
    )
    evaluation = FrameEvaluation(
        num_truth_stars=1,
        num_candidate_truth_matches=1,
        centroid_mae_pix=0.3,
        centroid_max_pix=0.3,
        matched_catalog_truth_support=1,
        matched_catalog_truth_ratio=1.0,
        boresight_error_arcsec=0.5,
        non_roll_error_arcsec=0.5,
        roll_error_arcsec=0.7,
        total_attitude_error_arcsec=0.9,
        centroid_rms_dx_pix=0.2,
        centroid_rms_dy_pix=0.4,
    )
    return raw, preprocessed, [candidate], observed, matching, solution, evaluation


def test_error_budget_ledger_marks_empirical_noise_and_fake_assets():
    raw, preprocessed, candidates, observed, matching, solution, evaluation = _base_inputs()

    ledger = build_error_budget_ledger(
        raw=raw,
        preprocessed=preprocessed,
        candidates=candidates,
        observed=observed,
        matching=matching,
        solution=solution,
        evaluation=evaluation,
        dataset_ctx=None,
        cfg={"evaluation": {"error_budget": {"enabled": True}}},
    )

    assert ledger.enabled is True
    assert ledger.frame_id == "frame000001"
    assert _term(ledger, "detector.noise.empirical_rms").value == 2.0
    assert _term(ledger, "detector.noise.empirical_rms").unit == "adu"
    assert _term(ledger, "detector.noise.read").available is False
    assert _term(ledger, "detector.noise.read").reason == "requires_poisson_read_noise_variance_model"
    fake_assets = _term(ledger, "preprocess.calibration.fake_asset_count")
    assert fake_assets.value == 2
    assert "fake" in fake_assets.assumption
    assert ledger.summary["dominant_angular_term"]["name"] == "attitude.residual.rms"
    assert ledger.summary["dominant_angular_term"]["angular_equivalent_arcsec"] == 2.5
    assert ledger.per_star[0]["source_id"] == "0:1"
    assert ledger.per_star[0]["sigma_angle_arcsec"] == 1.5
    assert ledger.per_star[0]["residual_arcsec"] == 2.5


def test_error_budget_ledger_decomposes_poisson_read_noise_components():
    raw, preprocessed, candidates, observed, matching, solution, evaluation = _base_inputs(
        variance_model="poisson_read_noise"
    )

    ledger = build_error_budget_ledger(
        raw=raw,
        preprocessed=preprocessed,
        candidates=candidates,
        observed=observed,
        matching=matching,
        solution=solution,
        evaluation=evaluation,
        dataset_ctx=None,
        cfg={"evaluation": {"error_budget": {"enabled": True}}},
    )

    assert _term(ledger, "detector.noise.read").value == 4.0
    assert _term(ledger, "detector.noise.quantization").value == 1.0
    assert _term(ledger, "detector.noise.photon_shot").available is True
    assert _term(ledger, "detector.noise.flat_residual").available is False
    assert _term(ledger, "detector.noise.flat_residual").reason == "flat_uncertainty_not_propagated"


def test_error_budget_aggregate_summary_reports_percentiles():
    raw, preprocessed, candidates, observed, matching, solution, evaluation = _base_inputs()
    ledgers = []
    for residual in (2.0, 4.0):
        solution.residual_rms_arcsec = residual
        solution.residual_max_arcsec = residual
        matching.matched[0].residual_arcsec = residual
        ledgers.append(
            build_error_budget_ledger(
                raw=raw,
                preprocessed=preprocessed,
                candidates=candidates,
                observed=observed,
                matching=matching,
                solution=solution,
                evaluation=evaluation,
                dataset_ctx=None,
                cfg={"evaluation": {"error_budget": {"enabled": True}}},
            )
        )

    summary = summarize_error_budget_ledgers(ledgers)

    assert summary["num_ledgers"] == 2
    assert summary["terms"]["attitude.residual.rms"]["unit"] == "arcsec"
    assert summary["terms"]["attitude.residual.rms"]["p50"] == 3.0
    assert summary["terms"]["attitude.residual.rms"]["p95"] > 3.0
