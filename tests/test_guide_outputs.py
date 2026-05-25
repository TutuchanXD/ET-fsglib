import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fsglib.common.debug import _write_attitude_debug_artifacts
from fsglib.common.types import (
    AttitudeSolution,
    MatchedStar,
    MatchingResult,
    ObservedStar,
    RawFrame,
)
from fsglib.ephemeris.types import ReferenceStar
from fsglib.pipeline import run_guide_init, run_guide_truth_noise
from fsglib.pipeline.guide_outputs import (
    resolve_debug_output_dir,
    resolve_figures_output_dir,
    resolve_fsg_results_root,
    save_matching_overlays,
)


def test_resolve_fsg_results_root_uses_dataset_sibling_for_default_output(tmp_path):
    dataset_root = tmp_path / "microlens_guide_photsim7_6s"
    dataset_root.mkdir()
    cfg = {
        "project": {"output_dir": "outputs/debug"},
        "guide_init": {"dataset_root": str(dataset_root), "frame_index": 0},
    }

    assert resolve_fsg_results_root(cfg) == tmp_path / "microlens_guide_photsim7_6s_fsg-results"
    assert resolve_debug_output_dir(cfg) == tmp_path / "microlens_guide_photsim7_6s_fsg-results" / "frame000000" / "debug"
    assert resolve_figures_output_dir(cfg) == tmp_path / "microlens_guide_photsim7_6s_fsg-results" / "frame000000" / "figures"


def test_resolve_output_dirs_respect_explicit_project_output_dir(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    explicit_debug_dir = tmp_path / "custom-debug"
    cfg = {
        "project": {"output_dir": str(explicit_debug_dir)},
        "guide_init": {"dataset_root": str(dataset_root)},
    }

    assert resolve_fsg_results_root(cfg) == explicit_debug_dir
    assert resolve_debug_output_dir(cfg) == explicit_debug_dir
    assert resolve_figures_output_dir(cfg) == explicit_debug_dir / "figures"


def test_resolve_output_dirs_treat_parent_relative_output_as_explicit(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    cfg = {
        "project": {"output_dir": "../outputs/debug"},
        "guide_init": {"dataset_root": str(dataset_root), "frame_index": 0},
    }

    assert resolve_fsg_results_root(cfg) == Path("../outputs/debug")
    assert resolve_debug_output_dir(cfg) == Path("../outputs/debug")
    assert resolve_figures_output_dir(cfg) == Path("../outputs/debug") / "figures"


def test_resolve_output_dirs_keep_default_when_dataset_root_missing(tmp_path):
    missing_dataset_root = tmp_path / "missing-dataset"
    cfg = {
        "project": {"output_dir": "outputs/debug"},
        "guide_init": {"dataset_root": str(missing_dataset_root)},
    }

    assert resolve_fsg_results_root(cfg) == Path("outputs/debug")
    assert resolve_debug_output_dir(cfg) == Path("outputs/debug")
    assert resolve_figures_output_dir(cfg) == Path("outputs/debug") / "figures"


def test_attitude_debug_artifacts_write_attitude_subdirectory(tmp_path):
    solution_payload = {
        "valid": True,
        "num_matched": 8,
        "num_rejected": 1,
        "q_ib": [1.0, 0.0, 0.0, 0.0],
        "residual_rms_arcsec": 0.5,
        "residual_max_arcsec": 1.0,
        "quality_flag": "VALID",
        "degraded_level": "NORMAL_4D",
        "active_detector_ids": [0, 1, 2, 3],
        "solver_iterations": 2,
        "covariance_rad2": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        "sigma_non_roll_arcsec": 0.1,
        "sigma_roll_arcsec": 0.2,
        "attitude_condition_number": 3.0,
        "quality": {
            "meta": {
                "attitude_covariance": {"available": True, "reason": None},
                "robust_rejection": {
                    "enabled": True,
                    "mode": "sigma_clip_iterative",
                    "num_rejected": 1,
                    "rejected_stars": [{"catalog_id": 42, "reasons": ["sigma_clip"]}],
                },
            }
        },
    }

    _write_attitude_debug_artifacts(tmp_path, solution_payload)

    attitude_dir = tmp_path / "attitude"
    assert attitude_dir.is_dir()
    robust = json.loads((attitude_dir / "robust_rejection.json").read_text())
    covariance = json.loads((attitude_dir / "covariance.json").read_text())
    summary = json.loads((attitude_dir / "solution_summary.json").read_text())
    assert robust["rejected_stars"][0]["catalog_id"] == 42
    assert covariance["attitude_condition_number"] == 3.0
    assert summary["quality_flag"] == "VALID"


def test_save_matching_overlays_writes_pngs_and_summary_counts(tmp_path):
    observed = [
        ObservedStar(
            detector_id="detA",
            source_id="detA:1",
            x=5.0,
            y=6.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=100.0,
            snr=20.0,
        ),
        ObservedStar(
            detector_id="detA",
            source_id="detA:2",
            x=15.0,
            y=16.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=50.0,
            snr=10.0,
        ),
    ]
    reference = [
        ReferenceStar(
            catalog_id=101,
            time_s=0.0,
            los_inertial=np.array([0.0, 0.0, 1.0]),
            mag_g=10.0,
            detector_ids_visible=["detA"],
            predicted_xy={"detA": (5.5, 6.5)},
            predicted_valid={"detA": True},
            weight_hint=1.0,
        ),
        ReferenceStar(
            catalog_id=102,
            time_s=0.0,
            los_inertial=np.array([0.0, 0.0, 1.0]),
            mag_g=11.0,
            detector_ids_visible=["detA"],
            predicted_xy={"detA": (10.0, 11.0)},
            predicted_valid={"detA": True},
            weight_hint=1.0,
        ),
    ]
    matching = MatchingResult(
        matched=[
            MatchedStar(
                detector_id="detA",
                source_id="detA:1",
                catalog_id=101,
                los_body=observed[0].los_body,
                los_inertial=reference[0].los_inertial,
                match_score=1.0,
                flags={
                    "match_mode": "predicted_position",
                    "observed_xy": (5.0, 6.0),
                    "predicted_xy": (5.5, 6.5),
                },
            )
        ],
        unmatched_observed_ids=["detA:2"],
        unmatched_catalog_ids=[102],
        mode="init",
        success=True,
        score=1.0,
        debug={"selected_strategy": "predicted_position"},
    )
    result = {
        "matching": matching,
        "debug_context": {
            "detectors": {"detA": {"image": np.ones((24, 24), dtype=np.float64)}},
            "observed_stars": observed,
            "reference_stars": reference,
        },
    }

    summary = save_matching_overlays(result, tmp_path)

    assert summary["detectors"]["detA"]["num_matched_observed"] == 1
    assert summary["detectors"]["detA"]["num_unmatched_observed"] == 1
    assert summary["detectors"]["detA"]["num_matched_reference"] == 1
    assert summary["detectors"]["detA"]["num_unmatched_reference"] == 1
    assert summary["detectors"]["detA"]["selected_strategy"] == "predicted_position"
    assert Path(summary["detectors"]["detA"]["overlay_path"]).exists()
    assert Path(summary["detectors"]["detA"]["overlay_path"]).stat().st_size > 0
    assert (tmp_path / "matching_overlay_summary.json").exists()


def test_save_matching_overlays_handles_integer_detector_keys(tmp_path):
    observed = [
        ObservedStar(
            detector_id=0,
            source_id=1,
            x=5.0,
            y=6.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=100.0,
            snr=20.0,
        )
    ]
    reference = [
        ReferenceStar(
            catalog_id=101,
            time_s=0.0,
            los_inertial=np.array([0.0, 0.0, 1.0]),
            mag_g=10.0,
            detector_ids_visible=[0],
            predicted_xy={0: (5.5, 6.5)},
            predicted_valid={0: True},
            weight_hint=1.0,
        )
    ]
    matching = MatchingResult(
        matched=[
            MatchedStar(
                detector_id=0,
                source_id=1,
                catalog_id=101,
                los_body=observed[0].los_body,
                los_inertial=reference[0].los_inertial,
                match_score=1.0,
                flags={"observed_xy": (5.0, 6.0), "predicted_xy": (5.5, 6.5)},
            )
        ],
        unmatched_observed_ids=[],
        unmatched_catalog_ids=[],
        mode="init",
        success=True,
        score=1.0,
        debug={"selected_strategy": "predicted_position"},
    )
    result = {
        "matching": matching,
        "debug_context": {
            "detectors": {0: {"image": np.ones((24, 24), dtype=np.float64)}},
            "observed_stars": observed,
            "reference_stars": reference,
        },
    }

    summary = save_matching_overlays(result, tmp_path)

    assert summary["detectors"]["0"]["num_matched_observed"] == 1
    assert summary["detectors"]["0"]["num_unmatched_reference"] == 0
    assert Path(summary["detectors"]["0"]["overlay_path"]).exists()


def test_run_guide_first_frame_init_debug_context_is_opt_in(tmp_path, monkeypatch):
    observed = [
        ObservedStar(
            detector_id="detA",
            source_id="detA:1",
            x=5.0,
            y=6.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=100.0,
            snr=20.0,
        )
    ]
    reference = [
        ReferenceStar(
            catalog_id=101,
            time_s=0.0,
            los_inertial=np.array([0.0, 0.0, 1.0]),
            mag_g=10.0,
            detector_ids_visible=["detA"],
            predicted_xy={"detA": (5.5, 6.5)},
            predicted_valid={"detA": True},
            weight_hint=1.0,
        )
    ]
    matching = MatchingResult(
        matched=[
            MatchedStar(
                detector_id="detA",
                source_id="detA:1",
                catalog_id=101,
                los_body=observed[0].los_body,
                los_inertial=reference[0].los_inertial,
                match_score=1.0,
            )
        ],
        unmatched_observed_ids=[],
        unmatched_catalog_ids=[],
        mode="init",
        success=True,
        score=1.0,
    )
    cfg = {
        "guide_init": {
            "dataset_root": str(tmp_path),
            "detector_batches": [{"detector_id": "detA", "batch_name": "batch0"}],
            "frame_index": 0,
            "reference_topk_per_detector": 1,
            "catalog_g_mag_max": 16.0,
        },
        "match": {},
        "attitude": {},
    }
    detector_stats = {
        "detA": {
            "batch_name": "batch0",
            "frame_path": str(tmp_path / "frame.npz"),
            "num_candidates_raw": 1,
            "num_candidates_selected": 1,
        }
    }
    detector_contexts = {
        "detA": {
            "raw": SimpleNamespace(image=np.ones((2, 2))),
            "preprocessed": SimpleNamespace(image=np.ones((2, 2))),
            "frame_path": str(tmp_path / "frame.npz"),
            "batch_path": str(tmp_path / "batch0"),
            "num_candidates_raw": 1,
            "num_candidates_selected": 1,
        }
    }
    reference_stats = {
        "detA": {
            "num_reference_stars": 1,
            "num_reference_preselected": 1,
            "num_reference_isolated": 1,
            "preselect_topk": 1,
            "isolation_radius_pix": None,
        }
    }
    geometry_adapter = SimpleNamespace(
        serialize=lambda: {
            "mode": "exact_et_focalplane",
            "rotation_body_from_eq": np.eye(3).tolist(),
            "frame_alignment_grid_size": 1,
            "frame_alignment_fit_rms_arcsec": 0.0,
            "frame_alignment_fit_max_arcsec": 0.0,
        }
    )

    monkeypatch.setattr(run_guide_init, "_load_et_coord", lambda _cfg: (object(), object(), object(), object()))
    monkeypatch.setattr(
        run_guide_init,
        "build_exact_focalplane_geometry_adapter",
        lambda *_args, **_kwargs: geometry_adapter,
    )
    monkeypatch.setattr(
        run_guide_init,
        "_build_sim_to_detector_map",
        lambda *_args: {
            "kind": "offset",
            "schema_version": 2,
            "offset_x_pix": 0.0,
            "offset_y_pix": 0.0,
            "image_center_pix": 1.0,
            "guide_query_target_center_xpix": 1.0,
            "guide_query_target_center_ypix": 1.0,
        },
    )
    monkeypatch.setattr(run_guide_init, "_build_observed_stars", lambda *_args: (observed, detector_stats.copy(), detector_contexts))
    monkeypatch.setattr(run_guide_init, "_build_reference_stars", lambda *_args: (reference, reference_stats))
    monkeypatch.setattr(run_guide_init, "match_stars", lambda *_args: matching)
    monkeypatch.setattr(
        run_guide_init,
        "solve_attitude",
        lambda *_args: AttitudeSolution(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            c_ib=np.eye(3),
            euler_zyx=None,
            valid=True,
            mode="init",
            num_matched=1,
            residual_rms_arcsec=0.0,
            residual_max_arcsec=0.0,
        ),
    )
    monkeypatch.setattr(run_guide_init, "compute_guide_error_audit", lambda *_args: {"enabled": False})

    default_result = run_guide_init.run_guide_first_frame_init(cfg)
    debug_result = run_guide_init.run_guide_first_frame_init(cfg, include_debug_context=True)

    assert "debug_context" not in default_result
    assert "body_model" not in default_result
    assert "geometry_model" not in default_result
    assert default_result["geometry_adapter"]["mode"] == "exact_et_focalplane"
    assert debug_result["debug_context"]["observed_stars"] is observed


def test_build_observed_stars_passes_loaded_calibration_to_preprocess(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    frame_dir = dataset_root / "batch0" / "frames"
    frame_dir.mkdir(parents=True)
    (frame_dir / "frame000.npz").write_bytes(b"placeholder")
    raw = SimpleNamespace(image=np.ones((3, 3)), detector_id="detA")
    candidate = SimpleNamespace(
        source_id=7,
        x=1.0,
        y=2.0,
        flux=50.0,
        snr=9.0,
        area=1,
        flags={"centroid_method": "test"},
    )
    transformed = SimpleNamespace(
        x_mm=0.1,
        y_mm=0.2,
        field_x_deg=0.3,
        field_y_deg=0.4,
    )
    geometry_adapter = SimpleNamespace(
        pixel_to_focal=lambda *_args: transformed,
        pixel_to_body_los=lambda *_args: np.array([0.0, 0.0, 1.0]),
    )
    expected_calib = {"bias": np.ones((3, 3))}
    seen = {}

    def fake_preprocess(raw_arg, calib, cfg):
        seen["raw"] = raw_arg
        seen["calib"] = calib
        seen["cfg"] = cfg
        return SimpleNamespace(image=np.ones((3, 3)))

    monkeypatch.setattr(run_guide_init, "load_npz_frame", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(run_guide_init, "preprocess_frame", fake_preprocess)
    monkeypatch.setattr(run_guide_init, "extract_stars", lambda *_args, **_kwargs: [candidate])

    observed, detector_stats, detector_contexts = run_guide_init._build_observed_stars(
        {
            "guide_init": {
                "dataset_root": str(dataset_root),
                "detector_batches": [{"detector_id": "detA", "batch_name": "batch0"}],
                "frame_index": 0,
            }
        },
        transformer=object(),
        sim_to_detector_map={
            "detA": {
                "kind": "offset",
                "offset_x_pix": 10.0,
                "offset_y_pix": 20.0,
            }
        },
        geometry_adapter=geometry_adapter,
        calib=expected_calib,
    )

    assert seen["calib"] is expected_calib
    assert observed[0].x == 11.0
    assert observed[0].y == 22.0
    assert detector_stats["detA"]["num_candidates_selected"] == 1
    assert detector_contexts["detA"]["raw"] is raw


def test_build_observed_stars_converts_adu_frame_before_extraction(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    frame_dir = dataset_root / "batch0" / "frames"
    frame_dir.mkdir(parents=True)
    (frame_dir / "frame000.npz").write_bytes(b"placeholder")

    image = np.zeros((9, 9), dtype=np.float64)
    image[4, 4] = 100.0
    image[4, 5] = 60.0
    image[5, 4] = 40.0
    raw = RawFrame(
        detector_id="detA",
        image=image,
        time_s=0.0,
        cadence_s=1.0,
        unit=None,
    )

    def pixel_to_body_los(_detector_id, x, y):
        los = np.array([float(x) * 1e-3, float(y) * 1e-3, 1.0], dtype=np.float64)
        return los / np.linalg.norm(los)

    geometry_adapter = SimpleNamespace(
        pixel_to_focal=lambda _detector_id, x, y: SimpleNamespace(
            x_mm=float(x) * 0.01,
            y_mm=float(y) * 0.01,
            field_x_deg=float(x) * 1e-3,
            field_y_deg=float(y) * 1e-3,
        ),
        pixel_to_body_los=pixel_to_body_los,
    )

    cfg = {
        "guide_init": {
            "dataset_root": str(dataset_root),
            "detector_batches": [{"detector_id": "detA", "batch_name": "batch0"}],
            "frame_index": 0,
            "max_observed_per_detector": 5,
        },
        "detector": {
            "adc_bit_depth": 12,
            "adc_min_value": 0.0,
        },
        "preprocess": {
            "enable_adc_clip": True,
            "enable_saturation_guard": False,
            "enable_bias_subtraction": False,
            "enable_fpn_subtraction": False,
            "convert_to_electrons": True,
            "gain_e_per_dn": 0.5,
            "enable_dark_subtraction": False,
            "enable_flat_field": False,
            "enable_bad_pixel_mask": False,
            "enable_background_subtraction": False,
            "variance_model": "poisson_read_noise",
            "read_noise_e": 1.0,
            "quantization_noise_e": 0.0,
        },
        "extract": {
            "detection_image": "snr",
            "seed_threshold_sigma": 5.0,
            "grow_threshold_sigma": 3.0,
            "min_area": 1,
            "max_area": 20,
            "centroid_method": "weighted_centroid",
            "centroid_window": {"center": "peak", "size": 5},
            "bbox_expand": 0,
            "reject_edge_margin": 0,
            "max_ellipticity": 1.0,
            "reject_degenerate_sources": False,
            "max_sharpness": None,
            "reject_artifact_mask_overlap": False,
            "centroid_covariance": {"min_sigma_pix": 0.0, "jacobian_step_pix": 0.01},
            "deblend": {"enabled": True, "policy": "flag_only"},
        },
        "attitude": {"weight_mode": "variance_snr_hybrid"},
    }

    monkeypatch.setattr(run_guide_init, "load_npz_frame", lambda *_args, **_kwargs: raw)

    observed, detector_stats, detector_contexts = run_guide_init._build_observed_stars(
        cfg,
        transformer=object(),
        sim_to_detector_map={
            "detA": {
                "kind": "offset",
                "offset_x_pix": 0.0,
                "offset_y_pix": 0.0,
            }
        },
        geometry_adapter=geometry_adapter,
        calib={},
    )

    preprocessed = detector_contexts["detA"]["preprocessed"]
    conversion = preprocessed.preprocess_meta["adu_to_electron_conversion"]
    assert conversion["applied"] is True
    assert conversion["reason"] == "missing_unit_defaulted_to_adu"
    assert conversion["gain_e_per_dn"] == 0.5
    assert preprocessed.preprocess_meta["output_unit"] == "electron"
    assert np.isclose(preprocessed.image[4, 4], 50.0)
    assert detector_stats["detA"]["num_candidates_selected"] == 1
    assert len(observed) == 1
    assert observed[0].flux > 0.0
    assert observed[0].flags["weight_source"] == "variance_snr_hybrid"


def test_run_guide_first_frame_truth_noise_uses_adapter_output_only(tmp_path, monkeypatch):
    observed = [
        ObservedStar(
            detector_id="detA",
            source_id="detA:1",
            x=5.0,
            y=6.0,
            los_body=np.array([0.0, 0.0, 1.0]),
            flux=100.0,
            snr=20.0,
        )
    ]
    reference = [
        ReferenceStar(
            catalog_id=101,
            time_s=0.0,
            los_inertial=np.array([0.0, 0.0, 1.0]),
            mag_g=10.0,
            detector_ids_visible=["detA"],
            predicted_xy={"detA": (5.5, 6.5)},
            predicted_valid={"detA": True},
            weight_hint=1.0,
        )
    ]
    matching = MatchingResult(
        matched=[
            MatchedStar(
                detector_id="detA",
                source_id="detA:1",
                catalog_id=101,
                los_body=observed[0].los_body,
                los_inertial=reference[0].los_inertial,
                match_score=1.0,
            )
        ],
        unmatched_observed_ids=[],
        unmatched_catalog_ids=[],
        mode="init",
        success=True,
        score=1.0,
    )
    cfg = {
        "guide_truth_noise": {
            "dataset_root": str(tmp_path),
            "detector_batches": [{"detector_id": "detA", "batch_name": "batch0"}],
            "frame_index": 0,
            "reference_topk_per_detector": 1,
            "catalog_g_mag_max": 16.0,
            "centroid_noise_sigma_pix": 0.0,
        },
        "match": {},
        "attitude": {},
    }
    detector_stats = {
        "detA": {
            "batch_name": "batch0",
            "frame_path": str(tmp_path / "frame.npz"),
            "num_truth_stars_visible": 1,
            "num_candidates_raw": 1,
            "num_candidates_selected": 1,
        }
    }
    detector_contexts = {
        "detA": {
            "raw": SimpleNamespace(image=np.ones((2, 2))),
            "frame_path": str(tmp_path / "frame.npz"),
            "batch_path": str(tmp_path / "batch0"),
            "all_candidates": [SimpleNamespace(source_id=1)],
            "selected_candidates": [SimpleNamespace(source_id=1)],
            "num_candidates_raw": 1,
            "num_candidates_selected": 1,
        }
    }
    reference_stats = {
        "detA": {
            "num_reference_stars": 1,
            "num_reference_preselected": 1,
            "num_reference_isolated": 1,
            "preselect_topk": 1,
            "isolation_radius_pix": None,
        }
    }
    geometry_adapter = SimpleNamespace(
        serialize=lambda: {
            "mode": "exact_et_focalplane",
            "rotation_body_from_eq": np.eye(3).tolist(),
            "frame_alignment_grid_size": 1,
            "frame_alignment_fit_rms_arcsec": 0.0,
            "frame_alignment_fit_max_arcsec": 0.0,
        }
    )
    sim_to_detector_map = {
        "kind": "offset",
        "schema_version": 2,
        "offset_x_pix": 0.0,
        "offset_y_pix": 0.0,
        "image_center_pix": 1.0,
        "guide_query_target_center_xpix": 1.0,
        "guide_query_target_center_ypix": 1.0,
    }

    monkeypatch.setattr(
        run_guide_truth_noise,
        "_load_et_coord",
        lambda _cfg: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        run_guide_truth_noise,
        "build_exact_focalplane_geometry_adapter",
        lambda *_args, **_kwargs: geometry_adapter,
    )
    monkeypatch.setattr(
        run_guide_truth_noise,
        "_build_sim_to_detector_map",
        lambda *_args: sim_to_detector_map,
    )
    monkeypatch.setattr(
        run_guide_truth_noise,
        "_build_truth_noise_observed",
        lambda *_args: (observed, detector_stats.copy(), detector_contexts),
    )
    monkeypatch.setattr(run_guide_truth_noise, "_build_reference_stars", lambda *_args: (reference, reference_stats))
    monkeypatch.setattr(run_guide_truth_noise, "match_stars", lambda *_args: matching)
    monkeypatch.setattr(
        run_guide_truth_noise,
        "solve_attitude",
        lambda *_args: AttitudeSolution(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            c_ib=np.eye(3),
            euler_zyx=None,
            valid=True,
            mode="init",
            num_matched=1,
            residual_rms_arcsec=0.0,
            residual_max_arcsec=0.0,
        ),
    )
    monkeypatch.setattr(run_guide_truth_noise, "compute_guide_error_audit", lambda *_args: {"enabled": False})

    result = run_guide_truth_noise.run_guide_first_frame_truth_noise(cfg)
    debug_result = run_guide_truth_noise.run_guide_first_frame_truth_noise(cfg, include_debug_context=True)

    assert "body_model" not in result
    assert "geometry_model" not in result
    assert "debug_context" not in result
    assert result["geometry_adapter"]["mode"] == "exact_et_focalplane"
    assert debug_result["debug_context"]["observed_stars"] is observed
    assert debug_result["debug_context"]["reference_stars"] is reference
    assert debug_result["debug_context"]["detectors"]["detA"]["all_candidates"] == detector_contexts["detA"]["all_candidates"]
