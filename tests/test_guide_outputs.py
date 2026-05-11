from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fsglib.common.types import AttitudeSolution, MatchedStar, MatchingResult, ObservedStar
from fsglib.ephemeris.types import ReferenceStar
from fsglib.pipeline import run_guide_init
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
    assert default_result["geometry_adapter"]["mode"] == "exact_et_focalplane"
    assert debug_result["debug_context"]["observed_stars"] is observed
