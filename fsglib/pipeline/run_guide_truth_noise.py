from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

from fsglib.attitude.solver import solve_attitude
from fsglib.common.io import load_npz_frame
from fsglib.common.types import AttitudeSolveInput, MatchingContext, ObservedStar, StarCandidate
from fsglib.ephemeris.guide_geometry import build_exact_focalplane_geometry_adapter
from fsglib.match.pipeline import match_stars
from fsglib.pipeline.convert import observed_weight_from_sigma, propagate_centroid_covariance
from fsglib.pipeline.guide_error_audit import compute_guide_error_audit
from fsglib.pipeline.run_guide_init import (
    _build_reference_stars,
    _build_sim_to_detector_map,
    _frame_path,
    _guide_entries,
    _load_et_coord,
    _select_candidates_for_attitude,
)


def _helper_cfg(cfg: dict) -> dict:
    helper_cfg = dict(cfg)
    helper_cfg["guide_init"] = dict(cfg["guide_truth_noise"])
    return helper_cfg


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truth_table_lookup(batch_path: Path) -> dict[int, dict[str, Any]]:
    table_path = batch_path / "stars.ecsv"
    if not table_path.exists():
        return {}

    table = Table.read(table_path)
    lookup: dict[int, dict[str, Any]] = {}
    for row in table:
        truth_index = _safe_int(row["Truth Index"]) if "Truth Index" in table.colnames else None
        if truth_index is None:
            continue
        photon_count = _safe_float(row["Photon Count"]) if "Photon Count" in table.colnames else None
        kepler_mag = _safe_float(row["Kepler Mag"]) if "Kepler Mag" in table.colnames else None
        gaia_g_mag = _safe_float(row["Gaia G Mag"]) if "Gaia G Mag" in table.colnames else None
        brightness_proxy = photon_count
        if brightness_proxy is None:
            mag_value = kepler_mag if kepler_mag is not None else gaia_g_mag
            brightness_proxy = None if mag_value is None else float(10.0 ** (-0.4 * mag_value))
        snr_proxy = None if brightness_proxy is None else float(max(np.sqrt(max(brightness_proxy, 0.0)), 1.0))
        lookup[truth_index] = {
            "catalog_source_id": _safe_int(row["Source ID"]) if "Source ID" in table.colnames else None,
            "truth_star_id": _safe_int(row["Star ID"]) if "Star ID" in table.colnames else None,
            "photon_count": photon_count,
            "kepler_mag": kepler_mag,
            "gaia_g_mag": gaia_g_mag,
            "brightness_proxy": brightness_proxy,
            "snr_proxy": snr_proxy,
        }
    return lookup


def _resolve_truth_detector_xy(truth_star) -> tuple[float | None, float | None]:
    meta = truth_star.meta or {}
    for x_key, y_key in (
        ("truth_x_detector_pix", "truth_y_detector_pix"),
        ("truth_abs_x_detector_pix", "truth_abs_y_detector_pix"),
        ("truth_static_x_detector_pix", "truth_static_y_detector_pix"),
    ):
        x_val = _safe_float(meta.get(x_key))
        y_val = _safe_float(meta.get(y_key))
        if x_val is not None and y_val is not None:
            return x_val, y_val
    return None, None


def _build_truth_noise_observed(
    cfg: dict,
    transformer,
    sim_to_detector_map: dict[str, dict],
    geometry_adapter,
) -> tuple[list[ObservedStar], dict, dict]:
    guide_cfg = cfg["guide_truth_noise"]
    dataset_root = Path(guide_cfg["dataset_root"]).expanduser().resolve()
    frame_index = int(guide_cfg.get("frame_index", 0))
    noise_mean = float(guide_cfg.get("centroid_noise_mean_pix", 0.0))
    noise_sigma = float(guide_cfg["centroid_noise_sigma_pix"])
    random_seed = int(guide_cfg.get("random_seed", 0))
    rng = np.random.default_rng(random_seed)

    observed: list[ObservedStar] = []
    detector_stats: dict[str, dict] = {}
    detector_contexts: dict[str, dict] = {}

    for entry in _guide_entries(_helper_cfg(cfg)):
        detector_id = str(entry["detector_id"])
        batch_name = str(entry["batch_name"])
        batch_path = dataset_root / batch_name
        frame_path = _frame_path(batch_path, frame_index)
        raw = load_npz_frame(str(frame_path), detector_id=detector_id)
        truth_lookup = _truth_table_lookup(batch_path)

        all_candidates: list[StarCandidate] = []
        selected_truth: dict[int, dict[str, Any]] = {}

        for truth_star in list((raw.meta or {}).get("truth_stars", [])):
            truth_index = _safe_int((truth_star.meta or {}).get("truth_index", truth_star.source_id))
            if truth_index is None:
                continue

            truth_detector_x, truth_detector_y = _resolve_truth_detector_xy(truth_star)
            if truth_detector_x is None or truth_detector_y is None:
                continue

            truth_meta = truth_lookup.get(truth_index, {})
            brightness_proxy = truth_meta.get("brightness_proxy")
            if brightness_proxy is None:
                brightness_proxy = float(max(10.0 ** (-0.4 * float(truth_star.mag or 0.0)), 1e-6))
            snr_proxy = truth_meta.get("snr_proxy")
            if snr_proxy is None:
                snr_proxy = float(max(np.sqrt(max(brightness_proxy, 0.0)), 1.0))

            dx_pix, dy_pix = rng.normal(loc=noise_mean, scale=noise_sigma, size=2)
            sim_x = float(truth_star.x_pix) + float(dx_pix)
            sim_y = float(truth_star.y_pix) + float(dy_pix)

            peak_x = int(round(sim_x))
            peak_y = int(round(sim_y))
            candidate = StarCandidate(
                detector_id=detector_id,
                source_id=truth_index,
                x=sim_x,
                y=sim_y,
                flux=float(brightness_proxy),
                peak=float(brightness_proxy),
                area=1,
                snr=float(snr_proxy),
                bbox=(peak_x, peak_y, peak_x, peak_y),
                centroid_cov_pix=np.eye(2, dtype=np.float64) * noise_sigma**2,
                shape={},
                flags={
                    "synthetic_from_truth": True,
                    "truth_index": truth_index,
                    "truth_catalog_source_id": truth_meta.get("catalog_source_id"),
                    "truth_star_id": truth_meta.get("truth_star_id"),
                    "truth_detector_x_pix": float(truth_detector_x),
                    "truth_detector_y_pix": float(truth_detector_y),
                    "truth_sim_x_pix": float(truth_star.x_pix),
                    "truth_sim_y_pix": float(truth_star.y_pix),
                    "photon_count": truth_meta.get("photon_count"),
                    "kepler_mag": truth_meta.get("kepler_mag"),
                    "gaia_g_mag": truth_meta.get("gaia_g_mag"),
                    "brightness_proxy": float(brightness_proxy),
                    "injected_dx_pix": float(dx_pix),
                    "injected_dy_pix": float(dy_pix),
                    "noise_mean_pix": noise_mean,
                    "noise_sigma_pix": noise_sigma,
                    "noise_space": "detector_pixel",
                },
            )
            all_candidates.append(candidate)
            selected_truth[truth_index] = {
                "truth_detector_x_pix": float(truth_detector_x),
                "truth_detector_y_pix": float(truth_detector_y),
                "truth_sim_x_pix": float(truth_star.x_pix),
                "truth_sim_y_pix": float(truth_star.y_pix),
                "truth_catalog_source_id": truth_meta.get("catalog_source_id"),
                "injected_dx_pix": float(dx_pix),
                "injected_dy_pix": float(dy_pix),
            }

        candidates = _select_candidates_for_attitude(all_candidates, _helper_cfg(cfg))

        detector_stats[detector_id] = {
            "batch_name": batch_name,
            "frame_path": str(frame_path),
            "num_truth_stars_visible": len(all_candidates),
            "num_candidates_raw": len(all_candidates),
            "num_candidates_selected": len(candidates),
            "selection_mode": "truth_brightness_proxy",
            "noise_mean_pix": noise_mean,
            "noise_sigma_pix": noise_sigma,
            "random_seed": random_seed,
        }
        detector_contexts[detector_id] = {
            "batch_path": str(batch_path),
            "frame_path": str(frame_path),
            "raw": raw,
            "all_candidates": all_candidates,
            "selected_candidates": candidates,
            "num_candidates_raw": len(all_candidates),
            "num_candidates_selected": len(candidates),
        }

        for candidate in candidates:
            truth_record = selected_truth[int(candidate.source_id)]
            observed_x = float(truth_record["truth_detector_x_pix"]) + float(
                candidate.flags["injected_dx_pix"]
            )
            observed_y = float(truth_record["truth_detector_y_pix"]) + float(
                candidate.flags["injected_dy_pix"]
            )
            transformed = geometry_adapter.pixel_to_focal(detector_id, observed_x, observed_y)
            los_cov_body, sigma_angle_arcsec = propagate_centroid_covariance(
                geometry_adapter,
                detector_id,
                observed_x,
                observed_y,
                candidate.centroid_cov_pix,
                cfg,
            )
            weight, weight_flags = observed_weight_from_sigma(
                candidate.snr,
                sigma_angle_arcsec,
                cfg,
            )
            observed.append(
                ObservedStar(
                    detector_id=detector_id,
                    source_id=f"{detector_id}:{candidate.source_id}",
                    x=observed_x,
                    y=observed_y,
                    los_body=geometry_adapter.pixel_to_body_los(detector_id, observed_x, observed_y),
                    flux=float(candidate.flux),
                    snr=float(candidate.snr),
                    weight=weight,
                    centroid_cov_pix=candidate.centroid_cov_pix,
                    los_cov_body=los_cov_body,
                    sigma_angle_arcsec=sigma_angle_arcsec,
                    flags={
                        **candidate.flags,
                        **weight_flags,
                        "sigma_angle_arcsec": sigma_angle_arcsec,
                        "sim_x_pix": float(candidate.x),
                        "sim_y_pix": float(candidate.y),
                        "et_x_pix": observed_x,
                        "et_y_pix": observed_y,
                        "focal_x_mm": float(transformed.x_mm),
                        "focal_y_mm": float(transformed.y_mm),
                        "field_x_deg": float(transformed.field_x_deg),
                        "field_y_deg": float(transformed.field_y_deg),
                    },
                )
            )

    return observed, detector_stats, detector_contexts


def run_guide_first_frame_truth_noise(cfg: dict) -> dict:
    helper_cfg = _helper_cfg(cfg)
    registry, transformer, catalog, GaiaSourceFilter = _load_et_coord(helper_cfg)
    geometry_adapter = build_exact_focalplane_geometry_adapter(helper_cfg, registry, transformer)

    dataset_root = Path(cfg["guide_truth_noise"]["dataset_root"]).expanduser().resolve()
    sim_to_detector_map: dict[str, dict] = {}
    for entry in _guide_entries(helper_cfg):
        detector_id = str(entry["detector_id"])
        batch_path = dataset_root / str(entry["batch_name"])
        sim_to_detector_map[detector_id] = _build_sim_to_detector_map(batch_path, detector_id, transformer)

    observed, detector_stats, detector_contexts = _build_truth_noise_observed(
        cfg,
        transformer,
        sim_to_detector_map,
        geometry_adapter,
    )
    reference, reference_stats = _build_reference_stars(helper_cfg, registry, catalog, GaiaSourceFilter)

    match_ctx = MatchingContext(
        mode="init",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout=cfg.get("layout", {}),
        optical_model=cfg.get("layout", {}),
        matching_cfg=cfg.get("match", {}),
        reference_stars=reference,
    )
    matching = match_stars(match_ctx, reference, cfg)
    solution = solve_attitude(
        AttitudeSolveInput(
            time_s=0.0,
            matched_stars=matching.matched,
            prior_q_ib=None,
            mode="init",
            solver_cfg=cfg["attitude"],
        ),
        cfg,
    )

    error_audit = compute_guide_error_audit(
        helper_cfg,
        transformer,
        sim_to_detector_map,
        geometry_adapter,
        detector_contexts,
        observed,
        matching,
        solution,
    )

    matched_per_detector: dict[str, int] = {}
    for matched_star in matching.matched:
        key = str(matched_star.detector_id)
        matched_per_detector[key] = matched_per_detector.get(key, 0) + 1

    for detector_id, stats in detector_stats.items():
        reference_detector_stats = reference_stats[detector_id]
        stats["num_matched"] = matched_per_detector.get(detector_id, 0)
        stats["num_reference_stars"] = reference_detector_stats["num_reference_stars"]
        stats["num_reference_preselected"] = reference_detector_stats["num_reference_preselected"]
        stats["num_reference_isolated"] = reference_detector_stats["num_reference_isolated"]
        stats["reference_preselect_topk"] = reference_detector_stats["preselect_topk"]
        stats["reference_isolation_radius_pix"] = reference_detector_stats["isolation_radius_pix"]
        stats["sim_to_detector_kind"] = sim_to_detector_map[detector_id]["kind"]
        stats["schema_version"] = sim_to_detector_map[detector_id]["schema_version"]
        if sim_to_detector_map[detector_id]["kind"] == "offset":
            stats["offset_x_pix"] = sim_to_detector_map[detector_id]["offset_x_pix"]
            stats["offset_y_pix"] = sim_to_detector_map[detector_id]["offset_y_pix"]

    guide_cfg = cfg["guide_truth_noise"]
    geometry_payload = geometry_adapter.serialize()
    return {
        "solution": solution,
        "matching": matching,
        "observed_count": len(observed),
        "reference_count": len(reference),
        "detector_stats": detector_stats,
        "sim_to_detector_map": {
            detector_id: (
                {
                    "kind": "offset",
                    "schema_version": int(mapping["schema_version"]),
                    "offset_x_pix": float(mapping["offset_x_pix"]),
                    "offset_y_pix": float(mapping["offset_y_pix"]),
                    "image_center_pix": float(mapping["image_center_pix"]),
                    "guide_query_target_center_xpix": float(mapping["guide_query_target_center_xpix"]),
                    "guide_query_target_center_ypix": float(mapping["guide_query_target_center_ypix"]),
                }
                if mapping["kind"] == "offset"
                else {
                    "kind": "affine",
                    "schema_version": int(mapping["schema_version"]),
                    "image_center_pix": float(mapping["image_center_pix"]),
                    "x_coeffs": [float(value) for value in mapping["x_coeffs"]],
                    "y_coeffs": [float(value) for value in mapping["y_coeffs"]],
                    "num_fit_stars": int(mapping["num_fit_stars"]),
                    "fit_rms_pix": float(mapping["fit_rms_pix"]),
                    "fit_max_pix": float(mapping["fit_max_pix"]),
                }
            )
            for detector_id, mapping in sim_to_detector_map.items()
        },
        "geometry_adapter": geometry_payload,
        "synthetic_centroid_model": {
            "mode": "truth_detector_gaussian",
            "noise_mean_pix": float(guide_cfg.get("centroid_noise_mean_pix", 0.0)),
            "noise_sigma_pix": float(guide_cfg["centroid_noise_sigma_pix"]),
            "noise_space": str(guide_cfg.get("centroid_noise_space", "detector_pixel")),
            "random_seed": int(guide_cfg.get("random_seed", 0)),
            "selection_mode": "truth_brightness_proxy",
            "los_geometry_mode": str(guide_cfg.get("los_geometry_mode", "exact_et_focalplane")),
        },
        "error_audit": error_audit,
        "meta": {
            "dataset_root": str(dataset_root),
            "frame_index": int(guide_cfg.get("frame_index", 0)),
            "los_geometry_mode": str(guide_cfg.get("los_geometry_mode", "exact_et_focalplane")),
            "reference_topk_per_detector": int(guide_cfg["reference_topk_per_detector"]),
            "reference_preselect_topk_per_detector": int(
                guide_cfg.get(
                    "reference_preselect_topk_per_detector",
                    guide_cfg["reference_topk_per_detector"],
                )
            ),
            "reference_isolation_radius_pix": (
                None
                if guide_cfg.get("reference_isolation_radius_pix") is None
                else float(guide_cfg["reference_isolation_radius_pix"])
            ),
            "catalog_g_mag_min": (
                None
                if guide_cfg.get("catalog_g_mag_min") is None
                else float(guide_cfg["catalog_g_mag_min"])
            ),
            "catalog_g_mag_max": float(guide_cfg["catalog_g_mag_max"]),
            "max_observed_per_detector": int(guide_cfg.get("max_observed_per_detector", 0)),
        },
    }
