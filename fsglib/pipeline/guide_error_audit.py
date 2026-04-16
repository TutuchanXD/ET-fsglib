from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table
from scipy.spatial.transform import Rotation

from fsglib.attitude.solver import quat_to_dcm, solve_attitude
from fsglib.common.coords import radec_to_unit_vector


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


def _scalar_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "rms": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _vector_stats(dx_values: list[float], dy_values: list[float]) -> dict[str, float | int | None]:
    dx = np.asarray([float(value) for value in dx_values if value is not None and np.isfinite(value)], dtype=np.float64)
    dy = np.asarray([float(value) for value in dy_values if value is not None and np.isfinite(value)], dtype=np.float64)
    if dx.size == 0 or dy.size == 0:
        return {
            "count": 0,
            "mean_dx": None,
            "mean_dy": None,
            "mean_abs_dx": None,
            "mean_abs_dy": None,
            "rms_dx": None,
            "rms_dy": None,
            "mean_radial": None,
            "median_radial": None,
            "rms_radial": None,
            "p95_radial": None,
            "max_radial": None,
        }
    radial = np.hypot(dx, dy)
    return {
        "count": int(dx.size),
        "mean_dx": float(np.mean(dx)),
        "mean_dy": float(np.mean(dy)),
        "mean_abs_dx": float(np.mean(np.abs(dx))),
        "mean_abs_dy": float(np.mean(np.abs(dy))),
        "rms_dx": float(np.sqrt(np.mean(dx**2))),
        "rms_dy": float(np.sqrt(np.mean(dy**2))),
        "mean_radial": float(np.mean(radial)),
        "median_radial": float(np.median(radial)),
        "rms_radial": float(np.sqrt(np.mean(radial**2))),
        "p95_radial": float(np.percentile(radial, 95)),
        "max_radial": float(np.max(radial)),
    }


def _angle_arcsec_between(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.asarray(a, dtype=np.float64)
    b_norm = np.asarray(b, dtype=np.float64)
    a_norm /= np.linalg.norm(a_norm)
    b_norm /= np.linalg.norm(b_norm)
    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)) * 3600.0)


def _rotation_delta_arcsec(c_a: np.ndarray | None, c_b: np.ndarray | None) -> float | None:
    if c_a is None or c_b is None:
        return None
    delta = Rotation.from_matrix(np.asarray(c_a, dtype=np.float64) @ np.asarray(c_b, dtype=np.float64).T)
    return float(np.degrees(delta.magnitude()) * 3600.0)


def _rotation_align_vector(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_norm = np.asarray(src, dtype=np.float64)
    dst_norm = np.asarray(dst, dtype=np.float64)
    src_norm /= np.linalg.norm(src_norm)
    dst_norm /= np.linalg.norm(dst_norm)
    cross = np.cross(src_norm, dst_norm)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(src_norm, dst_norm), -1.0, 1.0)

    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(src_norm, fallback)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src_norm, fallback)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()

    axis = cross / cross_norm
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _rotation_error_components(c_est: np.ndarray | None, c_ref: np.ndarray | None) -> dict[str, float | None]:
    if c_est is None or c_ref is None:
        return {
            "non_roll_arcsec": None,
            "roll_arcsec": None,
            "total_arcsec": None,
        }

    c_est = np.asarray(c_est, dtype=np.float64)
    c_ref = np.asarray(c_ref, dtype=np.float64)
    z_ref = c_ref[2]
    z_est = c_est[2]
    non_roll_arcsec = _angle_arcsec_between(z_ref, z_est)
    total_arcsec = _rotation_delta_arcsec(c_est, c_ref)

    align_rot = _rotation_align_vector(z_est, z_ref)
    x_ref = c_ref[0]
    x_est_aligned = align_rot @ c_est[0]
    x_est_proj = x_est_aligned - np.dot(x_est_aligned, z_ref) * z_ref
    norm = np.linalg.norm(x_est_proj)
    if norm < 1e-12:
        roll_arcsec = None
    else:
        x_est_proj /= norm
        signed_roll_rad = np.arctan2(
            np.dot(z_ref, np.cross(x_ref, x_est_proj)),
            np.clip(np.dot(x_ref, x_est_proj), -1.0, 1.0),
        )
        roll_arcsec = float(abs(np.degrees(signed_roll_rad) * 3600.0))

    return {
        "non_roll_arcsec": non_roll_arcsec,
        "roll_arcsec": roll_arcsec,
        "total_arcsec": total_arcsec,
    }


def _solution_summary(solution) -> dict[str, Any]:
    c_ib = solution.c_ib if solution.c_ib is not None else quat_to_dcm(solution.q_ib)
    return {
        "valid": bool(solution.valid),
        "num_matched": int(solution.num_matched),
        "num_rejected": int(solution.num_rejected),
        "residual_rms_arcsec": float(solution.residual_rms_arcsec),
        "residual_max_arcsec": float(solution.residual_max_arcsec),
        "quality_flag": str(solution.quality_flag),
        "degraded_level": str(solution.degraded_level),
        "q_ib": [float(value) for value in solution.q_ib],
        "c_ib": [[float(value) for value in row] for row in np.asarray(c_ib, dtype=np.float64)],
    }


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
        lookup[truth_index] = {
            "catalog_source_id": _safe_int(row["Source ID"]) if "Source ID" in table.colnames else None,
            "star_id": _safe_int(row["Star ID"]) if "Star ID" in table.colnames else None,
            "ecsv_detector_x_pix": _safe_float(row["Detector Xpix"]) if "Detector Xpix" in table.colnames else None,
            "ecsv_detector_y_pix": _safe_float(row["Detector Ypix"]) if "Detector Ypix" in table.colnames else None,
            "ecsv_detector_x_shifted_pix": (
                _safe_float(row["Detector Xpix Shifted"]) if "Detector Xpix Shifted" in table.colnames else None
            ),
            "ecsv_detector_y_shifted_pix": (
                _safe_float(row["Detector Ypix Shifted"]) if "Detector Ypix Shifted" in table.colnames else None
            ),
        }
    return lookup


def _resolve_truth_detector_xy(truth_star, mapping: dict) -> tuple[float | None, float | None]:
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

    x_image = _safe_float(meta.get("truth_x_image_pix", truth_star.x_pix))
    y_image = _safe_float(meta.get("truth_y_image_pix", truth_star.y_pix))
    if x_image is None or y_image is None:
        return None, None

    if mapping["kind"] == "offset":
        return (
            float(x_image) + float(mapping["offset_x_pix"]),
            float(y_image) + float(mapping["offset_y_pix"]),
        )

    sample = np.array([float(x_image), float(y_image), 1.0], dtype=np.float64)
    return (
        float(sample @ np.asarray(mapping["x_coeffs"], dtype=np.float64)),
        float(sample @ np.asarray(mapping["y_coeffs"], dtype=np.float64)),
    )


def _build_truth_records(
    detector_id: str,
    batch_path: Path,
    raw,
    mapping: dict,
    transformer,
    body_model: dict,
) -> list[dict[str, Any]]:
    truth_stars = list((raw.meta or {}).get("truth_stars", []))
    table_lookup = _truth_table_lookup(batch_path)
    body_rotation = np.asarray(body_model["rotation_body_from_eq"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    for truth_star in truth_stars:
        truth_index = _safe_int((truth_star.meta or {}).get("truth_index", truth_star.source_id))
        truth_detector_x, truth_detector_y = _resolve_truth_detector_xy(truth_star, mapping)
        mapped_detector_x = _safe_float(truth_star.x_pix)
        mapped_detector_y = _safe_float(truth_star.y_pix)
        if mapped_detector_x is not None and mapped_detector_y is not None:
            if mapping["kind"] == "offset":
                mapped_detector_x += float(mapping["offset_x_pix"])
                mapped_detector_y += float(mapping["offset_y_pix"])
            else:
                sample = np.array([mapped_detector_x, mapped_detector_y, 1.0], dtype=np.float64)
                mapped_detector_x = float(sample @ np.asarray(mapping["x_coeffs"], dtype=np.float64))
                mapped_detector_y = float(sample @ np.asarray(mapping["y_coeffs"], dtype=np.float64))

        focal_x_mm = None
        focal_y_mm = None
        truth_model_los = None
        if truth_detector_x is not None and truth_detector_y is not None:
            transformed = transformer.pixel_to_focal(detector_id, truth_detector_x, truth_detector_y)
            focal_x_mm = float(transformed.x_mm)
            focal_y_mm = float(transformed.y_mm)
            if str(body_model.get("mode", "body_model_proxy")) == "exact_et_focalplane":
                sky = transformer.pixel_to_sky(
                    detector_id,
                    truth_detector_x,
                    truth_detector_y,
                    frame="equatorial",
                )
                if sky.vector_xyz is not None:
                    los_eq = np.asarray(sky.vector_xyz, dtype=np.float64)
                    los_eq /= np.linalg.norm(los_eq)
                    truth_model_los = np.asarray(body_model["rotation_body_from_eq"], dtype=np.float64) @ los_eq
                    truth_model_los /= np.linalg.norm(truth_model_los)
            else:
                truth_model_los = np.asarray(body_model["focal_mm_to_body_vector"](focal_x_mm, focal_y_mm), dtype=np.float64)

        truth_los_inertial = radec_to_unit_vector(float(truth_star.ra_deg), float(truth_star.dec_deg))
        truth_exact_body = body_rotation @ truth_los_inertial
        truth_exact_body /= np.linalg.norm(truth_exact_body)

        table_meta = table_lookup.get(truth_index if truth_index is not None else -1, {})
        records.append(
            {
                "truth_index": truth_index,
                "catalog_source_id": table_meta.get("catalog_source_id"),
                "star_id": table_meta.get("star_id"),
                "truth_mag": _safe_float(truth_star.mag),
                "truth_ra_deg": float(truth_star.ra_deg),
                "truth_dec_deg": float(truth_star.dec_deg),
                "truth_sim_x_pix": _safe_float(truth_star.x_pix),
                "truth_sim_y_pix": _safe_float(truth_star.y_pix),
                "truth_detector_x_pix": truth_detector_x,
                "truth_detector_y_pix": truth_detector_y,
                "mapped_detector_x_pix": mapped_detector_x,
                "mapped_detector_y_pix": mapped_detector_y,
                "ecsv_detector_x_pix": table_meta.get("ecsv_detector_x_pix"),
                "ecsv_detector_y_pix": table_meta.get("ecsv_detector_y_pix"),
                "ecsv_detector_x_shifted_pix": table_meta.get("ecsv_detector_x_shifted_pix"),
                "ecsv_detector_y_shifted_pix": table_meta.get("ecsv_detector_y_shifted_pix"),
                "truth_los_inertial": truth_los_inertial,
                "truth_exact_body": truth_exact_body,
                "truth_model_los_body": truth_model_los,
                "truth_focal_x_mm": focal_x_mm,
                "truth_focal_y_mm": focal_y_mm,
                "truth_meta": dict(truth_star.meta or {}),
            }
        )
    return records


def _assign_candidates_to_truth(
    selected_candidates: list,
    truth_records: list[dict[str, Any]],
    truth_match_radius_pix: float,
) -> dict[int, dict[str, Any]]:
    if not selected_candidates or not truth_records:
        return {}

    truth_xy = np.asarray(
        [[record["truth_sim_x_pix"], record["truth_sim_y_pix"]] for record in truth_records],
        dtype=np.float64,
    )
    assignments: list[tuple[float, int, int]] = []
    for candidate_idx, candidate in enumerate(selected_candidates):
        dx = truth_xy[:, 0] - float(candidate.x)
        dy = truth_xy[:, 1] - float(candidate.y)
        dist = np.sqrt(dx * dx + dy * dy)
        truth_idx = int(np.argmin(dist))
        min_dist = float(dist[truth_idx])
        if min_dist <= float(truth_match_radius_pix):
            assignments.append((min_dist, truth_idx, candidate_idx))

    assignments.sort(key=lambda item: item[0])
    used_truth: set[int] = set()
    used_candidates: set[int] = set()
    result: dict[int, dict[str, Any]] = {}
    for _, truth_idx, candidate_idx in assignments:
        if truth_idx in used_truth or candidate_idx in used_candidates:
            continue
        used_truth.add(truth_idx)
        used_candidates.add(candidate_idx)
        result[candidate_idx] = truth_records[truth_idx]
    return result


def _collect_detector_summary(per_star_entries: list[dict[str, Any]], truth_records: list[dict[str, Any]]) -> dict[str, Any]:
    centroid_sim_dx = [entry["centroid_error_sim_dx_pix"] for entry in per_star_entries if entry["centroid_error_sim_dx_pix"] is not None]
    centroid_sim_dy = [entry["centroid_error_sim_dy_pix"] for entry in per_star_entries if entry["centroid_error_sim_dy_pix"] is not None]
    centroid_det_dx = [entry["centroid_error_detector_dx_pix"] for entry in per_star_entries if entry["centroid_error_detector_dx_pix"] is not None]
    centroid_det_dy = [entry["centroid_error_detector_dy_pix"] for entry in per_star_entries if entry["centroid_error_detector_dy_pix"] is not None]
    focal_dx = [entry["focal_error_dx_mm"] for entry in per_star_entries if entry["focal_error_dx_mm"] is not None]
    focal_dy = [entry["focal_error_dy_mm"] for entry in per_star_entries if entry["focal_error_dy_mm"] is not None]
    model_map_dx = [
        record["mapped_detector_x_pix"] - record["truth_detector_x_pix"]
        for record in truth_records
        if record["mapped_detector_x_pix"] is not None and record["truth_detector_x_pix"] is not None
    ]
    model_map_dy = [
        record["mapped_detector_y_pix"] - record["truth_detector_y_pix"]
        for record in truth_records
        if record["mapped_detector_y_pix"] is not None and record["truth_detector_y_pix"] is not None
    ]
    ecsv_npz_dx = [
        record["ecsv_detector_x_pix"] - record["truth_detector_x_pix"]
        for record in truth_records
        if record["ecsv_detector_x_pix"] is not None and record["truth_detector_x_pix"] is not None
    ]
    ecsv_npz_dy = [
        record["ecsv_detector_y_pix"] - record["truth_detector_y_pix"]
        for record in truth_records
        if record["ecsv_detector_y_pix"] is not None and record["truth_detector_y_pix"] is not None
    ]
    npz_minus_ecsv_dx = [
        record["truth_detector_x_pix"] - record["ecsv_detector_x_pix"]
        for record in truth_records
        if record["ecsv_detector_x_pix"] is not None and record["truth_detector_x_pix"] is not None
    ]
    npz_minus_ecsv_dy = [
        record["truth_detector_y_pix"] - record["ecsv_detector_y_pix"]
        for record in truth_records
        if record["ecsv_detector_y_pix"] is not None and record["truth_detector_y_pix"] is not None
    ]
    predicted_vs_ecsv_dx = [
        entry["predicted_vs_ecsv_detector_dx_pix"]
        for entry in per_star_entries
        if entry["predicted_vs_ecsv_detector_dx_pix"] is not None
    ]
    predicted_vs_ecsv_dy = [
        entry["predicted_vs_ecsv_detector_dy_pix"]
        for entry in per_star_entries
        if entry["predicted_vs_ecsv_detector_dy_pix"] is not None
    ]
    return {
        "counts": {
            "num_truth_stars": int(len(truth_records)),
            "num_selected_assigned": int(len(per_star_entries)),
            "num_matched_assigned": int(sum(bool(entry["matched"]) for entry in per_star_entries)),
            "num_correct_matches": int(sum(bool(entry["match_is_correct"]) for entry in per_star_entries if entry["matched"])),
            "num_incorrect_matches": int(sum(entry["matched"] and not entry["match_is_correct"] for entry in per_star_entries)),
            "num_selected_unmatched": int(sum(not entry["matched"] for entry in per_star_entries)),
        },
        "sim_to_detector_map_error_pix": _vector_stats(model_map_dx, model_map_dy),
        "ecsv_vs_npz_truth_detector_error_pix": _vector_stats(ecsv_npz_dx, ecsv_npz_dy),
        "npz_minus_ecsv_detector_offset_pix": _vector_stats(npz_minus_ecsv_dx, npz_minus_ecsv_dy),
        "centroid_error_sim_pix": _vector_stats(centroid_sim_dx, centroid_sim_dy),
        "centroid_error_detector_pix": _vector_stats(centroid_det_dx, centroid_det_dy),
        "centroid_error_focal_mm": _vector_stats(focal_dx, focal_dy),
        "body_error_centroid_arcsec": _scalar_stats(
            [entry["body_error_centroid_arcsec"] for entry in per_star_entries if entry["body_error_centroid_arcsec"] is not None]
        ),
        "body_error_geometry_arcsec": _scalar_stats(
            [entry["body_error_geometry_arcsec"] for entry in per_star_entries if entry["body_error_geometry_arcsec"] is not None]
        ),
        "body_error_total_arcsec": _scalar_stats(
            [entry["body_error_total_arcsec"] for entry in per_star_entries if entry["body_error_total_arcsec"] is not None]
        ),
        "body_error_gain_arcsec_per_pix": _scalar_stats(
            [entry["body_error_gain_arcsec_per_pix"] for entry in per_star_entries if entry["body_error_gain_arcsec_per_pix"] is not None]
        ),
        "match_observed_vs_predicted_pix": _scalar_stats(
            [entry["match_residual_pix"] for entry in per_star_entries if entry["match_residual_pix"] is not None]
        ),
        "match_predicted_vs_truth_pix": _vector_stats(
            [entry["predicted_vs_truth_dx_pix"] for entry in per_star_entries if entry["predicted_vs_truth_dx_pix"] is not None],
            [entry["predicted_vs_truth_dy_pix"] for entry in per_star_entries if entry["predicted_vs_truth_dy_pix"] is not None],
        ),
        "match_predicted_vs_ecsv_detector_pix": _vector_stats(predicted_vs_ecsv_dx, predicted_vs_ecsv_dy),
        "match_truth_catalog_sep_arcsec": _scalar_stats(
            [entry["truth_catalog_sep_arcsec"] for entry in per_star_entries if entry["truth_catalog_sep_arcsec"] is not None]
        ),
        "solution_residual_model_arcsec": _scalar_stats(
            [entry["solution_residual_model_arcsec"] for entry in per_star_entries if entry["solution_residual_model_arcsec"] is not None]
        ),
        "solution_residual_exact_body_arcsec": _scalar_stats(
            [entry["solution_residual_exact_body_arcsec"] for entry in per_star_entries if entry["solution_residual_exact_body_arcsec"] is not None]
        ),
    }


def _counterfactual_solutions(cfg: dict, matching, solution, per_star_by_source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth_pixel_matches = []
    exact_body_matches = []
    oracle_matches = []
    truth_backed_sources = 0

    for matched_star in matching.matched:
        observed_source_id = str(matched_star.source_id)
        entry = per_star_by_source.get(observed_source_id)
        truth_model_los = None if entry is None else entry.get("_truth_model_los_body")
        truth_exact_body = None if entry is None else entry.get("_truth_exact_body")
        truth_los_inertial = None if entry is None else entry.get("_truth_los_inertial")
        truth_catalog_source_id = None if entry is None else entry.get("truth_catalog_source_id")

        truth_pixel_matches.append(
            replace(
                matched_star,
                los_body=np.asarray(
                    truth_model_los if truth_model_los is not None else matched_star.los_body,
                    dtype=np.float64,
                ),
            )
        )
        exact_body_matches.append(
            replace(
                matched_star,
                los_body=np.asarray(
                    truth_exact_body if truth_exact_body is not None else matched_star.los_body,
                    dtype=np.float64,
                ),
            )
        )

        if truth_exact_body is not None and truth_los_inertial is not None:
            truth_backed_sources += 1
            oracle_matches.append(
                replace(
                    matched_star,
                    catalog_id=int(truth_catalog_source_id)
                    if truth_catalog_source_id is not None
                    else matched_star.catalog_id,
                    los_body=np.asarray(truth_exact_body, dtype=np.float64),
                    los_inertial=np.asarray(truth_los_inertial, dtype=np.float64),
                )
            )

    truth_pixel_solution = solve_attitude(truth_pixel_matches, cfg)
    exact_body_solution = solve_attitude(exact_body_matches, cfg)
    oracle_solution = solve_attitude(oracle_matches, cfg) if oracle_matches else None

    current_c = solution.c_ib if solution.c_ib is not None else quat_to_dcm(solution.q_ib)
    truth_pixel_c = (
        truth_pixel_solution.c_ib
        if truth_pixel_solution.c_ib is not None
        else quat_to_dcm(truth_pixel_solution.q_ib)
    )
    exact_body_c = (
        exact_body_solution.c_ib
        if exact_body_solution.c_ib is not None
        else quat_to_dcm(exact_body_solution.q_ib)
    )
    oracle_c = None
    if oracle_solution is not None:
        oracle_c = oracle_solution.c_ib if oracle_solution.c_ib is not None else quat_to_dcm(oracle_solution.q_ib)

    delta_current_to_truth_pixel = _rotation_error_components(current_c, truth_pixel_c)
    delta_truth_pixel_to_exact_body = _rotation_error_components(truth_pixel_c, exact_body_c)
    delta_current_to_exact_body = _rotation_error_components(current_c, exact_body_c)
    delta_current_to_oracle = _rotation_error_components(current_c, oracle_c)
    delta_exact_body_to_oracle = _rotation_error_components(exact_body_c, oracle_c)

    payload = {
        "num_current_matches": int(len(matching.matched)),
        "num_truth_backed_matches": int(truth_backed_sources),
        "current": _solution_summary(solution),
        # In exact_et_focalplane mode this is the closest proxy to the actual
        # simulated frame attitude, because it is built from NPZ detector truth.
        "frame_truth_same_matches": _solution_summary(truth_pixel_solution),
        # This keeps the old name for backward compatibility.
        "truth_pixel_same_matches": _solution_summary(truth_pixel_solution),
        # This is the nominal et_focalplane/body convention without the
        # simulator's telescope FOV offset baked into detector truth.
        "nominal_body_same_matches": _solution_summary(exact_body_solution),
        "exact_body_same_matches": _solution_summary(exact_body_solution),
        "delta_current_to_frame_truth_arcsec": delta_current_to_truth_pixel["total_arcsec"],
        "delta_current_to_truth_pixel_arcsec": delta_current_to_truth_pixel["total_arcsec"],
        "delta_frame_truth_to_nominal_body_arcsec": delta_truth_pixel_to_exact_body["total_arcsec"],
        "delta_truth_pixel_to_exact_body_arcsec": delta_truth_pixel_to_exact_body["total_arcsec"],
        "delta_current_to_nominal_body_arcsec": delta_current_to_exact_body["total_arcsec"],
        "delta_current_to_exact_body_arcsec": delta_current_to_exact_body["total_arcsec"],
        "delta_components": {
            "current_to_frame_truth": delta_current_to_truth_pixel,
            "current_to_truth_pixel": delta_current_to_truth_pixel,
            "frame_truth_to_nominal_body": delta_truth_pixel_to_exact_body,
            "truth_pixel_to_exact_body": delta_truth_pixel_to_exact_body,
            "current_to_nominal_body": delta_current_to_exact_body,
            "current_to_exact_body": delta_current_to_exact_body,
            "current_to_oracle": delta_current_to_oracle,
            "exact_body_to_oracle_match": delta_exact_body_to_oracle,
        },
        "interpretation": {
            "frame_truth_same_matches": (
                "Uses NPZ detector truth and is the recommended simulated-frame attitude reference."
            ),
            "nominal_body_same_matches": (
                "Uses nominal et_focalplane/body geometry without simulator telescope FOV offset."
            ),
        },
    }
    if oracle_solution is not None:
        payload["oracle_truth_body_and_match"] = _solution_summary(oracle_solution)
        payload["delta_exact_body_to_oracle_match_arcsec"] = delta_exact_body_to_oracle["total_arcsec"]
        payload["delta_current_to_oracle_arcsec"] = delta_current_to_oracle["total_arcsec"]
    else:
        payload["oracle_truth_body_and_match"] = None
        payload["delta_exact_body_to_oracle_match_arcsec"] = None
        payload["delta_current_to_oracle_arcsec"] = None
    return payload


def compute_guide_error_audit(
    cfg: dict,
    transformer,
    sim_to_detector_map: dict[str, dict],
    body_model: dict,
    detector_contexts: dict[str, dict[str, Any]],
    observed: list,
    matching,
    solution,
) -> dict[str, Any]:
    audit_cfg = dict(cfg.get("evaluation", {}).get("guide_error_audit", {}))
    if not audit_cfg.get("enabled", False):
        return {"enabled": False}

    truth_match_radius_pix = float(audit_cfg.get("truth_match_radius_pix", 3.0))
    observed_by_source = {str(star.source_id): star for star in observed}
    matched_by_source = {str(star.source_id): star for star in matching.matched}

    per_detector: dict[str, Any] = {}
    per_star: list[dict[str, Any]] = []
    per_star_by_source: dict[str, dict[str, Any]] = {}
    unassigned_selected: list[dict[str, Any]] = []

    for detector_id, ctx in detector_contexts.items():
        batch_path = Path(ctx["batch_path"])
        raw = ctx["raw"]
        selected_candidates = list(ctx["selected_candidates"])
        truth_records = _build_truth_records(
            detector_id,
            batch_path,
            raw,
            sim_to_detector_map[detector_id],
            transformer,
            body_model,
        )
        assigned = _assign_candidates_to_truth(selected_candidates, truth_records, truth_match_radius_pix)
        detector_entries: list[dict[str, Any]] = []

        for candidate_idx, candidate in enumerate(selected_candidates):
            observed_source_id = f"{detector_id}:{candidate.source_id}"
            observed_star = observed_by_source.get(observed_source_id)
            matched_star = matched_by_source.get(observed_source_id)
            truth_record = assigned.get(candidate_idx)
            if truth_record is None:
                unassigned_selected.append(
                    {
                        "detector_id": detector_id,
                        "observed_source_id": observed_source_id,
                        "candidate_source_id": int(candidate.source_id),
                        "snr": float(candidate.snr),
                        "flux": float(candidate.flux),
                        "sim_x_pix": float(candidate.x),
                        "sim_y_pix": float(candidate.y),
                    }
                )
                continue

            truth_detector_x = truth_record["truth_detector_x_pix"]
            truth_detector_y = truth_record["truth_detector_y_pix"]
            observed_x = _safe_float(observed_star.x if observed_star is not None else None)
            observed_y = _safe_float(observed_star.y if observed_star is not None else None)
            truth_focal_x = truth_record["truth_focal_x_mm"]
            truth_focal_y = truth_record["truth_focal_y_mm"]
            observed_focal_x = None if observed_star is None else _safe_float(observed_star.flags.get("focal_x_mm"))
            observed_focal_y = None if observed_star is None else _safe_float(observed_star.flags.get("focal_y_mm"))

            centroid_sim_dx = float(candidate.x) - float(truth_record["truth_sim_x_pix"])
            centroid_sim_dy = float(candidate.y) - float(truth_record["truth_sim_y_pix"])
            centroid_detector_dx = None
            centroid_detector_dy = None
            if observed_x is not None and observed_y is not None and truth_detector_x is not None and truth_detector_y is not None:
                centroid_detector_dx = observed_x - truth_detector_x
                centroid_detector_dy = observed_y - truth_detector_y

            focal_error_dx = None
            focal_error_dy = None
            if (
                observed_focal_x is not None
                and observed_focal_y is not None
                and truth_focal_x is not None
                and truth_focal_y is not None
            ):
                focal_error_dx = observed_focal_x - truth_focal_x
                focal_error_dy = observed_focal_y - truth_focal_y

            truth_model_los = truth_record["truth_model_los_body"]
            truth_exact_body = truth_record["truth_exact_body"]
            observed_los = None if observed_star is None else np.asarray(observed_star.los_body, dtype=np.float64)
            body_error_centroid = None
            body_error_geometry = None
            body_error_total = None
            body_error_gain = None
            if observed_los is not None and truth_model_los is not None:
                body_error_centroid = _angle_arcsec_between(observed_los, truth_model_los)
                if centroid_detector_dx is not None and centroid_detector_dy is not None:
                    radial_pix = float(np.hypot(centroid_detector_dx, centroid_detector_dy))
                    if radial_pix > 1e-9:
                        body_error_gain = float(body_error_centroid / radial_pix)
            if truth_model_los is not None and truth_exact_body is not None:
                body_error_geometry = _angle_arcsec_between(truth_model_los, truth_exact_body)
            if observed_los is not None and truth_exact_body is not None:
                body_error_total = _angle_arcsec_between(observed_los, truth_exact_body)

            predicted_xy = None if matched_star is None else matched_star.flags.get("predicted_xy")
            predicted_vs_truth_dx = None
            predicted_vs_truth_dy = None
            predicted_vs_ecsv_dx = None
            predicted_vs_ecsv_dy = None
            if predicted_xy is not None and truth_detector_x is not None and truth_detector_y is not None:
                predicted_vs_truth_dx = float(predicted_xy[0]) - truth_detector_x
                predicted_vs_truth_dy = float(predicted_xy[1]) - truth_detector_y
            if (
                predicted_xy is not None
                and truth_record["ecsv_detector_x_pix"] is not None
                and truth_record["ecsv_detector_y_pix"] is not None
            ):
                predicted_vs_ecsv_dx = float(predicted_xy[0]) - float(truth_record["ecsv_detector_x_pix"])
                predicted_vs_ecsv_dy = float(predicted_xy[1]) - float(truth_record["ecsv_detector_y_pix"])

            truth_catalog_sep_arcsec = None
            match_is_correct = None
            solution_residual_model = None
            solution_residual_exact = None
            if matched_star is not None:
                truth_catalog_sep_arcsec = _angle_arcsec_between(
                    np.asarray(matched_star.los_inertial, dtype=np.float64),
                    np.asarray(truth_record["truth_los_inertial"], dtype=np.float64),
                )
                truth_catalog_source_id = truth_record["catalog_source_id"]
                match_is_correct = (
                    truth_catalog_source_id is not None
                    and int(matched_star.catalog_id) == int(truth_catalog_source_id)
                )
                c_ib = solution.c_ib if solution.c_ib is not None else quat_to_dcm(solution.q_ib)
                model_body = np.asarray(matched_star.los_body, dtype=np.float64)
                solution_residual_model = _angle_arcsec_between(c_ib @ matched_star.los_inertial, model_body)
                solution_residual_exact = _angle_arcsec_between(c_ib @ matched_star.los_inertial, truth_exact_body)

            entry = {
                "detector_id": detector_id,
                "observed_source_id": observed_source_id,
                "candidate_source_id": int(candidate.source_id),
                "truth_index": truth_record["truth_index"],
                "truth_catalog_source_id": truth_record["catalog_source_id"],
                "truth_star_id": truth_record["star_id"],
                "truth_mag": truth_record["truth_mag"],
                "truth_ra_deg": truth_record["truth_ra_deg"],
                "truth_dec_deg": truth_record["truth_dec_deg"],
                "snr": float(candidate.snr),
                "flux": float(candidate.flux),
                "peak": float(candidate.peak),
                "area_pix": int(candidate.area),
                "matched": bool(matched_star is not None),
                "matched_catalog_id": None if matched_star is None else int(matched_star.catalog_id),
                "match_is_correct": match_is_correct,
                "truth_catalog_sep_arcsec": truth_catalog_sep_arcsec,
                "match_residual_pix": None if matched_star is None else _safe_float(matched_star.flags.get("residual_pix")),
                "solution_residual_model_arcsec": solution_residual_model,
                "solution_residual_exact_body_arcsec": solution_residual_exact,
                "truth_sim_x_pix": truth_record["truth_sim_x_pix"],
                "truth_sim_y_pix": truth_record["truth_sim_y_pix"],
                "sim_x_pix": float(candidate.x),
                "sim_y_pix": float(candidate.y),
                "raw_centroid_x_pix": _safe_float(candidate.flags.get("raw_centroid_x_pix")),
                "raw_centroid_y_pix": _safe_float(candidate.flags.get("raw_centroid_y_pix")),
                "centroid_error_sim_dx_pix": float(centroid_sim_dx),
                "centroid_error_sim_dy_pix": float(centroid_sim_dy),
                "centroid_error_sim_radial_pix": float(np.hypot(centroid_sim_dx, centroid_sim_dy)),
                "truth_detector_x_pix": truth_detector_x,
                "truth_detector_y_pix": truth_detector_y,
                "observed_detector_x_pix": observed_x,
                "observed_detector_y_pix": observed_y,
                "centroid_error_detector_dx_pix": centroid_detector_dx,
                "centroid_error_detector_dy_pix": centroid_detector_dy,
                "centroid_error_detector_radial_pix": (
                    None
                    if centroid_detector_dx is None or centroid_detector_dy is None
                    else float(np.hypot(centroid_detector_dx, centroid_detector_dy))
                ),
                "truth_focal_x_mm": truth_focal_x,
                "truth_focal_y_mm": truth_focal_y,
                "observed_focal_x_mm": observed_focal_x,
                "observed_focal_y_mm": observed_focal_y,
                "focal_error_dx_mm": focal_error_dx,
                "focal_error_dy_mm": focal_error_dy,
                "focal_error_radial_mm": (
                    None if focal_error_dx is None or focal_error_dy is None else float(np.hypot(focal_error_dx, focal_error_dy))
                ),
                "body_error_centroid_arcsec": body_error_centroid,
                "body_error_geometry_arcsec": body_error_geometry,
                "body_error_total_arcsec": body_error_total,
                "body_error_gain_arcsec_per_pix": body_error_gain,
                "predicted_x_pix": None if predicted_xy is None else float(predicted_xy[0]),
                "predicted_y_pix": None if predicted_xy is None else float(predicted_xy[1]),
                "predicted_vs_truth_dx_pix": predicted_vs_truth_dx,
                "predicted_vs_truth_dy_pix": predicted_vs_truth_dy,
                "predicted_vs_truth_radial_pix": (
                    None
                    if predicted_vs_truth_dx is None or predicted_vs_truth_dy is None
                    else float(np.hypot(predicted_vs_truth_dx, predicted_vs_truth_dy))
                ),
                "predicted_vs_ecsv_detector_dx_pix": predicted_vs_ecsv_dx,
                "predicted_vs_ecsv_detector_dy_pix": predicted_vs_ecsv_dy,
                "predicted_vs_ecsv_detector_radial_pix": (
                    None
                    if predicted_vs_ecsv_dx is None or predicted_vs_ecsv_dy is None
                    else float(np.hypot(predicted_vs_ecsv_dx, predicted_vs_ecsv_dy))
                ),
                "truth_map_dx_pix": (
                    None
                    if truth_record["mapped_detector_x_pix"] is None or truth_detector_x is None
                    else float(truth_record["mapped_detector_x_pix"] - truth_detector_x)
                ),
                "truth_map_dy_pix": (
                    None
                    if truth_record["mapped_detector_y_pix"] is None or truth_detector_y is None
                    else float(truth_record["mapped_detector_y_pix"] - truth_detector_y)
                ),
                "ecsv_vs_truth_detector_dx_pix": (
                    None
                    if truth_record["ecsv_detector_x_pix"] is None or truth_detector_x is None
                    else float(truth_record["ecsv_detector_x_pix"] - truth_detector_x)
                ),
                "ecsv_vs_truth_detector_dy_pix": (
                    None
                    if truth_record["ecsv_detector_y_pix"] is None or truth_detector_y is None
                    else float(truth_record["ecsv_detector_y_pix"] - truth_detector_y)
                ),
                "candidate_flags": dict(candidate.flags),
                "_truth_model_los_body": None
                if truth_model_los is None
                else [float(value) for value in np.asarray(truth_model_los, dtype=np.float64)],
                "_truth_exact_body": [float(value) for value in np.asarray(truth_exact_body, dtype=np.float64)],
                "_truth_los_inertial": [float(value) for value in np.asarray(truth_record["truth_los_inertial"], dtype=np.float64)],
            }
            detector_entries.append(entry)
            per_star.append(entry)
            per_star_by_source[observed_source_id] = entry

        assigned_ids = {f"{detector_id}:{candidate.source_id}" for idx, candidate in enumerate(selected_candidates) if idx in assigned}
        detector_summary = _collect_detector_summary(detector_entries, truth_records)
        detector_summary["counts"].update(
            {
                "num_selected_candidates": int(len(selected_candidates)),
                "num_selected_without_truth": int(len(selected_candidates) - len(assigned_ids)),
                "num_selected_with_truth": int(len(assigned_ids)),
            }
        )
        per_detector[detector_id] = detector_summary

    counterfactuals = _counterfactual_solutions(cfg, matching, solution, per_star_by_source)
    summary = {
        "counts": {
            "num_detectors": int(len(per_detector)),
            "num_observed_selected": int(sum(ctx["num_candidates_selected"] for ctx in detector_contexts.values())),
            "num_assigned_selected": int(len(per_star)),
            "num_selected_without_truth": int(len(unassigned_selected)),
            "num_matched": int(len(matching.matched)),
            "num_correct_matches": int(sum(bool(entry["match_is_correct"]) for entry in per_star if entry["matched"])),
            "num_incorrect_matches": int(sum(entry["matched"] and not entry["match_is_correct"] for entry in per_star)),
        },
        "sim_to_detector_map_error_pix": _vector_stats(
            [entry["truth_map_dx_pix"] for entry in per_star if entry["truth_map_dx_pix"] is not None],
            [entry["truth_map_dy_pix"] for entry in per_star if entry["truth_map_dy_pix"] is not None],
        ),
        "ecsv_vs_npz_truth_detector_error_pix": _vector_stats(
            [entry["ecsv_vs_truth_detector_dx_pix"] for entry in per_star if entry["ecsv_vs_truth_detector_dx_pix"] is not None],
            [entry["ecsv_vs_truth_detector_dy_pix"] for entry in per_star if entry["ecsv_vs_truth_detector_dy_pix"] is not None],
        ),
        "npz_minus_ecsv_detector_offset_pix": _vector_stats(
            [-entry["ecsv_vs_truth_detector_dx_pix"] for entry in per_star if entry["ecsv_vs_truth_detector_dx_pix"] is not None],
            [-entry["ecsv_vs_truth_detector_dy_pix"] for entry in per_star if entry["ecsv_vs_truth_detector_dy_pix"] is not None],
        ),
        "centroid_error_sim_pix": _vector_stats(
            [entry["centroid_error_sim_dx_pix"] for entry in per_star],
            [entry["centroid_error_sim_dy_pix"] for entry in per_star],
        ),
        "centroid_error_detector_pix": _vector_stats(
            [entry["centroid_error_detector_dx_pix"] for entry in per_star if entry["centroid_error_detector_dx_pix"] is not None],
            [entry["centroid_error_detector_dy_pix"] for entry in per_star if entry["centroid_error_detector_dy_pix"] is not None],
        ),
        "centroid_error_focal_mm": _vector_stats(
            [entry["focal_error_dx_mm"] for entry in per_star if entry["focal_error_dx_mm"] is not None],
            [entry["focal_error_dy_mm"] for entry in per_star if entry["focal_error_dy_mm"] is not None],
        ),
        "body_error_centroid_arcsec": _scalar_stats(
            [entry["body_error_centroid_arcsec"] for entry in per_star if entry["body_error_centroid_arcsec"] is not None]
        ),
        "body_error_geometry_arcsec": _scalar_stats(
            [entry["body_error_geometry_arcsec"] for entry in per_star if entry["body_error_geometry_arcsec"] is not None]
        ),
        "body_error_total_arcsec": _scalar_stats(
            [entry["body_error_total_arcsec"] for entry in per_star if entry["body_error_total_arcsec"] is not None]
        ),
        "body_error_gain_arcsec_per_pix": _scalar_stats(
            [entry["body_error_gain_arcsec_per_pix"] for entry in per_star if entry["body_error_gain_arcsec_per_pix"] is not None]
        ),
        "match_observed_vs_predicted_pix": _scalar_stats(
            [entry["match_residual_pix"] for entry in per_star if entry["match_residual_pix"] is not None]
        ),
        "match_predicted_vs_truth_pix": _vector_stats(
            [entry["predicted_vs_truth_dx_pix"] for entry in per_star if entry["predicted_vs_truth_dx_pix"] is not None],
            [entry["predicted_vs_truth_dy_pix"] for entry in per_star if entry["predicted_vs_truth_dy_pix"] is not None],
        ),
        "match_predicted_vs_ecsv_detector_pix": _vector_stats(
            [
                entry["predicted_vs_ecsv_detector_dx_pix"]
                for entry in per_star
                if entry["predicted_vs_ecsv_detector_dx_pix"] is not None
            ],
            [
                entry["predicted_vs_ecsv_detector_dy_pix"]
                for entry in per_star
                if entry["predicted_vs_ecsv_detector_dy_pix"] is not None
            ],
        ),
        "match_truth_catalog_sep_arcsec": _scalar_stats(
            [entry["truth_catalog_sep_arcsec"] for entry in per_star if entry["truth_catalog_sep_arcsec"] is not None]
        ),
        "solution_residual_model_arcsec": _scalar_stats(
            [entry["solution_residual_model_arcsec"] for entry in per_star if entry["solution_residual_model_arcsec"] is not None]
        ),
        "solution_residual_exact_body_arcsec": _scalar_stats(
            [entry["solution_residual_exact_body_arcsec"] for entry in per_star if entry["solution_residual_exact_body_arcsec"] is not None]
        ),
        "counterfactual_solutions": counterfactuals,
        "interpretation": {
            "match_predicted_vs_truth_pix": (
                "Reference predicted_xy compared against NPZ detector truth. In this dataset NPZ detector truth includes "
                "the simulator telescope FOV offset."
            ),
            "match_predicted_vs_ecsv_detector_pix": (
                "Reference predicted_xy compared against raw et_focalplane detector coordinates from stars.ecsv."
            ),
            "frame_truth_reference": (
                "Use counterfactual_solutions.frame_truth_same_matches and delta_current_to_frame_truth_arcsec as the "
                "primary simulated-frame attitude accuracy metric."
            ),
        },
    }

    for entry in per_star:
        entry.pop("_truth_model_los_body", None)
        entry.pop("_truth_exact_body", None)
        entry.pop("_truth_los_inertial", None)

    return {
        "enabled": True,
        "truth_match_radius_pix": truth_match_radius_pix,
        "summary": summary,
        "per_detector": per_detector,
        "per_star": per_star,
        "selected_without_truth": unassigned_selected,
    }
