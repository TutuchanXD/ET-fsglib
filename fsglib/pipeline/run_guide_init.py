from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table
from scipy.optimize import minimize

from fsglib.attitude.solver import solve_attitude
from fsglib.common.coords import radec_to_unit_vector
from fsglib.common.io import load_npz_frame
from fsglib.common.types import AttitudeSolveInput, MatchingContext, ObservedStar
from fsglib.ephemeris.types import ReferenceStar
from fsglib.extract.pipeline import extract_stars
from fsglib.match.pipeline import match_stars
from fsglib.pipeline.guide_error_audit import compute_guide_error_audit
from fsglib.preprocess.pipeline import preprocess_frame


def _load_et_coord(cfg: dict) -> tuple[Any, Any, Any, Any]:
    et_cfg = cfg["et_coord"]
    src_dir = Path(et_cfg["src_dir"]).expanduser().resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from et_coord import GaiaCatalog, GaiaSourceFilter, Transformer, load_registry

    registry = load_registry(Path(et_cfg["data_dir"]).expanduser().resolve())
    transformer = Transformer(registry)
    catalog = GaiaCatalog(Path(et_cfg["gaia_root_dir"]).expanduser().resolve())
    return registry, transformer, catalog, GaiaSourceFilter


def _guide_entries(cfg: dict) -> list[dict]:
    entries = cfg.get("guide_init", {}).get("detector_batches", [])
    if not entries:
        raise ValueError("guide_init.detector_batches is empty")
    return entries


def _frame_path(batch_path: Path, frame_index: int) -> Path:
    frame_paths = sorted((batch_path / "frames").glob("*.npz"))
    if frame_index < 0 or frame_index >= len(frame_paths):
        raise IndexError(
            f"frame_index={frame_index} out of range for {batch_path}; "
            f"available frames={len(frame_paths)}"
        )
    return frame_paths[frame_index]


def _image_center_from_run_meta(batch_path: Path) -> float:
    run_meta = json.loads((batch_path / "run_meta.json").read_text(encoding="utf-8"))
    width = int(run_meta["detector_width_pix"])
    return (width - 1) / 2.0


def _load_run_meta(batch_path: Path) -> dict:
    return json.loads((batch_path / "run_meta.json").read_text(encoding="utf-8"))


def _fit_sim_to_et_affine(batch_path: Path, detector_id: str, transformer) -> dict:
    table = Table.read(batch_path / "stars.ecsv")
    image_center = _image_center_from_run_meta(batch_path)

    sim_xy: list[list[float]] = []
    et_xy: list[list[float]] = []

    for row in table:
        sim_x = float(row["x0"]) + image_center
        sim_y = float(row["y0"]) + image_center
        mapped = transformer.sky_to_focal(ra=float(row["RA"]), dec=float(row["Dec"]))
        if mapped.status != "ok" or mapped.detector_id != detector_id:
            continue
        if mapped.xpix is None or mapped.ypix is None:
            continue
        sim_xy.append([sim_x, sim_y, 1.0])
        et_xy.append([float(mapped.xpix), float(mapped.ypix)])

    if len(sim_xy) < 3:
        raise RuntimeError(
            f"Not enough calibration stars to fit affine bridge for {detector_id} in {batch_path}"
        )

    sim_arr = np.asarray(sim_xy, dtype=np.float64)
    et_arr = np.asarray(et_xy, dtype=np.float64)
    x_coeffs, *_ = np.linalg.lstsq(sim_arr, et_arr[:, 0], rcond=None)
    y_coeffs, *_ = np.linalg.lstsq(sim_arr, et_arr[:, 1], rcond=None)

    fit_x = sim_arr @ x_coeffs
    fit_y = sim_arr @ y_coeffs
    residual = np.hypot(fit_x - et_arr[:, 0], fit_y - et_arr[:, 1])
    return {
        "kind": "affine",
        "x_coeffs": x_coeffs,
        "y_coeffs": y_coeffs,
        "num_fit_stars": int(sim_arr.shape[0]),
        "fit_rms_pix": float(np.sqrt(np.mean(np.square(residual)))),
        "fit_max_pix": float(np.max(residual)),
    }


def _build_sim_to_detector_map(batch_path: Path, detector_id: str, transformer) -> dict:
    run_meta = _load_run_meta(batch_path)
    schema_version = int(run_meta.get("frame_truth_schema_version", 1))
    image_center = _image_center_from_run_meta(batch_path)
    center_x = run_meta.get("guide_query_target_center_xpix_shifted")
    center_y = run_meta.get("guide_query_target_center_ypix_shifted")

    if schema_version >= 2 and center_x is not None and center_y is not None:
        offset_x = float(center_x) - image_center
        offset_y = float(center_y) - image_center
        return {
            "kind": "offset",
            "schema_version": schema_version,
            "offset_x_pix": offset_x,
            "offset_y_pix": offset_y,
            "image_center_pix": image_center,
            "guide_query_target_center_xpix": float(center_x),
            "guide_query_target_center_ypix": float(center_y),
        }

    affine = _fit_sim_to_et_affine(batch_path, detector_id, transformer)
    affine["schema_version"] = schema_version
    affine["image_center_pix"] = image_center
    return affine


def _apply_sim_to_detector_map(x_pix: float, y_pix: float, mapping: dict) -> tuple[float, float]:
    if mapping["kind"] == "offset":
        return (
            float(x_pix) + float(mapping["offset_x_pix"]),
            float(y_pix) + float(mapping["offset_y_pix"]),
        )

    sample = np.array([float(x_pix), float(y_pix), 1.0], dtype=np.float64)
    x_et = float(sample @ mapping["x_coeffs"])
    y_et = float(sample @ mapping["y_coeffs"])
    return x_et, y_et


def _field_angles_to_body_vector(field_x_deg: float, field_y_deg: float) -> np.ndarray:
    field_x_rad = np.radians(float(field_x_deg))
    field_y_rad = np.radians(float(field_y_deg))
    # et_coord defines focal/field +X toward the left of the focal-plane view.
    # fsglib attitude math assumes a conventional right-handed camera frame,
    # so the X axis needs to be flipped before building the LOS vector.
    tan_x = -np.tan(field_x_rad)
    tan_y = np.tan(field_y_rad)
    z = 1.0 / np.sqrt(1.0 + tan_x**2 + tan_y**2)
    return np.array([z * tan_x, z * tan_y, z], dtype=np.float64)


def _fit_focal_body_model(cfg: dict, registry, transformer) -> dict:
    grid_size = int(cfg.get("guide_init", {}).get("body_model_fit_grid_size", 13))
    initial_f_mm = float(cfg.get("guide_init", {}).get("body_model_initial_f_mm", 428.0))

    samples: list[tuple[float, float, np.ndarray]] = []
    for entry in _guide_entries(cfg):
        detector_id = str(entry["detector_id"])
        detector = registry.get_detector(detector_id)
        xs = np.linspace(0.0, detector.pixel_width, grid_size)
        ys = np.linspace(0.0, detector.pixel_height, grid_size)
        for x_pix in xs:
            for y_pix in ys:
                focal = transformer.pixel_to_focal(detector_id, float(x_pix), float(y_pix))
                sky = transformer.focal_to_sky(
                    detector_id,
                    float(focal.x_mm),
                    float(focal.y_mm),
                    frame="equatorial",
                )
                samples.append(
                    (
                        float(focal.x_mm),
                        float(focal.y_mm),
                        np.asarray(sky.vector_xyz, dtype=np.float64),
                    )
                )

    def _solve_alignment(body_vectors: list[np.ndarray], inertial_vectors: list[np.ndarray]) -> np.ndarray:
        B = np.zeros((3, 3), dtype=np.float64)
        for w, v in zip(body_vectors, inertial_vectors):
            B += np.outer(w, v)
        U, _, Vt = np.linalg.svd(B)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    def _evaluate(params: np.ndarray) -> tuple[float, float, np.ndarray]:
        k1, k3, k5 = [float(value) for value in params]
        body_vectors: list[np.ndarray] = []
        inertial_vectors: list[np.ndarray] = []
        for x_mm, y_mm, v in samples:
            r2 = x_mm * x_mm + y_mm * y_mm
            scale = k1 + (k3 * r2) + (k5 * r2 * r2)
            w = np.array([-x_mm * scale, y_mm * scale, 1.0], dtype=np.float64)
            w /= np.linalg.norm(w)
            body_vectors.append(w)
            inertial_vectors.append(v)

        R = _solve_alignment(body_vectors, inertial_vectors)
        W = np.asarray(body_vectors, dtype=np.float64)
        V = np.asarray(inertial_vectors, dtype=np.float64)
        dots = np.sum(W * (R @ V.T).T, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        residual_arcsec = np.degrees(np.arccos(dots)) * 3600.0
        return (
            float(np.sqrt(np.mean(np.square(residual_arcsec)))),
            float(np.max(residual_arcsec)),
            R,
        )

    result = minimize(
        lambda params: _evaluate(np.asarray(params, dtype=np.float64))[0],
        x0=np.array([1.0 / initial_f_mm, 0.0, 0.0], dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-12, "fatol": 1e-12},
    )
    coeffs = np.asarray(result.x, dtype=np.float64)
    fit_rms_arcsec, fit_max_arcsec, rotation_body_from_eq = _evaluate(coeffs)
    return {
        "coeffs": coeffs,
        "grid_size": grid_size,
        "fit_rms_arcsec": fit_rms_arcsec,
        "fit_max_arcsec": fit_max_arcsec,
        "rotation_body_from_eq": rotation_body_from_eq,
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
    }


def _focal_mm_to_body_vector(x_mm: float, y_mm: float, body_model: dict) -> np.ndarray:
    k1, k3, k5 = [float(value) for value in body_model["coeffs"]]
    r2 = float(x_mm) * float(x_mm) + float(y_mm) * float(y_mm)
    scale = k1 + (k3 * r2) + (k5 * r2 * r2)
    los_body = np.array([-float(x_mm) * scale, float(y_mm) * scale, 1.0], dtype=np.float64)
    return los_body / np.linalg.norm(los_body)


def _build_geometry_model(cfg: dict, registry, transformer) -> dict[str, Any]:
    guide_cfg = cfg["guide_init"]
    mode = str(guide_cfg.get("los_geometry_mode", "body_model_proxy"))

    proxy_model = _fit_focal_body_model(cfg, registry, transformer)
    rotation_body_from_eq = np.asarray(proxy_model["rotation_body_from_eq"], dtype=np.float64)

    if mode == "exact_et_focalplane":
        return {
            "mode": mode,
            "rotation_body_from_eq": rotation_body_from_eq,
            "frame_alignment_reference_grid_size": int(proxy_model["grid_size"]),
            "frame_alignment_reference_fit_rms_arcsec": float(proxy_model["fit_rms_arcsec"]),
            "frame_alignment_reference_fit_max_arcsec": float(proxy_model["fit_max_arcsec"]),
        }

    proxy_model["mode"] = "body_model_proxy"
    return proxy_model


def _geometry_model_body_vector(
    geometry_model: dict[str, Any],
    transformer,
    detector_id: str,
    observed_x_pix: float,
    observed_y_pix: float,
    transformed=None,
) -> np.ndarray:
    mode = str(geometry_model.get("mode", "body_model_proxy"))
    if mode == "exact_et_focalplane":
        sky = transformer.pixel_to_sky(
            detector_id,
            float(observed_x_pix),
            float(observed_y_pix),
            frame="equatorial",
        )
        if sky.vector_xyz is None:
            raise ValueError(f"Missing equatorial vector for detector {detector_id!r} exact geometry mapping.")
        los_eq = np.asarray(sky.vector_xyz, dtype=np.float64)
        los_eq /= np.linalg.norm(los_eq)
        los_body = np.asarray(geometry_model["rotation_body_from_eq"], dtype=np.float64) @ los_eq
        return los_body / np.linalg.norm(los_body)

    if transformed is None:
        transformed = transformer.pixel_to_focal(detector_id, observed_x_pix, observed_y_pix)
    return _focal_mm_to_body_vector(
        float(transformed.x_mm),
        float(transformed.y_mm),
        geometry_model,
    )


def _serialize_geometry_model(geometry_model: dict[str, Any]) -> dict[str, Any]:
    mode = str(geometry_model.get("mode", "body_model_proxy"))
    rotation = [
        [float(value) for value in row]
        for row in np.asarray(geometry_model["rotation_body_from_eq"], dtype=np.float64)
    ]
    if mode == "exact_et_focalplane":
        return {
            "mode": mode,
            "rotation_body_from_eq": rotation,
            "frame_alignment_reference_grid_size": int(geometry_model["frame_alignment_reference_grid_size"]),
            "frame_alignment_reference_fit_rms_arcsec": float(
                geometry_model["frame_alignment_reference_fit_rms_arcsec"]
            ),
            "frame_alignment_reference_fit_max_arcsec": float(
                geometry_model["frame_alignment_reference_fit_max_arcsec"]
            ),
        }

    return {
        "mode": mode,
        "coeffs": [float(value) for value in geometry_model["coeffs"]],
        "grid_size": int(geometry_model["grid_size"]),
        "fit_rms_arcsec": float(geometry_model["fit_rms_arcsec"]),
        "fit_max_arcsec": float(geometry_model["fit_max_arcsec"]),
        "rotation_body_from_eq": rotation,
        "optimization_success": bool(geometry_model["optimization_success"]),
        "optimization_message": geometry_model["optimization_message"],
    }


def _select_candidates_for_attitude(candidates: list, cfg: dict) -> list:
    max_per_detector = cfg.get("guide_init", {}).get("max_observed_per_detector")
    if max_per_detector is None:
        return candidates

    limit = int(max_per_detector)
    if limit <= 0 or len(candidates) <= limit:
        return candidates

    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.snr),
            float(candidate.flux),
            -float(candidate.area),
        ),
        reverse=True,
    )[:limit]


def _build_observed_stars(
    cfg: dict,
    transformer,
    sim_to_detector_map: dict[str, dict],
    geometry_model: dict[str, Any],
) -> tuple[list[ObservedStar], dict, dict]:
    dataset_root = Path(cfg["guide_init"]["dataset_root"]).expanduser().resolve()
    frame_index = int(cfg["guide_init"].get("frame_index", 0))
    observed: list[ObservedStar] = []
    detector_stats: dict[str, dict] = {}
    detector_contexts: dict[str, dict] = {}

    for entry in _guide_entries(cfg):
        detector_id = str(entry["detector_id"])
        batch_name = str(entry["batch_name"])
        batch_path = dataset_root / batch_name
        frame_path = _frame_path(batch_path, frame_index)

        raw = load_npz_frame(str(frame_path), detector_id=detector_id)
        pre = preprocess_frame(raw, calib={}, cfg=cfg)
        all_candidates = extract_stars(pre, cfg=cfg)
        candidates = _select_candidates_for_attitude(all_candidates, cfg=cfg)

        detector_stats[detector_id] = {
            "batch_name": batch_name,
            "frame_path": str(frame_path),
            "num_candidates_raw": len(all_candidates),
            "num_candidates_selected": len(candidates),
        }
        detector_contexts[detector_id] = {
            "batch_path": str(batch_path),
            "frame_path": str(frame_path),
            "raw": raw,
            "preprocessed": pre,
            "all_candidates": all_candidates,
            "selected_candidates": candidates,
            "num_candidates_raw": len(all_candidates),
            "num_candidates_selected": len(candidates),
        }

        mapping = sim_to_detector_map[detector_id]
        for candidate in candidates:
            x_et, y_et = _apply_sim_to_detector_map(candidate.x, candidate.y, mapping)
            transformed = transformer.pixel_to_focal(detector_id, x_et, y_et)
            observed.append(
                ObservedStar(
                    detector_id=detector_id,
                    source_id=f"{detector_id}:{candidate.source_id}",
                    x=x_et,
                    y=y_et,
                    los_body=_geometry_model_body_vector(
                        geometry_model,
                        transformer,
                        detector_id,
                        x_et,
                        y_et,
                        transformed=transformed,
                    ),
                    flux=candidate.flux,
                    snr=candidate.snr,
                    weight=max(candidate.snr, 1.0),
                    flags={
                        **candidate.flags,
                        "sim_x_pix": float(candidate.x),
                        "sim_y_pix": float(candidate.y),
                        "et_x_pix": x_et,
                        "et_y_pix": y_et,
                        "focal_x_mm": float(transformed.x_mm),
                        "focal_y_mm": float(transformed.y_mm),
                        "field_x_deg": float(transformed.field_x_deg),
                        "field_y_deg": float(transformed.field_y_deg),
                    },
                )
            )

    return observed, detector_stats, detector_contexts


def _build_reference_stars(cfg: dict, registry, catalog, GaiaSourceFilter) -> tuple[list[ReferenceStar], dict]:
    guide_cfg = cfg["guide_init"]
    g_mag_max = float(guide_cfg["catalog_g_mag_max"])
    topk = int(guide_cfg["reference_topk_per_detector"])
    target_epoch = float(guide_cfg.get("target_epoch", 2000.0))

    from et_coord import query_detector_sources

    reference: list[ReferenceStar] = []
    per_detector_stats: dict[str, dict] = {}

    filters = GaiaSourceFilter(g_mean_mag_max=g_mag_max)
    for entry in _guide_entries(cfg):
        detector_id = str(entry["detector_id"])
        frame = query_detector_sources(
            registry,
            catalog,
            detector_id,
            filters=filters,
            include_coords=("pixel",),
            target_epoch=target_epoch,
        )
        frame = frame.sort_values("g_mean_mag", ascending=True).head(topk)
        per_detector_stats[detector_id] = {
            "num_reference_stars": int(len(frame)),
            "catalog_g_mag_max": g_mag_max,
            "topk": topk,
        }
        for row in frame.itertuples(index=False):
            reference.append(
                ReferenceStar(
                    catalog_id=int(row.source_id),
                    time_s=0.0,
                    los_inertial=radec_to_unit_vector(float(row.ra_deg), float(row.dec_deg)),
                    mag_g=float(row.g_mean_mag),
                    detector_ids_visible=[detector_id],
                    predicted_xy={detector_id: (float(row.xpix), float(row.ypix))},
                    predicted_valid={detector_id: True},
                    weight_hint=1.0,
                    meta={"ra_deg": float(row.ra_deg), "dec_deg": float(row.dec_deg)},
                )
            )
    return reference, per_detector_stats


def run_guide_first_frame_init(cfg: dict) -> dict:
    registry, transformer, catalog, GaiaSourceFilter = _load_et_coord(cfg)
    geometry_model = _build_geometry_model(cfg, registry, transformer)

    sim_to_detector_map: dict[str, dict] = {}
    dataset_root = Path(cfg["guide_init"]["dataset_root"]).expanduser().resolve()
    for entry in _guide_entries(cfg):
        detector_id = str(entry["detector_id"])
        batch_path = dataset_root / str(entry["batch_name"])
        sim_to_detector_map[detector_id] = _build_sim_to_detector_map(batch_path, detector_id, transformer)

    observed, detector_stats, detector_contexts = _build_observed_stars(
        cfg,
        transformer,
        sim_to_detector_map,
        geometry_model,
    )
    reference, reference_stats = _build_reference_stars(cfg, registry, catalog, GaiaSourceFilter)

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
        cfg,
        transformer,
        sim_to_detector_map,
        geometry_model,
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
        stats["num_matched"] = matched_per_detector.get(detector_id, 0)
        stats["sim_to_detector_kind"] = sim_to_detector_map[detector_id]["kind"]
        stats["schema_version"] = sim_to_detector_map[detector_id]["schema_version"]
        if sim_to_detector_map[detector_id]["kind"] == "offset":
            stats["offset_x_pix"] = sim_to_detector_map[detector_id]["offset_x_pix"]
            stats["offset_y_pix"] = sim_to_detector_map[detector_id]["offset_y_pix"]
        else:
            stats["affine_fit_rms_pix"] = sim_to_detector_map[detector_id]["fit_rms_pix"]
            stats["affine_fit_max_pix"] = sim_to_detector_map[detector_id]["fit_max_pix"]
            stats["num_affine_fit_stars"] = sim_to_detector_map[detector_id]["num_fit_stars"]
        stats["num_reference_stars"] = reference_stats[detector_id]["num_reference_stars"]

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
        "geometry_model": _serialize_geometry_model(geometry_model),
        "body_model": _serialize_geometry_model(geometry_model),
        "error_audit": error_audit,
        "meta": {
            "dataset_root": str(dataset_root),
            "frame_index": int(cfg["guide_init"].get("frame_index", 0)),
            "reference_topk_per_detector": int(cfg["guide_init"]["reference_topk_per_detector"]),
            "catalog_g_mag_max": float(cfg["guide_init"]["catalog_g_mag_max"]),
            "max_observed_per_detector": int(cfg["guide_init"].get("max_observed_per_detector", 0)),
        },
    }
