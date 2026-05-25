from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

from fsglib.attitude.solver import solve_attitude
from fsglib.common.coords import radec_to_unit_vector
from fsglib.common.io import load_npz_frame
from fsglib.common.types import AttitudeSolveInput, MatchingContext, ObservedStar
from fsglib.ephemeris.guide_geometry import build_exact_focalplane_geometry_adapter
from fsglib.ephemeris.pipeline import reference_weight_from_magnitudes
from fsglib.ephemeris.types import ReferenceStar
from fsglib.extract.pipeline import extract_stars
from fsglib.match.pipeline import match_stars
from fsglib.pipeline.convert import (
    observed_weight_from_sigma,
    propagate_centroid_covariance,
)
from fsglib.pipeline.error_budget import build_error_budget_ledger
from fsglib.pipeline.guide_error_audit import compute_guide_error_audit
from fsglib.preprocess.calibration import load_calibration_products
from fsglib.preprocess.pipeline import preprocess_frame


def _load_et_coord(cfg: dict) -> tuple[Any, Any, Any, Any]:
    et_cfg = cfg["et_coord"]
    src_dir = Path(et_cfg["src_dir"]).expanduser().resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from et_coord import ETCoordConfig, GaiaCatalog, GaiaSourceFilter, Transformer, load_registry

    data_dir = Path(et_cfg["data_dir"]).expanduser().resolve()
    config_factory = et_cfg.get("config_factory")
    if config_factory is None:
        registry = load_registry(data_dir)
    else:
        registry = load_registry(data_dir, getattr(ETCoordConfig, str(config_factory))())
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


def _apply_sim_to_detector_covariance(cov_pix, mapping: dict):
    if cov_pix is None:
        return None
    cov = np.asarray(cov_pix, dtype=np.float64)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    if mapping["kind"] == "offset":
        return cov
    transform = np.array(
        [
            [float(mapping["x_coeffs"][0]), float(mapping["x_coeffs"][1])],
            [float(mapping["y_coeffs"][0]), float(mapping["y_coeffs"][1])],
        ],
        dtype=np.float64,
    )
    return transform @ cov @ transform.T


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
    geometry_adapter,
    calib: dict | None = None,
) -> tuple[list[ObservedStar], dict, dict]:
    calib = {} if calib is None else calib
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
        pre = preprocess_frame(raw, calib=calib, cfg=cfg)
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
            centroid_cov_pix = _apply_sim_to_detector_covariance(
                getattr(candidate, "centroid_cov_pix", None),
                mapping,
            )
            transformed = geometry_adapter.pixel_to_focal(detector_id, x_et, y_et)
            los_cov_body, sigma_angle_arcsec = propagate_centroid_covariance(
                geometry_adapter,
                detector_id,
                x_et,
                y_et,
                centroid_cov_pix,
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
                    x=x_et,
                    y=y_et,
                    los_body=geometry_adapter.pixel_to_body_los(detector_id, x_et, y_et),
                    flux=candidate.flux,
                    snr=candidate.snr,
                    weight=weight,
                    centroid_cov_pix=centroid_cov_pix,
                    los_cov_body=los_cov_body,
                    sigma_angle_arcsec=sigma_angle_arcsec,
                    flags={
                        **candidate.flags,
                        **weight_flags,
                        "sigma_angle_arcsec": sigma_angle_arcsec,
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
    g_mag_min = guide_cfg.get("catalog_g_mag_min")
    if g_mag_min is not None:
        g_mag_min = float(g_mag_min)
    g_mag_max = float(guide_cfg["catalog_g_mag_max"])
    topk = int(guide_cfg["reference_topk_per_detector"])
    preselect_topk = int(guide_cfg.get("reference_preselect_topk_per_detector", topk))
    isolation_radius_pix = guide_cfg.get("reference_isolation_radius_pix")
    if isolation_radius_pix is not None:
        isolation_radius_pix = float(isolation_radius_pix)
    target_epoch = float(guide_cfg.get("target_epoch", 2000.0))

    from et_coord import query_detector_sources

    reference: list[ReferenceStar] = []
    per_detector_stats: dict[str, dict] = {}

    filter_kwargs = {"g_mean_mag_max": g_mag_max}
    if g_mag_min is not None:
        filter_kwargs["g_mean_mag_min"] = g_mag_min
    filters = GaiaSourceFilter(**filter_kwargs)
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
        frame = frame.sort_values("g_mean_mag", ascending=True)
        frame = frame.head(preselect_topk).copy()

        num_preselected = int(len(frame))
        num_isolated = num_preselected
        if isolation_radius_pix is not None and isolation_radius_pix > 0.0 and num_preselected > 1:
            coords = frame[["xpix", "ypix"]].to_numpy(dtype=np.float64)
            dx = coords[:, 0][:, None] - coords[:, 0][None, :]
            dy = coords[:, 1][:, None] - coords[:, 1][None, :]
            dist2 = (dx * dx) + (dy * dy)
            np.fill_diagonal(dist2, np.inf)
            nearest_dist = np.sqrt(np.min(dist2, axis=1))
            frame = frame.loc[nearest_dist > isolation_radius_pix].copy()
            num_isolated = int(len(frame))

        frame = frame.head(topk)
        per_detector_stats[detector_id] = {
            "num_reference_stars": int(len(frame)),
            "num_reference_preselected": num_preselected,
            "num_reference_isolated": num_isolated,
            "catalog_g_mag_min": g_mag_min,
            "catalog_g_mag_max": g_mag_max,
            "topk": topk,
            "preselect_topk": preselect_topk,
            "isolation_radius_pix": isolation_radius_pix,
        }
        for row in frame.itertuples(index=False):
            mag_g = float(row.g_mean_mag)
            weight_hint, weight_meta = reference_weight_from_magnitudes(
                mag_g=mag_g,
                mag_kp=None,
            )
            ra_deg = float(row.ra_deg)
            dec_deg = float(row.dec_deg)
            reference.append(
                ReferenceStar(
                    catalog_id=int(row.source_id),
                    time_s=0.0,
                    los_inertial=radec_to_unit_vector(ra_deg, dec_deg),
                    mag_g=mag_g,
                    detector_ids_visible=[detector_id],
                    predicted_xy={detector_id: (float(row.xpix), float(row.ypix))},
                    predicted_valid={detector_id: True},
                    weight_hint=weight_hint,
                    meta={
                        "ra_deg": ra_deg,
                        "dec_deg": dec_deg,
                        "propagated_ra_deg": ra_deg,
                        "propagated_dec_deg": dec_deg,
                        "target_epoch": target_epoch,
                        "astrometry_source": "et_coord",
                        **weight_meta,
                    },
                )
            )
    return reference, per_detector_stats


def run_guide_first_frame_init(cfg: dict, *, include_debug_context: bool = False) -> dict:
    registry, transformer, catalog, GaiaSourceFilter = _load_et_coord(cfg)
    geometry_adapter = build_exact_focalplane_geometry_adapter(cfg, registry, transformer)
    calib = load_calibration_products(cfg)

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
        geometry_adapter,
        calib,
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
        geometry_adapter,
        detector_contexts,
        observed,
        matching,
        solution,
    )
    error_budget = build_error_budget_ledger(
        raw=None,
        preprocessed=None,
        candidates=[],
        observed=observed,
        matching=matching,
        solution=solution,
        evaluation=None,
        dataset_ctx=None,
        cfg=cfg,
        detector_contexts=detector_contexts,
    )

    matched_per_detector: dict[str, int] = {}
    for matched_star in matching.matched:
        key = str(matched_star.detector_id)
        matched_per_detector[key] = matched_per_detector.get(key, 0) + 1

    for detector_id, stats in detector_stats.items():
        reference_detector_stats = reference_stats[detector_id]
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
        stats["num_reference_stars"] = reference_detector_stats["num_reference_stars"]
        stats["num_reference_preselected"] = reference_detector_stats["num_reference_preselected"]
        stats["num_reference_isolated"] = reference_detector_stats["num_reference_isolated"]
        stats["reference_preselect_topk"] = reference_detector_stats["preselect_topk"]
        stats["reference_isolation_radius_pix"] = reference_detector_stats["isolation_radius_pix"]

    geometry_payload = geometry_adapter.serialize()
    result = {
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
        "error_audit": error_audit,
        "error_budget": error_budget.to_dict(),
        "meta": {
            "dataset_root": str(dataset_root),
            "frame_index": int(cfg["guide_init"].get("frame_index", 0)),
            "los_geometry_mode": str(cfg["guide_init"].get("los_geometry_mode", "exact_et_focalplane")),
            "reference_topk_per_detector": int(cfg["guide_init"]["reference_topk_per_detector"]),
            "reference_preselect_topk_per_detector": int(
                cfg["guide_init"].get(
                    "reference_preselect_topk_per_detector",
                    cfg["guide_init"]["reference_topk_per_detector"],
                )
            ),
            "reference_isolation_radius_pix": (
                None
                if cfg["guide_init"].get("reference_isolation_radius_pix") is None
                else float(cfg["guide_init"]["reference_isolation_radius_pix"])
            ),
            "catalog_g_mag_min": (
                None
                if cfg["guide_init"].get("catalog_g_mag_min") is None
                else float(cfg["guide_init"]["catalog_g_mag_min"])
            ),
            "catalog_g_mag_max": float(cfg["guide_init"]["catalog_g_mag_max"]),
            "max_observed_per_detector": int(cfg["guide_init"].get("max_observed_per_detector", 0)),
        },
    }
    if include_debug_context:
        result["debug_context"] = {
            "detectors": {
                detector_id: {
                    "raw": context["raw"],
                    "preprocessed": context["preprocessed"],
                    "image": context["raw"].image,
                    "preprocessed_image": context["preprocessed"].image,
                    "frame_path": context["frame_path"],
                    "batch_path": context["batch_path"],
                    "all_candidates": context.get("all_candidates", []),
                    "selected_candidates": context.get("selected_candidates", []),
                    "num_candidates_raw": context["num_candidates_raw"],
                    "num_candidates_selected": context["num_candidates_selected"],
                }
                for detector_id, context in detector_contexts.items()
            },
            "observed_stars": observed,
            "reference_stars": reference,
        }

    return result
