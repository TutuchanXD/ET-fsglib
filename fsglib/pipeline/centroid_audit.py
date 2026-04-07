import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from fsglib.common.types import PreprocessedFrame, RawFrame, StarCandidate, TruthStar
from fsglib.extract.pipeline import extract_stars


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _vector_error_stats(dx_values: list[float], dy_values: list[float]) -> dict[str, float | None]:
    if not dx_values or not dy_values:
        return {
            "count": 0,
            "mean_dx_pix": None,
            "mean_dy_pix": None,
            "mean_abs_dx_pix": None,
            "mean_abs_dy_pix": None,
            "rms_dx_pix": None,
            "rms_dy_pix": None,
            "mean_radial_pix": None,
            "rms_radial_pix": None,
            "max_radial_pix": None,
        }

    dx = np.asarray(dx_values, dtype=np.float64)
    dy = np.asarray(dy_values, dtype=np.float64)
    radial = np.sqrt(dx * dx + dy * dy)
    return {
        "count": int(dx.size),
        "mean_dx_pix": float(np.mean(dx)),
        "mean_dy_pix": float(np.mean(dy)),
        "mean_abs_dx_pix": float(np.mean(np.abs(dx))),
        "mean_abs_dy_pix": float(np.mean(np.abs(dy))),
        "rms_dx_pix": float(np.sqrt(np.mean(dx**2))),
        "rms_dy_pix": float(np.sqrt(np.mean(dy**2))),
        "mean_radial_pix": float(np.mean(radial)),
        "rms_radial_pix": float(np.sqrt(np.mean(radial**2))),
        "max_radial_pix": float(np.max(radial)),
    }


def _prepare_photsim6_alias(ft_root: Path, data_dir: Path) -> None:
    if str(ft_root) not in sys.path:
        sys.path.insert(0, str(ft_root))

    for module_name in list(sys.modules):
        if module_name == "photsim6" or module_name.startswith("photsim6."):
            sys.modules.pop(module_name)

    photsim6_alias_pkg = types.ModuleType("photsim6")
    photsim6_alias_pkg.__path__ = [str(ft_root / "photsim6ft")]
    photsim6_alias_pkg.__package__ = "photsim6"
    sys.modules["photsim6"] = photsim6_alias_pkg
    os.environ["ET_DATA_DIR"] = str(data_dir)


@lru_cache(maxsize=4)
def _load_photsim6_context(
    ft_root_str: str,
    data_dir_str: str,
    config_xlsx_str: str,
    stamp_npix: int,
    psf_field_id: int,
) -> dict[str, Any]:
    ft_root = Path(ft_root_str)
    data_dir = Path(data_dir_str)
    config_xlsx = Path(config_xlsx_str)
    _prepare_photsim6_alias(ft_root, data_dir)

    from astropy import units as u
    from photsim6.configurator import ConfigurationManager
    from photsim6.field import Stars
    from photsim6.psf.model import PSFModelManager

    config_manager = ConfigurationManager(filepath=str(config_xlsx))
    parameters = config_manager.parameters
    actor_config = {
        "bundle_name": str(parameters["PSF Bundle Name"]),
        "pixel_scale": parameters["Pixel Scale"],
        "n_pixels": int(stamp_npix),
        "n_subpixels": int(parameters["Subpixels Per Pixel Dim"]),
        "n_jitter_integrated_psf_models": 1,
        "n_jitter_frames": int(parameters["N Jitter Frames Per Model"]),
        "integrate_jitter": False,
        "compute_device": "cpu",
        "float_precision": 32,
    }
    psf_manager = PSFModelManager(
        config=actor_config,
        warp_frame_batch_size=256,
        xy_jitter_pix=None,
        intialize=True,
        build_jit_int_models=True,
    )
    return {
        "u": u,
        "Stars": Stars,
        "config_manager": config_manager,
        "actor_config": actor_config,
        "selected_psf_model": psf_manager.models[int(psf_field_id)],
    }


def _extract_stamp(
    arr: np.ndarray,
    x_pix: float,
    y_pix: float,
    size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    half = size // 2
    cx = int(round(float(x_pix)))
    cy = int(round(float(y_pix)))
    x0 = cx - half
    y0 = cy - half
    x1 = cx + half + 1
    y1 = cy + half + 1
    if x0 < 0 or y0 < 0 or x1 > arr.shape[1] or y1 > arr.shape[0]:
        raise ValueError("Requested stamp crosses image boundary")
    return (
        np.asarray(arr[y0:y1, x0:x1], dtype=np.float64).copy(),
        {"cx": cx, "cy": cy, "x0": x0, "y0": y0, "x1": x1, "y1": y1},
    )


def _downsample_subpixel_images(subpixel_images: np.ndarray, n_subpixels: int) -> np.ndarray:
    subpixel_images = np.asarray(subpixel_images, dtype=np.float64)
    n_frames, ny, nx = subpixel_images.shape
    return subpixel_images.reshape(
        n_frames,
        ny // n_subpixels,
        n_subpixels,
        nx // n_subpixels,
        n_subpixels,
    ).sum(axis=(2, 4))


def _weighted_centroid_full_window(image_2d: np.ndarray) -> tuple[float, float] | None:
    image_2d = np.asarray(image_2d, dtype=np.float64)
    total_flux = float(np.sum(image_2d))
    if not np.isfinite(total_flux) or total_flux <= 0.0:
        return None
    ys, xs = np.indices(image_2d.shape, dtype=np.float64)
    x = float(np.sum(xs * image_2d) / total_flux)
    y = float(np.sum(ys * image_2d) / total_flux)
    return x, y


def _local_segment_centroid(
    image_2d: np.ndarray,
    noise_2d: np.ndarray,
    cfg: dict,
    truth_local_xy: np.ndarray,
) -> dict[str, Any] | None:
    image_2d = np.asarray(image_2d, dtype=np.float64)
    noise_2d = np.asarray(noise_2d, dtype=np.float64)
    extract_cfg = cfg.get("extract", {})
    snr_map = image_2d / np.maximum(noise_2d, 1e-6)
    detect_mask = snr_map > float(extract_cfg.get("seed_threshold_sigma", 5.0))
    labeled, num_labels = ndimage.label(detect_mask)
    if num_labels <= 0:
        return None

    truth_xy = np.asarray(truth_local_xy, dtype=np.float64)
    truth_ix = int(np.clip(round(float(truth_xy[0])), 0, image_2d.shape[1] - 1))
    truth_iy = int(np.clip(round(float(truth_xy[1])), 0, image_2d.shape[0] - 1))

    candidate_label_ids: set[int] = set()
    y0 = max(0, truth_iy - 1)
    y1 = min(image_2d.shape[0], truth_iy + 2)
    x0 = max(0, truth_ix - 1)
    x1 = min(image_2d.shape[1], truth_ix + 2)
    local_patch = labeled[y0:y1, x0:x1]
    candidate_label_ids.update(int(val) for val in np.unique(local_patch) if int(val) > 0)

    if not candidate_label_ids:
        distances: list[tuple[float, int]] = []
        for label_id in range(1, num_labels + 1):
            seg = labeled == label_id
            ys, xs = np.where(seg)
            if xs.size == 0:
                continue
            centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
            distances.append((float(np.linalg.norm(centroid - truth_xy)), int(label_id)))
        if not distances:
            return None
        candidate_label_ids = {min(distances, key=lambda item: item[0])[1]}

    best_payload = None
    best_distance = np.inf
    for label_id in sorted(candidate_label_ids):
        seg = labeled == label_id
        flux = float(np.sum(image_2d[seg]))
        if flux <= 0.0 or not np.isfinite(flux):
            continue
        ys, xs = np.where(seg)
        x = float(np.sum(xs * image_2d[seg]) / flux)
        y = float(np.sum(ys * image_2d[seg]) / flux)
        distance = float(np.hypot(x - truth_xy[0], y - truth_xy[1]))
        if distance >= best_distance:
            continue
        best_distance = distance
        best_payload = {
            "x_local_pix": x,
            "y_local_pix": y,
            "flux": flux,
            "area_pix": int(xs.size),
            "distance_to_truth_pix": distance,
            "label_id": int(label_id),
        }

    return best_payload


def _run_local_extract(
    image_2d: np.ndarray,
    noise_2d: np.ndarray,
    cfg: dict,
    truth_local_xy: np.ndarray,
) -> StarCandidate | None:
    frame = PreprocessedFrame(
        detector_id=0,
        image=np.asarray(image_2d, dtype=np.float64),
        background=np.zeros_like(image_2d, dtype=np.float64),
        noise_map=np.asarray(noise_2d, dtype=np.float64),
        valid_mask=np.isfinite(image_2d),
        preprocess_meta={},
    )
    candidates = extract_stars(frame, cfg)
    if not candidates:
        return None
    target = np.asarray(truth_local_xy, dtype=np.float64)
    return min(candidates, key=lambda cand: float(np.hypot(cand.x - target[0], cand.y - target[1])))


def _build_single_star_stamp(
    truth_star: TruthStar,
    stamp_meta: dict[str, int],
    audit_cfg: dict,
) -> np.ndarray:
    mag = truth_star.mag
    if mag is None or not np.isfinite(mag):
        raise ValueError("Truth star is missing magnitude required for single-star stamp synthesis")

    stamp_npix = int(audit_cfg["stamp_size"])
    ctx = _load_photsim6_context(
        str(Path(audit_cfg["ft_root"]).expanduser()),
        str(Path(audit_cfg["data_dir"]).expanduser()),
        str(Path(audit_cfg["config_xlsx"]).expanduser()),
        stamp_npix,
        int(audit_cfg["psf_field_id"]),
    )

    u = ctx["u"]
    Stars = ctx["Stars"]
    config_manager = ctx["config_manager"]
    parameters = config_manager.parameters
    selected_psf_model = ctx["selected_psf_model"]
    actor_config = ctx["actor_config"]

    stars = Stars()
    star_data = {
        "x0": np.array([0.0], dtype=np.float64),
        "y0": np.array([0.0], dtype=np.float64),
        "ra": np.array([float(truth_star.ra_deg)], dtype=np.float64),
        "dec": np.array([float(truth_star.dec_deg)], dtype=np.float64),
        "kp_mag": np.array([float(mag)], dtype=np.float64),
    }
    optical_eff_ratio = parameters["Optical Efficiency Ratio"].to(u.percent).value / 100.0
    aperture_diameter_cm = parameters["Aperture Diameter"].to(u.cm).value
    aperture_area_cm2 = np.pi * (aperture_diameter_cm / 2.0) ** 2
    reference_et_aperture_cm2 = 384.8451000647498
    relative_aperture_area = aperture_area_cm2 / reference_et_aperture_cm2
    stars.build_catalog(
        star_data,
        parameters["Exposure Duration"],
        optical_eff_ratio,
        relative_aperture_area,
        mag_type=str(audit_cfg.get("mag_type", "ET")),
    )
    photon_count = float(stars.catalog["Photon Count"][0].to(u.electron).value)

    dx = float(truth_star.x_pix - stamp_meta["cx"])
    dy = float(truth_star.y_pix - stamp_meta["cy"])
    atms = selected_psf_model.generate_affine_transform_matrices(
        x_offset=np.array([dx], dtype=np.float64),
        y_offset=np.array([dy], dtype=np.float64),
        x_scale=np.ones(1, dtype=np.float64),
        y_scale=np.ones(1, dtype=np.float64),
        theta_deg=np.zeros(1, dtype=np.float64),
    )
    subpixel_images = selected_psf_model.generate_images(
        affine_transform_matrices=atms,
        photon_count=photon_count,
        return_gpu_arr=False,
        jitter_enabled=False,
    )
    detector_images = _downsample_subpixel_images(
        np.asarray(subpixel_images, dtype=np.float64),
        n_subpixels=int(actor_config["n_subpixels"]),
    )
    return np.asarray(detector_images[0], dtype=np.float64)


def _stage_entry(
    truth_idx: int,
    truth_star: TruthStar,
    stage_name: str,
    stage_xy_global: tuple[float, float] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if stage_xy_global is None:
        return None
    x_pix = float(stage_xy_global[0])
    y_pix = float(stage_xy_global[1])
    dx_pix = x_pix - float(truth_star.x_pix)
    dy_pix = y_pix - float(truth_star.y_pix)
    entry = {
        "truth_index": int(truth_idx),
        "truth_source_id": truth_star.source_id,
        "stage": stage_name,
        "x_pix": x_pix,
        "y_pix": y_pix,
        "dx_pix": dx_pix,
        "dy_pix": dy_pix,
        "radial_pix": float(np.hypot(dx_pix, dy_pix)),
    }
    if extra:
        entry["meta"] = extra
    return entry


def _summarize_stage_entries(entries: list[dict[str, Any]], stage_name: str) -> dict[str, float | None]:
    stage_entries = [entry for entry in entries if entry["stage"] == stage_name]
    if not stage_entries:
        return _vector_error_stats([], [])
    return _vector_error_stats(
        [float(entry["dx_pix"]) for entry in stage_entries],
        [float(entry["dy_pix"]) for entry in stage_entries],
    )


def _summarize_transition(
    entries_by_star: dict[int, dict[str, dict[str, Any]]],
    from_stage: str,
    to_stage: str,
) -> dict[str, float | None]:
    dx_values: list[float] = []
    dy_values: list[float] = []
    radial_delta_values: list[float] = []
    for stage_map in entries_by_star.values():
        from_entry = stage_map.get(from_stage)
        to_entry = stage_map.get(to_stage)
        if from_entry is None or to_entry is None:
            continue
        dx = float(to_entry["x_pix"]) - float(from_entry["x_pix"])
        dy = float(to_entry["y_pix"]) - float(from_entry["y_pix"])
        dx_values.append(dx)
        dy_values.append(dy)
        radial_delta_values.append(float(to_entry["radial_pix"]) - float(from_entry["radial_pix"]))

    summary = _vector_error_stats(dx_values, dy_values)
    if radial_delta_values:
        radial_delta = np.asarray(radial_delta_values, dtype=np.float64)
        summary["mean_radial_error_change_pix"] = float(np.mean(radial_delta))
        summary["max_radial_error_change_pix"] = float(np.max(radial_delta))
        summary["min_radial_error_change_pix"] = float(np.min(radial_delta))
    else:
        summary["mean_radial_error_change_pix"] = None
        summary["max_radial_error_change_pix"] = None
        summary["min_radial_error_change_pix"] = None
    return summary


def _candidate_truth_pairs(
    candidates: list[StarCandidate],
    truth_stars: list[TruthStar],
    truth_match_radius_pix: float,
) -> dict[int, StarCandidate]:
    if not candidates or not truth_stars:
        return {}
    truth_xy = np.asarray([[star.x_pix, star.y_pix] for star in truth_stars], dtype=np.float64)
    assignments: list[tuple[float, int, int]] = []
    for cand_idx, candidate in enumerate(candidates):
        dx = truth_xy[:, 0] - float(candidate.x)
        dy = truth_xy[:, 1] - float(candidate.y)
        dist = np.sqrt(dx * dx + dy * dy)
        truth_idx = int(np.argmin(dist))
        min_dist = float(dist[truth_idx])
        if min_dist <= float(truth_match_radius_pix):
            assignments.append((min_dist, truth_idx, cand_idx))

    assignments.sort(key=lambda item: item[0])
    used_truth: set[int] = set()
    used_candidates: set[int] = set()
    pairs: dict[int, StarCandidate] = {}
    for _, truth_idx, cand_idx in assignments:
        if truth_idx in used_truth or cand_idx in used_candidates:
            continue
        used_truth.add(truth_idx)
        used_candidates.add(cand_idx)
        pairs[truth_idx] = candidates[cand_idx]
    return pairs


def _nearest_neighbor_distance(truth_xy: np.ndarray, truth_idx: int) -> float | None:
    if truth_xy.shape[0] <= 1:
        return None
    dx = truth_xy[:, 0] - truth_xy[truth_idx, 0]
    dy = truth_xy[:, 1] - truth_xy[truth_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[truth_idx] = np.inf
    return float(np.min(dist))


def compute_centroid_step_audit(
    raw: RawFrame,
    preprocessed: PreprocessedFrame,
    candidates: list[StarCandidate],
    cfg: dict,
    truth_stars: list[TruthStar],
) -> dict[str, Any]:
    audit_cfg = dict(cfg.get("evaluation", {}).get("centroid_step_audit", {}))
    if not audit_cfg.get("enabled", False):
        return {"enabled": False}

    stamp_size = int(audit_cfg.get("stamp_size", 31))
    truth_match_radius_pix = float(audit_cfg.get("truth_match_radius_pix", 3.0))
    max_stars = audit_cfg.get("max_stars")
    if max_stars is not None:
        max_stars = int(max_stars)

    truth_xy = np.asarray([[star.x_pix, star.y_pix] for star in truth_stars], dtype=np.float64)
    candidate_pairs = _candidate_truth_pairs(candidates, truth_stars, truth_match_radius_pix)

    sortable_indices = list(range(len(truth_stars)))
    sortable_indices.sort(
        key=lambda idx: (
            np.inf if truth_stars[idx].mag is None else float(truth_stars[idx].mag),
            idx,
        )
    )
    if max_stars is not None:
        sortable_indices = sortable_indices[:max_stars]

    entries: list[dict[str, Any]] = []
    entries_by_star: dict[int, dict[str, dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []

    for truth_idx in sortable_indices:
        truth_star = truth_stars[truth_idx]
        try:
            raw_stamp, stamp_meta = _extract_stamp(raw.image, truth_star.x_pix, truth_star.y_pix, stamp_size)
            pre_stamp, _ = _extract_stamp(preprocessed.image, truth_star.x_pix, truth_star.y_pix, stamp_size)
            noise_stamp, _ = _extract_stamp(np.asarray(preprocessed.noise_map, dtype=np.float64), truth_star.x_pix, truth_star.y_pix, stamp_size)
        except Exception as exc:
            skipped.append(
                {
                    "truth_index": int(truth_idx),
                    "truth_source_id": truth_star.source_id,
                    "reason": f"stamp_extract_failed: {exc}",
                }
            )
            continue

        truth_local = np.array(
            [
                float(truth_star.x_pix) - stamp_meta["x0"],
                float(truth_star.y_pix) - stamp_meta["y0"],
            ],
            dtype=np.float64,
        )

        star_meta = {
            "nearest_neighbor_pix": _safe_float(_nearest_neighbor_distance(truth_xy, truth_idx)),
            "truth_mag": _safe_float(truth_star.mag),
            "truth_dx_pointing_pix": _safe_float(truth_star.meta.get("truth_dx_pointing_pix")),
            "truth_dy_pointing_pix": _safe_float(truth_star.meta.get("truth_dy_pointing_pix")),
            "truth_dx_dva_pix": _safe_float(truth_star.meta.get("truth_dx_dva_pix")),
            "truth_dy_dva_pix": _safe_float(truth_star.meta.get("truth_dy_dva_pix")),
            "truth_dx_thermal_pix": _safe_float(truth_star.meta.get("truth_dx_thermal_pix")),
            "truth_dy_thermal_pix": _safe_float(truth_star.meta.get("truth_dy_thermal_pix")),
            "truth_dx_jitter_mean_pix": _safe_float(truth_star.meta.get("truth_dx_jitter_mean_pix")),
            "truth_dy_jitter_mean_pix": _safe_float(truth_star.meta.get("truth_dy_jitter_mean_pix")),
        }

        stage_map: dict[str, dict[str, Any]] = {}

        def add_entry(stage_name: str, xy_local: tuple[float, float] | None, extra: dict[str, Any] | None = None) -> None:
            if xy_local is None:
                return
            stage_xy_global = (stamp_meta["x0"] + float(xy_local[0]), stamp_meta["y0"] + float(xy_local[1]))
            payload = _stage_entry(
                truth_idx,
                truth_star,
                stage_name,
                stage_xy_global,
                extra={**star_meta, **(extra or {})},
            )
            if payload is None:
                return
            entries.append(payload)
            stage_map[stage_name] = payload

        raw_positive = np.clip(raw_stamp, 0.0, None)
        add_entry("multi_raw_full_window", _weighted_centroid_full_window(raw_positive))

        pre_positive = np.clip(pre_stamp, 0.0, None)
        add_entry("multi_preprocessed_full_window", _weighted_centroid_full_window(pre_positive))

        local_segment = _local_segment_centroid(pre_positive, noise_stamp, cfg, truth_local)
        add_entry(
            "multi_local_segment",
            None
            if local_segment is None
            else (float(local_segment["x_local_pix"]), float(local_segment["y_local_pix"])),
            extra=None if local_segment is None else {
                "segment_area_pix": int(local_segment["area_pix"]),
                "distance_to_truth_pix": float(local_segment["distance_to_truth_pix"]),
            },
        )

        matched_candidate = candidate_pairs.get(truth_idx)
        if matched_candidate is not None:
            raw_x = float(matched_candidate.flags.get("raw_centroid_x_pix", matched_candidate.x))
            raw_y = float(matched_candidate.flags.get("raw_centroid_y_pix", matched_candidate.y))
            add_entry(
                "multi_pipeline_raw",
                (raw_x - stamp_meta["x0"], raw_y - stamp_meta["y0"]),
                extra={"candidate_source_id": int(matched_candidate.source_id)},
            )
            add_entry(
                "multi_pipeline_final",
                (float(matched_candidate.x) - stamp_meta["x0"], float(matched_candidate.y) - stamp_meta["y0"]),
                extra={"candidate_source_id": int(matched_candidate.source_id)},
            )

        try:
            model_stamp = _build_single_star_stamp(truth_star, stamp_meta, audit_cfg)
        except Exception as exc:
            skipped.append(
                {
                    "truth_index": int(truth_idx),
                    "truth_source_id": truth_star.source_id,
                    "reason": f"single_star_synthesis_failed: {exc}",
                }
            )
            entries_by_star[truth_idx] = stage_map
            continue

        add_entry("single_model_full_window", _weighted_centroid_full_window(model_stamp))
        model_bg = float(np.median(np.concatenate([model_stamp[0], model_stamp[-1], model_stamp[1:-1, 0], model_stamp[1:-1, -1]])))
        model_bgsub = np.clip(model_stamp - model_bg, 0.0, None)
        add_entry("single_model_bgsub_full_window", _weighted_centroid_full_window(model_bgsub))

        single_segment = _local_segment_centroid(model_bgsub, noise_stamp, cfg, truth_local)
        add_entry(
            "single_local_segment",
            None
            if single_segment is None
            else (float(single_segment["x_local_pix"]), float(single_segment["y_local_pix"])),
            extra=None if single_segment is None else {
                "segment_area_pix": int(single_segment["area_pix"]),
                "distance_to_truth_pix": float(single_segment["distance_to_truth_pix"]),
            },
        )

        single_candidate = _run_local_extract(model_bgsub, noise_stamp, cfg, truth_local)
        if single_candidate is not None:
            raw_x = float(single_candidate.flags.get("raw_centroid_x_pix", single_candidate.x))
            raw_y = float(single_candidate.flags.get("raw_centroid_y_pix", single_candidate.y))
            add_entry("single_pipeline_raw", (raw_x, raw_y))
            add_entry("single_pipeline_final", (float(single_candidate.x), float(single_candidate.y)))

        entries_by_star[truth_idx] = stage_map

    stage_names = [
        "multi_raw_full_window",
        "multi_preprocessed_full_window",
        "multi_local_segment",
        "multi_pipeline_raw",
        "multi_pipeline_final",
        "single_model_full_window",
        "single_model_bgsub_full_window",
        "single_local_segment",
        "single_pipeline_raw",
        "single_pipeline_final",
    ]
    stage_stats = {stage_name: _summarize_stage_entries(entries, stage_name) for stage_name in stage_names}
    transition_pairs = [
        ("multi_raw_full_window", "multi_preprocessed_full_window"),
        ("multi_preprocessed_full_window", "multi_local_segment"),
        ("multi_local_segment", "multi_pipeline_raw"),
        ("multi_pipeline_raw", "multi_pipeline_final"),
        ("single_model_full_window", "single_model_bgsub_full_window"),
        ("single_model_bgsub_full_window", "single_local_segment"),
        ("single_local_segment", "single_pipeline_raw"),
        ("single_pipeline_raw", "single_pipeline_final"),
        ("single_pipeline_raw", "multi_pipeline_raw"),
        ("single_pipeline_final", "multi_pipeline_final"),
        ("single_local_segment", "multi_local_segment"),
    ]
    transition_stats = {
        f"{from_stage}__to__{to_stage}": _summarize_transition(entries_by_star, from_stage, to_stage)
        for from_stage, to_stage in transition_pairs
    }

    worst_multi_final = sorted(
        [entry for entry in entries if entry["stage"] == "multi_pipeline_final"],
        key=lambda entry: float(entry["radial_pix"]),
        reverse=True,
    )[:10]

    return {
        "enabled": True,
        "stamp_size": stamp_size,
        "truth_match_radius_pix": truth_match_radius_pix,
        "num_truth_stars": len(truth_stars),
        "num_audited_stars": len(entries_by_star),
        "num_candidate_pairs": len(candidate_pairs),
        "stage_stats": stage_stats,
        "transition_stats": transition_stats,
        "worst_multi_pipeline_final": worst_multi_final,
        "entries": entries,
        "skipped": skipped,
    }
