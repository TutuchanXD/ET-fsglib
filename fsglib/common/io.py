import json
from pathlib import Path
import numpy as np
from astropy.table import Table

from fsglib.common.types import DatasetContext, RawFrame, TruthStar


def _centered_truth_to_pixel(
    x_value: float,
    y_value: float,
    width: int,
    height: int,
    y_axis_up: bool,
) -> tuple[float, float]:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_pix = cx + float(x_value)
    if y_axis_up:
        y_pix = cy - float(y_value)
    else:
        y_pix = cy + float(y_value)
    return x_pix, y_pix


def _resolve_default_principal_point(
    cfg: dict | None,
    detector_width: int | None,
    detector_height: int | None,
) -> tuple[float | None, float | None]:
    layout = (cfg or {}).get("layout", {})
    default_detector_id = int(layout.get("default_detector_id", 0))
    for detector in layout.get("detectors", []):
        try:
            detector_id = int(detector.get("detector_id", -1))
        except (TypeError, ValueError):
            continue
        if detector_id != default_detector_id:
            continue
        principal_point = detector.get("principal_point_pix")
        if principal_point is not None and len(principal_point) >= 2:
            return float(principal_point[0]), float(principal_point[1])

    if detector_width is None or detector_height is None:
        return None, None
    return (detector_width - 1) / 2.0, (detector_height - 1) / 2.0


def _estimate_field_offset_from_truth(
    truth_stars: list[TruthStar],
    batch_center_ra_deg: float | None,
    batch_center_dec_deg: float | None,
    pixel_scale_arcsec_per_pix: float | None,
    principal_point_x_pix: float | None,
    principal_point_y_pix: float | None,
) -> tuple[float | None, float | None, dict]:
    if (
        not truth_stars
        or batch_center_ra_deg is None
        or batch_center_dec_deg is None
        or pixel_scale_arcsec_per_pix is None
        or pixel_scale_arcsec_per_pix <= 0
        or principal_point_x_pix is None
        or principal_point_y_pix is None
    ):
        return None, None, {}

    dec0_rad = np.radians(float(batch_center_dec_deg))
    dx_values: list[float] = []
    dy_values: list[float] = []

    for truth_star in truth_stars:
        dra_rad = np.arctan2(
            np.sin(np.radians(float(truth_star.ra_deg) - float(batch_center_ra_deg))),
            np.cos(np.radians(float(truth_star.ra_deg) - float(batch_center_ra_deg))),
        )
        x_deg = np.degrees(dra_rad * np.cos(dec0_rad))
        y_deg = float(truth_star.dec_deg) - float(batch_center_dec_deg)

        expected_x_pix = principal_point_x_pix + (x_deg * 3600.0 / pixel_scale_arcsec_per_pix)
        expected_y_pix = principal_point_y_pix + (y_deg * 3600.0 / pixel_scale_arcsec_per_pix)
        dx_values.append(float(truth_star.x_pix) - expected_x_pix)
        dy_values.append(float(truth_star.y_pix) - expected_y_pix)

    dx_array = np.asarray(dx_values, dtype=np.float64)
    dy_array = np.asarray(dy_values, dtype=np.float64)
    return (
        float(np.mean(dx_array)),
        float(np.mean(dy_array)),
        {
            "field_offset_estimate_std_x_pix": float(np.std(dx_array)),
            "field_offset_estimate_std_y_pix": float(np.std(dy_array)),
            "field_offset_estimate_num_truth_stars": int(dx_array.size),
        },
    )


def _select_truth_source_ids(
    table: Table,
    run_meta: dict,
) -> tuple[list[int | None], dict]:
    colnames = set(table.colnames)

    if "Truth Index" in colnames:
        raw_ids = [int(row["Truth Index"]) for row in table]
        if len(set(raw_ids)) == len(raw_ids):
            return raw_ids, {
                **run_meta,
                "truth_star_id_status": "truth_index_unique",
                "truth_star_id_column": "Truth Index",
            }

    if "Star ID" in colnames:
        raw_ids = [int(row["Star ID"]) for row in table]
        if len(set(raw_ids)) == len(raw_ids):
            return raw_ids, {
                **run_meta,
                "truth_star_id_status": "star_id_unique",
                "truth_star_id_column": "Star ID",
            }
        return [None] * len(raw_ids), {
            **run_meta,
            "truth_star_id_status": "non_unique_or_invalid",
        }

    return [None] * len(table), run_meta


def _extract_npz_truth_payload(
    data,
    variant_index: int = 0,
    frame_index: int = 0,
) -> tuple[list[TruthStar], dict] | tuple[None, None]:
    required = [
        "truth_x_image_pix",
        "truth_y_image_pix",
        "truth_ra_deg",
        "truth_dec_deg",
    ]
    if any(key not in data for key in required):
        return None, None

    x_image = np.asarray(data["truth_x_image_pix"][variant_index, frame_index], dtype=np.float64)
    y_image = np.asarray(data["truth_y_image_pix"][variant_index, frame_index], dtype=np.float64)
    ra_deg = np.asarray(data["truth_ra_deg"], dtype=np.float64)
    dec_deg = np.asarray(data["truth_dec_deg"], dtype=np.float64)
    mag = (
        np.asarray(data["truth_mag"], dtype=np.float64)
        if "truth_mag" in data
        else np.full_like(ra_deg, np.nan, dtype=np.float64)
    )
    source_ids = (
        np.asarray(data["truth_star_index"], dtype=np.int64)
        if "truth_star_index" in data
        else np.arange(ra_deg.shape[0], dtype=np.int64)
    )
    valid_mask = (
        np.asarray(data["truth_valid_mask"][variant_index, frame_index], dtype=bool)
        if "truth_valid_mask" in data
        else np.ones_like(x_image, dtype=bool)
    )

    def _optional_static_1d(key: str):
        if key not in data:
            return None
        return np.asarray(data[key], dtype=np.float64)

    def _optional_dynamic_1d(key: str):
        if key not in data:
            return None
        return np.asarray(data[key][variant_index, frame_index], dtype=np.float64)

    optional_static = {
        "truth_static_x_centered_pix": _optional_static_1d("truth_static_x_centered_pix"),
        "truth_static_y_centered_pix": _optional_static_1d("truth_static_y_centered_pix"),
        "truth_static_x_image_pix": _optional_static_1d("truth_static_x_image_pix"),
        "truth_static_y_image_pix": _optional_static_1d("truth_static_y_image_pix"),
        "truth_static_x_detector_pix": _optional_static_1d("truth_static_x_detector_pix"),
        "truth_static_y_detector_pix": _optional_static_1d("truth_static_y_detector_pix"),
    }
    optional_dynamic = {
        "truth_x_centered_pix": _optional_dynamic_1d("truth_x_centered_pix"),
        "truth_y_centered_pix": _optional_dynamic_1d("truth_y_centered_pix"),
        "truth_x_image_pix": _optional_dynamic_1d("truth_x_image_pix"),
        "truth_y_image_pix": _optional_dynamic_1d("truth_y_image_pix"),
        "truth_abs_x_image_pix": _optional_dynamic_1d("truth_abs_x_image_pix"),
        "truth_abs_y_image_pix": _optional_dynamic_1d("truth_abs_y_image_pix"),
        "truth_x_detector_pix": _optional_dynamic_1d("truth_x_detector_pix"),
        "truth_y_detector_pix": _optional_dynamic_1d("truth_y_detector_pix"),
        "truth_abs_x_detector_pix": _optional_dynamic_1d("truth_abs_x_detector_pix"),
        "truth_abs_y_detector_pix": _optional_dynamic_1d("truth_abs_y_detector_pix"),
        "truth_dx_pointing_pix": _optional_dynamic_1d("truth_dx_pointing_pix"),
        "truth_dy_pointing_pix": _optional_dynamic_1d("truth_dy_pointing_pix"),
        "truth_dx_dva_pix": _optional_dynamic_1d("truth_dx_dva_pix"),
        "truth_dy_dva_pix": _optional_dynamic_1d("truth_dy_dva_pix"),
        "truth_dx_thermal_pix": _optional_dynamic_1d("truth_dx_thermal_pix"),
        "truth_dy_thermal_pix": _optional_dynamic_1d("truth_dy_thermal_pix"),
        "truth_dx_jitter_mean_pix": _optional_dynamic_1d("truth_dx_jitter_mean_pix"),
        "truth_dy_jitter_mean_pix": _optional_dynamic_1d("truth_dy_jitter_mean_pix"),
    }

    truth_stars: list[TruthStar] = []
    for idx in range(x_image.shape[0]):
        if idx >= valid_mask.shape[0] or not bool(valid_mask[idx]):
            continue

        meta = {
            "truth_index": int(source_ids[idx]),
            "truth_valid": bool(valid_mask[idx]),
        }
        for key, values in optional_static.items():
            if values is not None:
                meta[key] = float(values[idx])
        for key, values in optional_dynamic.items():
            if values is not None:
                meta[key] = float(values[idx])

        mag_value = None
        if idx < mag.shape[0] and np.isfinite(mag[idx]):
            mag_value = float(mag[idx])

        truth_stars.append(
            TruthStar(
                source_id=int(source_ids[idx]),
                x_pix=float(x_image[idx]),
                y_pix=float(y_image[idx]),
                ra_deg=float(ra_deg[idx]),
                dec_deg=float(dec_deg[idx]),
                mag=mag_value,
                meta=meta,
            )
        )

    payload = {
        "variant_index": int(variant_index),
        "frame_index": int(frame_index),
        "truth_star_index": source_ids,
        "truth_valid_mask": valid_mask,
        "truth_x_image_pix": x_image,
        "truth_y_image_pix": y_image,
    }
    for key, values in optional_static.items():
        if values is not None:
            payload[key] = values
    for key, values in optional_dynamic.items():
        if values is not None:
            payload[key] = values

    return truth_stars, payload


def load_dataset_batch(batch_root: str | Path, cfg: dict | None = None) -> DatasetContext:
    batch_path = Path(batch_root)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    frames_dir = batch_path / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.npz"))
    if not frame_paths:
        raise FileNotFoundError(f"No NPZ frames found under {frames_dir}")

    run_meta_path = batch_path / "run_meta.json"
    run_meta = {}
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    detector_width = run_meta.get("detector_width_pix")
    detector_height = run_meta.get("detector_height_pix", detector_width)
    truth_cfg = (cfg or {}).get("dataset", {})
    truth_origin = truth_cfg.get("truth_origin", "centered_pixels")
    truth_y_axis_up = bool(truth_cfg.get("truth_y_axis_up", False))

    truth_stars: list[TruthStar] = []
    truth_path = batch_path / "stars.ecsv"
    if truth_path.exists():
        table = Table.read(truth_path)
        colnames = set(table.colnames)
        truth_source_ids, run_meta = _select_truth_source_ids(table, run_meta)
        if detector_width is None:
            detector_width = int(run_meta.get("image_width", 0) or 0) or None
        if detector_height is None:
            detector_height = int(run_meta.get("image_height", 0) or 0) or detector_width

        for row_idx, row in enumerate(table):
            x_raw = float(row["x0"])
            y_raw = float(row["y0"])
            if truth_origin == "centered_pixels" and detector_width is not None and detector_height is not None:
                x_pix, y_pix = _centered_truth_to_pixel(
                    x_raw,
                    y_raw,
                    int(detector_width),
                    int(detector_height),
                    truth_y_axis_up,
                )
            else:
                x_pix = x_raw
                y_pix = y_raw

            mag = float(row["Kepler Mag"]) if "Kepler Mag" in colnames else None
            source_id = truth_source_ids[row_idx] if row_idx < len(truth_source_ids) else None

            truth_stars.append(
                TruthStar(
                    source_id=source_id,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    ra_deg=float(row["RA"]),
                    dec_deg=float(row["Dec"]),
                    mag=mag,
                    meta={
                        "truth_index": (
                            int(row["Truth Index"]) if "Truth Index" in colnames else None
                        ),
                        "field_id": int(row["Field ID"]) if "Field ID" in colnames else None,
                        "raw_star_id": int(row["Star ID"]) if "Star ID" in colnames else None,
                    },
                )
            )

    field_offset_x_pix = run_meta.get("field_offset_x_pix")
    field_offset_y_pix = run_meta.get("field_offset_y_pix")
    field_offset_source = None

    if field_offset_x_pix is not None or field_offset_y_pix is not None:
        field_offset_x_pix = 0.0 if field_offset_x_pix is None else float(field_offset_x_pix)
        field_offset_y_pix = 0.0 if field_offset_y_pix is None else float(field_offset_y_pix)
        field_offset_source = "run_meta"
    else:
        projection_model = (cfg or {}).get("layout", {}).get("projection_model", "sky_patch_linearized")
        if projection_model == "sky_patch_linearized":
            principal_point_x_pix, principal_point_y_pix = _resolve_default_principal_point(
                cfg,
                int(detector_width) if detector_width is not None else None,
                int(detector_height) if detector_height is not None else None,
            )
            estimated_x, estimated_y, estimate_meta = _estimate_field_offset_from_truth(
                truth_stars,
                run_meta.get("field_center_ra_deg"),
                run_meta.get("field_center_dec_deg"),
                run_meta.get("pixel_scale_arcsec_per_pix"),
                principal_point_x_pix,
                principal_point_y_pix,
            )
            if estimated_x is not None and estimated_y is not None:
                field_offset_x_pix = estimated_x
                field_offset_y_pix = estimated_y
                field_offset_source = "estimated_truth"
                run_meta = {
                    **run_meta,
                    **estimate_meta,
                }

    return DatasetContext(
        batch_root=batch_path,
        frame_paths=frame_paths,
        batch_center_ra_deg=run_meta.get("field_center_ra_deg"),
        batch_center_dec_deg=run_meta.get("field_center_dec_deg"),
        pixel_scale_arcsec_per_pix=run_meta.get("pixel_scale_arcsec_per_pix"),
        detector_width_pix=detector_width,
        detector_height_pix=detector_height,
        field_offset_x_pix=field_offset_x_pix,
        field_offset_y_pix=field_offset_y_pix,
        field_offset_source=field_offset_source,
        truth_stars=truth_stars,
        run_meta=run_meta,
    )


def load_dataset_batch_for_frame(npz_path: str | Path, cfg: dict | None = None) -> DatasetContext:
    frame_path = Path(npz_path)
    batch_root = frame_path.parent.parent
    return load_dataset_batch(batch_root, cfg=cfg)

def load_npz_frame(npz_path: str, detector_id: int = 0) -> RawFrame:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(path, allow_pickle=False)
    required = ["images", "time_s"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in {npz_path}")

    image = data["images"]
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    time_s = float(np.atleast_1d(data["time_s"])[0])

    variant_id = None
    if "variant_ids" in data:
        variant_id = int(np.atleast_1d(data["variant_ids"])[0])

    cadence_s = None
    if "cadence_s" in data:
        cadence_s = float(data["cadence_s"])

    coadd_start = int(data["coadd_start"]) if "coadd_start" in data else None
    coadd_stop = int(data["coadd_stop"]) if "coadd_stop" in data else None
    unit = str(data["unit"]) if "unit" in data else None

    truth_stars = None
    truth_payload = None
    truth_source = None
    extracted_truth_stars, extracted_truth_payload = _extract_npz_truth_payload(data)
    if extracted_truth_stars is not None and extracted_truth_payload is not None:
        truth_stars = extracted_truth_stars
        truth_payload = extracted_truth_payload
        truth_source = "npz_frame_truth"

    meta = {"npz_path": str(path)}
    if truth_stars is not None:
        meta["truth_stars"] = truth_stars
        meta["truth_source"] = truth_source
        meta["truth_payload"] = truth_payload

    return RawFrame(
        detector_id=detector_id,
        image=np.asarray(image, dtype=np.float64),
        time_s=time_s,
        cadence_s=cadence_s,
        coadd_start=coadd_start,
        coadd_stop=coadd_stop,
        unit=unit,
        variant_id=variant_id,
        meta=meta,
    )

def load_npz_sequence(npz_paths: list[str], detector_id: int = 0) -> list[RawFrame]:
    frames = []
    for path in npz_paths:
        frames.append(load_npz_frame(path, detector_id))
    return frames
