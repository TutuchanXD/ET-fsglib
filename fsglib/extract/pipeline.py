import numpy as np
from scipy import ndimage

from fsglib.common.types import PreprocessedFrame, StarCandidate
from fsglib.extract.bias import predict_centroid_bias, resolve_bias_correction_config


def _expanded_bbox(
    xs: np.ndarray, ys: np.ndarray, image_shape: tuple[int, int], expand: int
) -> tuple[int, int, int, int]:
    height, width = image_shape
    return (
        max(int(xs.min()) - expand, 0),
        max(int(ys.min()) - expand, 0),
        min(int(xs.max()) + expand, width - 1),
        min(int(ys.max()) + expand, height - 1),
    )


def _fixed_window_bbox(
    center_x: int, center_y: int, image_shape: tuple[int, int], size: int
) -> tuple[int, int, int, int]:
    if size <= 0 or size % 2 == 0:
        raise ValueError(
            f"centroid_window.size must be a positive odd integer, got {size}"
        )

    height, width = image_shape
    half = size // 2
    x0 = center_x - half
    y0 = center_y - half
    x1 = center_x + half
    y1 = center_y + half

    if x0 < 0:
        x1 = min(width - 1, x1 - x0)
        x0 = 0
    if y0 < 0:
        y1 = min(height - 1, y1 - y0)
        y0 = 0
    if x1 >= width:
        shift = x1 - (width - 1)
        x0 = max(0, x0 - shift)
        x1 = width - 1
    if y1 >= height:
        shift = y1 - (height - 1)
        y0 = max(0, y0 - shift)
        y1 = height - 1

    return int(x0), int(y0), int(x1), int(y1)


def _bbox_touches_edge(
    bbox: tuple[int, int, int, int], image_shape: tuple[int, int], margin: int
) -> bool:
    if margin <= 0:
        return False
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    return x0 < margin or y0 < margin or x1 >= width - margin or y1 >= height - margin


def _finite_nonnegative_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extract.{name} must be a finite non-negative value") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"extract.{name} must be a finite non-negative value")
    return result


def _hysteresis_segments(
    snr_map: np.ndarray,
    valid_mask: np.ndarray,
    extract_cfg: dict,
) -> list[tuple[np.ndarray, int]]:
    seed_th = _finite_nonnegative_float(
        extract_cfg["seed_threshold_sigma"],
        "seed_threshold_sigma",
    )
    grow_th = _finite_nonnegative_float(
        extract_cfg.get("grow_threshold_sigma", seed_th),
        "grow_threshold_sigma",
    )
    if grow_th > seed_th:
        raise ValueError(
            "extract.grow_threshold_sigma must be less than or equal to "
            "extract.seed_threshold_sigma"
        )

    seed_mask = valid_mask & (snr_map > seed_th)
    grow_mask = valid_mask & (snr_map > grow_th)
    structure = np.ones((3, 3), dtype=bool)
    labeled_grow, _ = ndimage.label(grow_mask, structure=structure)
    seed_labels = np.unique(labeled_grow[seed_mask])

    segments: list[tuple[np.ndarray, int]] = []
    for label_id in seed_labels:
        if int(label_id) == 0:
            continue
        seg = labeled_grow == label_id
        num_seed_pixels = int(np.count_nonzero(seed_mask & seg))
        if num_seed_pixels > 0:
            segments.append((seg, num_seed_pixels))
    return segments


def _weighted_centroid_from_mask(
    image: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    flux = float(np.sum(image[mask]))
    if flux <= 0.0 or not np.isfinite(flux):
        return np.nan, np.nan, flux
    ys, xs = np.where(mask)
    x = float(np.sum(xs * image[mask]) / flux)
    y = float(np.sum(ys * image[mask]) / flux)
    return x, y, flux


def _first_moment_in_bbox(
    image: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[float, float, float]:
    x0, y0, x1, y1 = bbox
    window = np.asarray(image[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64)
    flux = float(np.sum(window))
    if flux <= 0.0 or not np.isfinite(flux):
        return np.nan, np.nan, flux

    ys, xs = np.indices(window.shape, dtype=np.float64)
    x = float(x0 + np.sum(xs * window) / flux)
    y = float(y0 + np.sum(ys * window) / flux)
    return x, y, flux


def _shape_metrics_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    peak: float,
) -> dict:
    ys, xs = np.where(mask)
    weights = np.maximum(np.asarray(image[mask], dtype=np.float64), 0.0)
    flux = float(np.sum(weights))
    area = int(xs.size)
    if area <= 1 or flux <= 0.0 or not np.isfinite(flux):
        return {
            "shape_degenerate": True,
            "sigma_major_pix": 0.0,
            "sigma_minor_pix": 0.0,
            "theta_rad": 0.0,
            "ellipticity": 0.0,
            "fwhm_pix": 0.0,
            "sharpness": 1.0 if peak > 0.0 else 0.0,
            "roundness": 1.0,
        }

    xbar = float(np.sum(xs * weights) / flux)
    ybar = float(np.sum(ys * weights) / flux)
    dx = xs - xbar
    dy = ys - ybar
    mxx = float(np.sum(weights * dx * dx) / flux)
    myy = float(np.sum(weights * dy * dy) / flux)
    mxy = float(np.sum(weights * dx * dy) / flux)
    cov = np.array([[mxx, mxy], [mxy, myy]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    lambda_min = float(eigvals[0])
    lambda_max = float(eigvals[1])

    if lambda_max <= 0.0 or not np.isfinite(lambda_max):
        ellipticity = 0.0
        sigma_major = 0.0
        sigma_minor = 0.0
        theta = 0.0
        degenerate = True
    else:
        sigma_major = float(np.sqrt(lambda_max))
        sigma_minor = float(np.sqrt(lambda_min))
        ellipticity = float(1.0 - np.sqrt(lambda_min / lambda_max))
        major_vec = eigvecs[:, 1]
        theta = float(np.arctan2(major_vec[1], major_vec[0]))
        degenerate = bool(lambda_min <= 0.0)

    fwhm_pix = float(2.354820045 * sigma_major)
    mean_surface_brightness = flux / float(area)
    sharpness = float(peak / mean_surface_brightness) if mean_surface_brightness > 0.0 else 0.0
    roundness = float(sigma_minor / sigma_major) if sigma_major > 0.0 else 1.0
    return {
        "shape_degenerate": degenerate,
        "sigma_major_pix": sigma_major,
        "sigma_minor_pix": sigma_minor,
        "theta_rad": theta,
        "ellipticity": ellipticity,
        "fwhm_pix": fwhm_pix,
        "sharpness": sharpness,
        "roundness": roundness,
    }


def extract_stars(frame: PreprocessedFrame, cfg: dict) -> list[StarCandidate]:
    image = frame.image
    noise = frame.noise_map
    mask = frame.valid_mask
    extract_cfg = cfg["extract"]
    centroid_method = str(extract_cfg.get("centroid_method", "weighted_centroid"))
    centroid_window_cfg = extract_cfg.get("centroid_window", {})
    bias_cfg = resolve_bias_correction_config(cfg)

    snr_map = np.where(mask, image / np.maximum(noise, 1e-6), 0.0)

    seed_th = _finite_nonnegative_float(
        extract_cfg["seed_threshold_sigma"],
        "seed_threshold_sigma",
    )
    grow_th = _finite_nonnegative_float(
        extract_cfg.get("grow_threshold_sigma", seed_th),
        "grow_threshold_sigma",
    )
    segments = _hysteresis_segments(snr_map, mask, extract_cfg)
    candidates = []

    for seg, num_seed_pixels in segments:
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue

        area = len(xs)
        if area < extract_cfg["min_area"] or area > extract_cfg["max_area"]:
            continue

        peak = float(np.max(image[seg]))
        peak_index = int(np.argmax(image[seg]))
        peak_x = int(xs[peak_index])
        peak_y = int(ys[peak_index])
        shape = _shape_metrics_from_mask(image, seg, peak)
        max_ellipticity = extract_cfg.get("max_ellipticity")
        if max_ellipticity is not None:
            max_ellipticity = _finite_nonnegative_float(
                max_ellipticity,
                "max_ellipticity",
            )
            if float(shape["ellipticity"]) > max_ellipticity:
                continue

        segment_bbox = _expanded_bbox(
            xs, ys, image.shape, int(extract_cfg.get("bbox_expand", 0))
        )

        if centroid_method == "weighted_centroid":
            x, y, flux = _weighted_centroid_from_mask(image, seg)
            centroid_bbox = segment_bbox
        elif centroid_method in {
            "fixed_window_first_moment",
            "full_window_first_moment",
        }:
            window_size = int(centroid_window_cfg.get("size", 31))
            centroid_bbox = _fixed_window_bbox(peak_x, peak_y, image.shape, window_size)
            x, y, flux = _first_moment_in_bbox(image, centroid_bbox)
        else:
            raise ValueError(f"Unsupported extract.centroid_method: {centroid_method}")

        if flux <= 0 or not np.isfinite(x) or not np.isfinite(y):
            continue

        if _bbox_touches_edge(
            centroid_bbox, image.shape, int(extract_cfg.get("reject_edge_margin", 0))
        ):
            continue

        snr = float(np.sum(image[seg]) / np.sqrt(np.sum(noise[seg] ** 2)))
        corrected_x = x
        corrected_y = y
        flags = {
            "centroid_method": centroid_method,
            "raw_centroid_x_pix": float(x),
            "raw_centroid_y_pix": float(y),
            "segment_bbox": segment_bbox,
            "centroid_bbox": centroid_bbox,
            "peak_x_pix": peak_x,
            "peak_y_pix": peak_y,
            "segmentation_mode": "hysteresis",
            "seed_threshold_sigma": seed_th,
            "grow_threshold_sigma": grow_th,
            "num_seed_pixels": num_seed_pixels,
            "shape_filter_passed": True,
            "shape_ellipticity": float(shape["ellipticity"]),
            "shape_fwhm_pix": float(shape["fwhm_pix"]),
            "shape_sharpness": float(shape["sharpness"]),
            "shape_roundness": float(shape["roundness"]),
            "shape_degenerate": bool(shape["shape_degenerate"]),
        }
        if bias_cfg is not None:
            bias_x, bias_y = predict_centroid_bias(x, y, bias_cfg)
            corrected_x = float(x - bias_x)
            corrected_y = float(y - bias_y)
            flags.update(
                {
                    "centroid_bias_corrected": True,
                    "predicted_bias_x_pix": float(bias_x),
                    "predicted_bias_y_pix": float(bias_y),
                    "bias_profile": bias_cfg["profile_name"],
                }
            )
        else:
            flags["centroid_bias_corrected"] = False

        candidates.append(
            StarCandidate(
                detector_id=frame.detector_id,
                source_id=len(candidates),
                x=corrected_x,
                y=corrected_y,
                flux=flux,
                peak=peak,
                area=area,
                snr=snr,
                bbox=centroid_bbox,
                shape=shape,
                flags=flags,
            )
        )

    return candidates
