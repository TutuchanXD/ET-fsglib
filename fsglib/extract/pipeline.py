import numpy as np
from scipy import ndimage

from fsglib.common.types import PreprocessedFrame, StarCandidate


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


def _expand_existing_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    return (
        max(int(x0) - margin, 0),
        max(int(y0) - margin, 0),
        min(int(x1) + margin, width - 1),
        min(int(y1) + margin, height - 1),
    )


def _artifact_overlap_reason(
    artifact_masks: dict,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin: int,
) -> str | None:
    if not artifact_masks:
        return None
    x0, y0, x1, y1 = _expand_existing_bbox(bbox, image_shape, margin)
    for name, mask in artifact_masks.items():
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != image_shape:
            raise ValueError(
                f"artifact mask {name!r} shape {mask_arr.shape} does not match image shape {image_shape}"
            )
        if np.any(mask_arr[y0 : y1 + 1, x0 : x1 + 1]):
            return str(name)
    return None


def _finite_nonnegative_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extract.{name} must be a finite non-negative value") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"extract.{name} must be a finite non-negative value")
    return result


def _finite_nonnegative_int(value: object, name: str, default: int = 0) -> int:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extract.{name} must be a non-negative integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"extract.{name} must be a non-negative integer")
    return int(numeric)


def _hysteresis_segments(
    snr_map: np.ndarray,
    valid_mask: np.ndarray,
    seed_th: float,
    grow_th: float,
) -> tuple[np.ndarray, list[tuple[int, int, tuple[slice, slice]]]]:
    seed_mask = valid_mask & (snr_map > seed_th)
    grow_mask = valid_mask & (snr_map > grow_th)
    structure = np.ones((3, 3), dtype=bool)
    labeled_grow, _ = ndimage.label(grow_mask, structure=structure)
    seed_label_values = np.asarray(labeled_grow[seed_mask], dtype=np.int64)
    if seed_label_values.size == 0:
        return labeled_grow, []

    seed_counts = np.bincount(seed_label_values)
    object_slices = ndimage.find_objects(labeled_grow)

    segments: list[tuple[int, int, tuple[slice, slice]]] = []
    for label_id, num_seed_pixels in enumerate(seed_counts):
        if label_id == 0 or num_seed_pixels <= 0:
            continue
        object_index = label_id - 1
        if object_index >= len(object_slices):
            continue
        slices = object_slices[object_index]
        if slices is None:
            continue
        segments.append((int(label_id), int(num_seed_pixels), (slices[0], slices[1])))
    return labeled_grow, segments


def _variance_array(frame: PreprocessedFrame) -> np.ndarray:
    source = frame.variance_map
    if source is None:
        source = np.asarray(frame.noise_map, dtype=np.float64) ** 2
    variance = np.asarray(source, dtype=np.float64)
    if variance.shape == ():
        variance = np.full_like(frame.image, float(variance), dtype=np.float64)
    if variance.shape != frame.image.shape:
        raise ValueError(
            f"frame variance/noise shape {variance.shape} does not match image shape {frame.image.shape}"
        )
    return np.maximum(variance, 0.0)


def _centroid_covariance_cfg(cfg: dict) -> dict:
    return dict(cfg.get("extract", {}).get("centroid_covariance", {}))


def _centroid_min_sigma_pix(cfg: dict) -> float:
    cov_cfg = _centroid_covariance_cfg(cfg)
    value = cov_cfg.get("min_sigma_pix", 0.03)
    return _finite_nonnegative_float(value, "centroid_covariance.min_sigma_pix")


def _apply_centroid_covariance_floor(cov: np.ndarray, min_sigma_pix: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        cov = np.eye(2, dtype=np.float64) * min_sigma_pix**2
    cov = 0.5 * (cov + cov.T)
    if min_sigma_pix > 0.0:
        cov = cov + np.eye(2, dtype=np.float64) * min_sigma_pix**2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, min_sigma_pix**2)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _centroid_from_values(
    signal: np.ndarray,
    noise_var: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    min_sigma_pix: float,
    kernel_values: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    noise_var = np.asarray(noise_var, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if kernel_values is None:
        kernel_values = np.ones_like(signal, dtype=np.float64)
    else:
        kernel_values = np.asarray(kernel_values, dtype=np.float64)
    weights = signal * kernel_values
    flux = float(np.sum(weights))
    if flux <= 0.0 or not np.isfinite(flux):
        return np.nan, np.nan, flux, _apply_centroid_covariance_floor(
            np.full((2, 2), np.nan, dtype=np.float64),
            min_sigma_pix,
        )
    x = float(np.sum(xs * weights) / flux)
    y = float(np.sum(ys * weights) / flux)
    dx_dI = kernel_values * (xs - x) / flux
    dy_dI = kernel_values * (ys - y) / flux
    cov = np.array(
        [
            [np.sum(noise_var * dx_dI * dx_dI), np.sum(noise_var * dx_dI * dy_dI)],
            [np.sum(noise_var * dx_dI * dy_dI), np.sum(noise_var * dy_dI * dy_dI)],
        ],
        dtype=np.float64,
    )
    return x, y, flux, _apply_centroid_covariance_floor(cov, min_sigma_pix)


def _centroid_from_mask(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    *,
    min_sigma_pix: float,
    kernel: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    ys, xs = np.where(mask)
    kernel_values = None if kernel is None else kernel[mask]
    return _centroid_from_values(
        image[mask],
        variance[mask],
        xs,
        ys,
        min_sigma_pix=min_sigma_pix,
        kernel_values=kernel_values,
    )


def _weighted_centroid_from_mask(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    *,
    min_sigma_pix: float,
) -> tuple[float, float, float, np.ndarray]:
    return _centroid_from_mask(
        image,
        variance,
        mask,
        min_sigma_pix=min_sigma_pix,
    )


def _first_moment_in_bbox(
    image: np.ndarray,
    variance: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    min_sigma_pix: float,
) -> tuple[float, float, float, np.ndarray]:
    x0, y0, x1, y1 = bbox
    window = np.asarray(image[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64)
    variance_window = np.asarray(variance[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64)
    ys, xs = np.indices(window.shape, dtype=np.float64)
    return _centroid_from_values(
        window.ravel(),
        variance_window.ravel(),
        (x0 + xs).ravel(),
        (y0 + ys).ravel(),
        min_sigma_pix=min_sigma_pix,
    )


def _adaptive_moment_centroid_from_mask(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    *,
    min_sigma_pix: float,
) -> tuple[float, float, float, np.ndarray]:
    x0, y0, flux0, _ = _weighted_centroid_from_mask(
        image,
        variance,
        mask,
        min_sigma_pix=min_sigma_pix,
    )
    if flux0 <= 0.0 or not np.isfinite(x0) or not np.isfinite(y0):
        return x0, y0, flux0, _apply_centroid_covariance_floor(
            np.full((2, 2), np.nan, dtype=np.float64),
            min_sigma_pix,
        )

    ys, xs = np.where(mask)
    weights = np.maximum(np.asarray(image[mask], dtype=np.float64), 0.0)
    if float(np.sum(weights)) <= 0.0:
        return _weighted_centroid_from_mask(
            image,
            variance,
            mask,
            min_sigma_pix=min_sigma_pix,
        )
    dx = xs.astype(np.float64) - x0
    dy = ys.astype(np.float64) - y0
    norm = float(np.sum(weights))
    mxx = float(np.sum(weights * dx * dx) / norm)
    myy = float(np.sum(weights * dy * dy) / norm)
    mxy = float(np.sum(weights * dx * dy) / norm)
    moment_cov = np.array([[mxx, mxy], [mxy, myy]], dtype=np.float64)
    moment_cov = _apply_centroid_covariance_floor(moment_cov, min_sigma_pix)
    try:
        inv_cov = np.linalg.inv(moment_cov)
    except np.linalg.LinAlgError:
        return _weighted_centroid_from_mask(
            image,
            variance,
            mask,
            min_sigma_pix=min_sigma_pix,
        )
    kernel = np.zeros_like(image, dtype=np.float64)
    q = (
        inv_cov[0, 0] * dx * dx
        + 2.0 * inv_cov[0, 1] * dx * dy
        + inv_cov[1, 1] * dy * dy
    )
    kernel[mask] = np.exp(-0.5 * np.clip(q, 0.0, 100.0))
    return _centroid_from_mask(
        image,
        variance,
        mask,
        min_sigma_pix=min_sigma_pix,
        kernel=kernel,
    )


def _validate_psf_template_fit_interface(cfg: dict) -> None:
    template_path = cfg.get("psf", {}).get("template_bundle_path")
    if not template_path:
        raise ValueError(
            "extract.centroid_method=psf_template_fit requires psf.template_bundle_path"
        )
    raise NotImplementedError(
        "extract.centroid_method=psf_template_fit is reserved for #89; "
        "PR13 only defines the YAML/interface contract."
    )


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


def _blend_config(cfg: dict) -> dict:
    default_cfg = {
        "enabled": True,
        "policy": "flag_only",
        "peak_threshold_sigma": None,
    }
    default_cfg.update(dict(cfg.get("extract", {}).get("deblend", {})))
    policy = str(default_cfg.get("policy", "flag_only"))
    if policy not in {"flag_only", "reject"}:
        raise ValueError("extract.deblend.policy must be 'flag_only' or 'reject'")
    default_cfg["policy"] = policy
    return default_cfg


def _local_peak_summary(
    image: np.ndarray,
    seg: np.ndarray,
    snr_map: np.ndarray,
    local_max: np.ndarray,
    threshold_sigma: float,
) -> dict:
    peak_mask = seg & local_max & (snr_map > threshold_sigma)
    labeled, num = ndimage.label(peak_mask, structure=np.ones((3, 3), dtype=bool))
    peaks = []
    for label_id in range(1, num + 1):
        ys, xs = np.where(labeled == label_id)
        if xs.size == 0:
            continue
        values = image[ys, xs]
        idx = int(np.argmax(values))
        peaks.append(
            {
                "x_pix": int(xs[idx]),
                "y_pix": int(ys[idx]),
                "peak": float(values[idx]),
                "snr": float(snr_map[ys[idx], xs[idx]]),
            }
        )
    peaks.sort(key=lambda item: item["peak"], reverse=True)
    return {
        "num_local_peaks": len(peaks),
        "local_peaks": peaks,
    }


def extract_stars(frame: PreprocessedFrame, cfg: dict) -> list[StarCandidate]:
    image = frame.image
    noise = frame.noise_map
    mask = frame.valid_mask
    extract_cfg = cfg["extract"]
    centroid_method = str(extract_cfg.get("centroid_method", "weighted_centroid"))
    centroid_window_cfg = extract_cfg.get("centroid_window", {})
    if centroid_method == "psf_template_fit":
        _validate_psf_template_fit_interface(cfg)
    variance = _variance_array(frame)
    min_sigma_pix = _centroid_min_sigma_pix(cfg)
    blend_cfg = _blend_config(cfg)

    snr_map = np.where(mask, image / np.maximum(noise, 1e-6), 0.0)
    local_max = image == ndimage.maximum_filter(image, size=3, mode="nearest")

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

    labeled_segments, segments = _hysteresis_segments(snr_map, mask, seed_th, grow_th)
    candidates = []

    for label_id, num_seed_pixels, segment_slices in segments:
        y_slice, x_slice = segment_slices
        local_seg = labeled_segments[y_slice, x_slice] == label_id
        ys_local, xs_local = np.where(local_seg)
        y_offset = 0 if y_slice.start is None else int(y_slice.start)
        x_offset = 0 if x_slice.start is None else int(x_slice.start)
        ys = ys_local + y_offset
        xs = xs_local + x_offset
        if len(xs) == 0:
            continue

        area = len(xs)
        if area < extract_cfg["min_area"] or area > extract_cfg["max_area"]:
            continue

        seg = np.zeros_like(mask, dtype=bool)
        seg[y_slice, x_slice] = local_seg

        peak = float(np.max(image[seg]))
        peak_index = int(np.argmax(image[seg]))
        peak_x = int(xs[peak_index])
        peak_y = int(ys[peak_index])
        segment_bbox = _expanded_bbox(
            xs, ys, image.shape, int(extract_cfg.get("bbox_expand", 0))
        )
        peak_threshold_sigma = blend_cfg.get("peak_threshold_sigma")
        if peak_threshold_sigma is None:
            peak_threshold_sigma = seed_th
        else:
            peak_threshold_sigma = _finite_nonnegative_float(
                peak_threshold_sigma,
                "deblend.peak_threshold_sigma",
            )
        peak_summary = _local_peak_summary(
            image,
            seg,
            snr_map,
            local_max,
            peak_threshold_sigma,
        )
        blend_flag = bool(blend_cfg.get("enabled", True)) and peak_summary["num_local_peaks"] > 1
        if blend_flag and blend_cfg["policy"] == "reject":
            continue

        shape = _shape_metrics_from_mask(image, seg, peak)
        if extract_cfg.get("reject_degenerate_sources", False) and bool(
            shape["shape_degenerate"]
        ):
            continue
        min_fwhm_pix = extract_cfg.get("min_fwhm_pix")
        if min_fwhm_pix is not None:
            min_fwhm_pix = _finite_nonnegative_float(min_fwhm_pix, "min_fwhm_pix")
            if float(shape["fwhm_pix"]) < min_fwhm_pix:
                continue
        max_sharpness = extract_cfg.get("max_sharpness")
        if max_sharpness is not None:
            max_sharpness = _finite_nonnegative_float(max_sharpness, "max_sharpness")
            if float(shape["sharpness"]) > max_sharpness:
                continue
        if extract_cfg.get("reject_artifact_mask_overlap", False):
            artifact_mask_margin_pix = _finite_nonnegative_int(
                extract_cfg.get("artifact_mask_margin_pix", 0),
                "artifact_mask_margin_pix",
            )
            artifact_reason = _artifact_overlap_reason(
                getattr(frame, "artifact_masks", {}) or {},
                segment_bbox,
                image.shape,
                artifact_mask_margin_pix,
            )
            if artifact_reason is not None:
                continue
        max_ellipticity = extract_cfg.get("max_ellipticity")
        if max_ellipticity is not None:
            max_ellipticity = _finite_nonnegative_float(
                max_ellipticity,
                "max_ellipticity",
            )
            if float(shape["ellipticity"]) > max_ellipticity:
                continue

        if centroid_method == "weighted_centroid":
            x, y, flux, centroid_cov_pix = _weighted_centroid_from_mask(
                image,
                variance,
                seg,
                min_sigma_pix=min_sigma_pix,
            )
            centroid_bbox = segment_bbox
        elif centroid_method == "adaptive_moment_centroid":
            x, y, flux, centroid_cov_pix = _adaptive_moment_centroid_from_mask(
                image,
                variance,
                seg,
                min_sigma_pix=min_sigma_pix,
            )
            centroid_bbox = segment_bbox
        elif centroid_method in {
            "fixed_window_first_moment",
            "full_window_first_moment",
        }:
            window_size = int(centroid_window_cfg.get("size", 31))
            centroid_bbox = _fixed_window_bbox(peak_x, peak_y, image.shape, window_size)
            x, y, flux, centroid_cov_pix = _first_moment_in_bbox(
                image,
                variance,
                centroid_bbox,
                min_sigma_pix=min_sigma_pix,
            )
        else:
            raise ValueError(f"Unsupported extract.centroid_method: {centroid_method}")

        if flux <= 0 or not np.isfinite(x) or not np.isfinite(y):
            continue

        if _bbox_touches_edge(
            centroid_bbox, image.shape, int(extract_cfg.get("reject_edge_margin", 0))
        ):
            continue

        snr_denominator = max(float(np.sqrt(np.sum(noise[seg] ** 2))), 1e-12)
        snr = float(np.sum(image[seg]) / snr_denominator)
        cov_eigvals = np.linalg.eigvalsh(np.asarray(centroid_cov_pix, dtype=np.float64))
        sigma_x_pix = float(np.sqrt(max(float(centroid_cov_pix[0, 0]), 0.0)))
        sigma_y_pix = float(np.sqrt(max(float(centroid_cov_pix[1, 1]), 0.0)))
        sigma_radial_pix = float(np.sqrt(max(float(np.trace(centroid_cov_pix)), 0.0)))
        flags = {
            "centroid_method": centroid_method,
            "raw_centroid_x_pix": float(x),
            "raw_centroid_y_pix": float(y),
            "centroid_covariance_source": "noise_propagation",
            "centroid_cov_xx_pix2": float(centroid_cov_pix[0, 0]),
            "centroid_cov_xy_pix2": float(centroid_cov_pix[0, 1]),
            "centroid_cov_yy_pix2": float(centroid_cov_pix[1, 1]),
            "centroid_sigma_x_pix": sigma_x_pix,
            "centroid_sigma_y_pix": sigma_y_pix,
            "centroid_sigma_radial_pix": sigma_radial_pix,
            "centroid_sigma_major_pix": float(np.sqrt(max(float(cov_eigvals[-1]), 0.0))),
            "centroid_sigma_minor_pix": float(np.sqrt(max(float(cov_eigvals[0]), 0.0))),
            "centroid_min_sigma_pix": min_sigma_pix,
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
            "blend_flag": blend_flag,
            "deblend_policy": blend_cfg["policy"],
            "num_local_peaks": int(peak_summary["num_local_peaks"]),
            "local_peaks": peak_summary["local_peaks"],
        }

        candidates.append(
            StarCandidate(
                detector_id=frame.detector_id,
                source_id=len(candidates),
                x=x,
                y=y,
                flux=flux,
                peak=peak,
                area=area,
                snr=snr,
                bbox=centroid_bbox,
                centroid_cov_pix=centroid_cov_pix,
                shape=shape,
                flags=flags,
            )
        )

    return candidates
