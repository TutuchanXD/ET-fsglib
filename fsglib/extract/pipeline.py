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


def extract_stars(frame: PreprocessedFrame, cfg: dict) -> list[StarCandidate]:
    image = frame.image
    noise = frame.noise_map
    mask = frame.valid_mask
    extract_cfg = cfg["extract"]
    centroid_method = str(extract_cfg.get("centroid_method", "weighted_centroid"))
    centroid_window_cfg = extract_cfg.get("centroid_window", {})
    bias_cfg = resolve_bias_correction_config(cfg)

    snr_map = np.where(mask, image / np.maximum(noise, 1e-6), 0.0)

    seed_th = extract_cfg["seed_threshold_sigma"]
    detect_mask = snr_map > seed_th

    labeled, num = ndimage.label(detect_mask)
    candidates = []

    for label_id in range(1, num + 1):
        seg = labeled == label_id
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
                shape={},
                flags=flags,
            )
        )

    return candidates
