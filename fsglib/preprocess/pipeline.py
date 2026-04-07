import numpy as np
from fsglib.common.types import RawFrame, PreprocessedFrame

def preprocess_frame(raw: RawFrame, calib: dict, cfg: dict) -> PreprocessedFrame:
    image = raw.image.copy()
    valid_mask = np.isfinite(image)

    image = np.where(valid_mask, image, 0.0)

    if cfg["preprocess"].get("enable_background_subtraction", True):
        background = estimate_background(image, valid_mask, cfg)
        image_sub = image - background
    else:
        background = 0.0
        image_sub = image

    noise_map = estimate_noise_map(image_sub, valid_mask, cfg)

    return PreprocessedFrame(
        detector_id=raw.detector_id,
        image=image_sub,
        background=background,
        noise_map=noise_map,
        valid_mask=valid_mask,
        preprocess_meta={}
    )

def estimate_background(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    vals = image[valid_mask]
    median = np.median(vals)
    return median

def estimate_noise_map(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    vals = image[valid_mask]
    sigma = np.std(vals)
    return np.full_like(image, fill_value=max(sigma, 1e-6), dtype=np.float64)
