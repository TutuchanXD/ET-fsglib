import numpy as np
from fsglib.common.types import RawFrame, PreprocessedFrame


_CALIBRATION_ORDER = [
    "finite_mask",
    "bias_subtraction",
    "dark_subtraction",
    "fpn_subtraction",
    "flat_field",
    "bad_pixel_mask",
    "background_subtraction",
    "noise_estimation",
]


def _preprocess_cfg(cfg: dict) -> dict:
    return cfg.get("preprocess", {})


def _calibration_meta(calib: dict, name: str) -> dict:
    return dict(calib.get("meta", {}).get(name, {}))


def _record_calibration(
    calib: dict,
    preprocess_meta: dict,
    name: str,
    *,
    enabled: bool,
    applied: bool,
    details: dict | None = None,
) -> None:
    payload = _calibration_meta(calib, name)
    payload.update(
        {
            "enabled": bool(enabled),
            "applied": bool(applied),
        }
    )
    if details:
        payload.update(details)
    preprocess_meta["calibration"][name] = payload


def _require_calibration_product(
    calib: dict,
    name: str,
    flag_name: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    if name not in calib:
        raise ValueError(
            f"preprocess.{flag_name} is true but calibration product {name!r} is missing"
        )
    product = np.asarray(calib[name])
    if product.shape != image_shape:
        raise ValueError(
            f"{name} calibration shape {product.shape} does not match raw image shape {image_shape}"
        )
    return product


def _require_finite_float_map(
    calib: dict,
    name: str,
    flag_name: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    product = _require_calibration_product(calib, name, flag_name, image_shape).astype(
        np.float64,
        copy=False,
    )
    if not np.all(np.isfinite(product)):
        raise ValueError(f"{name} calibration product must contain only finite values")
    return product


def _require_bad_pixel_mask(calib: dict, image_shape: tuple[int, int]) -> np.ndarray:
    product = _require_calibration_product(
        calib,
        "bad_pixel_mask",
        "enable_bad_pixel_mask",
        image_shape,
    )
    if product.dtype == bool:
        return product.astype(bool, copy=False)

    numeric = np.asarray(product, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("bad_pixel_mask calibration product must not contain non-finite values")
    unique = set(np.unique(numeric).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("bad_pixel_mask calibration product must be bool or numeric 0/1")
    return numeric.astype(bool)


def preprocess_frame(raw: RawFrame, calib: dict, cfg: dict) -> PreprocessedFrame:
    preprocess_cfg = _preprocess_cfg(cfg)
    image = np.asarray(raw.image, dtype=np.float64).copy()
    valid_mask = np.isfinite(image)
    image_shape = image.shape

    image = np.where(valid_mask, image, 0.0)
    preprocess_meta = {
        "input_unit": raw.unit,
        "output_unit": raw.unit,
        "raw_image_shape": tuple(image_shape),
        "calibration_order": list(_CALIBRATION_ORDER),
        "calibration": {},
    }

    if preprocess_cfg.get("enable_bias_subtraction", False):
        bias = _require_finite_float_map(
            calib,
            "bias",
            "enable_bias_subtraction",
            image_shape,
        )
        image = image - bias
        _record_calibration(
            calib,
            preprocess_meta,
            "bias",
            enabled=True,
            applied=True,
            details={"shape": tuple(bias.shape)},
        )
    else:
        _record_calibration(calib, preprocess_meta, "bias", enabled=False, applied=False)

    if preprocess_cfg.get("enable_dark_subtraction", False):
        if raw.cadence_s is None:
            raise ValueError(
                "raw.cadence_s is required when preprocess.enable_dark_subtraction is true"
            )
        dark = _require_finite_float_map(
            calib,
            "dark",
            "enable_dark_subtraction",
            image_shape,
        )
        image = image - dark * float(raw.cadence_s)
        _record_calibration(
            calib,
            preprocess_meta,
            "dark",
            enabled=True,
            applied=True,
            details={"shape": tuple(dark.shape), "cadence_s": float(raw.cadence_s)},
        )
    else:
        _record_calibration(calib, preprocess_meta, "dark", enabled=False, applied=False)

    if preprocess_cfg.get("enable_fpn_subtraction", False):
        fpn = _require_finite_float_map(
            calib,
            "fpn_residual",
            "enable_fpn_subtraction",
            image_shape,
        )
        image = image - fpn
        _record_calibration(
            calib,
            preprocess_meta,
            "fpn_residual",
            enabled=True,
            applied=True,
            details={"shape": tuple(fpn.shape)},
        )
    else:
        _record_calibration(
            calib,
            preprocess_meta,
            "fpn_residual",
            enabled=False,
            applied=False,
        )

    if preprocess_cfg.get("enable_flat_field", False):
        flat = _require_calibration_product(
            calib,
            "flat",
            "enable_flat_field",
            image_shape,
        ).astype(np.float64, copy=False)
        flat_valid = np.isfinite(flat) & (flat > 0.0)
        corrected = np.zeros_like(image, dtype=np.float64)
        corrected[flat_valid] = image[flat_valid] / flat[flat_valid]
        image = corrected
        valid_mask &= flat_valid
        num_invalid_flat_pixels = int(
            np.size(flat_valid) - np.count_nonzero(flat_valid)
        )
        _record_calibration(
            calib,
            preprocess_meta,
            "flat",
            enabled=True,
            applied=True,
            details={
                "shape": tuple(flat.shape),
                "num_invalid_flat_pixels": num_invalid_flat_pixels,
            },
        )
    else:
        _record_calibration(calib, preprocess_meta, "flat", enabled=False, applied=False)

    bad_pixel_count = 0
    if preprocess_cfg.get("enable_bad_pixel_mask", False):
        bad_pixel_mask = _require_bad_pixel_mask(calib, image_shape)
        bad_pixel_count = int(np.count_nonzero(bad_pixel_mask))
        valid_mask &= ~bad_pixel_mask
        _record_calibration(
            calib,
            preprocess_meta,
            "bad_pixel_mask",
            enabled=True,
            applied=True,
            details={
                "shape": tuple(bad_pixel_mask.shape),
                "num_bad_pixels": bad_pixel_count,
            },
        )
    else:
        _record_calibration(
            calib,
            preprocess_meta,
            "bad_pixel_mask",
            enabled=False,
            applied=False,
        )

    image = np.where(valid_mask, image, 0.0)

    if preprocess_cfg.get("enable_background_subtraction", True):
        background = estimate_background(image, valid_mask, cfg)
        image_sub = image - background
    else:
        background = 0.0
        image_sub = image

    image_sub = np.where(valid_mask, image_sub, 0.0)
    noise_map = estimate_noise_map(image_sub, valid_mask, cfg)
    variance_map = np.asarray(noise_map, dtype=np.float64) ** 2

    preprocess_meta["num_finite_input_pixels"] = int(np.count_nonzero(np.isfinite(raw.image)))
    preprocess_meta["num_bad_pixels"] = bad_pixel_count
    preprocess_meta["num_invalid_pixels"] = int(valid_mask.size - np.count_nonzero(valid_mask))
    preprocess_meta["background_subtraction_enabled"] = bool(
        preprocess_cfg.get("enable_background_subtraction", True)
    )
    preprocess_meta["background_method"] = preprocess_cfg.get("background_method", "median")

    return PreprocessedFrame(
        detector_id=raw.detector_id,
        image=image_sub,
        background=background,
        noise_map=noise_map,
        valid_mask=valid_mask,
        variance_map=variance_map,
        preprocess_meta=preprocess_meta,
    )


def estimate_background(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    vals = image[valid_mask]
    if vals.size == 0:
        return 0.0
    median = np.median(vals)
    return median


def estimate_noise_map(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    vals = image[valid_mask]
    if vals.size == 0:
        return np.full_like(image, fill_value=1e-6, dtype=np.float64)
    sigma = np.std(vals)
    return np.full_like(image, fill_value=max(sigma, 1e-6), dtype=np.float64)
