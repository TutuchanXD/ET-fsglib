import numpy as np
from scipy import ndimage

from fsglib.common.types import RawFrame, PreprocessedFrame


_MIN_NOISE = 1.0e-6
_ELECTRON_UNITS = {"e", "electron", "electrons"}

_CALIBRATION_ORDER = [
    "finite_mask",
    "adc_clip",
    "saturation_guard",
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


def _detector_cfg(cfg: dict) -> dict:
    return cfg.get("detector", {})


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


def _positive_float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"preprocess.{name} must be configured")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"preprocess.{name} must be a positive finite value")
    return result


def _nonnegative_float(value: object, name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"preprocess.{name} must be a non-negative finite value")
    return result


def _required_nonnegative_float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"preprocess.{name} must be configured")
    return _nonnegative_float(value, name)


def _nonnegative_int(value: object, name: str, default: int = 0) -> int:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preprocess.{name} must be a non-negative integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"preprocess.{name} must be a non-negative integer")
    result = int(numeric)
    if result < 0:
        raise ValueError(f"preprocess.{name} must be a non-negative integer")
    return result


def _detector_finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"detector.{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"detector.{name} must be finite")
    return result


def _detector_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"detector.{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"detector.{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"detector.{name} must be a positive integer")
    return int(numeric)


def _unit_is_electron(unit: str | None) -> bool:
    if unit is None:
        return False
    return str(unit).strip().lower() in _ELECTRON_UNITS


def _gain_e_per_image_unit(unit: str | None, cfg: dict) -> float:
    if _unit_is_electron(unit):
        return 1.0
    return _positive_float(
        _preprocess_cfg(cfg).get("gain_e_per_dn"),
        "gain_e_per_dn",
    )


def _variance_unit(unit: str | None) -> str:
    if unit is None:
        return "image_unit^2"
    return f"{unit}^2"


def _robust_sigma(vals: np.ndarray) -> float:
    finite = np.asarray(vals, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if sigma > 0.0 and np.isfinite(sigma):
        return sigma
    std = float(np.std(finite))
    if std > 0.0 and np.isfinite(std):
        return std
    return 0.0


def _adc_clip_limits(cfg: dict) -> tuple[bool, float, float, int | None]:
    preprocess_cfg = _preprocess_cfg(cfg)
    detector_cfg = _detector_cfg(cfg)
    enabled = bool(preprocess_cfg.get("enable_adc_clip", True))

    min_value = _detector_finite_float(detector_cfg.get("adc_min_value", 0.0), "adc_min_value")

    bit_depth_value = detector_cfg.get("adc_bit_depth", 12)
    bit_depth = _detector_positive_int(bit_depth_value, "adc_bit_depth")
    max_value_config = detector_cfg.get("saturation_value")
    if max_value_config is None:
        if bit_depth is None or bit_depth <= 0:
            raise ValueError("detector.adc_bit_depth must be a positive integer")
        max_value = float((1 << bit_depth) - 1)
    else:
        max_value = _detector_finite_float(max_value_config, "saturation_value")

    if max_value <= min_value:
        if max_value_config is None:
            raise ValueError(
                "detector maximum derived from adc_bit_depth must be greater than "
                "detector.adc_min_value"
            )
        raise ValueError("detector.saturation_value must be greater than detector.adc_min_value")
    return enabled, min_value, max_value, bit_depth


def _apply_adc_clip(
    image: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    enabled, min_value, max_value, bit_depth = _adc_clip_limits(cfg)
    finite = valid_mask & np.isfinite(image)
    saturated_mask = finite & (image >= max_value)
    clipped_low = finite & (image < min_value)
    clipped_high = finite & (image > max_value)

    if enabled:
        clipped = np.asarray(image, dtype=np.float64).copy()
        clipped[finite] = np.clip(clipped[finite], min_value, max_value)
    else:
        clipped = image

    return clipped, saturated_mask, {
        "enabled": bool(enabled),
        "min_value": float(min_value),
        "max_value": float(max_value),
        "adc_bit_depth": bit_depth,
        "num_clipped_low_pixels": int(np.count_nonzero(clipped_low)) if enabled else 0,
        "num_clipped_high_pixels": int(np.count_nonzero(clipped_high)) if enabled else 0,
        "num_saturated_pixels": int(np.count_nonzero(saturated_mask)),
    }


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool, copy=True)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure)


def _sigma_clip_values(
    vals: np.ndarray,
    *,
    sigma_clip_k: float,
    max_iters: int,
) -> tuple[np.ndarray, int]:
    clipped = np.asarray(vals, dtype=np.float64)
    clipped = clipped[np.isfinite(clipped)]
    if clipped.size == 0:
        return clipped, 0

    original_count = clipped.size
    for _ in range(max_iters):
        if clipped.size == 0:
            break
        median = float(np.median(clipped))
        sigma = _robust_sigma(clipped)
        if sigma <= 0.0:
            keep = np.isclose(clipped, median)
        else:
            keep = np.abs(clipped - median) <= sigma_clip_k * sigma
        if np.all(keep):
            break
        clipped = clipped[keep]

    return clipped, int(original_count - clipped.size)


def _scalar_background(
    image: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict,
    method: str,
) -> tuple[float, dict]:
    preprocess_cfg = _preprocess_cfg(cfg)
    vals = image[valid_mask]
    if vals.size == 0:
        return 0.0, {
            "background_method_effective": method,
            "background_rms": 0.0,
            "background_num_clipped_pixels": 0,
        }

    if method == "median":
        used = np.asarray(vals, dtype=np.float64)
        num_clipped = 0
    elif method == "sigma_clip_global":
        used, num_clipped = _sigma_clip_values(
            vals,
            sigma_clip_k=float(preprocess_cfg.get("sigma_clip_k", 3.0)),
            max_iters=int(preprocess_cfg.get("sigma_clip_max_iters", 3)),
        )
        if used.size == 0:
            used = np.asarray(vals, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported preprocess.background_method: {method}")

    background = float(np.median(used))
    rms = max(_robust_sigma(used - background), _MIN_NOISE)
    return background, {
        "background_method_effective": method,
        "background_rms": rms,
        "background_num_clipped_pixels": int(num_clipped),
    }


def _mesh_background(
    image: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    preprocess_cfg = _preprocess_cfg(cfg)
    mesh_size = int(preprocess_cfg.get("background_mesh_size", 64))
    if mesh_size <= 0:
        raise ValueError("preprocess.background_mesh_size must be a positive integer")

    height, width = image.shape
    background = np.zeros_like(image, dtype=np.float64)
    rms_map = np.full_like(image, fill_value=_MIN_NOISE, dtype=np.float64)
    fallback, fallback_meta = _scalar_background(
        image,
        valid_mask,
        cfg,
        "sigma_clip_global",
    )
    total_clipped = 0

    for y0 in range(0, height, mesh_size):
        y1 = min(y0 + mesh_size, height)
        for x0 in range(0, width, mesh_size):
            x1 = min(x0 + mesh_size, width)
            block_valid = valid_mask[y0:y1, x0:x1]
            block_vals = image[y0:y1, x0:x1][block_valid]
            if block_vals.size == 0:
                block_background = float(fallback)
                block_rms = float(fallback_meta["background_rms"])
            else:
                used, num_clipped = _sigma_clip_values(
                    block_vals,
                    sigma_clip_k=float(preprocess_cfg.get("sigma_clip_k", 3.0)),
                    max_iters=int(preprocess_cfg.get("sigma_clip_max_iters", 3)),
                )
                if used.size == 0:
                    used = np.asarray(block_vals, dtype=np.float64)
                block_background = float(np.median(used))
                block_rms = max(_robust_sigma(used - block_background), _MIN_NOISE)
                total_clipped += int(num_clipped)
            background[y0:y1, x0:x1] = block_background
            rms_map[y0:y1, x0:x1] = block_rms

    return background, {
        "background_method_effective": "mesh_median",
        "background_rms": float(np.median(rms_map[valid_mask])) if np.any(valid_mask) else 0.0,
        "background_rms_map": rms_map,
        "background_mesh_size": mesh_size,
        "background_num_clipped_pixels": int(total_clipped),
    }


def _estimate_background_model(
    image: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray | float, dict]:
    preprocess_cfg = _preprocess_cfg(cfg)
    method = str(preprocess_cfg.get("background_method", "sigma_clip_global"))
    if method in {"median", "sigma_clip_global"}:
        return _scalar_background(image, valid_mask, cfg, method)
    if method == "mesh_median":
        return _mesh_background(image, valid_mask, cfg)
    raise ValueError(f"Unsupported preprocess.background_method: {method}")


def _empirical_noise_map(
    image: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    preprocess_cfg = _preprocess_cfg(cfg)
    background_method = str(preprocess_cfg.get("background_method", "sigma_clip_global"))
    mesh_size = int(preprocess_cfg.get("background_mesh_size", 64))

    if background_method == "mesh_median" and mesh_size > 0:
        noise_map = np.full_like(image, fill_value=_MIN_NOISE, dtype=np.float64)
        height, width = image.shape
        for y0 in range(0, height, mesh_size):
            y1 = min(y0 + mesh_size, height)
            for x0 in range(0, width, mesh_size):
                x1 = min(x0 + mesh_size, width)
                block_vals = image[y0:y1, x0:x1][valid_mask[y0:y1, x0:x1]]
                if block_vals.size == 0:
                    sigma = _MIN_NOISE
                else:
                    used, _ = _sigma_clip_values(
                        block_vals,
                        sigma_clip_k=float(preprocess_cfg.get("sigma_clip_k", 3.0)),
                        max_iters=int(preprocess_cfg.get("sigma_clip_max_iters", 3)),
                    )
                    sigma = max(_robust_sigma(used if used.size else block_vals), _MIN_NOISE)
                noise_map[y0:y1, x0:x1] = sigma
        return noise_map, {"empirical_noise_scope": "mesh"}

    vals = image[valid_mask]
    if vals.size == 0:
        sigma = _MIN_NOISE
    else:
        used, _ = _sigma_clip_values(
            vals,
            sigma_clip_k=float(preprocess_cfg.get("sigma_clip_k", 3.0)),
            max_iters=int(preprocess_cfg.get("sigma_clip_max_iters", 3)),
        )
        sigma = max(_robust_sigma(used if used.size else vals), _MIN_NOISE)
    return np.full_like(image, fill_value=sigma, dtype=np.float64), {
        "empirical_noise_scope": "global"
    }


def _poisson_read_noise_variance_map(
    image_for_photon_noise: np.ndarray,
    flat_response_for_variance: np.ndarray | None,
    valid_mask: np.ndarray,
    raw: RawFrame,
    output_unit: str | None,
    dark_current_map: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    preprocess_cfg = _preprocess_cfg(cfg)
    gain_e_per_output_unit = _gain_e_per_image_unit(output_unit, cfg)
    read_noise_e = _required_nonnegative_float(
        preprocess_cfg.get("read_noise_e"),
        "read_noise_e",
    )
    quantization_noise_e = _nonnegative_float(
        preprocess_cfg.get("quantization_noise_e"),
        "quantization_noise_e",
        default=0.0,
    )

    if flat_response_for_variance is None:
        flat_response = np.ones_like(image_for_photon_noise, dtype=np.float64)
        flat_response_propagated = False
    else:
        flat_response = np.asarray(flat_response_for_variance, dtype=np.float64)
        flat_response_propagated = True

    signal_e = np.maximum(image_for_photon_noise, 0.0) * gain_e_per_output_unit
    dark_current_source = "none"
    dark_e = np.zeros_like(image_for_photon_noise, dtype=np.float64)
    if dark_current_map is not None:
        if raw.cadence_s is None:
            raise ValueError(
                "raw.cadence_s is required to propagate dark-current shot noise"
            )
        dark_e = np.maximum(dark_current_map, 0.0) * float(raw.cadence_s)
        dark_e *= gain_e_per_output_unit
        dark_current_source = "calib.dark"
    elif (
        preprocess_cfg.get("dark_current_e_per_s") is not None
        and preprocess_cfg.get("enable_dark_subtraction", False)
    ):
        if raw.cadence_s is None:
            raise ValueError(
                "raw.cadence_s is required when preprocess.dark_current_e_per_s is configured"
            )
        dark_e.fill(
            _nonnegative_float(
                preprocess_cfg.get("dark_current_e_per_s"),
                "dark_current_e_per_s",
            )
            * float(raw.cadence_s)
        )
        dark_current_source = "preprocess.dark_current_e_per_s"

    variance_e2 = signal_e + dark_e + read_noise_e**2 + quantization_noise_e**2
    denominator = (gain_e_per_output_unit * flat_response) ** 2
    variance = np.zeros_like(variance_e2, dtype=np.float64)
    safe_denominator = np.isfinite(denominator) & (denominator > 0.0)
    variance[safe_denominator] = variance_e2[safe_denominator] / denominator[
        safe_denominator
    ]
    variance = np.where(valid_mask, np.maximum(variance, _MIN_NOISE**2), _MIN_NOISE**2)
    return variance, {
        "gain_e_per_output_unit": float(gain_e_per_output_unit),
        "read_noise_e": float(read_noise_e),
        "quantization_noise_e": float(quantization_noise_e),
        "dark_current_source": dark_current_source,
        "flat_response_propagated": flat_response_propagated,
        "flat_uncertainty_included": False,
    }


def _estimate_variance_and_noise(
    image_sub: np.ndarray,
    image_for_photon_noise: np.ndarray,
    flat_response_for_variance: np.ndarray | None,
    valid_mask: np.ndarray,
    raw: RawFrame,
    output_unit: str | None,
    dark_current_map: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    preprocess_cfg = _preprocess_cfg(cfg)
    model = str(preprocess_cfg.get("variance_model", "empirical_robust"))

    if model == "empirical_robust":
        noise_map, details = _empirical_noise_map(image_sub, valid_mask, cfg)
        variance_map = np.asarray(noise_map, dtype=np.float64) ** 2
    elif model == "poisson_read_noise":
        variance_map, details = _poisson_read_noise_variance_map(
            image_for_photon_noise,
            flat_response_for_variance,
            valid_mask,
            raw,
            output_unit,
            dark_current_map,
            cfg,
        )
        noise_map = np.sqrt(variance_map)
    else:
        raise ValueError(f"Unsupported preprocess.variance_model: {model}")

    meta = {
        "variance_model_configured": model,
        "variance_model_effective": model,
        "variance_unit": _variance_unit(output_unit),
        "noise_unit": output_unit or "image_unit",
        "variance_components": details,
    }
    return variance_map, noise_map, meta


def preprocess_frame(raw: RawFrame, calib: dict, cfg: dict) -> PreprocessedFrame:
    preprocess_cfg = _preprocess_cfg(cfg)
    image = np.asarray(raw.image, dtype=np.float64).copy()
    valid_mask = np.isfinite(image)
    image_shape = image.shape
    output_unit = raw.unit
    dark_current_map = None
    artifact_masks: dict[str, np.ndarray] = {}

    image = np.where(valid_mask, image, 0.0)
    image, saturated_mask, adc_meta = _apply_adc_clip(image, valid_mask, cfg)
    artifact_masks["saturated"] = saturated_mask
    saturation_guard_enabled = bool(preprocess_cfg.get("enable_saturation_guard", True))
    saturation_guard_radius = _nonnegative_int(
        preprocess_cfg.get("saturation_mask_dilation_pix", 0),
        "saturation_mask_dilation_pix",
    )
    saturation_guard_mask = _dilate_mask(saturated_mask, saturation_guard_radius)
    artifact_masks["saturation_guard"] = saturation_guard_mask
    if saturation_guard_enabled:
        valid_mask &= ~saturation_guard_mask

    preprocess_meta = {
        "input_unit": raw.unit,
        "output_unit": output_unit,
        "raw_image_shape": tuple(image_shape),
        "calibration_order": list(_CALIBRATION_ORDER),
        "calibration": {},
        "adc_clip": adc_meta,
        "artifact_counts": {
            "saturated": int(np.count_nonzero(saturated_mask)),
            "saturation_guard": int(np.count_nonzero(saturation_guard_mask)),
        },
        "artifact_policy": {
            "saturation_guard_applied": saturation_guard_enabled,
            "saturation_mask_dilation_pix": saturation_guard_radius,
        },
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
        dark_current_map = dark
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

    image_for_photon_noise = image
    flat_response_for_variance = None
    if preprocess_cfg.get("enable_flat_field", False):
        flat = _require_calibration_product(
            calib,
            "flat",
            "enable_flat_field",
            image_shape,
        ).astype(np.float64, copy=False)
        flat_valid = np.isfinite(flat) & (flat > 0.0)
        flat_response_for_variance = flat
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

    conversion_enabled = bool(preprocess_cfg.get("convert_to_electrons", False))
    conversion_meta = {
        "enabled": conversion_enabled,
        "applied": False,
        "input_unit": raw.unit,
        "output_unit": output_unit,
    }
    if conversion_enabled:
        gain_e_per_dn = _gain_e_per_image_unit(raw.unit, cfg)
        conversion_meta["gain_e_per_dn"] = float(gain_e_per_dn)
        if _unit_is_electron(raw.unit):
            output_unit = "electron"
            conversion_meta["output_unit"] = output_unit
            conversion_meta["reason"] = "input_already_electron"
        else:
            photon_noise_aliases_image = image_for_photon_noise is image
            image *= gain_e_per_dn
            if not photon_noise_aliases_image:
                image_for_photon_noise *= gain_e_per_dn
            if dark_current_map is not None:
                dark_current_map = dark_current_map * gain_e_per_dn
            output_unit = "electron"
            conversion_meta["applied"] = True
            conversion_meta["output_unit"] = output_unit
    preprocess_meta["output_unit"] = output_unit
    preprocess_meta["adu_to_electron_conversion"] = conversion_meta

    image = np.where(valid_mask, image, 0.0)

    if preprocess_cfg.get("enable_background_subtraction", True):
        background, background_meta = _estimate_background_model(image, valid_mask, cfg)
        image_sub = image - background
    else:
        background = 0.0
        background_meta = {
            "background_method_effective": "none",
            "background_rms": 0.0,
            "background_num_clipped_pixels": 0,
        }
        image_sub = image

    image_sub = np.where(valid_mask, image_sub, 0.0)
    variance_map, noise_map, variance_meta = _estimate_variance_and_noise(
        image_sub,
        image_for_photon_noise,
        flat_response_for_variance,
        valid_mask,
        raw,
        output_unit,
        dark_current_map,
        cfg,
    )

    preprocess_meta["num_finite_input_pixels"] = int(np.count_nonzero(np.isfinite(raw.image)))
    preprocess_meta["num_bad_pixels"] = bad_pixel_count
    preprocess_meta["num_invalid_pixels"] = int(valid_mask.size - np.count_nonzero(valid_mask))
    background_enabled = bool(preprocess_cfg.get("enable_background_subtraction", True))
    background_method_configured = preprocess_cfg.get(
        "background_method",
        "sigma_clip_global",
    )
    preprocess_meta["background_subtraction_enabled"] = background_enabled
    preprocess_meta["background_method_configured"] = background_method_configured
    preprocess_meta["background_method_effective"] = background_meta[
        "background_method_effective"
    ]
    preprocess_meta["background_method"] = background_meta["background_method_effective"]
    for key, value in background_meta.items():
        if key != "background_rms_map":
            preprocess_meta[key] = value
    preprocess_meta.update(variance_meta)

    return PreprocessedFrame(
        detector_id=raw.detector_id,
        image=image_sub,
        background=background,
        noise_map=noise_map,
        valid_mask=valid_mask,
        variance_map=variance_map,
        preprocess_meta=preprocess_meta,
        artifact_masks=artifact_masks,
    )


def estimate_background(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    background, _ = _estimate_background_model(image, valid_mask, cfg)
    return background


def estimate_noise_map(image: np.ndarray, valid_mask: np.ndarray, cfg: dict):
    noise_map, _ = _empirical_noise_map(image, valid_mask, cfg)
    return noise_map