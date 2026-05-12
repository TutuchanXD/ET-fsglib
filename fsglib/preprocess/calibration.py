from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]

_CALIBRATION_PRODUCTS = {
    "bias": {
        "enabled_key": "enable_bias_subtraction",
        "path_key": "bias_frame_path",
        "kind": "additive",
    },
    "dark": {
        "enabled_key": "enable_dark_subtraction",
        "path_key": "dark_current_path",
        "kind": "additive",
    },
    "flat": {
        "enabled_key": "enable_flat_field",
        "path_key": "flat_field_path",
        "kind": "flat",
    },
    "bad_pixel_mask": {
        "enabled_key": "enable_bad_pixel_mask",
        "path_key": "bad_pixel_mask_path",
        "kind": "mask",
    },
    "fpn_residual": {
        "enabled_key": "enable_fpn_subtraction",
        "path_key": "fpn_residual_map_path",
        "kind": "additive",
    },
}


def _preprocess_cfg(cfg: dict) -> dict:
    return cfg.get("preprocess", {})


def _resolve_calibration_path(path_value: str | Path, path_key: str) -> Path:
    if path_value is None or str(path_value).strip() == "":
        raise ValueError(
            f"preprocess.{path_key} must be configured when its calibration is enabled"
        )

    raw_path = Path(path_value).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [Path.cwd() / raw_path, _REPO_ROOT / raw_path]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Calibration asset for preprocess.{path_key} was not found; tried: {tried}"
    )


def _load_calibration_array(path: Path) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False), "npy"
    if suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if "data" in payload.files:
            return payload["data"], "npz:data"
        if len(payload.files) == 1:
            key = payload.files[0]
            return payload[key], f"npz:{key}"
        keys = ", ".join(payload.files)
        raise ValueError(
            f"Calibration asset {path} is an npz with multiple arrays; "
            f"expected key 'data', got keys: {keys}"
        )
    raise ValueError(
        f"Unsupported calibration asset format for {path}; expected .npy or .npz"
    )


def _as_2d_numeric(name: str, array: Any) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(
            f"{name} calibration product must be a 2-D array, got shape {result.shape}"
        )
    return result


def _as_additive_map(name: str, array: Any) -> np.ndarray:
    result = _as_2d_numeric(name, array)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} calibration product must contain only finite values")
    return result


def _as_bad_pixel_mask(array: Any) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim != 2:
        raise ValueError(
            "bad_pixel_mask calibration product must be a 2-D array, "
            f"got shape {result.shape}"
        )
    if result.dtype == bool:
        return result.astype(bool, copy=False)

    numeric = np.asarray(result, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(
            "bad_pixel_mask calibration product must not contain non-finite values"
        )
    unique = set(np.unique(numeric).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("bad_pixel_mask calibration product must be bool or numeric 0/1")
    return numeric.astype(bool)


def _normalize_product(name: str, kind: str, array: Any) -> np.ndarray:
    if kind == "mask":
        return _as_bad_pixel_mask(array)
    if kind == "flat":
        return _as_2d_numeric(name, array)
    return _as_additive_map(name, array)


def load_calibration_products(cfg: dict) -> dict:
    """Load configured detector calibration products from YAML paths."""
    preprocess_cfg = _preprocess_cfg(cfg)
    products: dict[str, Any] = {"meta": {}}

    for name, spec in _CALIBRATION_PRODUCTS.items():
        enabled_key = spec["enabled_key"]
        if not bool(preprocess_cfg.get(enabled_key, False)):
            continue

        path_key = spec["path_key"]
        resolved_path = _resolve_calibration_path(preprocess_cfg.get(path_key), path_key)
        raw_array, array_format = _load_calibration_array(resolved_path)
        products[name] = _normalize_product(name, spec["kind"], raw_array)
        products["meta"][name] = {
            "path": str(resolved_path),
            "format": array_format.split(":", 1)[0],
            "array_key": array_format.split(":", 1)[1] if ":" in array_format else None,
            "shape": tuple(products[name].shape),
            "dtype": str(products[name].dtype),
        }

    return products
