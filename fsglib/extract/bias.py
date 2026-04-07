import json
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def local_phase_from_pixel_coord(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.round(values)


def _resolve_bias_table_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_path = Path.cwd() / candidate
    if cwd_path.exists():
        return cwd_path

    repo_path = REPO_ROOT / candidate
    return repo_path


@lru_cache(maxsize=16)
def _load_bias_table(table_path: str) -> dict:
    resolved = _resolve_bias_table_path(table_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Bias correction table not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _load_compiled_profile(table_path: str, calibration_key: str) -> dict:
    payload = _load_bias_table(table_path)
    rows = payload.get("profile_rows", [])
    if not rows:
        raise ValueError(f"Bias correction table is empty: {table_path}")

    measured_x_key = f"{calibration_key}_x_pix"
    measured_y_key = f"{calibration_key}_y_pix"
    bias_x_key = f"{calibration_key}_dx_err_pix"
    bias_y_key = f"{calibration_key}_dy_err_pix"

    try:
        measured_x = np.asarray([row[measured_x_key] for row in rows], dtype=np.float64)
        measured_y = np.asarray([row[measured_y_key] for row in rows], dtype=np.float64)
        bias_x = np.asarray([row[bias_x_key] for row in rows], dtype=np.float64)
        bias_y = np.asarray([row[bias_y_key] for row in rows], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(
            f"Bias correction table {table_path} is missing calibration field {exc!s} "
            f"for calibration_key={calibration_key!r}"
        ) from exc

    return {
        "payload": payload,
        "train_phase_x": local_phase_from_pixel_coord(measured_x),
        "train_phase_y": local_phase_from_pixel_coord(measured_y),
        "bias_x": bias_x,
        "bias_y": bias_y,
    }


def resolve_bias_correction_config(cfg: dict) -> dict | None:
    extract_cfg = cfg.get("extract", {})
    runtime_cfg = dict(extract_cfg.get("bias_correction", {}))
    if not runtime_cfg.get("enabled", False):
        return None

    profile_name = runtime_cfg.get("profile")
    bias_profiles_cfg = cfg.get("bias_profiles", {})
    profile_cfg: dict
    resolved_name: str

    if profile_name:
        profile_cfg = dict(bias_profiles_cfg.get("profiles", {}).get(profile_name, {}))
        if not profile_cfg:
            raise KeyError(f"Unknown bias correction profile: {profile_name}")
        resolved_name = str(profile_name)
    else:
        psf_model_key = str(
            runtime_cfg.get("psf_model_key")
            or cfg.get("psf", {}).get("active_model_key")
            or ""
        ).strip()
        if not psf_model_key:
            raise ValueError(
                "extract.bias_correction.enabled is true but no profile or psf.active_model_key was configured"
            )

        profile_cfg = dict(
            bias_profiles_cfg.get("by_psf_model", {}).get(psf_model_key, {})
        )
        if not profile_cfg:
            raise KeyError(
                f"No bias correction table registered for psf.active_model_key={psf_model_key!r}"
            )
        resolved_name = psf_model_key

    merged_cfg = {**profile_cfg, **runtime_cfg}
    calibration_key = str(merged_cfg.get("calibration_key", "full_window"))
    table_path = merged_cfg.get("bias_table_path")
    if not table_path:
        raise ValueError(
            f"Bias correction profile {profile_name} does not define bias_table_path"
        )

    compiled = _load_compiled_profile(str(table_path), calibration_key)
    merged_cfg["profile_name"] = resolved_name
    merged_cfg["calibration_key"] = calibration_key
    merged_cfg["compiled_profile"] = compiled

    if merged_cfg.get("strict_centroid_check", True):
        expected_method = merged_cfg.get("centroid_method")
        actual_method = extract_cfg.get("centroid_method")
        if expected_method and actual_method and expected_method != actual_method:
            raise ValueError(
                "Bias correction profile centroid_method mismatch: "
                f"profile expects {expected_method!r}, extractor configured {actual_method!r}"
            )

        expected_window_size = merged_cfg.get("window_size")
        actual_window_size = extract_cfg.get("centroid_window", {}).get("size")
        if expected_window_size is not None and actual_window_size is not None:
            if int(expected_window_size) != int(actual_window_size):
                raise ValueError(
                    "Bias correction profile window_size mismatch: "
                    f"profile expects {expected_window_size}, extractor configured {actual_window_size}"
                )

    return merged_cfg


def predict_centroid_bias(
    measured_x: float | np.ndarray,
    measured_y: float | np.ndarray,
    bias_cfg: dict,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    compiled = bias_cfg["compiled_profile"]
    train_phase_x = compiled["train_phase_x"]
    train_phase_y = compiled["train_phase_y"]
    bias_x = compiled["bias_x"]
    bias_y = compiled["bias_y"]

    query_x = np.asarray(measured_x, dtype=np.float64)
    query_y = np.asarray(measured_y, dtype=np.float64)
    scalar_input = query_x.ndim == 0 and query_y.ndim == 0
    query_x = np.atleast_1d(query_x)
    query_y = np.atleast_1d(query_y)
    if query_x.shape != query_y.shape:
        raise ValueError("measured_x and measured_y must have the same shape")

    phase_x = local_phase_from_pixel_coord(query_x)
    phase_y = local_phase_from_pixel_coord(query_y)
    out_x = np.empty_like(phase_x, dtype=np.float64)
    out_y = np.empty_like(phase_y, dtype=np.float64)

    k = max(1, min(int(bias_cfg.get("idw_k", 12)), train_phase_x.size))
    power = float(bias_cfg.get("idw_power", 2.0))

    for idx, (qx, qy) in enumerate(zip(phase_x, phase_y)):
        dist2 = (train_phase_x - qx) ** 2 + (train_phase_y - qy) ** 2
        nearest = np.argsort(dist2)[:k]
        nearest_dist2 = dist2[nearest]
        if np.any(nearest_dist2 < 1e-18):
            exact_idx = nearest[int(np.argmin(nearest_dist2))]
            out_x[idx] = bias_x[exact_idx]
            out_y[idx] = bias_y[exact_idx]
            continue

        weights = 1.0 / np.maximum(nearest_dist2, 1e-18) ** (power / 2.0)
        weights /= np.sum(weights)
        out_x[idx] = float(np.sum(weights * bias_x[nearest]))
        out_y[idx] = float(np.sum(weights * bias_y[nearest]))

    if scalar_input:
        return float(out_x[0]), float(out_y[0])
    return out_x, out_y
