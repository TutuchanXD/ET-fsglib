"""
PR21 resource-limited error-budget smoke.

Default mode:
- truth_noise_exact: uses configs/guide_truth_noise_0065pix_exact_etcoord.yaml
  and the legacy guide simulation dataset, but bypasses real image extraction.

Optional mode:
- real_image_no_calib: uses configs/guide_v1_noise_psf_etcoord.yaml and real
  simulated images, but disables calibration products because the legacy images
  are 1947x1947 while the default fake calibration assets are 2049x2049.

Resource controls:
- FSGLIB_SMOKE_MAX_MEMORY_GB, explicit memory cap override
- FSGLIB_SMOKE_MEMORY_FRACTION, default 0.60 of current MemAvailable after reserve
- FSGLIB_SMOKE_RESERVE_MEMORY_GB, default 8
- FSGLIB_SMOKE_MAX_CPU_SECONDS, default 180
- FSGLIB_SMOKE_MAX_OBS_PER_DETECTOR, default 25
- FSGLIB_SMOKE_REFERENCE_TOPK_PER_DETECTOR, default 60
- FSGLIB_SMOKE_CATALOG_G_MAG_MAX, default 12.0
- FSGLIB_PR21_SMOKE_MODE, default truth_noise_exact
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from _smoke_resources import apply_guide_smoke_scale, configure_smoke_process

SMOKE_MODE = os.environ.get("FSGLIB_PR21_SMOKE_MODE", "truth_noise_exact")
DEFAULT_MEMORY_GB = 24.0 if SMOKE_MODE == "real_image_no_calib" else 12.0
RESOURCE_LIMITS = configure_smoke_process(default_memory_gb=DEFAULT_MEMORY_GB)

import yaml

sys.path.append(str(Path(__file__).parent.parent))

from fsglib.pipeline import run_guide_first_frame_truth_noise
from fsglib.pipeline.error_budget import error_budget_csv_rows
from fsglib.pipeline.run_guide_init import run_guide_first_frame_init


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_cfg(overlay: str) -> dict:
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    return _deep_update(cfg, yaml.safe_load(Path(overlay).read_text(encoding="utf-8")))


def _write_budget_outputs(prefix: str, payload: dict, budget: dict) -> dict[str, str]:
    output_dir = Path("outputs/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{prefix}_result.json"
    budget_path = output_dir / f"{prefix}_error_budget.json"
    budget_csv_path = output_dir / f"{prefix}_error_budget_terms.csv"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    rows = error_budget_csv_rows(budget)
    if rows:
        with budget_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "result_json": str(result_path),
        "error_budget_json": str(budget_path),
        "error_budget_csv": str(budget_csv_path) if rows else None,
    }


def _run_truth_noise_exact() -> tuple[dict, dict]:
    cfg = _load_cfg("configs/guide_truth_noise_0065pix_exact_etcoord.yaml")
    scale = apply_guide_smoke_scale(cfg, section="guide_truth_noise")
    result = run_guide_first_frame_truth_noise(cfg)
    solution = result["solution"]
    matching = result["matching"]
    budget = result["error_budget"]
    payload = {
        "mode": "truth_noise_exact",
        "resource_limits": RESOURCE_LIMITS,
        "smoke_scale": scale,
        "input": {
            "dataset_root": cfg["guide_truth_noise"]["dataset_root"],
            "frame_index": cfg["guide_truth_noise"]["frame_index"],
            "detectors": cfg["guide_truth_noise"]["detector_batches"],
            "centroid_noise_sigma_pix": cfg["guide_truth_noise"]["centroid_noise_sigma_pix"],
        },
        "solution": {
            "valid": bool(solution.valid),
            "quality_flag": solution.quality_flag,
            "degraded_level": solution.degraded_level,
            "num_matched": int(solution.num_matched),
            "num_rejected": int(solution.num_rejected),
            "residual_rms_arcsec": float(solution.residual_rms_arcsec),
            "residual_max_arcsec": float(solution.residual_max_arcsec),
            "active_detector_ids": list(solution.active_detector_ids),
        },
        "matching": {
            "success": bool(matching.success),
            "selected_strategy": matching.debug.get("selected_strategy"),
            "mean_residual_pix": matching.debug.get("mean_residual_pix"),
        },
        "observed_count": int(result["observed_count"]),
        "reference_count": int(result["reference_count"]),
        "detector_stats": result["detector_stats"],
        "error_budget": budget,
    }
    return payload, budget


def _run_real_image_no_calib() -> tuple[dict, dict]:
    cfg = _load_cfg("configs/guide_v1_noise_psf_etcoord.yaml")
    scale = apply_guide_smoke_scale(cfg, section="guide_init")
    cfg["preprocess"].update(
        enable_bias_subtraction=False,
        enable_dark_subtraction=False,
        enable_flat_field=False,
        enable_bad_pixel_mask=False,
        enable_fpn_subtraction=False,
    )
    result = run_guide_first_frame_init(cfg)
    solution = result["solution"]
    matching = result["matching"]
    budget = result["error_budget"]
    payload = {
        "mode": "real_image_no_calib",
        "resource_limits": RESOURCE_LIMITS,
        "smoke_scale": scale,
        "input": {
            "dataset_root": cfg["guide_init"]["dataset_root"],
            "frame_index": cfg["guide_init"]["frame_index"],
            "detectors": cfg["guide_init"]["detector_batches"],
            "preprocess_smoke_override": (
                "disabled bias/dark/flat/bad-pixel/FPN calibration because legacy image shape is 1947x1947 "
                "while the default fake assets are 2049x2049"
            ),
        },
        "solution": {
            "valid": bool(solution.valid),
            "quality_flag": solution.quality_flag,
            "degraded_level": solution.degraded_level,
            "num_matched": int(solution.num_matched),
            "num_rejected": int(solution.num_rejected),
            "residual_rms_arcsec": float(solution.residual_rms_arcsec),
            "residual_max_arcsec": float(solution.residual_max_arcsec),
            "active_detector_ids": list(solution.active_detector_ids),
        },
        "matching": {
            "success": bool(matching.success),
            "selected_strategy": matching.debug.get("selected_strategy"),
            "mean_residual_pix": matching.debug.get("mean_residual_pix"),
        },
        "observed_count": int(result["observed_count"]),
        "reference_count": int(result["reference_count"]),
        "detector_stats": result["detector_stats"],
        "error_budget": budget,
    }
    return payload, budget


def main() -> None:
    mode = SMOKE_MODE
    try:
        if mode == "truth_noise_exact":
            payload, budget = _run_truth_noise_exact()
        elif mode == "real_image_no_calib":
            payload, budget = _run_real_image_no_calib()
        else:
            raise ValueError(f"Unsupported FSGLIB_PR21_SMOKE_MODE: {mode}")
    except MemoryError as exc:
        failure = {
            "mode": mode,
            "status": "failed_resource_limited",
            "resource_limits": RESOURCE_LIMITS,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hint": (
                "The smoke process hit its memory cap. Increase FSGLIB_SMOKE_MAX_MEMORY_GB "
                "or use the default truth_noise_exact smoke for PR21 ledger validation."
            ),
        }
        output_dir = Path("outputs/debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / f"pr21_error_budget_smoke_{mode}_failure.json"
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print("PR21 error-budget smoke failed under resource limits")
        print("failure=", failure)
        print("failure_json=", failure_path)
        raise SystemExit(1) from None

    paths = _write_budget_outputs(f"pr21_error_budget_smoke_{mode}", payload, budget)
    print("PR21 error-budget smoke")
    print("mode=", payload["mode"])
    print("resource_limits=", payload["resource_limits"])
    print("smoke_scale=", payload["smoke_scale"])
    print("solution=", payload["solution"])
    print("matching=", payload["matching"])
    print("observed_count=", payload["observed_count"])
    print("reference_count=", payload["reference_count"])
    print("dominant_budget_term=", budget["summary"].get("dominant_angular_term"))
    print("outputs=", paths)


if __name__ == "__main__":
    main()
