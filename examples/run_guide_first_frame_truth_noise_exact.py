"""
导星首帧理想质心噪声 + exact et_focalplane 几何联合解算示例。

用途：
- 基于 truth detector 质心注入高斯噪声构造理想观测质心；
- 使用 et_focalplane 的精确几何映射生成 LOS；
- 走 QUEST 姿态解算，并输出首帧结果与误差审计。

使用配置：
- configs/base.yaml
- configs/guide_truth_noise_0065pix_exact_etcoord.yaml

结果输出：
- project.output_dir 指定目录下的完整 debug bundle
- 默认由 configs/guide_truth_noise_0065pix_exact_etcoord.yaml 写到 Results-sshfs 结果目录
"""

import json
import sys
import csv
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add the parent directory of fsglib to python path so we can run this directly
sys.path.append(str(Path(__file__).parent.parent))

from _smoke_resources import configure_smoke_process

RESOURCE_LIMITS = configure_smoke_process(default_memory_gb=0.0, default_cpu_seconds=0)

from fsglib.pipeline import run_guide_first_frame_truth_noise
from fsglib.pipeline.error_budget import error_budget_csv_rows
from fsglib.pipeline.guide_outputs import save_matching_overlays


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _geometry_summary_lines(geometry_adapter: dict) -> list[str]:
    return [
        f"LOS geometry: {geometry_adapter['mode']}",
        (
            "Frame alignment RMS/max (arcsec): "
            f"{geometry_adapter['frame_alignment_fit_rms_arcsec']:.4f} / "
            f"{geometry_adapter['frame_alignment_fit_max_arcsec']:.4f}"
        ),
    ]


def _write_error_budget_csv(path: Path, error_budget: dict) -> None:
    rows = error_budget_csv_rows(error_budget)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _solution_payload(solution) -> dict:
    return {
        "valid": bool(solution.valid),
        "quality_flag": solution.quality_flag,
        "degraded_level": solution.degraded_level,
        "active_detector_ids": list(solution.active_detector_ids),
        "num_matched": int(solution.num_matched),
        "num_rejected": int(solution.num_rejected),
        "residual_rms_arcsec": float(solution.residual_rms_arcsec),
        "residual_max_arcsec": float(solution.residual_max_arcsec),
        "q_ib": np.asarray(solution.q_ib).tolist(),
        "c_ib": None if solution.c_ib is None else np.asarray(solution.c_ib).tolist(),
        "euler_zyx": None if solution.euler_zyx is None else np.asarray(solution.euler_zyx).tolist(),
        "quality": solution.quality,
        "solver_iterations": int(solution.solver_iterations),
        "covariance_rad2": (
            None if solution.covariance_rad2 is None else np.asarray(solution.covariance_rad2).tolist()
        ),
        "sigma_non_roll_arcsec": solution.sigma_non_roll_arcsec,
        "sigma_roll_arcsec": solution.sigma_roll_arcsec,
        "attitude_condition_number": solution.attitude_condition_number,
    }


def _write_truth_noise_bundle(
    result: dict,
    cfg: dict,
    *,
    base_cfg_path: Path,
    guide_cfg_path: Path,
    output_dir: Path,
) -> dict:
    solution = result["solution"]
    matching = result["matching"]
    error_audit = result["error_audit"]
    error_budget = result["error_budget"]
    debug_context = result.get("debug_context", {})
    detectors = debug_context.get("detectors", {})
    observed_stars = list(debug_context.get("observed_stars", []))
    reference_stars = list(debug_context.get("reference_stars", []))
    stem = "guide_truth_noise_0065pix_exact_etcoord"

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("config", "detectors", "matching", "audit", "validation", "figures", "geometry"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    error_audit_summary = (
        {
            "enabled": True,
            "truth_match_radius_pix": float(error_audit["truth_match_radius_pix"]),
            "summary": error_audit["summary"],
            "per_detector": error_audit["per_detector"],
            "selected_without_truth_count": int(len(error_audit["selected_without_truth"])),
        }
        if error_audit.get("enabled", False)
        else {"enabled": False}
    )
    payload = {
        "solution": _solution_payload(solution),
        "matching": {
            "success": bool(matching.success),
            "score": float(matching.score),
            "mode": matching.mode,
            "debug": matching.debug,
        },
        "observed_count": int(result["observed_count"]),
        "reference_count": int(result["reference_count"]),
        "detector_stats": result["detector_stats"],
        "sim_to_detector_map": result["sim_to_detector_map"],
        "geometry_adapter": result["geometry_adapter"],
        "synthetic_centroid_model": result["synthetic_centroid_model"],
        "error_audit": error_audit_summary,
        "error_budget": error_budget,
        "meta": {**result["meta"], "config_path": str(guide_cfg_path), "output_dir": str(output_dir)},
    }

    _write_json(output_dir / f"{stem}_result.json", payload)
    _write_json(output_dir / f"{stem}_error_audit.json", error_audit)
    _write_json(output_dir / f"{stem}_error_budget.json", error_budget)
    _write_error_budget_csv(output_dir / f"{stem}_error_budget_terms.csv", error_budget)

    _write_json(output_dir / "run_summary.json", payload)
    _write_json(output_dir / "solution.json", payload["solution"])
    _write_json(output_dir / "synthetic_centroid_model.json", result["synthetic_centroid_model"])
    _write_json(output_dir / "geometry" / "geometry_adapter.json", result["geometry_adapter"])
    _write_json(output_dir / "geometry" / "sim_to_detector_map.json", result["sim_to_detector_map"])
    _write_json(output_dir / "matching" / "matching_result.json", payload["matching"])
    _write_json(output_dir / "matching" / "matched_stars.json", matching.matched)
    _write_json(output_dir / "matching" / "observed_stars.json", observed_stars)
    _write_json(output_dir / "matching" / "reference_stars.json", reference_stars)
    _write_json(output_dir / "matching" / "detector_stats.json", result["detector_stats"])
    _write_json(output_dir / "audit" / "guide_error_audit.json", error_audit)
    _write_json(output_dir / "validation" / "error_budget.json", error_budget)
    _write_error_budget_csv(output_dir / "validation" / "error_budget_terms.csv", error_budget)

    (output_dir / "config" / base_cfg_path.name).write_text(
        base_cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "config" / guide_cfg_path.name).write_text(
        guide_cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "config" / "merged_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(output_dir / "config" / "run_meta.json", payload["meta"])

    for detector_id, context in detectors.items():
        detector_dir = output_dir / "detectors" / str(detector_id)
        detector_dir.mkdir(parents=True, exist_ok=True)
        raw = context.get("raw")
        if raw is not None:
            np.save(detector_dir / "raw_image.npy", raw.image)
            _write_json(detector_dir / "raw_meta.json", raw.meta)
        else:
            np.save(detector_dir / "raw_image.npy", context["image"])
            _write_json(detector_dir / "raw_meta.json", {})
        _write_json(detector_dir / "all_candidates.json", context.get("all_candidates", []))
        _write_json(detector_dir / "selected_candidates.json", context.get("selected_candidates", []))
        _write_json(detector_dir / "detector_stats.json", result["detector_stats"].get(detector_id, {}))

    overlay_summary = save_matching_overlays(result, output_dir / "figures")
    _write_json(output_dir / "figures" / "matching_overlay_summary.json", overlay_summary)

    readme = f"""# fsglib truth-noise exact-etcoord debug bundle

Configuration: `{guide_cfg_path}`
Dataset root: `{result["meta"]["dataset_root"]}`

## Result

- valid: `{bool(solution.valid)}`
- matched: `{int(solution.num_matched)}` / observed `{int(result["observed_count"])}`
- reference stars: `{int(result["reference_count"])}`
- residual_rms_arcsec: `{float(solution.residual_rms_arcsec):.6f}`
- matching strategy: `{matching.debug.get("selected_strategy")}`
- predicted-position matches: `{matching.debug.get("num_predicted_position_matches")}`
- local-pyramid matches: `{matching.debug.get("num_local_pyramid_matches")}`

## Contents

- `run_summary.json`: full top-level result payload.
- `solution.json`: attitude solution details.
- `matching/`: observed/reference/matched stars and matcher debug.
- `detectors/<detector_id>/`: raw image arrays, raw metadata, synthetic truth candidates.
- `geometry/`: exact ET focal-plane geometry and sim-to-detector bridge.
- `audit/guide_error_audit.json`: guide-specific truth/matching/LOS/attitude audit.
- `validation/error_budget.json` and `validation/error_budget_terms.csv`: error-budget ledger.
- `figures/matching_overlay_*.png`: per-detector matching overlays.
- `config/`: base, customer, and merged configuration snapshots.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "result_json": str(output_dir / f"{stem}_result.json"),
        "error_audit_json": str(output_dir / f"{stem}_error_audit.json"),
        "error_budget_json": str(output_dir / f"{stem}_error_budget.json"),
        "error_budget_csv": str(output_dir / f"{stem}_error_budget_terms.csv"),
        "overlay_summary": str(output_dir / "figures" / "matching_overlay_summary.json"),
    }


def main() -> None:
    base_cfg_path = Path("configs/base.yaml")
    guide_cfg_path = Path("configs/guide_truth_noise_0065pix_exact_etcoord.yaml")

    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg = _deep_update(cfg, yaml.safe_load(guide_cfg_path.read_text(encoding="utf-8")))

    result = run_guide_first_frame_truth_noise(cfg, include_debug_context=True)
    solution = result["solution"]
    matching = result["matching"]
    synth = result["synthetic_centroid_model"]
    geometry_adapter = result["geometry_adapter"]

    print("----------------------------------------")
    print("Guide First Frame Truth-Noise Exact Solve:")
    print("----------------------------------------")
    print(f"Resource limits: {RESOURCE_LIMITS}")
    print(f"Centroid model: {synth['mode']}")
    print(
        "Noise assumption: dx,dy ~ N({mean:.4f}, {sigma:.4f}^2) pix in {space}".format(
            mean=synth["noise_mean_pix"],
            sigma=synth["noise_sigma_pix"],
            space=synth["noise_space"],
        )
    )
    print(f"Random seed: {synth['random_seed']}")
    for line in _geometry_summary_lines(geometry_adapter):
        print(line)
    print(f"Attitude valid: {solution.valid}")
    print(f"Matched stars:  {solution.num_matched}")
    print(f"Residual RMS (arcsec): {solution.residual_rms_arcsec:.6f}")
    print(f"Quality flag: {solution.quality_flag}")
    print(f"Degraded level: {solution.degraded_level}")
    print(f"Active detectors: {solution.active_detector_ids}")
    print(f"Quaternion [w, x, y, z]: {solution.q_ib}")
    print(f"Observed stars: {result['observed_count']}")
    print(f"Reference stars: {result['reference_count']}")
    print(f"Matching strategy: {matching.debug.get('selected_strategy')}")
    print(f"Predicted-position matches: {matching.debug.get('num_predicted_position_matches')}")
    print(f"Local-pyramid matches: {matching.debug.get('num_local_pyramid_matches')}")
    print(f"Mean residual (pix): {matching.debug.get('mean_residual_pix')}")
    if result["error_audit"].get("enabled", False):
        audit_summary = result["error_audit"]["summary"]
        counterfactuals = audit_summary["counterfactual_solutions"]
        print(
            "Centroid RMS (detector pix): "
            f"{audit_summary['centroid_error_detector_pix']['rms_radial']:.4f}"
        )
        print(
            "LOS geometry RMS (arcsec): "
            f"{audit_summary['body_error_geometry_arcsec']['rms']:.6f}"
        )
        print(
            "LOS total error RMS (arcsec): "
            f"{audit_summary['body_error_total_arcsec']['rms']:.6f}"
        )
        current_to_frame_truth = counterfactuals["delta_components"]["current_to_frame_truth"]
        frame_truth_to_nominal = counterfactuals["delta_components"]["frame_truth_to_nominal_body"]
        print(
            "Current->FrameTruth total/non-roll/roll (arcsec): "
            "{total:.6f} / {non_roll:.6f} / {roll:.6f}".format(
                total=current_to_frame_truth["total_arcsec"],
                non_roll=current_to_frame_truth["non_roll_arcsec"],
                roll=current_to_frame_truth["roll_arcsec"],
            )
        )
        print(
            "FrameTruth->NominalBody total/non-roll/roll (arcsec): "
            "{total:.6f} / {non_roll:.6f} / {roll:.6f}".format(
                total=frame_truth_to_nominal["total_arcsec"],
                non_roll=frame_truth_to_nominal["non_roll_arcsec"],
                roll=frame_truth_to_nominal["roll_arcsec"],
            )
        )
        print(
            "Predicted->RawDetector RMS (pix): "
            f"{audit_summary['match_predicted_vs_ecsv_detector_pix']['rms_radial']:.4e}"
        )
        print(
            "NPZ minus RawDetector offset RMS (pix): "
            f"{audit_summary['npz_minus_ecsv_detector_offset_pix']['rms_radial']:.6f}"
        )
    print("Per detector:")
    for detector_id, stats in result["detector_stats"].items():
        print(
            "  {det}: truth={truth} selected={cand} matched={matched} ref={ref}".format(
                det=detector_id,
                truth=stats["num_truth_stars_visible"],
                cand=stats["num_candidates_selected"],
                matched=stats["num_matched"],
                ref=stats["num_reference_stars"],
            )
        )

    output_dir = Path(cfg.get("project", {}).get("output_dir", "outputs/debug")).expanduser()
    bundle_paths = _write_truth_noise_bundle(
        result,
        cfg,
        base_cfg_path=base_cfg_path,
        guide_cfg_path=guide_cfg_path,
        output_dir=output_dir,
    )
    print(f"Output dir: {bundle_paths['output_dir']}")
    print(f"Result JSON: {bundle_paths['result_json']}")
    print(f"Error audit JSON: {bundle_paths['error_audit_json']}")
    print(f"Error budget JSON: {bundle_paths['error_budget_json']}")
    print(f"Error budget CSV: {bundle_paths['error_budget_csv']}")
    print(f"Overlay summary: {bundle_paths['overlay_summary']}")


if __name__ == "__main__":
    main()
