"""
微引力导星首帧真实质心联合解算示例。

用途：
- 读取 4 片微引力导星探测器首帧仿真图像；
- 走真实质心提取、参考星匹配和 QUEST 姿态解算；
- 输出首帧联合解算结果与误差审计 JSON。

使用配置：
- configs/base.yaml
- configs/guide_microlens_v1_noise_psf_etcoord.yaml

结果输出：
- <dataset_root>_fsg-results/frameXXXXXX/debug/microlens_guide_first_frame_v1_noise_psf_result.json
- <dataset_root>_fsg-results/frameXXXXXX/debug/microlens_guide_first_frame_v1_noise_psf_error_audit.json
- <dataset_root>_fsg-results/frameXXXXXX/figures/matching_overlay_*.png
"""

import json
import sys
from pathlib import Path

import yaml

# Add the parent directory of fsglib to python path so we can run this directly
sys.path.append(str(Path(__file__).parent.parent))

from fsglib.pipeline.run_guide_init import run_guide_first_frame_init
from fsglib.pipeline.guide_outputs import (
    resolve_debug_output_dir,
    resolve_figures_output_dir,
    resolve_fsg_results_root,
    save_matching_overlays,
)


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


def main() -> None:
    base_cfg_path = Path("configs/base.yaml")
    guide_cfg_path = Path("configs/guide_microlens_v1_noise_psf_etcoord.yaml")

    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg = _deep_update(cfg, yaml.safe_load(guide_cfg_path.read_text(encoding="utf-8")))

    result = run_guide_first_frame_init(cfg, include_debug_context=True)
    solution = result["solution"]
    matching = result["matching"]
    geometry_adapter = result["geometry_adapter"]

    print("----------------------------------------")
    print("Microlens Guide First Frame Joint Solve Results:")
    print("----------------------------------------")
    print(f"Attitude valid: {solution.valid}")
    print(f"Matched stars:  {solution.num_matched}")
    print(f"Residual RMS (arcsec): {solution.residual_rms_arcsec:.2f}")
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
    for line in _geometry_summary_lines(geometry_adapter):
        print(line)
    if result["error_audit"].get("enabled", False):
        audit_summary = result["error_audit"]["summary"]
        counterfactuals = audit_summary["counterfactual_solutions"]
        print(
            "Centroid RMS (detector pix): "
            f"{audit_summary['centroid_error_detector_pix']['rms_radial']:.4f}"
        )
        print(
            "LOS geometry RMS (arcsec): "
            f"{audit_summary['body_error_geometry_arcsec']['rms']:.4f}"
        )
        print(
            "LOS total error RMS (arcsec): "
            f"{audit_summary['body_error_total_arcsec']['rms']:.4f}"
        )
        current_to_frame_truth = counterfactuals["delta_components"]["current_to_frame_truth"]
        frame_truth_to_nominal = counterfactuals["delta_components"]["frame_truth_to_nominal_body"]
        print(
            "Current->FrameTruth total/non-roll/roll (arcsec): "
            "{total:.4f} / {non_roll:.4f} / {roll:.4f}".format(
                total=current_to_frame_truth["total_arcsec"],
                non_roll=current_to_frame_truth["non_roll_arcsec"],
                roll=current_to_frame_truth["roll_arcsec"],
            )
        )
        print(
            "FrameTruth->NominalBody total/non-roll/roll (arcsec): "
            "{total:.4f} / {non_roll:.4f} / {roll:.4f}".format(
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
        transform_desc = (
            "offset=({:.3f},{:.3f})".format(stats["offset_x_pix"], stats["offset_y_pix"])
            if stats["sim_to_detector_kind"] == "offset"
            else "affine_rms={:.3f}".format(stats["affine_fit_rms_pix"])
        )
        print(
            "  {det}: cand={cand} matched={matched} ref={ref} {transform_desc}".format(
                det=detector_id,
                cand=stats["num_candidates_selected"],
                matched=stats["num_matched"],
                ref=stats["num_reference_stars"],
                transform_desc=transform_desc,
            )
        )

    results_root = resolve_fsg_results_root(cfg)
    debug_dir = resolve_debug_output_dir(cfg)
    figures_dir = resolve_figures_output_dir(cfg)
    output_path = debug_dir / "microlens_guide_first_frame_v1_noise_psf_result.json"
    audit_path = debug_dir / "microlens_guide_first_frame_v1_noise_psf_error_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    matching_overlay_summary = save_matching_overlays(result, figures_dir)
    error_audit = result["error_audit"]
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
        "solution": {
            "valid": bool(solution.valid),
            "quality_flag": solution.quality_flag,
            "degraded_level": solution.degraded_level,
            "active_detector_ids": list(solution.active_detector_ids),
            "num_matched": int(solution.num_matched),
            "num_rejected": int(solution.num_rejected),
            "residual_rms_arcsec": float(solution.residual_rms_arcsec),
            "residual_max_arcsec": float(solution.residual_max_arcsec),
            "q_ib": [float(value) for value in solution.q_ib],
        },
        "matching": {
            "success": bool(matching.success),
            "score": float(matching.score),
            "debug": matching.debug,
        },
        "observed_count": int(result["observed_count"]),
        "reference_count": int(result["reference_count"]),
        "detector_stats": result["detector_stats"],
        "sim_to_detector_map": result["sim_to_detector_map"],
        "geometry_adapter": geometry_adapter,
        "error_audit": error_audit_summary,
        "error_audit_detail_path": str(audit_path),
        "results_root": str(results_root),
        "matching_overlay_summary_path": str(figures_dir / "matching_overlay_summary.json"),
        "matching_overlay_summary": matching_overlay_summary,
        "meta": result["meta"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(error_audit, indent=2), encoding="utf-8")
    print(f"Results root: {results_root}")
    print(f"Result JSON: {output_path}")
    print(f"Error audit JSON: {audit_path}")
    print(f"Matching overlays: {figures_dir}")


if __name__ == "__main__":
    main()
