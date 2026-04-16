import json
import sys
from pathlib import Path

import yaml

# Add the parent directory of fsglib to python path so we can run this directly
sys.path.append(str(Path(__file__).parent.parent))

from fsglib.pipeline import run_guide_first_frame_truth_noise


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _geometry_summary_lines(geometry_model: dict) -> list[str]:
    mode = str(geometry_model.get("mode", "body_model_proxy"))
    if mode == "exact_et_focalplane":
        return [
            "LOS geometry: exact_et_focalplane",
            (
                "Body-frame alignment reference RMS/max (arcsec): "
                f"{geometry_model['frame_alignment_reference_fit_rms_arcsec']:.4f} / "
                f"{geometry_model['frame_alignment_reference_fit_max_arcsec']:.4f}"
            ),
        ]
    return [
        "LOS geometry: body_model_proxy",
        (
            "Body model fit RMS/max (arcsec): "
            f"{geometry_model['fit_rms_arcsec']:.4f} / {geometry_model['fit_max_arcsec']:.4f}"
        ),
    ]


def main() -> None:
    base_cfg_path = Path("configs/base.yaml")
    guide_cfg_path = Path("configs/guide_truth_noise_0065pix_exact_etcoord.yaml")

    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg = _deep_update(cfg, yaml.safe_load(guide_cfg_path.read_text(encoding="utf-8")))

    result = run_guide_first_frame_truth_noise(cfg)
    solution = result["solution"]
    matching = result["matching"]
    synth = result["synthetic_centroid_model"]
    geometry_model = result["geometry_model"]

    print("----------------------------------------")
    print("Guide First Frame Truth-Noise Exact Solve:")
    print("----------------------------------------")
    print(f"Centroid model: {synth['mode']}")
    print(
        "Noise assumption: dx,dy ~ N({mean:.4f}, {sigma:.4f}^2) pix in {space}".format(
            mean=synth["noise_mean_pix"],
            sigma=synth["noise_sigma_pix"],
            space=synth["noise_space"],
        )
    )
    print(f"Random seed: {synth['random_seed']}")
    for line in _geometry_summary_lines(geometry_model):
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

    output_path = Path("outputs/debug/guide_first_frame_truth_noise_0065pix_exact_etcoord_result.json")
    audit_path = Path("outputs/debug/guide_first_frame_truth_noise_0065pix_exact_etcoord_error_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
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
        "geometry_model": result["geometry_model"],
        "synthetic_centroid_model": result["synthetic_centroid_model"],
        "error_audit": error_audit_summary,
        "error_audit_detail_path": str(audit_path),
        "meta": result["meta"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(error_audit, indent=2), encoding="utf-8")
    print(f"Result JSON: {output_path}")
    print(f"Error audit JSON: {audit_path}")


if __name__ == "__main__":
    main()
