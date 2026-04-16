#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fsglib.common.coords import radec_to_unit_vector
from fsglib.common.io import load_dataset_batch
from fsglib.models.mock import build_models
from fsglib.pipeline.evaluate import evaluate_dataset
from fsglib.pipeline.run_init import run_single_frame_init


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = "/home/cxgao/ET/FSG_images_sims/v2"
DEFAULT_BASE_CFG = str(REPO_ROOT / "configs" / "base.yaml")
DEFAULT_DET_CFG = str(REPO_ROOT / "configs" / "main_sim_v2.yaml")


def deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_cfg(base_cfg_path: str, det_cfg_path: str) -> dict:
    base_cfg = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))
    det_cfg = yaml.safe_load(Path(det_cfg_path).read_text(encoding="utf-8"))
    return deep_update(base_cfg, det_cfg)


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("fit_similarity expects Nx2 source and destination arrays")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_zero = src - mu_src
    dst_zero = dst - mu_dst

    cov = (dst_zero.T @ src_zero) / len(src)
    u_mat, sing_vals, vt_mat = np.linalg.svd(cov)
    s_mat = np.eye(2)
    if np.linalg.det(u_mat @ vt_mat) < 0:
        s_mat[-1, -1] = -1
    rot = u_mat @ s_mat @ vt_mat

    var_src = np.mean(np.sum(src_zero**2, axis=1))
    scale = np.trace(np.diag(sing_vals) @ s_mat) / var_src if var_src > 0 else 1.0
    trans = mu_dst - scale * (rot @ mu_src)
    return float(scale), rot, trans


def summarize_first_frame(batch_dir: Path, cfg: dict, models: dict) -> dict[str, Any]:
    dataset_ctx = load_dataset_batch(batch_dir, cfg=cfg)
    frame_result = run_single_frame_init(str(dataset_ctx.frame_paths[0]), cfg, models, dataset_ctx=dataset_ctx)

    truth_los = np.array(
        [radec_to_unit_vector(star.ra_deg, star.dec_deg) for star in dataset_ctx.truth_stars],
        dtype=np.float64,
    )
    truth_xy_all = np.array(
        [[star.x_pix, star.y_pix] for star in dataset_ctx.truth_stars],
        dtype=np.float64,
    )
    principal_point = np.array(cfg["layout"]["detectors"][0]["principal_point_pix"], dtype=np.float64)

    candidate_points = np.array([[cand.x, cand.y] for cand in frame_result.candidates], dtype=np.float64)
    if candidate_points.size and truth_xy_all.size:
        diff = candidate_points[:, None, :] - truth_xy_all[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        nearest_truth = truth_xy_all[nearest_idx]
        candidate_diff = candidate_points - nearest_truth
        candidate_common_shift = np.mean(candidate_diff, axis=0)
        candidate_local_diff = candidate_diff - candidate_common_shift
        candidate_residual = np.sqrt(np.sum(candidate_diff**2, axis=1))
        candidate_local_residual = np.sqrt(np.sum(candidate_local_diff**2, axis=1))
    else:
        candidate_common_shift = np.zeros(2, dtype=np.float64)
        candidate_residual = np.zeros(0, dtype=np.float64)
        candidate_local_residual = np.zeros(0, dtype=np.float64)

    truth_points: list[list[float]] = []
    observed_points: list[list[float]] = []
    radial_terms: list[float] = []
    tangential_terms: list[float] = []
    radii: list[float] = []
    catalog_truth_seps_arcsec: list[float] = []

    for matched_star in frame_result.matching.matched:
        dots = np.clip(truth_los @ matched_star.los_inertial, -1.0, 1.0)
        truth_idx = int(np.argmax(dots))
        truth_star = dataset_ctx.truth_stars[truth_idx]

        truth_xy = np.array([truth_star.x_pix, truth_star.y_pix], dtype=np.float64)
        observed_xy = np.array(matched_star.flags["observed_xy"], dtype=np.float64)
        error_xy = observed_xy - truth_xy

        rel_xy = truth_xy - principal_point
        radius_pix = float(np.linalg.norm(rel_xy))
        if radius_pix > 1e-9:
            e_radial = rel_xy / radius_pix
            e_tangential = np.array([-e_radial[1], e_radial[0]], dtype=np.float64)
            radial_terms.append(float(error_xy @ e_radial))
            tangential_terms.append(float(error_xy @ e_tangential))
            radii.append(radius_pix)

        truth_points.append(truth_xy.tolist())
        observed_points.append(observed_xy.tolist())
        catalog_truth_seps_arcsec.append(float(np.degrees(np.arccos(dots[truth_idx])) * 3600.0))

    truth_points_arr = np.asarray(truth_points, dtype=np.float64)
    observed_points_arr = np.asarray(observed_points, dtype=np.float64)
    radial_arr = np.asarray(radial_terms, dtype=np.float64)
    tangential_arr = np.asarray(tangential_terms, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64)
    if truth_points_arr.size and observed_points_arr.size:
        matched_diff = observed_points_arr - truth_points_arr
        matched_common_shift = np.mean(matched_diff, axis=0)
        matched_local_diff = matched_diff - matched_common_shift
        matched_local_residual = np.sqrt(np.sum(matched_local_diff**2, axis=1))
    else:
        matched_common_shift = np.zeros(2, dtype=np.float64)
        matched_local_residual = np.zeros(0, dtype=np.float64)

    if len(truth_points_arr) >= 2:
        scale, rot, trans = fit_similarity(truth_points_arr, observed_points_arr)
        rot_arcsec = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])) * 3600.0)
        mapped_points = (scale * (rot @ truth_points_arr.T)).T + trans
        postfit_residual_pix = np.sqrt(np.sum((mapped_points - observed_points_arr) ** 2, axis=1))
    else:
        scale = 1.0
        rot_arcsec = 0.0
        trans = np.zeros(2, dtype=np.float64)
        postfit_residual_pix = np.zeros(0, dtype=np.float64)

    if len(truth_points_arr) >= 2:
        centered_truth = truth_points_arr - principal_point
        cov = np.cov(centered_truth.T)
        eigvals = np.linalg.eigvalsh(cov)
        anisotropy = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else None
        azimuth_deg = np.mod(np.degrees(np.arctan2(centered_truth[:, 1], centered_truth[:, 0])), 360.0)
        azimuth_bins = np.histogram(azimuth_deg, bins=np.linspace(0.0, 360.0, 9))[0].tolist()
    else:
        anisotropy = None
        azimuth_bins = []

    evaluation = frame_result.evaluation
    return {
        "batch_name": batch_dir.name,
        "field_offset_source": dataset_ctx.field_offset_source,
        "field_offset_x_pix": dataset_ctx.field_offset_x_pix,
        "field_offset_y_pix": dataset_ctx.field_offset_y_pix,
        "truth_star_id_status": dataset_ctx.run_meta.get("truth_star_id_status"),
        "num_candidates": len(frame_result.candidates),
        "num_reference": len(frame_result.reference),
        "num_matched": frame_result.solution.num_matched,
        "valid": bool(frame_result.solution.valid),
        "residual_rms_arcsec": float(frame_result.solution.residual_rms_arcsec),
        "centroid_mae_pix_static_truth": (
            None if evaluation is None else evaluation.centroid_mae_pix
        ),
        "candidate_common_shift_pix": [float(candidate_common_shift[0]), float(candidate_common_shift[1])],
        "candidate_common_shift_mag_pix": float(np.linalg.norm(candidate_common_shift)),
        "candidate_local_rms_after_common_shift_pix": (
            float(np.sqrt(np.mean(candidate_local_residual**2))) if candidate_local_residual.size else None
        ),
        "non_roll_error_arcsec": None if evaluation is None else evaluation.non_roll_error_arcsec,
        "roll_error_arcsec": None if evaluation is None else evaluation.roll_error_arcsec,
        "total_attitude_error_arcsec": (
            None if evaluation is None else evaluation.total_attitude_error_arcsec
        ),
        "catalog_truth_sep_max_arcsec": (
            float(np.max(catalog_truth_seps_arcsec)) if catalog_truth_seps_arcsec else None
        ),
        "radial_rms_pix": float(np.sqrt(np.mean(radial_arr**2))) if radial_arr.size else None,
        "tangential_rms_pix": (
            float(np.sqrt(np.mean(tangential_arr**2))) if tangential_arr.size else None
        ),
        "radial_mean_pix": float(np.mean(radial_arr)) if radial_arr.size else None,
        "tangential_mean_pix": float(np.mean(tangential_arr)) if tangential_arr.size else None,
        "matched_common_shift_pix": [float(matched_common_shift[0]), float(matched_common_shift[1])],
        "matched_common_shift_mag_pix": float(np.linalg.norm(matched_common_shift)),
        "matched_local_rms_after_common_shift_pix": (
            float(np.sqrt(np.mean(matched_local_residual**2))) if matched_local_residual.size else None
        ),
        "mean_radius_pix": float(np.mean(radii_arr)) if radii_arr.size else None,
        "p90_radius_pix": float(np.percentile(radii_arr, 90)) if radii_arr.size else None,
        "similarity_rotation_arcsec": rot_arcsec,
        "similarity_scale_minus1_ppm": float((scale - 1.0) * 1.0e6),
        "similarity_translation_pix": [float(trans[0]), float(trans[1])],
        "postfit_residual_mean_pix": (
            float(np.mean(postfit_residual_pix)) if postfit_residual_pix.size else None
        ),
        "postfit_residual_rms_pix": (
            float(np.sqrt(np.mean(postfit_residual_pix**2))) if postfit_residual_pix.size else None
        ),
        "geometry_anisotropy": anisotropy,
        "azimuth_bins_45deg": azimuth_bins,
        "notes": [
            "centroid_mae_pix_static_truth uses static stars.ecsv and is affected by simulated DVA/pointing/jitter",
            "radial/tangential metrics are computed after nearest truth association in inertial RA/Dec space",
        ],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(summary["batch_name"])
    print(
        "  matched={num_matched} reference={num_reference} valid={valid} rms={residual_rms_arcsec:.6f} arcsec".format(
            **summary
        )
    )
    print(
        "  non_roll={non_roll_error_arcsec} roll={roll_error_arcsec} total={total_attitude_error_arcsec}".format(
            **summary
        )
    )
    print(
        "  centroid_mae_static_truth={centroid_mae_pix_static_truth} candidate_common_shift={candidate_common_shift_pix} candidate_local_rms={candidate_local_rms_after_common_shift_pix}".format(
            **summary
        )
    )
    print(
        "  radial_rms={radial_rms_pix} tangential_rms={tangential_rms_pix} matched_common_shift={matched_common_shift_pix} matched_local_rms={matched_local_rms_after_common_shift_pix}".format(
            **summary
        )
    )
    print(
        "  similarity_rot_arcsec={similarity_rotation_arcsec} scale_minus1_ppm={similarity_scale_minus1_ppm} trans_pix={similarity_translation_pix}".format(
            **summary
        )
    )
    print(
        "  anisotropy={geometry_anisotropy} azimuth_bins_45deg={azimuth_bins_45deg}".format(
            **summary
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze first-frame centroid and roll biases on v2 batches.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--base-cfg", default=DEFAULT_BASE_CFG)
    parser.add_argument("--det-cfg", default=DEFAULT_DET_CFG)
    parser.add_argument("--sequence-frames", type=int, default=2)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.base_cfg, args.det_cfg)
    models = build_models(cfg)
    dataset_root = Path(args.dataset_root)

    batch_summaries = [
        summarize_first_frame(batch_dir, cfg, models)
        for batch_dir in sorted(dataset_root.glob("batch*"))
    ]

    result: dict[str, Any] = {"batches": batch_summaries}
    if args.sequence_frames and args.sequence_frames > 0:
        cfg_eval = deep_update(cfg, {"evaluation": {"frame_stride": 1, "max_frames_per_batch": args.sequence_frames}})
        result["dataset_summary"] = evaluate_dataset(str(dataset_root), cfg_eval, build_models(cfg_eval))["summary"]

    for batch_summary in batch_summaries:
        print_summary(batch_summary)
    if "dataset_summary" in result:
        print("dataset_summary")
        print(json.dumps(result["dataset_summary"], indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
