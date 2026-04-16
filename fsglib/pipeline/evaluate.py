import numpy as np
from scipy.spatial.transform import Rotation

from fsglib.attitude.solver import quat_to_dcm
from fsglib.common.coords import radec_to_unit_vector
from fsglib.common.types import (
    DatasetContext,
    FrameEvaluation,
    MatchingResult,
    PreprocessedFrame,
    RawFrame,
    StarCandidate,
    TruthStar,
)
from fsglib.pipeline.centroid_audit import compute_centroid_step_audit


def _nearest_pixel_distance(
    candidate: StarCandidate,
    truth_xy: np.ndarray,
) -> float:
    dx = truth_xy[:, 0] - candidate.x
    dy = truth_xy[:, 1] - candidate.y
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def _truth_attitude_from_radec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra_rad = np.radians(float(ra_deg))
    dec_rad = np.radians(float(dec_deg))
    east = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0], dtype=np.float64)
    north = np.array(
        [
            -np.sin(dec_rad) * np.cos(ra_rad),
            -np.sin(dec_rad) * np.sin(ra_rad),
            np.cos(dec_rad),
        ],
        dtype=np.float64,
    )
    center = np.array(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ],
        dtype=np.float64,
    )
    return np.vstack([east, north, center])


def _angle_arcsec_between(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.asarray(a, dtype=np.float64)
    b_norm = np.asarray(b, dtype=np.float64)
    a_norm /= np.linalg.norm(a_norm)
    b_norm /= np.linalg.norm(b_norm)
    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)) * 3600.0)


def _rotation_align_vector(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_norm = np.asarray(src, dtype=np.float64)
    dst_norm = np.asarray(dst, dtype=np.float64)
    src_norm /= np.linalg.norm(src_norm)
    dst_norm /= np.linalg.norm(dst_norm)
    cross = np.cross(src_norm, dst_norm)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(src_norm, dst_norm), -1.0, 1.0)

    if cross_norm < 1e-12:
        if dot > 0:
            return np.eye(3)
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(src_norm, fallback)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src_norm, fallback)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()

    axis = cross / cross_norm
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _compute_attitude_error_metrics(
    solution,
    dataset_ctx: DatasetContext,
) -> tuple[float | None, float | None, float | None]:
    if (
        not solution.valid
        or dataset_ctx.batch_center_ra_deg is None
        or dataset_ctx.batch_center_dec_deg is None
    ):
        return None, None, None

    c_truth = _truth_attitude_from_radec(
        dataset_ctx.batch_center_ra_deg,
        dataset_ctx.batch_center_dec_deg,
    )
    c_est = solution.c_ib if solution.c_ib is not None else quat_to_dcm(solution.q_ib)

    z_truth = c_truth[2]
    z_est = c_est[2]
    non_roll_error_arcsec = _angle_arcsec_between(z_truth, z_est)

    delta_rot = Rotation.from_matrix(c_est @ c_truth.T)
    total_attitude_error_arcsec = float(np.degrees(delta_rot.magnitude()) * 3600.0)

    align_rot = _rotation_align_vector(z_est, z_truth)
    x_truth = c_truth[0]
    x_est_aligned = align_rot @ c_est[0]
    x_est_proj = x_est_aligned - np.dot(x_est_aligned, z_truth) * z_truth
    norm = np.linalg.norm(x_est_proj)
    if norm < 1e-12:
        roll_error_arcsec = None
    else:
        x_est_proj /= norm
        signed_roll_rad = np.arctan2(
            np.dot(z_truth, np.cross(x_truth, x_est_proj)),
            np.clip(np.dot(x_truth, x_est_proj), -1.0, 1.0),
        )
        roll_error_arcsec = float(abs(np.degrees(signed_roll_rad) * 3600.0))

    return non_roll_error_arcsec, roll_error_arcsec, total_attitude_error_arcsec


def _collect_eval_values(frame_results, field_name: str) -> np.ndarray:
    values = [
        getattr(frame.evaluation, field_name)
        for frame in frame_results
        if frame.evaluation is not None and getattr(frame.evaluation, field_name) is not None
    ]
    return np.asarray(values, dtype=np.float64)


def _resolve_truth_stars(
    raw: RawFrame | None,
    dataset_ctx: DatasetContext | None,
) -> tuple[list[TruthStar], str | None]:
    raw_meta = {} if raw is None else (raw.meta or {})
    raw_truth_stars = raw_meta.get("truth_stars")
    if raw_truth_stars:
        return list(raw_truth_stars), raw_meta.get("truth_source", "npz_frame_truth")
    if dataset_ctx is not None and dataset_ctx.truth_stars:
        return list(dataset_ctx.truth_stars), "stars_ecsv_static"
    return [], None


def evaluate_frame_result(
    raw: RawFrame | None,
    preprocessed: PreprocessedFrame | None,
    candidates: list[StarCandidate],
    matching: MatchingResult,
    solution,
    dataset_ctx: DatasetContext | None,
    cfg: dict | None = None,
) -> FrameEvaluation | None:
    truth_stars, truth_source = _resolve_truth_stars(raw, dataset_ctx)
    if not truth_stars:
        return None

    truth_xy = np.array([[star.x_pix, star.y_pix] for star in truth_stars], dtype=np.float64)
    centroid_distances = []
    centroid_dx = []
    centroid_dy = []
    for candidate in candidates:
        dx = truth_xy[:, 0] - candidate.x
        dy = truth_xy[:, 1] - candidate.y
        dist = np.sqrt(dx * dx + dy * dy)
        truth_idx = int(np.argmin(dist))
        centroid_distances.append(float(dist[truth_idx]))
        centroid_dx.append(float(candidate.x - truth_xy[truth_idx, 0]))
        centroid_dy.append(float(candidate.y - truth_xy[truth_idx, 1]))

    centroid_dx_arr = np.asarray(centroid_dx, dtype=np.float64)
    centroid_dy_arr = np.asarray(centroid_dy, dtype=np.float64)
    centroid_mae = float(np.mean(centroid_distances)) if centroid_distances else None
    centroid_max = float(np.max(centroid_distances)) if centroid_distances else None
    centroid_mean_dx = float(np.mean(centroid_dx_arr)) if centroid_dx else None
    centroid_mean_dy = float(np.mean(centroid_dy_arr)) if centroid_dy else None
    centroid_mean_abs_dx = float(np.mean(np.abs(centroid_dx_arr))) if centroid_dx else None
    centroid_mean_abs_dy = float(np.mean(np.abs(centroid_dy_arr))) if centroid_dy else None
    centroid_rms_dx = float(np.sqrt(np.mean(centroid_dx_arr**2))) if centroid_dx else None
    centroid_rms_dy = float(np.sqrt(np.mean(centroid_dy_arr**2))) if centroid_dy else None

    matched_catalog_truth_support = 0
    for matched_star in matching.matched:
        los_truth = np.array(
            [radec_to_unit_vector(star.ra_deg, star.dec_deg) for star in truth_stars],
            dtype=np.float64,
        )
        dots = np.clip(los_truth @ matched_star.los_inertial, -1.0, 1.0)
        min_sep_arcsec = np.degrees(np.arccos(np.max(dots))) * 3600.0
        if min_sep_arcsec <= 5.0:
            matched_catalog_truth_support += 1

    matched_ratio = None
    if matching.matched:
        matched_ratio = matched_catalog_truth_support / len(matching.matched)

    non_roll_error_arcsec, roll_error_arcsec, total_attitude_error_arcsec = _compute_attitude_error_metrics(
        solution,
        dataset_ctx,
    )
    boresight_error_arcsec = non_roll_error_arcsec

    meta = {
        "truth_source": truth_source,
        "field_offset_x_pix": None if dataset_ctx is None else dataset_ctx.field_offset_x_pix,
        "field_offset_y_pix": None if dataset_ctx is None else dataset_ctx.field_offset_y_pix,
        "field_offset_source": None if dataset_ctx is None else dataset_ctx.field_offset_source,
    }
    if cfg is not None and raw is not None and preprocessed is not None:
        try:
            audit_payload = compute_centroid_step_audit(raw, preprocessed, candidates, cfg, truth_stars)
        except Exception as exc:
            audit_payload = {
                "enabled": True,
                "error": str(exc),
            }
        meta["centroid_step_audit"] = audit_payload

    return FrameEvaluation(
        num_truth_stars=len(truth_stars),
        num_candidate_truth_matches=sum(d <= 3.0 for d in centroid_distances),
        centroid_mae_pix=centroid_mae,
        centroid_max_pix=centroid_max,
        matched_catalog_truth_support=matched_catalog_truth_support,
        matched_catalog_truth_ratio=matched_ratio,
        boresight_error_arcsec=boresight_error_arcsec,
        non_roll_error_arcsec=non_roll_error_arcsec,
        roll_error_arcsec=roll_error_arcsec,
        total_attitude_error_arcsec=total_attitude_error_arcsec,
        centroid_mean_dx_pix=centroid_mean_dx,
        centroid_mean_dy_pix=centroid_mean_dy,
        centroid_mean_abs_dx_pix=centroid_mean_abs_dx,
        centroid_mean_abs_dy_pix=centroid_mean_abs_dy,
        centroid_rms_dx_pix=centroid_rms_dx,
        centroid_rms_dy_pix=centroid_rms_dy,
        meta=meta,
    )


def summarize_sequence_result(sequence_result) -> dict:
    frame_results = sequence_result.frame_results
    if not frame_results:
        return {
            "num_frames": 0,
            "init_success_rate": 0.0,
            "tracking_keep_rate": 0.0,
            "reacquire_count": 0,
            "mean_num_matched": 0.0,
            "mean_rms_arcsec": np.inf,
            "p95_rms_arcsec": np.inf,
            "mean_boresight_error_arcsec": None,
            "mean_non_roll_error_arcsec": None,
            "p95_non_roll_error_arcsec": None,
            "mean_roll_error_arcsec": None,
            "p95_roll_error_arcsec": None,
            "mean_total_attitude_error_arcsec": None,
            "p95_total_attitude_error_arcsec": None,
            "mean_total_runtime_s": None,
            "p95_total_runtime_s": None,
            "mean_stage_runtime_s": {},
        }

    init_frames = [frame for frame in frame_results if frame.meta.get("requested_mode", frame.solution.mode) == "init"]
    tracking_frames = [frame for frame in frame_results if frame.meta.get("requested_mode", frame.solution.mode) == "tracking"]
    valid_frames = [frame for frame in frame_results if frame.solution.valid]
    rms_values = np.array([frame.solution.residual_rms_arcsec for frame in frame_results], dtype=np.float64)
    matched_values = np.array([frame.solution.num_matched for frame in frame_results], dtype=np.float64)
    boresight_errors = _collect_eval_values(frame_results, "boresight_error_arcsec")
    non_roll_errors = _collect_eval_values(frame_results, "non_roll_error_arcsec")
    roll_errors = _collect_eval_values(frame_results, "roll_error_arcsec")
    total_attitude_errors = _collect_eval_values(frame_results, "total_attitude_error_arcsec")
    total_runtimes = np.array(
        [
            frame.meta.get("timings_s", {}).get("total")
            for frame in frame_results
            if frame.meta.get("timings_s", {}).get("total") is not None
        ],
        dtype=np.float64,
    )
    stage_names = sorted(
        {
            stage
            for frame in frame_results
            for stage in frame.meta.get("timings_s", {})
            if stage != "total"
        }
    )
    mean_stage_runtime_s = {}
    for stage in stage_names:
        values = np.array(
            [
                frame.meta.get("timings_s", {}).get(stage)
                for frame in frame_results
                if frame.meta.get("timings_s", {}).get(stage) is not None
            ],
            dtype=np.float64,
        )
        if values.size:
            mean_stage_runtime_s[stage] = float(np.mean(values))

    return {
        "num_frames": len(frame_results),
        "init_success_rate": (
            sum(frame.solution.valid for frame in init_frames) / len(init_frames) if init_frames else 0.0
        ),
        "tracking_keep_rate": (
            sum(frame.solution.valid for frame in tracking_frames) / len(tracking_frames) if tracking_frames else 0.0
        ),
        "reacquire_count": int(sum(state.transition_reason == "reacquire_init" for state in sequence_result.state_history)),
        "mean_num_matched": float(np.mean(matched_values)) if matched_values.size else 0.0,
        "mean_rms_arcsec": float(np.mean(rms_values)) if rms_values.size else np.inf,
        "p95_rms_arcsec": float(np.percentile(rms_values, 95)) if rms_values.size else np.inf,
        "mean_boresight_error_arcsec": float(np.mean(boresight_errors)) if boresight_errors.size else None,
        "mean_non_roll_error_arcsec": float(np.mean(non_roll_errors)) if non_roll_errors.size else None,
        "p95_non_roll_error_arcsec": float(np.percentile(non_roll_errors, 95)) if non_roll_errors.size else None,
        "mean_roll_error_arcsec": float(np.mean(roll_errors)) if roll_errors.size else None,
        "p95_roll_error_arcsec": float(np.percentile(roll_errors, 95)) if roll_errors.size else None,
        "mean_total_attitude_error_arcsec": (
            float(np.mean(total_attitude_errors)) if total_attitude_errors.size else None
        ),
        "p95_total_attitude_error_arcsec": (
            float(np.percentile(total_attitude_errors, 95)) if total_attitude_errors.size else None
        ),
        "mean_total_runtime_s": float(np.mean(total_runtimes)) if total_runtimes.size else None,
        "p95_total_runtime_s": float(np.percentile(total_runtimes, 95)) if total_runtimes.size else None,
        "mean_stage_runtime_s": mean_stage_runtime_s,
        "num_valid_frames": len(valid_frames),
        "final_mode": sequence_result.state_history[-1].mode if sequence_result.state_history else None,
    }


def evaluate_dataset(
    dataset_root: str,
    cfg: dict,
    models: dict,
) -> dict:
    from pathlib import Path

    from fsglib.common.io import load_dataset_batch
    from fsglib.pipeline.run_tracking import run_sequence_tracking

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    eval_cfg = cfg.get("evaluation", {})
    frame_stride = max(1, int(eval_cfg.get("frame_stride", 1)))
    max_frames_per_batch = eval_cfg.get("max_frames_per_batch")
    batch_glob = eval_cfg.get("batch_glob", "batch*")

    if (root / "frames").exists():
        batch_dirs = [root]
    else:
        batch_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.match(batch_glob))
    batch_summaries: dict[str, dict] = {}
    sequence_results = {}

    for batch_dir in batch_dirs:
        dataset_ctx = load_dataset_batch(batch_dir, cfg=cfg)
        frame_paths = dataset_ctx.frame_paths[::frame_stride]
        if max_frames_per_batch is not None:
            frame_paths = frame_paths[: int(max_frames_per_batch)]
        npz_paths = [str(path) for path in frame_paths]
        sequence_result = run_sequence_tracking(npz_paths, cfg, models, dataset_ctx=dataset_ctx)
        batch_summaries[batch_dir.name] = summarize_sequence_result(sequence_result)
        sequence_results[batch_dir.name] = sequence_result

    if not batch_summaries:
        return {"dataset_root": str(root), "batches": {}, "summary": {"num_batches": 0}}

    mean_num_matched = np.array([item["mean_num_matched"] for item in batch_summaries.values()], dtype=np.float64)
    mean_rms_arcsec = np.array([item["mean_rms_arcsec"] for item in batch_summaries.values()], dtype=np.float64)
    p95_rms_arcsec = np.array([item["p95_rms_arcsec"] for item in batch_summaries.values()], dtype=np.float64)
    mean_boresight = np.array(
        [item["mean_boresight_error_arcsec"] for item in batch_summaries.values() if item["mean_boresight_error_arcsec"] is not None],
        dtype=np.float64,
    )
    mean_non_roll = np.array(
        [item["mean_non_roll_error_arcsec"] for item in batch_summaries.values() if item["mean_non_roll_error_arcsec"] is not None],
        dtype=np.float64,
    )
    p95_non_roll = np.array(
        [item["p95_non_roll_error_arcsec"] for item in batch_summaries.values() if item["p95_non_roll_error_arcsec"] is not None],
        dtype=np.float64,
    )
    mean_roll = np.array(
        [item["mean_roll_error_arcsec"] for item in batch_summaries.values() if item["mean_roll_error_arcsec"] is not None],
        dtype=np.float64,
    )
    p95_roll = np.array(
        [item["p95_roll_error_arcsec"] for item in batch_summaries.values() if item["p95_roll_error_arcsec"] is not None],
        dtype=np.float64,
    )
    mean_total_attitude = np.array(
        [
            item["mean_total_attitude_error_arcsec"]
            for item in batch_summaries.values()
            if item["mean_total_attitude_error_arcsec"] is not None
        ],
        dtype=np.float64,
    )
    p95_total_attitude = np.array(
        [
            item["p95_total_attitude_error_arcsec"]
            for item in batch_summaries.values()
            if item["p95_total_attitude_error_arcsec"] is not None
        ],
        dtype=np.float64,
    )
    mean_runtime = np.array(
        [item["mean_total_runtime_s"] for item in batch_summaries.values() if item["mean_total_runtime_s"] is not None],
        dtype=np.float64,
    )
    stage_names = sorted(
        {
            stage
            for item in batch_summaries.values()
            for stage in item.get("mean_stage_runtime_s", {})
        }
    )
    mean_stage_runtime_s = {}
    for stage in stage_names:
        values = np.array(
            [
                item.get("mean_stage_runtime_s", {}).get(stage)
                for item in batch_summaries.values()
                if item.get("mean_stage_runtime_s", {}).get(stage) is not None
            ],
            dtype=np.float64,
        )
        if values.size:
            mean_stage_runtime_s[stage] = float(np.mean(values))

    summary = {
        "num_batches": len(batch_summaries),
        "frame_stride": frame_stride,
        "max_frames_per_batch": int(max_frames_per_batch) if max_frames_per_batch is not None else None,
        "init_success_rate": float(np.mean([item["init_success_rate"] for item in batch_summaries.values()])),
        "tracking_keep_rate": float(np.mean([item["tracking_keep_rate"] for item in batch_summaries.values()])),
        "reacquire_count": int(sum(item["reacquire_count"] for item in batch_summaries.values())),
        "mean_num_matched": float(np.mean(mean_num_matched)) if mean_num_matched.size else 0.0,
        "mean_rms_arcsec": float(np.mean(mean_rms_arcsec)) if mean_rms_arcsec.size else np.inf,
        "p95_rms_arcsec": float(np.max(p95_rms_arcsec)) if p95_rms_arcsec.size else np.inf,
        "mean_boresight_error_arcsec": float(np.mean(mean_boresight)) if mean_boresight.size else None,
        "mean_non_roll_error_arcsec": float(np.mean(mean_non_roll)) if mean_non_roll.size else None,
        "p95_non_roll_error_arcsec": float(np.max(p95_non_roll)) if p95_non_roll.size else None,
        "mean_roll_error_arcsec": float(np.mean(mean_roll)) if mean_roll.size else None,
        "p95_roll_error_arcsec": float(np.max(p95_roll)) if p95_roll.size else None,
        "mean_total_attitude_error_arcsec": (
            float(np.mean(mean_total_attitude)) if mean_total_attitude.size else None
        ),
        "p95_total_attitude_error_arcsec": (
            float(np.max(p95_total_attitude)) if p95_total_attitude.size else None
        ),
        "mean_total_runtime_s": float(np.mean(mean_runtime)) if mean_runtime.size else None,
        "p95_total_runtime_s": float(np.percentile(mean_runtime, 95)) if mean_runtime.size else None,
        "mean_stage_runtime_s": mean_stage_runtime_s,
    }

    return {
        "dataset_root": str(root),
        "batches": batch_summaries,
        "summary": summary,
        "sequence_results": sequence_results,
    }
