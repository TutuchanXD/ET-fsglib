import json
from pathlib import Path
from typing import Any

import numpy as np

from fsglib.common.coords import radec_to_unit_vector
from fsglib.common.io import load_dataset_batch


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "rms": None,
            "max_abs": None,
            "p90_abs": None,
        }
    abs_arr = np.abs(arr)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "rms": float(np.sqrt(np.mean(arr**2))),
        "max_abs": float(np.max(abs_arr)),
        "p90_abs": float(np.percentile(abs_arr, 90)),
    }


def _array_rms(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr**2)))


def _nearest_truth_indices(points_xy: np.ndarray, truth_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0 or truth_xy.size == 0:
        return np.zeros(0, dtype=np.int64)
    diff = points_xy[:, None, :] - truth_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1)


def _truth_arrays(truth_stars: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    truth_xy = np.asarray(
        [[star.x_pix, star.y_pix] for star in truth_stars],
        dtype=np.float64,
    )
    truth_los = np.asarray(
        [radec_to_unit_vector(star.ra_deg, star.dec_deg) for star in truth_stars],
        dtype=np.float64,
    )
    return truth_xy, truth_los


def _resolve_truth_stars(raw: Any | None, dataset_ctx: Any | None) -> tuple[list[Any], str | None]:
    raw_meta = {} if raw is None else (_get_field(raw, "meta", {}) or {})
    truth_stars = raw_meta.get("truth_stars")
    if truth_stars:
        return list(truth_stars), raw_meta.get("truth_source", "npz_frame_truth")
    if dataset_ctx is not None and getattr(dataset_ctx, "truth_stars", None):
        return list(dataset_ctx.truth_stars), "stars_ecsv_static"
    return [], None


def _truth_source_label(truth_source: str | None) -> str:
    if truth_source == "npz_frame_truth":
        return "frame truth (npz)"
    if truth_source == "stars_ecsv_static":
        return "static truth (stars.ecsv)"
    return "truth"


def _load_dataset_ctx_for_result(result: Any, cfg: dict) -> Any | None:
    meta = _get_field(result, "meta", {}) or {}
    batch_root = meta.get("dataset_batch_root")
    if not batch_root:
        return None
    try:
        return load_dataset_batch(batch_root, cfg=cfg)
    except Exception:
        return None


def _serialize_candidates(candidates: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for candidate in candidates:
        payload.append(
            {
                "source_id": candidate.source_id,
                "detector_id": candidate.detector_id,
                "x_pix": float(candidate.x),
                "y_pix": float(candidate.y),
                "flux": float(candidate.flux),
                "peak": float(candidate.peak),
                "area_pix": int(candidate.area),
                "snr": float(candidate.snr),
                "bbox": list(candidate.bbox),
                "shape": _to_builtin(candidate.shape),
                "flags": _to_builtin(candidate.flags),
            }
        )
    return payload


def _serialize_truth_stars(truth_stars: list[Any]) -> list[dict[str, Any]]:
    if not truth_stars:
        return []
    payload: list[dict[str, Any]] = []
    for star in truth_stars:
        payload.append(
            {
                "source_id": star.source_id,
                "x_pix": float(star.x_pix),
                "y_pix": float(star.y_pix),
                "ra_deg": float(star.ra_deg),
                "dec_deg": float(star.dec_deg),
                "mag": _safe_float(star.mag),
                "meta": _to_builtin(star.meta),
            }
        )
    return payload


def _serialize_reference_stars(reference: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for star in reference:
        payload.append(
            {
                "catalog_id": star.catalog_id,
                "time_s": float(star.time_s),
                "mag_g": _safe_float(star.mag_g),
                "detector_ids_visible": list(star.detector_ids_visible),
                "predicted_xy": {
                    str(detector_id): [float(xy[0]), float(xy[1])]
                    for detector_id, xy in star.predicted_xy.items()
                },
                "predicted_valid": {str(key): bool(val) for key, val in star.predicted_valid.items()},
                "weight_hint": float(star.weight_hint),
                "meta": _to_builtin(star.meta),
            }
        )
    return payload


def _serialize_matches(matching: Any, truth_stars: list[Any]) -> list[dict[str, Any]]:
    if matching is None:
        return []

    truth_xy = np.zeros((0, 2), dtype=np.float64)
    truth_los = np.zeros((0, 3), dtype=np.float64)
    if truth_stars:
        truth_xy, truth_los = _truth_arrays(truth_stars)

    payload: list[dict[str, Any]] = []
    for matched_star in matching.matched:
        truth_payload = None
        if truth_los.size:
            dots = np.clip(truth_los @ matched_star.los_inertial, -1.0, 1.0)
            truth_idx = int(np.argmax(dots))
            truth_star = truth_stars[truth_idx]
            truth_x, truth_y = truth_xy[truth_idx]
            min_sep_arcsec = float(np.degrees(np.arccos(dots[truth_idx])) * 3600.0)
            observed_xy = matched_star.flags.get("observed_xy")
            predicted_xy = matched_star.flags.get("predicted_xy")
            truth_payload = {
                "truth_index": truth_idx,
                "truth_source_id": truth_star.source_id,
                "truth_x_pix": float(truth_x),
                "truth_y_pix": float(truth_y),
                "truth_ra_deg": float(truth_star.ra_deg),
                "truth_dec_deg": float(truth_star.dec_deg),
                "truth_mag": _safe_float(truth_star.mag),
                "catalog_truth_sep_arcsec": min_sep_arcsec,
            }
            if observed_xy is not None:
                truth_payload["truth_to_observed_dx_pix"] = float(observed_xy[0] - truth_x)
                truth_payload["truth_to_observed_dy_pix"] = float(observed_xy[1] - truth_y)
            if predicted_xy is not None:
                truth_payload["truth_to_predicted_dx_pix"] = float(predicted_xy[0] - truth_x)
                truth_payload["truth_to_predicted_dy_pix"] = float(predicted_xy[1] - truth_y)

        payload.append(
            {
                "source_id": matched_star.source_id,
                "detector_id": matched_star.detector_id,
                "catalog_id": matched_star.catalog_id,
                "weight": float(matched_star.weight),
                "match_score": float(matched_star.match_score),
                "residual_arcsec": _safe_float(matched_star.residual_arcsec),
                "flags": _to_builtin(matched_star.flags),
                "truth": truth_payload,
            }
        )
    return payload


def _build_analysis_payload(result: Any, dataset_ctx: Any | None) -> dict[str, Any]:
    candidates = _get_field(result, "candidates", []) or []
    matching = _get_field(result, "matching")
    solution = _get_field(result, "solution")
    evaluation = _get_field(result, "evaluation")
    reference = _get_field(result, "reference", []) or []
    meta = _get_field(result, "meta", {}) or {}
    raw = _get_field(result, "raw")
    truth_stars, truth_source = _resolve_truth_stars(raw, dataset_ctx)
    truth_label = _truth_source_label(truth_source)

    payload: dict[str, Any] = {
        "truth_source": truth_source,
        "counts": {
            "num_candidates": len(candidates),
            "num_reference_stars": len(reference),
            "num_matched": 0 if matching is None else len(matching.matched),
            "num_truth_stars": len(truth_stars),
        },
        "timings_s": _to_builtin(meta.get("timings_s", {})),
        "notes": [],
    }

    if truth_source == "npz_frame_truth":
        payload["notes"] = [
            "candidate_vs_truth compares extracted centroids with per-frame truth exported from the simulator NPZ",
            "per-frame truth already includes static offset, telescope offset, pointing drift, DVA, thermal drift, and jitter-mean shift",
            "matched_truth_alignment uses inertial RA/Dec association for matched stars and helps separate common image shift from local centroid scatter",
            "match_prediction_residual isolates how far extracted stars are from the reference positions used during matching",
        ]
    else:
        payload["notes"] = [
            "candidate_vs_truth compares extracted centroids with static stars.ecsv positions",
            "static stars.ecsv does not include per-frame simulated DVA/pointing/thermal drift, so its error is not pure centroid noise",
            "matched_truth_alignment uses inertial RA/Dec association for matched stars and is a better place to separate common image shift from local centroid scatter",
            "match_prediction_residual isolates how far extracted stars are from the reference positions used during matching",
        ]

    if dataset_ctx is not None:
        payload["dataset"] = {
            "batch_root": str(dataset_ctx.batch_root),
            "batch_center_ra_deg": _safe_float(dataset_ctx.batch_center_ra_deg),
            "batch_center_dec_deg": _safe_float(dataset_ctx.batch_center_dec_deg),
            "pixel_scale_arcsec_per_pix": _safe_float(dataset_ctx.pixel_scale_arcsec_per_pix),
            "field_offset_x_pix": _safe_float(dataset_ctx.field_offset_x_pix),
            "field_offset_y_pix": _safe_float(dataset_ctx.field_offset_y_pix),
            "field_offset_source": dataset_ctx.field_offset_source,
        }

    if solution is not None:
        payload["attitude"] = {
            "valid": bool(solution.valid),
            "mode": solution.mode,
            "residual_rms_arcsec": float(solution.residual_rms_arcsec),
            "residual_max_arcsec": float(solution.residual_max_arcsec),
            "quality_flag": solution.quality_flag,
            "degraded_level": solution.degraded_level,
            "num_rejected": int(solution.num_rejected),
            "active_detector_ids": list(solution.active_detector_ids),
            "solver_iterations": int(solution.solver_iterations),
            "q_ib": _to_builtin(solution.q_ib),
            "quality": _to_builtin(solution.quality),
        }

    if evaluation is not None:
        payload["evaluation"] = {
            "centroid_mae_pix": _safe_float(evaluation.centroid_mae_pix),
            "centroid_max_pix": _safe_float(evaluation.centroid_max_pix),
            "matched_catalog_truth_support": int(evaluation.matched_catalog_truth_support),
            "matched_catalog_truth_ratio": _safe_float(evaluation.matched_catalog_truth_ratio),
            "boresight_error_arcsec": _safe_float(evaluation.boresight_error_arcsec),
            "non_roll_error_arcsec": _safe_float(evaluation.non_roll_error_arcsec),
            "roll_error_arcsec": _safe_float(evaluation.roll_error_arcsec),
            "total_attitude_error_arcsec": _safe_float(evaluation.total_attitude_error_arcsec),
            "meta": _to_builtin(evaluation.meta),
        }

    if not truth_stars:
        return payload

    truth_xy, truth_los = _truth_arrays(truth_stars)

    if candidates:
        candidate_xy = np.asarray([[cand.x, cand.y] for cand in candidates], dtype=np.float64)
        truth_idx = _nearest_truth_indices(candidate_xy, truth_xy)
        diff_xy = candidate_xy - truth_xy[truth_idx]
        diff_r = np.sqrt(np.sum(diff_xy**2, axis=1))
        common_shift = diff_xy.mean(axis=0)
        local_diff_xy = diff_xy - common_shift
        local_diff_r = np.sqrt(np.sum(local_diff_xy**2, axis=1))

        candidate_truth_payload = {
            "association_mode": "nearest_pixel_truth",
            "truth_source": truth_source,
            "truth_label": truth_label,
            "mean_dx_pix": float(common_shift[0]),
            "mean_dy_pix": float(common_shift[1]),
            "mean_shift_mag_pix": float(np.linalg.norm(common_shift)),
            "dx_stats": _safe_stats(diff_xy[:, 0]),
            "dy_stats": _safe_stats(diff_xy[:, 1]),
            "radial_error_stats": _safe_stats(diff_r),
            "after_common_shift_radial_error_stats": _safe_stats(local_diff_r),
            "candidate_truth_match_within_1pix_ratio": float(np.mean(diff_r <= 1.0)),
            "candidate_truth_match_within_0p3pix_ratio": float(np.mean(diff_r <= 0.3)),
        }
        payload["candidate_vs_truth"] = candidate_truth_payload
        if truth_source == "stars_ecsv_static":
            payload["candidate_vs_static_truth"] = candidate_truth_payload

    if matching is None or not matching.matched:
        return payload

    truth_dx: list[float] = []
    truth_dy: list[float] = []
    truth_dr: list[float] = []
    pred_dx: list[float] = []
    pred_dy: list[float] = []
    pred_dr: list[float] = []
    truth_sep_arcsec: list[float] = []

    for matched_star in matching.matched:
        dots = np.clip(truth_los @ matched_star.los_inertial, -1.0, 1.0)
        truth_idx = int(np.argmax(dots))
        truth_xy_i = truth_xy[truth_idx]
        truth_sep_arcsec.append(float(np.degrees(np.arccos(dots[truth_idx])) * 3600.0))

        observed_xy = matched_star.flags.get("observed_xy")
        predicted_xy = matched_star.flags.get("predicted_xy")
        if observed_xy is None:
            continue

        obs_xy = np.asarray(observed_xy, dtype=np.float64)
        diff_truth = obs_xy - truth_xy_i
        truth_dx.append(float(diff_truth[0]))
        truth_dy.append(float(diff_truth[1]))
        truth_dr.append(float(np.linalg.norm(diff_truth)))

        if predicted_xy is not None:
            pred_xy = np.asarray(predicted_xy, dtype=np.float64)
            diff_pred = obs_xy - pred_xy
            pred_dx.append(float(diff_pred[0]))
            pred_dy.append(float(diff_pred[1]))
            pred_dr.append(float(np.linalg.norm(diff_pred)))

    truth_dx_arr = np.asarray(truth_dx, dtype=np.float64)
    truth_dy_arr = np.asarray(truth_dy, dtype=np.float64)
    truth_dr_arr = np.asarray(truth_dr, dtype=np.float64)
    pred_dx_arr = np.asarray(pred_dx, dtype=np.float64)
    pred_dy_arr = np.asarray(pred_dy, dtype=np.float64)
    pred_dr_arr = np.asarray(pred_dr, dtype=np.float64)

    if truth_dx_arr.size:
        common_shift = np.array(
            [float(np.mean(truth_dx_arr)), float(np.mean(truth_dy_arr))],
            dtype=np.float64,
        )
        local_dx = truth_dx_arr - common_shift[0]
        local_dy = truth_dy_arr - common_shift[1]
        local_dr = np.sqrt(local_dx**2 + local_dy**2)
        payload["matched_truth_alignment"] = {
            "association_mode": "nearest_inertial_truth",
            "truth_source": truth_source,
            "truth_label": truth_label,
            "mean_dx_pix": float(common_shift[0]),
            "mean_dy_pix": float(common_shift[1]),
            "mean_shift_mag_pix": float(np.linalg.norm(common_shift)),
            "catalog_truth_sep_arcsec_max": float(np.max(truth_sep_arcsec)),
            "dx_stats": _safe_stats(truth_dx_arr),
            "dy_stats": _safe_stats(truth_dy_arr),
            "radial_error_stats": _safe_stats(truth_dr_arr),
            "after_common_shift_dx_stats": _safe_stats(local_dx),
            "after_common_shift_dy_stats": _safe_stats(local_dy),
            "after_common_shift_radial_error_stats": _safe_stats(local_dr),
        }

    if pred_dx_arr.size:
        payload["match_prediction_residual"] = {
            "mean_dx_pix": float(np.mean(pred_dx_arr)),
            "mean_dy_pix": float(np.mean(pred_dy_arr)),
            "dx_stats": _safe_stats(pred_dx_arr),
            "dy_stats": _safe_stats(pred_dy_arr),
            "radial_error_stats": _safe_stats(pred_dr_arr),
        }

    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _save_overlay_truth_candidates(
    bundle_dir: Path,
    image: np.ndarray,
    truth_stars: list[Any],
    truth_source: str | None,
    candidates: list[Any],
    matching: Any | None,
) -> None:
    if not truth_stars:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    matched_ids = set()
    if matching is not None:
        matched_ids = {matched.source_id for matched in matching.matched}

    truth_xy = np.asarray([[star.x_pix, star.y_pix] for star in truth_stars], dtype=np.float64)
    candidate_xy = np.asarray([[cand.x, cand.y] for cand in candidates], dtype=np.float64) if candidates else np.zeros((0, 2))
    matched_xy = np.asarray(
        [[cand.x, cand.y] for cand in candidates if cand.source_id in matched_ids],
        dtype=np.float64,
    ) if candidates else np.zeros((0, 2))
    truth_label = _truth_source_label(truth_source)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    ax.scatter(
        truth_xy[:, 0],
        truth_xy[:, 1],
        s=42,
        facecolors="none",
        edgecolors="#00d5ff",
        linewidths=0.8,
        label=truth_label,
    )
    if candidate_xy.size:
        ax.scatter(
            candidate_xy[:, 0],
            candidate_xy[:, 1],
            s=22,
            c="#ff9f1c",
            marker="x",
            linewidths=0.8,
            label="extracted centroid",
        )
    if matched_xy.size:
        ax.scatter(
            matched_xy[:, 0],
            matched_xy[:, 1],
            s=52,
            facecolors="none",
            edgecolors="#7CFC00",
            linewidths=0.9,
            label="matched centroid",
        )
    ax.set_title(f"{truth_label.title()} vs Extracted Centroids")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(bundle_dir / "overlay_truth_candidates.png", bbox_inches="tight")
    plt.close(fig)


def _save_matched_truth_bias_plot(
    bundle_dir: Path,
    image: np.ndarray,
    matches_payload: list[dict[str, Any]],
    truth_source: str | None,
    title_suffix: str = "",
) -> None:
    usable = [item for item in matches_payload if item.get("truth") is not None and item["flags"].get("observed_xy") is not None]
    if not usable:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    truth_xy = np.asarray(
        [[item["truth"]["truth_x_pix"], item["truth"]["truth_y_pix"]] for item in usable],
        dtype=np.float64,
    )
    observed_xy = np.asarray([item["flags"]["observed_xy"] for item in usable], dtype=np.float64)
    delta_xy = observed_xy - truth_xy
    mean_dx = float(np.mean(delta_xy[:, 0]))
    mean_dy = float(np.mean(delta_xy[:, 1]))
    arrow_scale_factor = 20.0
    truth_label = _truth_source_label(truth_source)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    ax.scatter(
        truth_xy[:, 0],
        truth_xy[:, 1],
        s=40,
        facecolors="none",
        edgecolors="#00d5ff",
        linewidths=0.8,
        label=truth_label,
    )
    ax.scatter(
        observed_xy[:, 0],
        observed_xy[:, 1],
        s=20,
        c="#ff9f1c",
        marker="x",
        linewidths=0.8,
        label="matched centroid",
    )
    ax.quiver(
        truth_xy[:, 0],
        truth_xy[:, 1],
        delta_xy[:, 0],
        delta_xy[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / arrow_scale_factor,
        color="white",
        width=0.002,
        alpha=0.7,
    )
    ax.set_title(
        f"Matched {truth_label.title()} -> Observed Bias{title_suffix}\nmean shift = ({mean_dx:+.3f}, {mean_dy:+.3f}) pix, vectors x{arrow_scale_factor:.0f}"
    )
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(bundle_dir / "matched_truth_bias.png", bbox_inches="tight")
    plt.close(fig)


def _save_matched_prediction_plot(
    bundle_dir: Path,
    image: np.ndarray,
    matches_payload: list[dict[str, Any]],
) -> None:
    usable = [
        item
        for item in matches_payload
        if item["flags"].get("observed_xy") is not None and item["flags"].get("predicted_xy") is not None
    ]
    if not usable:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    predicted_xy = np.asarray([item["flags"]["predicted_xy"] for item in usable], dtype=np.float64)
    observed_xy = np.asarray([item["flags"]["observed_xy"] for item in usable], dtype=np.float64)
    delta_xy = observed_xy - predicted_xy
    rms_pix = _array_rms(np.sqrt(np.sum(delta_xy**2, axis=1)))
    arrow_scale_factor = 20.0

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    ax.scatter(
        predicted_xy[:, 0],
        predicted_xy[:, 1],
        s=34,
        facecolors="none",
        edgecolors="#ff5a5f",
        linewidths=0.8,
        label="predicted reference",
    )
    ax.scatter(
        observed_xy[:, 0],
        observed_xy[:, 1],
        s=18,
        c="#7CFC00",
        marker="x",
        linewidths=0.8,
        label="observed centroid",
    )
    ax.quiver(
        predicted_xy[:, 0],
        predicted_xy[:, 1],
        delta_xy[:, 0],
        delta_xy[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / arrow_scale_factor,
        color="#ffd166",
        width=0.002,
        alpha=0.75,
    )
    title = "Predicted Reference -> Observed Residual"
    if rms_pix is not None:
        title += f"\nRMS = {rms_pix:.3f} pix, vectors x{arrow_scale_factor:.0f}"
    ax.set_title(title)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(bundle_dir / "matched_prediction_overlay.png", bbox_inches="tight")
    plt.close(fig)


def _build_solution_payload(result: Any) -> dict[str, Any]:
    solution = _get_field(result, "solution")
    matching = _get_field(result, "matching")
    evaluation = _get_field(result, "evaluation")
    meta = _get_field(result, "meta", {}) or {}

    payload = {
        "valid": solution.valid,
        "num_matched": solution.num_matched,
        "num_rejected": solution.num_rejected,
        "q_ib": solution.q_ib.tolist(),
        "residual_rms_arcsec": solution.residual_rms_arcsec,
        "residual_max_arcsec": solution.residual_max_arcsec,
        "quality_flag": solution.quality_flag,
        "degraded_level": solution.degraded_level,
        "active_detector_ids": solution.active_detector_ids,
        "solver_iterations": solution.solver_iterations,
        "quality": solution.quality,
        "timings_s": meta.get("timings_s"),
    }
    if matching is not None:
        payload["matching"] = {
            "success": matching.success,
            "score": matching.score,
            "debug": matching.debug,
        }
    if evaluation is not None:
        centroid_step_audit = evaluation.meta.get("centroid_step_audit") if isinstance(evaluation.meta, dict) else None
        centroid_step_audit_summary = None
        if isinstance(centroid_step_audit, dict):
            centroid_step_audit_summary = {
                "enabled": centroid_step_audit.get("enabled"),
                "error": centroid_step_audit.get("error"),
                "num_audited_stars": centroid_step_audit.get("num_audited_stars"),
                "stage_stats": centroid_step_audit.get("stage_stats"),
                "transition_stats": centroid_step_audit.get("transition_stats"),
            }
        evaluation_meta = dict(evaluation.meta) if isinstance(evaluation.meta, dict) else {}
        if "centroid_step_audit" in evaluation_meta:
            evaluation_meta["centroid_step_audit"] = centroid_step_audit_summary
        payload["evaluation"] = {
            "num_truth_stars": evaluation.num_truth_stars,
            "num_candidate_truth_matches": evaluation.num_candidate_truth_matches,
            "centroid_mae_pix": evaluation.centroid_mae_pix,
            "centroid_max_pix": evaluation.centroid_max_pix,
            "centroid_mean_dx_pix": evaluation.centroid_mean_dx_pix,
            "centroid_mean_dy_pix": evaluation.centroid_mean_dy_pix,
            "centroid_mean_abs_dx_pix": evaluation.centroid_mean_abs_dx_pix,
            "centroid_mean_abs_dy_pix": evaluation.centroid_mean_abs_dy_pix,
            "centroid_rms_dx_pix": evaluation.centroid_rms_dx_pix,
            "centroid_rms_dy_pix": evaluation.centroid_rms_dy_pix,
            "matched_catalog_truth_support": evaluation.matched_catalog_truth_support,
            "matched_catalog_truth_ratio": evaluation.matched_catalog_truth_ratio,
            "boresight_error_arcsec": evaluation.boresight_error_arcsec,
            "non_roll_error_arcsec": evaluation.non_roll_error_arcsec,
            "roll_error_arcsec": evaluation.roll_error_arcsec,
            "total_attitude_error_arcsec": evaluation.total_attitude_error_arcsec,
            "meta": evaluation_meta,
            "centroid_step_audit_summary": centroid_step_audit_summary,
        }
    return payload


def _write_bundle_readme(bundle_dir: Path, result: Any, analysis: dict[str, Any]) -> None:
    raw = _get_field(result, "raw")
    solution = _get_field(result, "solution")
    truth_source = analysis.get("truth_source")
    truth_label = _truth_source_label(truth_source)
    lines = [
        f"# Debug Bundle: {bundle_dir.name}",
        "",
        "## 文件说明",
        "- `raw.npy`: 原始图像数组。",
        "- `preprocessed.npy`: 预处理后的图像数组。",
        "- `noise_map.npy`: 预处理估计得到的噪声图。",
        f"- `truth_stars.json`: 当前帧使用的真值星表（来源：`{truth_source or 'unknown'}`，标签：`{truth_label}`）。",
        "- `reference_stars.json`: 当前帧构造出来的参考星集合和预测像点。",
        "- `candidates.json`: 提取到的候选星点质心。",
        "- `matches.json`: 最终参与姿态解算的匹配对，以及 truth/预测/观测三者之间的关系。",
        "- `solution.json`: 主结果文件。",
        "- `analysis.json`: 误差分解文件，用于区分静态 truth 偏差、公共平移项、局部质心散布、匹配残差和姿态误差。",
        "- `centroid_step_audit.json`: 单星 vs 多星质心提取分步骤审计结果，重点看每一步的 `x / y / 总误差` 如何变化。",
        "- `overlay_truth_candidates.png`: 当前 truth 与提取质心叠加图。",
        "- `matched_truth_bias.png`: matched 星从当前 truth 到观测质心的偏差箭头图。",
        "- `matched_prediction_overlay.png`: 参考预测像点到观测质心的残差箭头图。",
        "",
        "## solution.json 字段含义",
        "- `valid`: 本帧姿态解是否通过当前有效性门限。",
        "- `num_matched`: 进入姿态解算的匹配星数。",
        "- `num_rejected`: 姿态求解阶段被剔除的星数。",
        "- `q_ib`: 四元数 `[w, x, y, z]`，表示惯性系到本体系的旋转。",
        "- `residual_rms_arcsec`: 姿态解算后，matched 星方向矢量残差的 RMS。它不是质心对 truth 的像素误差。",
        "- `residual_max_arcsec`: matched 星方向矢量残差的最大值。",
        "- `quality_flag`: 当前质量标签，例如 `VALID`、`INVALID`。",
        "- `degraded_level`: 当前解算是否处于降级模式。",
        "- `active_detector_ids`: 参与当前解算的探测器编号列表。",
        "- `solver_iterations`: 当前求解器迭代次数。",
        "- `quality`: 姿态解算质量摘要，例如输入星数、使用星数、残差门限。",
        "- `timings_s`: 各阶段耗时统计，单位秒。",
        "- `matching.debug.mean_residual_pix`: 匹配阶段中，参考预测像点到观测质心的平均像面残差。",
        f"- `evaluation.centroid_mae_pix`: 提取质心到当前 truth（`{truth_label}`）的平均最近邻距离。",
        "- `evaluation.centroid_mean_dx_pix / centroid_mean_dy_pix`: 候选质心相对 truth 的平均 `x / y` 偏差。",
        "- `evaluation.centroid_rms_dx_pix / centroid_rms_dy_pix`: 候选质心相对 truth 的 `x / y` 方向 RMS。",
        "- `evaluation.non_roll_error_arcsec`: 非绕光轴姿态误差。",
        "- `evaluation.roll_error_arcsec`: 绕光轴姿态误差。",
        "- `evaluation.total_attitude_error_arcsec`: 总姿态误差角。",
        "- `evaluation.meta.centroid_step_audit.stage_stats`: 单星/多星各阶段的 `x / y / 总误差` 汇总。",
        "- `evaluation.meta.centroid_step_audit.transition_stats`: 从一步到下一步时，位置和误差是如何变化的。",
        "",
        "## 当前帧快速解读",
        f"- 输入文件: `{raw.meta.get('npz_path', 'unknown')}`",
        f"- truth source: `{truth_source or 'unknown'}`",
        f"- `valid = {solution.valid}`",
        f"- `num_matched = {solution.num_matched}`",
        f"- `residual_rms_arcsec = {solution.residual_rms_arcsec:.6f}`",
    ]

    matched_truth = analysis.get("matched_truth_alignment", {})
    if matched_truth:
        lines.extend(
            [
                f"- matched 星相对当前 truth 的公共平移项约为 `({matched_truth.get('mean_dx_pix', 0.0):+.3f}, {matched_truth.get('mean_dy_pix', 0.0):+.3f}) pix`",
                "- `after_common_shift_radial_error_stats.rms` 更接近局部质心散布，而不是整帧公共平移。",
            ]
        )

    lines.extend(
        [
            "",
            "## 解读建议",
            "- 先看 `overlay_truth_candidates.png`，判断提取质心整体是否相对当前 truth 出现系统平移。",
            "- 再看 `matched_truth_bias.png`，如果箭头大致同向，说明主导项更像公共位移，不是随机质心噪声。",
            "- 再看 `matched_prediction_overlay.png`，判断匹配使用的参考像点和观测质心之间还剩多少局部残差。",
            "- 最后结合 `analysis.json` 和 `solution.json`，区分图像端误差与姿态解算端误差。",
            "",
        ]
    )

    (bundle_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def save_debug_bundle(result: Any, cfg: dict) -> Path | None:
    if not cfg["project"].get("save_debug", False):
        return None

    out_dir = Path(cfg["project"].get("output_dir", "outputs/debug"))

    raw = _get_field(result, "raw")
    preprocessed = _get_field(result, "preprocessed")
    candidates = _get_field(result, "candidates", []) or []
    reference = _get_field(result, "reference", []) or []
    matching = _get_field(result, "matching")

    npz_name = Path(raw.meta.get("npz_path", "unknown")).stem
    bundle_dir = out_dir / npz_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    dataset_ctx = _load_dataset_ctx_for_result(result, cfg)
    truth_stars, truth_source = _resolve_truth_stars(raw, dataset_ctx)
    truth_payload = {
        "truth_source": truth_source,
        "truth_label": _truth_source_label(truth_source),
        "stars": _serialize_truth_stars(truth_stars),
    }
    reference_payload = _serialize_reference_stars(reference)
    candidates_payload = _serialize_candidates(candidates)
    matches_payload = _serialize_matches(matching, truth_stars)
    solution_payload = _build_solution_payload(result)
    analysis_payload = _build_analysis_payload(result, dataset_ctx)

    if cfg["logging"].get("save_intermediate_arrays", True):
        np.save(bundle_dir / "raw.npy", raw.image)
        np.save(bundle_dir / "preprocessed.npy", preprocessed.image)
        np.save(bundle_dir / "noise_map.npy", preprocessed.noise_map)

    _write_json(bundle_dir / "truth_stars.json", truth_payload)
    _write_json(bundle_dir / "reference_stars.json", reference_payload)
    _write_json(bundle_dir / "candidates.json", candidates_payload)
    _write_json(bundle_dir / "matches.json", matches_payload)
    _write_json(bundle_dir / "solution.json", solution_payload)
    _write_json(bundle_dir / "analysis.json", analysis_payload)
    evaluation = _get_field(result, "evaluation")
    if evaluation is not None and isinstance(evaluation.meta, dict):
        centroid_step_audit = evaluation.meta.get("centroid_step_audit")
        if centroid_step_audit is not None:
            _write_json(bundle_dir / "centroid_step_audit.json", centroid_step_audit)

    image_for_plot = preprocessed.image if preprocessed is not None else raw.image
    _save_overlay_truth_candidates(
        bundle_dir,
        image_for_plot,
        truth_stars,
        truth_source,
        candidates,
        matching,
    )
    _save_matched_truth_bias_plot(bundle_dir, image_for_plot, matches_payload, truth_source)
    _save_matched_prediction_plot(bundle_dir, image_for_plot, matches_payload)
    _write_bundle_readme(bundle_dir, result, analysis_payload)

    print(f"[*] Debug bundle saved to {bundle_dir}")
    return bundle_dir
