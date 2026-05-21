import numpy as np
from scipy.spatial.transform import Rotation
from fsglib.common.types import AttitudeQuality, AttitudeSolution, AttitudeSolveInput, MatchedStar


ARCSEC_PER_RAD = 206264.80624709636


def _normalize_detector_id(value) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def scalar_first_quat_to_scipy_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def scipy_xyzw_to_scalar_first_quat(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """
    Convert an inertial-to-body DCM to a unit quaternion in scalar-first form [w, x, y, z].
    """
    rot = Rotation.from_matrix(dcm)
    return scipy_xyzw_to_scalar_first_quat(rot.as_quat())

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert a scalar-first quaternion [w, x, y, z] to an inertial-to-body DCM.
    """
    rot = Rotation.from_quat(scalar_first_quat_to_scipy_xyzw(q))
    return rot.as_matrix()

def compute_residuals(c_ib: np.ndarray, matched_stars: list[MatchedStar]) -> np.ndarray:
    """
    Compute angle residuals (in arcsec) between the rotated inertial vector and the body vector.
    """
    res = []
    for m in matched_stars:
        w = np.asarray(m.los_body, dtype=np.float64)
        v = np.asarray(m.los_inertial, dtype=np.float64)
        
        # v_body = C_ib * v_inertial
        v_rot = c_ib @ v
        
        # Angle between w and v_rot
        cos_theta = np.clip(np.dot(w, v_rot), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        res.append(np.rad2deg(angle_rad) * 3600.0) # convert to arcseconds
        
    return np.array(res, dtype=np.float64)


def compute_weights(
    matched_stars: list[MatchedStar],
    cfg: dict,
) -> np.ndarray:
    if not matched_stars:
        return np.zeros(0, dtype=np.float64)
    weights = np.array([max(float(star.weight), 1e-6) for star in matched_stars], dtype=np.float64)
    return weights


def _sigma_angle_arcsec(star: MatchedStar) -> float | None:
    value = star.flags.get("sigma_angle_arcsec")
    if value is None:
        return None
    try:
        sigma = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return sigma


def _empty_covariance_meta(reason: str, matched_stars: list[MatchedStar]) -> dict:
    return {
        "available": False,
        "reason": reason,
        "source": None,
        "num_used": len(matched_stars),
        "missing_sigma_count": sum(
            1 for star in matched_stars if _sigma_angle_arcsec(star) is None
        ),
    }


def _attitude_covariance_from_sigmas(
    matched_stars: list[MatchedStar],
    cfg: dict,
) -> tuple[np.ndarray | None, float | None, float | None, float | None, dict]:
    if not cfg.get("attitude", {}).get("estimate_covariance", True):
        return None, None, None, None, _empty_covariance_meta(
            "disabled",
            matched_stars,
        )
    if not matched_stars:
        return None, None, None, None, _empty_covariance_meta(
            "not_enough_stars",
            matched_stars,
        )

    sigmas_arcsec = [_sigma_angle_arcsec(star) for star in matched_stars]
    missing_count = sum(sigma is None for sigma in sigmas_arcsec)
    if missing_count:
        meta = _empty_covariance_meta("missing_sigma_angle_arcsec", matched_stars)
        meta["sigma_angle_arcsec"] = sigmas_arcsec
        return None, None, None, None, meta

    normal = np.zeros((3, 3), dtype=np.float64)
    for star, sigma_arcsec in zip(matched_stars, sigmas_arcsec):
        los = np.asarray(star.los_body, dtype=np.float64)
        norm = float(np.linalg.norm(los))
        if los.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
            meta = _empty_covariance_meta("invalid_los_body", matched_stars)
            meta["sigma_angle_arcsec"] = sigmas_arcsec
            return None, None, None, None, meta
        los = los / norm
        sigma_rad = float(sigma_arcsec) / ARCSEC_PER_RAD
        tangent_projector = np.eye(3, dtype=np.float64) - np.outer(los, los)
        normal += tangent_projector / max(sigma_rad**2, 1e-30)

    normal = 0.5 * (normal + normal.T)
    eigvals, eigvecs = np.linalg.eigh(normal)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
    rank_tol = float(cfg.get("attitude", {}).get("covariance_rank_tol", 1.0e-12))
    min_allowed = max(max_eig * rank_tol, 1e-30)
    if max_eig <= 0.0 or float(np.min(eigvals)) <= min_allowed:
        meta = _empty_covariance_meta("singular_attitude_normal_matrix", matched_stars)
        meta.update(
            {
                "sigma_angle_arcsec": sigmas_arcsec,
                "normal_eigenvalues": eigvals.tolist(),
            }
        )
        return None, None, None, None, meta

    covariance = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        meta = _empty_covariance_meta("nonfinite_attitude_covariance", matched_stars)
        meta["sigma_angle_arcsec"] = sigmas_arcsec
        return None, None, None, None, meta

    condition_number = float(max_eig / float(np.min(eigvals)))
    sigma_non_roll_arcsec = float(
        np.sqrt(max(float(covariance[0, 0] + covariance[1, 1]), 0.0))
        * ARCSEC_PER_RAD
    )
    sigma_roll_arcsec = float(
        np.sqrt(max(float(covariance[2, 2]), 0.0)) * ARCSEC_PER_RAD
    )
    meta = {
        "available": True,
        "reason": None,
        "source": "sigma_angle_arcsec",
        "num_used": len(matched_stars),
        "missing_sigma_count": 0,
        "sigma_angle_arcsec": [float(sigma) for sigma in sigmas_arcsec],
        "normal_eigenvalues": eigvals.tolist(),
        "condition_number": condition_number,
        "sigma_non_roll_arcsec": sigma_non_roll_arcsec,
        "sigma_roll_arcsec": sigma_roll_arcsec,
    }
    return covariance, sigma_non_roll_arcsec, sigma_roll_arcsec, condition_number, meta


def _active_detector_ids(matched_stars: list[MatchedStar]) -> list[int | str]:
    return sorted({_normalize_detector_id(star.detector_id) for star in matched_stars}, key=str)


def _star_audit_identity(star: MatchedStar) -> dict:
    return {
        "detector_id": _normalize_detector_id(star.detector_id),
        "source_id": _normalize_detector_id(star.source_id),
        "catalog_id": _normalize_detector_id(star.catalog_id),
    }


def _robust_sigma_arcsec(residuals: np.ndarray, cfg: dict) -> tuple[float | None, float | None, float | None]:
    if not cfg.get("attitude", {}).get("outlier_mad_fallback_enable", True):
        return None, None, None
    finite = np.asarray(residuals, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return None, None, None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    min_sigma = float(cfg.get("attitude", {}).get("outlier_mad_min_sigma_arcsec", 1.0e-6))
    if not np.isfinite(robust_sigma) or robust_sigma <= min_sigma:
        return None, median, robust_sigma
    clip = float(cfg.get("attitude", {}).get("outlier_sigma_clip", 3.0))
    return float(median + clip * robust_sigma), median, float(robust_sigma)


def _outlier_decisions(
    matched_stars: list[MatchedStar],
    residuals: np.ndarray,
    cfg: dict,
    mode: str,
) -> list[dict]:
    hard_gate = float(cfg.get("attitude", {}).get("outlier_max_residual_arcsec", np.inf))
    sigma_clip = float(cfg.get("attitude", {}).get("outlier_sigma_clip", np.inf))
    use_measurement_sigma = cfg.get("attitude", {}).get("outlier_use_measurement_sigma", True)
    sigma_floor = float(cfg.get("attitude", {}).get("outlier_sigma_floor_arcsec", 0.0))
    mad_threshold, mad_center, mad_sigma = _robust_sigma_arcsec(residuals, cfg)
    decisions: list[dict] = []

    for star, residual in zip(matched_stars, residuals):
        residual = float(residual)
        reasons: list[str] = []
        sigma_arcsec = _sigma_angle_arcsec(star)
        effective_sigma_arcsec = None
        normalized_residual = None
        sigma_threshold_arcsec = None
        mad_threshold_arcsec = None

        if np.isfinite(hard_gate) and residual > hard_gate:
            reasons.append("hard_gate")

        if mode == "sigma_clip_iterative":
            if use_measurement_sigma and sigma_arcsec is not None and np.isfinite(sigma_clip):
                effective_sigma_arcsec = max(float(sigma_arcsec), sigma_floor)
                normalized_residual = residual / effective_sigma_arcsec
                sigma_threshold_arcsec = sigma_clip * effective_sigma_arcsec
                if normalized_residual > sigma_clip:
                    reasons.append("sigma_clip")
            elif mad_threshold is not None:
                mad_threshold_arcsec = mad_threshold
                if residual > mad_threshold:
                    reasons.append("mad_sigma_clip")

        decision = {
            **_star_audit_identity(star),
            "reject": bool(reasons),
            "reasons": reasons,
            "reason": "+".join(reasons) if reasons else None,
            "residual_arcsec": residual,
            "hard_gate_arcsec": hard_gate if np.isfinite(hard_gate) else None,
            "sigma_angle_arcsec": sigma_arcsec,
            "effective_sigma_angle_arcsec": effective_sigma_arcsec,
            "sigma_floor_arcsec": sigma_floor,
            "sigma_clip": sigma_clip if np.isfinite(sigma_clip) else None,
            "sigma_threshold_arcsec": sigma_threshold_arcsec,
            "normalized_residual": normalized_residual,
            "mad_threshold_arcsec": mad_threshold_arcsec,
            "mad_center_arcsec": mad_center,
            "mad_sigma_arcsec": mad_sigma,
            "match_score": float(star.match_score),
        }
        decisions.append(decision)

    return decisions


def _rejection_severity(decision: dict) -> float:
    scores: list[float] = []
    normalized = decision.get("normalized_residual")
    sigma_clip = decision.get("sigma_clip")
    if normalized is not None and sigma_clip not in (None, 0.0):
        scores.append(float(normalized) / float(sigma_clip))
    residual = decision.get("residual_arcsec")
    hard_gate = decision.get("hard_gate_arcsec")
    if residual is not None and hard_gate not in (None, 0.0):
        scores.append(float(residual) / float(hard_gate))
    mad_threshold = decision.get("mad_threshold_arcsec")
    if residual is not None and mad_threshold not in (None, 0.0):
        scores.append(float(residual) / float(mad_threshold))
    return max(scores) if scores else 0.0


def _select_rejections_for_iteration(
    decisions: list[dict],
    cfg: dict,
    mode: str,
) -> list[dict]:
    rejected = [decision for decision in decisions if decision["reject"]]
    if mode == "single_pass":
        for decision in rejected:
            decision["deferred_rejection"] = False
        return rejected
    max_reject = int(
        cfg.get("attitude", {}).get("outlier_max_reject_per_iteration", 1)
    )
    if max_reject <= 0 or len(rejected) <= max_reject:
        for decision in rejected:
            decision["deferred_rejection"] = False
        return rejected

    selected_ids = {
        id(decision)
        for decision in sorted(
            rejected,
            key=lambda decision: (
                _rejection_severity(decision),
                float(decision["residual_arcsec"]),
            ),
            reverse=True,
        )[:max_reject]
    }
    selected = []
    for decision in rejected:
        if id(decision) in selected_ids:
            decision["deferred_rejection"] = False
            selected.append(decision)
        else:
            decision["deferred_rejection"] = True
            decision["reject"] = False
            decision["deferred_reasons"] = list(decision["reasons"])
            decision["reasons"] = []
            decision["reason"] = None
    return selected


def _robust_disabled_meta(
    matched_stars: list[MatchedStar],
    *,
    reason: str,
    mode: str,
    solver_iterations: int,
) -> dict:
    active_ids = _active_detector_ids(matched_stars)
    return {
        "enabled": False,
        "mode": mode,
        "reason": reason,
        "converged": True,
        "convergence_reason": reason,
        "robust_valid": True,
        "num_iterations": solver_iterations,
        "num_rejected": 0,
        "initial_support_count": len(matched_stars),
        "final_support_count": len(matched_stars),
        "final_active_detector_ids": active_ids,
        "final_active_detector_count": len(active_ids),
        "iterations": [],
        "rejected_stars": [],
        "pending_rejections": [],
    }


def _solve_with_robust_rejection(
    matched_stars: list[MatchedStar],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, list[MatchedStar], int, dict, int]:
    att_cfg = cfg.get("attitude", {})
    mode = str(att_cfg.get("outlier_reject_mode", "single_pass")).lower()
    if mode not in {"single_pass", "hard_gate_iterative", "sigma_clip_iterative"}:
        mode = "sigma_clip_iterative"

    if not att_cfg.get("outlier_reject_enable", False):
        q = solve_quest(matched_stars, cfg)
        c = quat_to_dcm(q)
        meta = _robust_disabled_meta(
            matched_stars,
            reason="disabled",
            mode=mode,
            solver_iterations=1,
        )
        return q, c, matched_stars, 0, meta, 1

    max_iterations = max(1, int(att_cfg.get("max_iterations", 2)))
    if mode == "single_pass":
        max_iterations = min(max_iterations, 2)

    min_math = int(att_cfg["min_stars_mathematical"])
    current = list(matched_stars)
    rejected_records: list[dict] = []
    pending_rejections: list[dict] = []
    iteration_records: list[dict] = []
    solver_iterations = 0
    converged = False
    convergence_reason = "max_iterations_reached"
    robust_valid = True

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    c = np.eye(3, dtype=np.float64)

    for iteration_idx in range(max_iterations):
        if len(current) < min_math:
            convergence_reason = "support_below_min_stars_mathematical"
            robust_valid = False
            break

        q = solve_quest(current, cfg)
        c = quat_to_dcm(q)
        solver_iterations += 1
        residuals = compute_residuals(c, current)
        decisions = _outlier_decisions(current, residuals, cfg, mode)
        rejected_this_iter = _select_rejections_for_iteration(decisions, cfg, mode)
        kept = [
            star
            for star, decision in zip(current, decisions)
            if not decision["reject"]
        ]
        iteration_records.append(
            {
                "iteration": iteration_idx,
                "num_input": len(current),
                "num_rejected": len(rejected_this_iter),
                "num_kept": len(kept),
                "residual_rms_arcsec": float(np.sqrt(np.mean(np.square(residuals)))),
                "residual_max_arcsec": float(np.max(residuals)),
                "decisions": decisions,
            }
        )

        if not rejected_this_iter:
            converged = True
            convergence_reason = "no_new_rejections"
            break

        if len(kept) < min_math:
            for record in rejected_this_iter:
                rejected_records.append({**record, "iteration": iteration_idx})
            current = kept
            robust_valid = False
            convergence_reason = "support_below_min_stars_mathematical"
            break

        if iteration_idx + 1 >= max_iterations:
            pending_rejections = rejected_this_iter
            robust_valid = False
            convergence_reason = "max_iterations_reached"
            break

        for record in rejected_this_iter:
            rejected_records.append({**record, "iteration": iteration_idx})
        current = kept

    active_ids = _active_detector_ids(current)
    meta = {
        "enabled": True,
        "mode": mode,
        "converged": converged,
        "convergence_reason": convergence_reason,
        "robust_valid": robust_valid,
        "num_iterations": solver_iterations,
        "num_rejected": len(rejected_records),
        "initial_support_count": len(matched_stars),
        "final_support_count": len(current),
        "final_active_detector_ids": active_ids,
        "final_active_detector_count": len(active_ids),
        "iterations": iteration_records,
        "rejected_stars": rejected_records,
        "pending_rejections": pending_rejections,
    }
    return q, c, current, len(rejected_records), meta, solver_iterations


def _build_b_matrix(
    matched_stars: list[MatchedStar],
    weights: np.ndarray,
) -> np.ndarray:
    B = np.zeros((3, 3), dtype=np.float64)
    for m, weight in zip(matched_stars, weights):
        w = np.asarray(m.los_body, dtype=np.float64)
        v = np.asarray(m.los_inertial, dtype=np.float64)
        w /= np.linalg.norm(w)
        v /= np.linalg.norm(v)
        B += np.outer(w, v) * weight
    return B


def _solve_quest_fallback(B: np.ndarray) -> np.ndarray:
    sigma = float(np.trace(B))
    S = B + B.T
    z = np.array(
        [
            B[2, 1] - B[1, 2],
            B[0, 2] - B[2, 0],
            B[1, 0] - B[0, 1],
        ],
        dtype=np.float64,
    )
    K = np.zeros((4, 4), dtype=np.float64)
    K[:3, :3] = S - sigma * np.eye(3, dtype=np.float64)
    K[:3, 3] = z
    K[3, :3] = z
    K[3, 3] = sigma

    eigvals, eigvecs = np.linalg.eigh(K)
    dominant = eigvecs[:, int(np.argmax(eigvals))]
    q = np.array([dominant[3], dominant[0], dominant[1], dominant[2]], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def solve_quest(matched_stars: list[MatchedStar], cfg: dict) -> np.ndarray:
    weights = compute_weights(matched_stars, cfg)
    B = _build_b_matrix(matched_stars, weights)

    sigma = float(np.trace(B))
    S = B + B.T
    z = np.array(
        [
            B[2, 1] - B[1, 2],
            B[0, 2] - B[2, 0],
            B[1, 0] - B[0, 1],
        ],
        dtype=np.float64,
    )

    kappa = float(
        S[0, 0] * S[1, 1]
        + S[1, 1] * S[2, 2]
        + S[2, 2] * S[0, 0]
        - S[0, 1] * S[1, 0]
        - S[1, 2] * S[2, 1]
        - S[2, 0] * S[0, 2]
    )
    delta = float(np.linalg.det(S))
    z_norm2 = float(z @ z)
    sz = S @ z
    a = sigma * sigma - kappa
    b = sigma * sigma + z_norm2
    c = delta + float(z @ sz)
    d = float(z @ (S @ sz))

    lambda_est = float(np.sum(weights))
    tol = float(cfg.get("attitude", {}).get("quest_tol", 1e-12))
    max_iter = int(cfg.get("attitude", {}).get("quest_max_iter", 50))
    converged = False
    for _ in range(max_iter):
        f = lambda_est**4 - (a + b) * lambda_est**2 - c * lambda_est + (a * b + c * sigma - d)
        fp = 4.0 * lambda_est**3 - 2.0 * (a + b) * lambda_est - c
        if not np.isfinite(f) or not np.isfinite(fp) or abs(fp) < 1e-15:
            break
        step = f / fp
        next_lambda = lambda_est - step
        if not np.isfinite(next_lambda):
            break
        if abs(next_lambda - lambda_est) <= tol:
            lambda_est = next_lambda
            converged = True
            break
        lambda_est = next_lambda

    if not converged and not np.isfinite(lambda_est):
        return _solve_quest_fallback(B)

    a_mat = (lambda_est + sigma) * np.eye(3, dtype=np.float64) - S
    try:
        q_vec = np.linalg.solve(a_mat, z)
    except np.linalg.LinAlgError:
        return _solve_quest_fallback(B)

    q = np.array([1.0, q_vec[0], q_vec[1], q_vec[2]], dtype=np.float64)

    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-15:
        return _solve_quest_fallback(B)

    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)

def _normalize_input(
    solve_input: AttitudeSolveInput | list[MatchedStar],
    cfg: dict,
) -> tuple[list[MatchedStar], str]:
    if isinstance(solve_input, AttitudeSolveInput):
        return solve_input.matched_stars, solve_input.mode
    return solve_input, cfg.get("project", {}).get("mode", "init")


def reject_outliers(
    matched_stars: list[MatchedStar],
    c_ib: np.ndarray,
    cfg: dict,
) -> tuple[list[MatchedStar], int]:
    if not cfg["attitude"].get("outlier_reject_enable", False):
        return matched_stars, 0
    if len(matched_stars) <= cfg["attitude"]["min_stars_operational"]:
        return matched_stars, 0

    residuals = compute_residuals(c_ib, matched_stars)
    threshold = float(cfg["attitude"].get("outlier_max_residual_arcsec", np.inf))
    kept = [m for m, residual in zip(matched_stars, residuals) if residual <= threshold]
    rejected = len(matched_stars) - len(kept)
    if len(kept) < cfg["attitude"]["min_stars_mathematical"]:
        return matched_stars, 0
    return kept, rejected


def solve_attitude(
    solve_input: AttitudeSolveInput | list[MatchedStar],
    cfg: dict,
) -> AttitudeSolution:
    matched_stars, mode = _normalize_input(solve_input, cfg)
    min_stars = cfg["attitude"]["min_stars_mathematical"]
    
    if len(matched_stars) < min_stars:
        return AttitudeSolution(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            c_ib=np.eye(3),
            euler_zyx=None,
            valid=False,
            mode="lost",
            num_matched=len(matched_stars),
            residual_rms_arcsec=np.inf,
            residual_max_arcsec=np.inf,
            quality={"reason": "not_enough_stars", "num_input": len(matched_stars)},
            num_rejected=0,
            quality_flag="LOST",
            degraded_level="LOST",
            active_detector_ids=sorted({_normalize_detector_id(m.detector_id) for m in matched_stars}, key=str),
            solver_iterations=0,
        )

    (
        q_ib,
        c_ib,
        matched_used,
        num_rejected,
        robust_rejection_meta,
        solver_iterations,
    ) = _solve_with_robust_rejection(matched_stars, cfg)

    if matched_used:
        residuals = compute_residuals(c_ib, matched_used)
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        rmax = float(np.max(residuals))
    else:
        rms = np.inf
        rmax = np.inf

    residual_gate = float(cfg["attitude"].get("outlier_max_residual_arcsec", np.inf))
    active_detector_ids = _active_detector_ids(matched_used)
    num_active_detectors = len(active_detector_ids)
    min_active_detectors_valid_cfg = cfg["attitude"].get("min_active_detectors_valid")
    min_active_detectors_valid = None
    active_detector_support_ok = True
    if min_active_detectors_valid_cfg is not None:
        min_active_detectors_valid = int(min_active_detectors_valid_cfg)
        active_detector_support_ok = num_active_detectors >= min_active_detectors_valid
    degraded_level = "LOST"
    if num_active_detectors >= 4:
        degraded_level = "NORMAL_4D"
    elif num_active_detectors == 3:
        degraded_level = "DEGRADED_3D"
    elif num_active_detectors == 2:
        degraded_level = "DEGRADED_2D"
    elif num_active_detectors == 1:
        degraded_level = "DEGRADED_1D"

    valid = (
        len(matched_used) >= cfg["attitude"]["min_stars_operational"]
        and rms <= residual_gate
        and robust_rejection_meta.get("robust_valid", True)
        and active_detector_support_ok
    )
    degraded = not valid and len(matched_used) >= min_stars
    quality_flag = "VALID" if valid else ("DEGRADED" if degraded else "LOST")
    weights_used = compute_weights(matched_used, cfg)
    (
        covariance_rad2,
        sigma_non_roll_arcsec,
        sigma_roll_arcsec,
        attitude_condition_number,
        covariance_meta,
    ) = _attitude_covariance_from_sigmas(matched_used, cfg)
    quality = AttitudeQuality(
        num_input=len(matched_stars),
        num_used=len(matched_used),
        num_rejected=num_rejected,
        residual_rms_arcsec=rms,
        residual_max_arcsec=rmax,
        degraded=degraded,
        mode=mode,
        meta={
            "active_detector_ids": active_detector_ids,
            "quality_flag": quality_flag,
            "weight_mode": cfg.get("attitude", {}).get("weight_mode", "variance_snr_hybrid"),
            "min_active_detectors_valid": min_active_detectors_valid,
            "active_detector_support_ok": active_detector_support_ok,
            "effective_weights": [float(weight) for weight in weights_used],
            "sigma_angle_arcsec": [
                m.flags.get("sigma_angle_arcsec") for m in matched_used
            ],
            "weight_sources": [m.flags.get("weight_source") for m in matched_used],
            "attitude_covariance": covariance_meta,
            "robust_rejection": robust_rejection_meta,
        },
    )
    quality_payload = {
        "num_input": quality.num_input,
        "num_used": quality.num_used,
        "num_rejected": quality.num_rejected,
        "degraded": quality.degraded,
        "mode": quality.mode,
        "residual_gate_arcsec": residual_gate,
        "meta": quality.meta,
        "covariance_rad2": (
            covariance_rad2.tolist() if covariance_rad2 is not None else None
        ),
        "sigma_non_roll_arcsec": sigma_non_roll_arcsec,
        "sigma_roll_arcsec": sigma_roll_arcsec,
        "attitude_condition_number": attitude_condition_number,
    }

    return AttitudeSolution(
        q_ib=q_ib,
        c_ib=c_ib,
        euler_zyx=None, # TBD Euler conversion if needed for debug printing
        valid=valid,
        mode="degraded" if degraded else mode,
        num_matched=len(matched_used),
        residual_rms_arcsec=rms,
        residual_max_arcsec=rmax,
        quality=quality_payload,
        num_rejected=num_rejected,
        quality_flag=quality_flag,
        degraded_level=degraded_level,
        active_detector_ids=active_detector_ids,
        solver_iterations=solver_iterations,
        covariance_rad2=covariance_rad2,
        sigma_non_roll_arcsec=sigma_non_roll_arcsec,
        sigma_roll_arcsec=sigma_roll_arcsec,
        attitude_condition_number=attitude_condition_number,
    )
