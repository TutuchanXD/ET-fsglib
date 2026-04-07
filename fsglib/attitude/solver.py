import numpy as np
from scipy.spatial.transform import Rotation
from fsglib.common.types import AttitudeQuality, AttitudeSolution, AttitudeSolveInput, MatchedStar


def _normalize_detector_id(value) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)

def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """
    Convert an inertial-to-body DCM to a unit quaternion in scalar-first form [w, x, y, z].
    """
    rot = Rotation.from_matrix(dcm)
    q_xyzw = rot.as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert a scalar-first quaternion [w, x, y, z] to an inertial-to-body DCM.
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rot = Rotation.from_quat([x, y, z, w])
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

    q_ib = solve_quest(matched_stars, cfg)
    c_ib = quat_to_dcm(q_ib)
    matched_used, num_rejected = reject_outliers(matched_stars, c_ib, cfg)
    if num_rejected > 0:
        q_ib = solve_quest(matched_used, cfg)
        c_ib = quat_to_dcm(q_ib)
    else:
        matched_used = matched_stars

    residuals = compute_residuals(c_ib, matched_used)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    rmax = float(np.max(residuals))

    residual_gate = float(cfg["attitude"].get("outlier_max_residual_arcsec", np.inf))
    active_detector_ids = sorted({_normalize_detector_id(m.detector_id) for m in matched_used}, key=str)
    num_active_detectors = len(active_detector_ids)
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
    )
    degraded = not valid and len(matched_used) >= min_stars
    quality_flag = "VALID" if valid else ("DEGRADED" if degraded else "LOST")
    quality = AttitudeQuality(
        num_input=len(matched_stars),
        num_used=len(matched_used),
        num_rejected=num_rejected,
        residual_rms_arcsec=rms,
        residual_max_arcsec=rmax,
        degraded=degraded,
        mode=mode,
        meta={"active_detector_ids": active_detector_ids, "quality_flag": quality_flag},
    )

    return AttitudeSolution(
        q_ib=q_ib,
        c_ib=c_ib,
        euler_zyx=None, # TBD Euler conversion if needed for debug printing
        valid=valid,
        mode="degraded" if degraded else mode,
        num_matched=len(matched_used),
        residual_rms_arcsec=rms,
        residual_max_arcsec=rmax,
        quality={
            "num_input": quality.num_input,
            "num_used": quality.num_used,
            "num_rejected": quality.num_rejected,
            "degraded": quality.degraded,
            "mode": quality.mode,
            "residual_gate_arcsec": residual_gate,
        },
        num_rejected=num_rejected,
        quality_flag=quality_flag,
        degraded_level=degraded_level,
        active_detector_ids=active_detector_ids,
        solver_iterations=2 if num_rejected > 0 else 1,
    )
