import numpy as np
from fsglib.common.types import StarCandidate, ObservedStar


ARCSEC_PER_RAD = 206264.80624709636


def _cfg_float(cfg: dict, section: str, key: str, default: float) -> float:
    value = cfg.get(section, {}).get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result) or result <= 0.0:
        return default
    return result


def _pixel_to_los(projector, detector_id, x: float, y: float) -> np.ndarray | None:
    if hasattr(projector, "pixel_to_los_body"):
        los = projector.pixel_to_los_body(detector_id, x, y)
    elif hasattr(projector, "pixel_to_body_los"):
        los = projector.pixel_to_body_los(detector_id, x, y)
    else:
        return None
    if los is None:
        return None
    los = np.asarray(los, dtype=np.float64)
    norm = float(np.linalg.norm(los))
    if los.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
        return None
    return los / norm


def _los_jacobian_numeric(
    projector,
    detector_id,
    x: float,
    y: float,
    step_pix: float,
) -> np.ndarray | None:
    plus_x = _pixel_to_los(projector, detector_id, x + step_pix, y)
    minus_x = _pixel_to_los(projector, detector_id, x - step_pix, y)
    plus_y = _pixel_to_los(projector, detector_id, x, y + step_pix)
    minus_y = _pixel_to_los(projector, detector_id, x, y - step_pix)
    if plus_x is None or minus_x is None or plus_y is None or minus_y is None:
        return None
    return np.column_stack(
        [
            (plus_x - minus_x) / (2.0 * step_pix),
            (plus_y - minus_y) / (2.0 * step_pix),
        ]
    )


def propagate_centroid_covariance(
    projector,
    detector_id,
    x: float,
    y: float,
    centroid_cov_pix: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray | None, float | None]:
    if centroid_cov_pix is None:
        return None, None
    cov_pix = np.asarray(centroid_cov_pix, dtype=np.float64)
    if cov_pix.shape != (2, 2) or not np.all(np.isfinite(cov_pix)):
        return None, None
    step_pix = _cfg_float(
        cfg.get("extract", {}),
        "centroid_covariance",
        "jacobian_step_pix",
        0.01,
    )
    jacobian = _los_jacobian_numeric(projector, detector_id, float(x), float(y), step_pix)
    if jacobian is None:
        return None, None
    los_cov = jacobian @ cov_pix @ jacobian.T
    los_cov = 0.5 * (los_cov + los_cov.T)
    if not np.all(np.isfinite(los_cov)):
        return None, None
    los = _pixel_to_los(projector, detector_id, float(x), float(y))
    if los is not None:
        tangent_projector = np.eye(3, dtype=np.float64) - np.outer(los, los)
        los_cov = tangent_projector @ los_cov @ tangent_projector.T
    tangent_trace = max(float(np.trace(los_cov)), 0.0)
    sigma_angle_arcsec = float(np.sqrt(0.5 * tangent_trace) * ARCSEC_PER_RAD)
    return los_cov, sigma_angle_arcsec


def observed_weight_from_sigma(
    snr: float,
    sigma_angle_arcsec: float | None,
    cfg: dict,
) -> tuple[float, dict]:
    mode = str(cfg.get("attitude", {}).get("weight_mode", "variance_snr_hybrid"))
    snr_weight = max(float(snr), 1.0)
    covariance_weight = None
    if sigma_angle_arcsec is not None and np.isfinite(sigma_angle_arcsec) and sigma_angle_arcsec > 0.0:
        covariance_weight = 1.0 / max(float(sigma_angle_arcsec) ** 2, 1e-12)

    if mode == "centroid_variance" and covariance_weight is not None:
        return covariance_weight, {
            "weight_mode": mode,
            "weight_source": "centroid_variance",
            "covariance_weight": covariance_weight,
        }
    if mode == "variance_snr_hybrid" and covariance_weight is not None:
        weight = snr_weight / max(float(sigma_angle_arcsec), 1e-6)
        return weight, {
            "weight_mode": mode,
            "weight_source": "variance_snr_hybrid",
            "covariance_weight": covariance_weight,
        }
    return snr_weight, {
        "weight_mode": mode,
        "weight_source": "snr",
        "covariance_weight": covariance_weight,
    }


def candidates_to_observed(candidates: list[StarCandidate], projector, cfg: dict) -> list[ObservedStar]:
    """
    Convert image plane StarCandidates into 3D ObservedStars using the projector / camera model.
    """
    observed = []
    
    for cand in candidates:
        los_body = projector.pixel_to_los_body(cand.detector_id, cand.x, cand.y)
        los_cov_body, sigma_angle_arcsec = propagate_centroid_covariance(
            projector,
            cand.detector_id,
            cand.x,
            cand.y,
            cand.centroid_cov_pix,
            cfg,
        )
        weight, weight_flags = observed_weight_from_sigma(
            cand.snr,
            sigma_angle_arcsec,
            cfg,
        )
        flags = {
            **cand.flags,
            **weight_flags,
            "sigma_angle_arcsec": sigma_angle_arcsec,
        }
        
        observed.append(ObservedStar(
            detector_id=cand.detector_id,
            source_id=cand.source_id,
            x=cand.x,
            y=cand.y,
            los_body=los_body,
            flux=cand.flux,
            snr=cand.snr,
            weight=weight,
            centroid_cov_pix=cand.centroid_cov_pix,
            los_cov_body=los_cov_body,
            sigma_angle_arcsec=sigma_angle_arcsec,
            flags=flags
        ))
        
    return observed
