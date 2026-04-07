from functools import lru_cache

import numpy as np
from fsglib.common.coords import radec_to_unit_vector
from fsglib.ephemeris.types import ReferenceStar


@lru_cache(maxsize=4)
def _load_gaia_to_kp_coefficients(poly_path: str) -> np.ndarray:
    return np.load(poly_path)


def _gaia_to_kepler_mag(mag_g: float | None, cfg: dict) -> float | None:
    if mag_g is None:
        return None
    poly_path = cfg.get("ephemeris", {}).get("gaia_to_kp_poly_path")
    if not poly_path:
        return None
    try:
        coeffs = _load_gaia_to_kp_coefficients(str(poly_path))
    except OSError:
        return None
    return float(np.polyval(coeffs, float(mag_g)))


def _apply_reference_selection(
    ref_stars: list[ReferenceStar],
    cfg: dict,
    mode: str,
) -> list[ReferenceStar]:
    eph_cfg = cfg.get("ephemeris", {})
    selection_mode = eph_cfg.get("reference_selection_mode", "visible_only")
    if mode != "init":
        return ref_stars
    if selection_mode != "sim_rect_topk":
        return ref_stars

    reference_topk = int(eph_cfg.get("reference_topk", 0) or 0)
    if reference_topk <= 0 or len(ref_stars) <= reference_topk:
        return ref_stars

    def _sort_key(star: ReferenceStar) -> tuple[float, float, int]:
        mag_kp = star.meta.get("mag_kp")
        mag_g = star.mag_g
        kp_key = float(mag_kp) if mag_kp is not None else np.inf
        g_key = float(mag_g) if mag_g is not None else np.inf
        return kp_key, g_key, int(star.catalog_id)

    return sorted(ref_stars, key=_sort_key)[:reference_topk]

def build_reference_stars(ctx, catalog_provider, projector, cfg):
    if ctx.mode == "init":
        catalog_stars = catalog_provider.query_region(
            boresight_vec=ctx.boresight_inertial,
            radius_deg=cfg["match"]["init_max_catalog_radius_deg"],
            mag_limit=cfg["ephemeris"]["mag_limit"],
        )
    else:
        catalog_stars = catalog_provider.query_tracking_targets(ctx)

    attitude_q = ctx.prior_attitude_q
    if attitude_q is None:
        attitude_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    ref_stars = []
    for star in catalog_stars:
        los_inertial = radec_to_unit_vector(star.ra_deg, star.dec_deg)
        predicted_xy, predicted_valid, visible_det_ids = projector.project_to_detectors(
            los_inertial=los_inertial,
            attitude_q=attitude_q,
        )
        mag_kp = _gaia_to_kepler_mag(star.mag_g, cfg)
        
        # Only add to reference list if it's visible on at least one detector,
        # or if we are skipping projection checks for now.
        if visible_det_ids:
            ref_stars.append(
                ReferenceStar(
                    catalog_id=star.catalog_id,
                    time_s=ctx.time_s,
                    los_inertial=los_inertial,
                    mag_g=star.mag_g,
                    detector_ids_visible=visible_det_ids,
                    predicted_xy=predicted_xy,
                    predicted_valid=predicted_valid,
                    weight_hint=1.0,
                    meta={"ra_deg": star.ra_deg, "dec_deg": star.dec_deg, "mag_kp": mag_kp},
                )
            )
    return _apply_reference_selection(ref_stars, cfg, ctx.mode)
