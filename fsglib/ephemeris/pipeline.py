from functools import lru_cache
import warnings

import numpy as np

from fsglib.common.coords import radec_to_unit_vector
from fsglib.ephemeris.types import CatalogStar, ReferenceStar


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


def _finite_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def magnitude_to_flux_weight(magnitude: float | None) -> float:
    magnitude = _finite_float(magnitude)
    if magnitude is None:
        return 1.0
    return float(10.0 ** (-0.4 * magnitude))


def reference_weight_from_magnitudes(
    *,
    mag_g: float | None,
    mag_kp: float | None,
) -> tuple[float, dict]:
    mag_kp = _finite_float(mag_kp)
    if mag_kp is not None:
        weight = magnitude_to_flux_weight(mag_kp)
        return weight, {
            "weight_source": "kepler",
            "weight_magnitude": mag_kp,
            "flux_weight": weight,
        }

    mag_g = _finite_float(mag_g)
    if mag_g is not None:
        weight = magnitude_to_flux_weight(mag_g)
        return weight, {
            "weight_source": "gaia_g",
            "weight_magnitude": mag_g,
            "flux_weight": weight,
        }

    return 1.0, {
        "weight_source": "default",
        "weight_magnitude": None,
        "flux_weight": 1.0,
    }


def _target_epoch_from_cfg(cfg: dict, target_epoch: float | None = None) -> float:
    if target_epoch is not None:
        return float(target_epoch)
    return float(cfg.get("ephemeris", {}).get("target_epoch", 2000.0))


def _reference_epoch_for_star(star: CatalogStar, cfg: dict) -> float:
    ref_epoch = _finite_float(getattr(star, "ref_epoch", None))
    if ref_epoch is not None:
        return ref_epoch
    return float(cfg.get("ephemeris", {}).get("reference_epoch_default", 2016.0))


def _has_motion_terms(star: CatalogStar) -> bool:
    values = (
        _finite_float(star.pm_ra_mas_per_yr),
        _finite_float(star.pm_dec_mas_per_yr),
        _finite_float(star.rv_km_s),
    )
    return any(value is not None and abs(value) > 0.0 for value in values)


def _catalog_star_astrometry(
    star: CatalogStar,
    cfg: dict,
    target_epoch: float | None = None,
) -> tuple[np.ndarray, dict]:
    eph_cfg = cfg.get("ephemeris", {})
    target_epoch_value = _target_epoch_from_cfg(cfg, target_epoch)
    ref_epoch = _reference_epoch_for_star(star, cfg)
    original_ra = float(star.ra_deg)
    original_dec = float(star.dec_deg)
    propagated_ra = original_ra
    propagated_dec = original_dec
    astrometry_applied = False

    enable_proper_motion = bool(eph_cfg.get("enable_proper_motion", False))
    if enable_proper_motion and _has_motion_terms(star):
        try:
            from astropy import units as astropy_units
            from astropy.coordinates import Distance, SkyCoord
            from astropy.time import Time
        except ImportError as exc:
            raise ImportError(
                "Astropy is required for ephemeris.enable_proper_motion but is not "
                "installed. Install it with: pip install astropy"
            ) from exc
        kwargs = {
            "ra": original_ra * astropy_units.deg,
            "dec": original_dec * astropy_units.deg,
            "pm_ra_cosdec": (
                float(_finite_float(star.pm_ra_mas_per_yr) or 0.0)
                * astropy_units.mas
                / astropy_units.yr
            ),
            "pm_dec": (
                float(_finite_float(star.pm_dec_mas_per_yr) or 0.0)
                * astropy_units.mas
                / astropy_units.yr
            ),
            "obstime": Time(ref_epoch, format="jyear"),
        }
        parallax = _finite_float(star.parallax_mas)
        if parallax is not None and parallax > 0.0:
            kwargs["distance"] = Distance(parallax=parallax * astropy_units.mas)
        rv_km_s = _finite_float(star.rv_km_s)
        if rv_km_s is not None:
            kwargs["radial_velocity"] = rv_km_s * astropy_units.km / astropy_units.s

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            moved = SkyCoord(**kwargs).apply_space_motion(
                Time(target_epoch_value, format="jyear")
            )
        propagated_ra = float(moved.ra.deg)
        propagated_dec = float(moved.dec.deg)
        astrometry_applied = True

    meta = {
        "ra_deg": propagated_ra,
        "dec_deg": propagated_dec,
        "original_ra_deg": original_ra,
        "original_dec_deg": original_dec,
        "propagated_ra_deg": propagated_ra,
        "propagated_dec_deg": propagated_dec,
        "ref_epoch": ref_epoch,
        "target_epoch": target_epoch_value,
        "astrometry_applied": astrometry_applied,
        "proper_motion_enabled": enable_proper_motion,
        "pm_ra_mas_per_yr": _finite_float(star.pm_ra_mas_per_yr),
        "pm_dec_mas_per_yr": _finite_float(star.pm_dec_mas_per_yr),
        "parallax_mas": _finite_float(star.parallax_mas),
        "rv_km_s": _finite_float(star.rv_km_s),
    }
    return radec_to_unit_vector(propagated_ra, propagated_dec), meta


def catalog_star_to_unit_vector(
    star: CatalogStar,
    cfg: dict,
    target_epoch: float | None = None,
) -> np.ndarray:
    los_inertial, _ = _catalog_star_astrometry(star, cfg, target_epoch)
    return los_inertial


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
        los_inertial, astrometry_meta = _catalog_star_astrometry(star, cfg)
        predicted_xy, predicted_valid, visible_det_ids = projector.project_to_detectors(
            los_inertial=los_inertial,
            attitude_q=attitude_q,
        )
        mag_kp = _gaia_to_kepler_mag(star.mag_g, cfg)
        weight_hint, weight_meta = reference_weight_from_magnitudes(
            mag_g=star.mag_g,
            mag_kp=mag_kp,
        )

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
                    weight_hint=weight_hint,
                    meta={
                        **astrometry_meta,
                        "mag_kp": mag_kp,
                        **weight_meta,
                    },
                )
            )
    return _apply_reference_selection(ref_stars, cfg, ctx.mode)
