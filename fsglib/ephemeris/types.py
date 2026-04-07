from dataclasses import dataclass, field
import numpy as np

@dataclass
class CatalogStar:
    catalog_id: int
    ra_deg: float
    dec_deg: float
    pm_ra_mas_per_yr: float | None
    pm_dec_mas_per_yr: float | None
    parallax_mas: float | None
    rv_km_s: float | None
    mag_g: float | None
    color_bp_rp: float | None
    meta: dict = field(default_factory=dict)

@dataclass
class ReferenceStar:
    catalog_id: int
    time_s: float
    los_inertial: np.ndarray      # (3,)
    mag_g: float | None
    detector_ids_visible: list[int]
    predicted_xy: dict[int, tuple[float, float]]
    predicted_valid: dict[int, bool]
    weight_hint: float
    meta: dict = field(default_factory=dict)

@dataclass
class EphemerisContext:
    mode: str                     # init | tracking
    time_s: float
    prior_attitude_q: np.ndarray | None
    boresight_inertial: np.ndarray | None
    angular_rate_body: np.ndarray | None
    detector_model: dict
    optical_model: dict
    catalog_cfg: dict
    correction_cfg: dict
    track_catalog_ids: list[int] = field(default_factory=list)
