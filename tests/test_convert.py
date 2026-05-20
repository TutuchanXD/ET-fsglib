import numpy as np

from fsglib.common.types import StarCandidate
from fsglib.pipeline.convert import candidates_to_observed


class SmallAngleProjector:
    def __init__(self, arcsec_per_pix: float = 1.0):
        self.scale_rad = np.deg2rad(arcsec_per_pix / 3600.0)

    def pixel_to_los_body(self, detector_id, x, y):
        vec = np.array([x * self.scale_rad, y * self.scale_rad, 1.0], dtype=np.float64)
        return vec / np.linalg.norm(vec)


def test_candidates_to_observed_propagates_centroid_covariance_to_angle_sigma():
    candidate = StarCandidate(
        detector_id=0,
        source_id=1,
        x=0.0,
        y=0.0,
        flux=100.0,
        peak=20.0,
        area=5,
        snr=10.0,
        bbox=(0, 0, 2, 2),
        centroid_cov_pix=np.eye(2, dtype=np.float64) * 0.25,
    )
    cfg = {
        "extract": {"centroid_covariance": {"jacobian_step_pix": 0.01}},
        "attitude": {"weight_mode": "snr"},
    }

    observed = candidates_to_observed([candidate], SmallAngleProjector(), cfg)

    assert len(observed) == 1
    assert observed[0].centroid_cov_pix.shape == (2, 2)
    assert observed[0].los_cov_body.shape == (3, 3)
    assert np.isclose(observed[0].sigma_angle_arcsec, 0.5, rtol=1e-4)
    assert observed[0].weight == 10.0
    assert observed[0].flags["weight_source"] == "snr"
    assert np.isclose(observed[0].flags["sigma_angle_arcsec"], 0.5, rtol=1e-4)


def test_candidates_to_observed_can_use_centroid_variance_weight_mode():
    candidate = StarCandidate(
        detector_id=0,
        source_id=1,
        x=0.0,
        y=0.0,
        flux=100.0,
        peak=20.0,
        area=5,
        snr=10.0,
        bbox=(0, 0, 2, 2),
        centroid_cov_pix=np.eye(2, dtype=np.float64) * 0.25,
    )
    cfg = {
        "extract": {"centroid_covariance": {"jacobian_step_pix": 0.01}},
        "attitude": {"weight_mode": "centroid_variance"},
    }

    observed = candidates_to_observed([candidate], SmallAngleProjector(), cfg)

    assert np.isclose(observed[0].weight, 4.0, rtol=1e-4)
    assert observed[0].flags["weight_source"] == "centroid_variance"


def test_candidates_to_observed_defaults_to_variance_snr_hybrid_weight_mode():
    candidate = StarCandidate(
        detector_id=0,
        source_id=1,
        x=0.0,
        y=0.0,
        flux=100.0,
        peak=20.0,
        area=5,
        snr=10.0,
        bbox=(0, 0, 2, 2),
        centroid_cov_pix=np.eye(2, dtype=np.float64) * 0.25,
    )
    cfg = {"extract": {"centroid_covariance": {"jacobian_step_pix": 0.01}}}

    observed = candidates_to_observed([candidate], SmallAngleProjector(), cfg)

    assert np.isclose(observed[0].weight, 20.0, rtol=1e-4)
    assert observed[0].flags["weight_mode"] == "variance_snr_hybrid"
    assert observed[0].flags["weight_source"] == "variance_snr_hybrid"
