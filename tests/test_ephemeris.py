import numpy as np

from fsglib.common.coords import radec_to_unit_vector
from fsglib.ephemeris.pipeline import build_reference_stars
from fsglib.ephemeris.types import CatalogStar, EphemerisContext


class FakeCatalogProvider:
    def __init__(self):
        self.last_branch = None

    def query_region(self, boresight_vec, radius_deg, mag_limit=None):
        self.last_branch = "init"
        return [
            CatalogStar(1, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0),
            CatalogStar(2, 10.1, 20.1, 0.0, 0.0, 0.0, 0.0, 11.2, 0.0),
        ]

    def query_tracking_targets(self, ctx):
        self.last_branch = "tracking"
        catalog_id = ctx.track_catalog_ids[0]
        return [CatalogStar(catalog_id, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0)]


class FakeProjector:
    def project_to_detectors(self, los_inertial, attitude_q):
        return {0: (100.0, 200.0)}, {0: True}, [0]


def test_build_reference_stars_uses_init_branch():
    ctx = EphemerisContext(
        mode="init",
        time_s=0.0,
        prior_attitude_q=np.array([1.0, 0.0, 0.0, 0.0]),
        boresight_inertial=radec_to_unit_vector(10.0, 20.0),
        angular_rate_body=None,
        detector_model={},
        optical_model={},
        catalog_cfg={"mag_limit": 15.0},
        correction_cfg={},
    )

    provider = FakeCatalogProvider()
    refs = build_reference_stars(ctx, provider, FakeProjector(), {"match": {"init_max_catalog_radius_deg": 1.5}, "ephemeris": {"mag_limit": 15.0}})

    assert provider.last_branch == "init"
    assert len(refs) == 2
    assert refs[0].predicted_xy[0] == (100.0, 200.0)


def test_build_reference_stars_uses_tracking_branch():
    ctx = EphemerisContext(
        mode="tracking",
        time_s=0.0,
        prior_attitude_q=np.array([1.0, 0.0, 0.0, 0.0]),
        boresight_inertial=radec_to_unit_vector(10.0, 20.0),
        angular_rate_body=None,
        detector_model={},
        optical_model={},
        catalog_cfg={"mag_limit": 15.0},
        correction_cfg={},
        track_catalog_ids=[42],
    )

    provider = FakeCatalogProvider()
    refs = build_reference_stars(ctx, provider, FakeProjector(), {"match": {"init_max_catalog_radius_deg": 1.5}, "ephemeris": {"mag_limit": 15.0}})

    assert provider.last_branch == "tracking"
    assert len(refs) == 1
    assert refs[0].catalog_id == 42


def test_build_reference_stars_sim_rect_topk_selects_brightest_kepler(tmp_path):
    poly_path = tmp_path / "gaia2kp.npy"
    np.save(poly_path, np.array([1.0, 0.0], dtype=np.float64))

    ctx = EphemerisContext(
        mode="init",
        time_s=0.0,
        prior_attitude_q=np.array([1.0, 0.0, 0.0, 0.0]),
        boresight_inertial=radec_to_unit_vector(10.0, 20.0),
        angular_rate_body=None,
        detector_model={},
        optical_model={},
        catalog_cfg={"mag_limit": 15.0},
        correction_cfg={},
    )

    provider = FakeCatalogProvider()
    refs = build_reference_stars(
        ctx,
        provider,
        FakeProjector(),
        {
            "match": {"init_max_catalog_radius_deg": 1.5},
            "ephemeris": {
                "mag_limit": 15.0,
                "reference_selection_mode": "sim_rect_topk",
                "reference_topk": 1,
                "gaia_to_kp_poly_path": str(poly_path),
            },
        },
    )

    assert len(refs) == 1
    assert refs[0].catalog_id == 1
    assert np.isclose(refs[0].meta["mag_kp"], 11.0)
