import numpy as np
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from fsglib.common.coords import radec_to_unit_vector
from fsglib.ephemeris.catalog import HealpixCatalogProvider
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


class SinglePixelHealpix:
    def cone_search_skycoord(self, center, radius):
        return [0]


class SingleStarCatalogProvider:
    def __init__(self, star: CatalogStar):
        self.star = star
        self.last_branch = None

    def query_region(self, boresight_vec, radius_deg, mag_limit=None):
        self.last_branch = "init"
        return [self.star]

    def query_tracking_targets(self, ctx):
        self.last_branch = "tracking"
        return [self.star]


def _expected_propagated_vector(star: CatalogStar, target_epoch: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord = SkyCoord(
            ra=float(star.ra_deg) * u.deg,
            dec=float(star.dec_deg) * u.deg,
            pm_ra_cosdec=float(star.pm_ra_mas_per_yr) * u.mas / u.yr,
            pm_dec=float(star.pm_dec_mas_per_yr) * u.mas / u.yr,
            obstime=Time(float(star.ref_epoch), format="jyear"),
        )
        moved = coord.apply_space_motion(Time(float(target_epoch), format="jyear"))
    return radec_to_unit_vector(float(moved.ra.deg), float(moved.dec.deg))


def test_healpix_catalog_provider_preserves_ref_epoch_and_missing_rv(tmp_path):
    partition = tmp_path / "healpix_n05_nested_00000.csv"
    partition.write_text(
        "\n".join(
            [
                "source_id,ra,dec,g_mean_mag,bp_mean_mag,rp_mean_mag,pmra,pmdec,ref_epoch,parallax",
                "101,10.0,20.0,11.0,11.2,10.7,123.0,-45.0,2015.5,7.0",
            ]
        ),
        encoding="utf-8",
    )
    provider = HealpixCatalogProvider(
        {"ephemeris": {"gaia_root_dir": str(tmp_path), "mag_limit": 15.0}}
    )
    provider.hp = SinglePixelHealpix()

    stars = provider.query_region(radec_to_unit_vector(10.0, 20.0), radius_deg=1.0)

    assert len(stars) == 1
    assert stars[0].ref_epoch == 2015.5
    assert stars[0].rv_km_s is None


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
    assert np.isclose(refs[0].weight_hint, 10.0 ** (-0.4 * 11.0))
    assert refs[0].meta["weight_source"] == "kepler"


def test_build_reference_stars_applies_proper_motion_to_target_epoch():
    star = CatalogStar(101, 10.0, 20.0, 100.0, 50.0, None, None, 11.0, 0.0)
    star.ref_epoch = 2016.0
    target_epoch = 2026.0
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

    refs = build_reference_stars(
        ctx,
        SingleStarCatalogProvider(star),
        FakeProjector(),
        {
            "match": {"init_max_catalog_radius_deg": 1.5},
            "ephemeris": {
                "mag_limit": 15.0,
                "enable_proper_motion": True,
                "target_epoch": target_epoch,
            },
        },
    )

    expected = _expected_propagated_vector(star, target_epoch)
    static = radec_to_unit_vector(star.ra_deg, star.dec_deg)
    assert np.allclose(refs[0].los_inertial, expected, atol=1.0e-12)
    assert not np.allclose(refs[0].los_inertial, static, atol=1.0e-12)
    assert refs[0].meta["astrometry_applied"] is True
    assert refs[0].meta["original_ra_deg"] == 10.0
    assert refs[0].meta["original_dec_deg"] == 20.0
    assert np.isclose(refs[0].meta["propagated_ra_deg"], 10.000295605197604)
    assert np.isclose(refs[0].meta["propagated_dec_deg"], 20.000138888643804)
    assert refs[0].meta["ref_epoch"] == 2016.0
    assert refs[0].meta["target_epoch"] == target_epoch


def test_build_reference_stars_can_disable_proper_motion():
    star = CatalogStar(101, 10.0, 20.0, 100.0, 50.0, None, None, 11.0, 0.0)
    star.ref_epoch = 2016.0
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

    refs = build_reference_stars(
        ctx,
        SingleStarCatalogProvider(star),
        FakeProjector(),
        {
            "match": {"init_max_catalog_radius_deg": 1.5},
            "ephemeris": {
                "mag_limit": 15.0,
                "enable_proper_motion": False,
                "target_epoch": 2026.0,
            },
        },
    )

    assert np.allclose(refs[0].los_inertial, radec_to_unit_vector(10.0, 20.0))
    assert refs[0].meta["astrometry_applied"] is False
    assert refs[0].meta["target_epoch"] == 2026.0
