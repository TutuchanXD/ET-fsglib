import numpy as np

from fsglib.common.types import AttitudeSolution, MatchingContext, ObservedStar
from fsglib.ephemeris.types import ReferenceStar
from fsglib.match.pipeline import associate_nearest, match_stars, validate_match_hypothesis


def _cfg():
    return {
        "match": {
            "validate_max_residual_pix": 5.0,
            "validate_min_support": 1,
            "algorithm": "local_triangle",
        },
        "attitude": {"outlier_max_residual_arcsec": 30.0},
        "tracking": {"max_attitude_jump_arcsec": 100.0},
    }


def test_associate_nearest_prefers_closest_reference():
    observed = [
        ObservedStar(0, 1, 10.2, 11.1, np.array([0.0, 0.0, 1.0]), flux=100.0, snr=20.0),
    ]
    reference = [
        ReferenceStar(100, 0.0, np.array([0.0, 0.0, 1.0]), 10.5, [0], {0: (10.0, 11.0)}, {0: True}, 1.0),
        ReferenceStar(200, 0.0, np.array([0.0, 0.0, 1.0]), 10.7, [0], {0: (20.0, 21.0)}, {0: True}, 1.0),
    ]

    result = associate_nearest(observed, reference, _cfg())

    assert result.success
    assert len(result.matched) == 1
    assert result.matched[0].catalog_id == 100
    assert result.debug["mean_residual_pix"] < 1.0
    assert result.debug["stars_per_detector"]["0"] == 1


def test_validate_match_hypothesis_rejects_large_attitude_jump():
    matching = associate_nearest(
        [ObservedStar(0, 1, 10.0, 11.0, np.array([0.0, 0.0, 1.0]), flux=50.0, snr=10.0)],
        [ReferenceStar(100, 0.0, np.array([0.0, 0.0, 1.0]), 10.5, [0], {0: (10.0, 11.0)}, {0: True}, 1.0)],
        _cfg(),
    )
    solution = AttitudeSolution(
        q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
        c_ib=np.eye(3),
        euler_zyx=None,
        valid=True,
        mode="tracking",
        num_matched=1,
        residual_rms_arcsec=1.0,
        residual_max_arcsec=1.0,
    )

    ok, debug = validate_match_hypothesis(matching, solution, _cfg(), attitude_delta_arcsec=200.0)
    assert not ok
    assert debug["reason"] == "attitude_jump"


def test_match_stars_uses_context_mode():
    observed = [ObservedStar(0, 1, 10.0, 11.0, np.array([0.0, 0.0, 1.0]), flux=50.0, snr=10.0)]
    reference = [ReferenceStar(100, 0.0, np.array([0.0, 0.0, 1.0]), 10.5, [0], {0: (10.0, 11.0)}, {0: True}, 1.0)]
    ctx = MatchingContext(
        mode="tracking",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=_cfg()["match"],
        reference_stars=reference,
    )

    result = match_stars(ctx, reference, _cfg())
    assert result.mode == "tracking"
    assert result.success
