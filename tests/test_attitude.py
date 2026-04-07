import numpy as np

from fsglib.attitude.solver import solve_attitude
from fsglib.common.types import AttitudeSolveInput, MatchedStar


def test_attitude_solution_exposes_quality_fields():
    stars = [
        MatchedStar(0, 0, 0, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), weight=3.0),
        MatchedStar(1, 1, 1, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), weight=2.0),
        MatchedStar(2, 2, 2, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), weight=4.0),
        MatchedStar(3, 3, 3, np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), weight=5.0),
    ]
    cfg = {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": 4,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 30.0,
        },
        "project": {"mode": "init"},
    }

    sol = solve_attitude(AttitudeSolveInput(0.0, stars, None, "tracking", cfg["attitude"]), cfg)
    assert sol.valid
    assert sol.quality_flag == "VALID"
    assert sol.degraded_level == "NORMAL_4D"
    assert sol.active_detector_ids == [0, 1, 2, 3]
    assert sol.solver_iterations == 1
    assert sol.num_rejected == 0


def test_attitude_solution_reports_lost_when_underconstrained():
    stars = [MatchedStar(0, 0, 0, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]
    cfg = {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": 4,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 30.0,
        },
        "project": {"mode": "init"},
    }

    sol = solve_attitude(stars, cfg)
    assert not sol.valid
    assert sol.quality_flag == "LOST"
    assert sol.degraded_level == "LOST"
