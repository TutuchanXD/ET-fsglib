import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from fsglib.common.types import RawFrame, PreprocessedFrame
from fsglib.preprocess.pipeline import preprocess_frame

def test_preprocess_frame():
    # Simple dummy input
    image = np.ones((10, 10))
    image[5, 5] = 100.0
    
    raw = RawFrame(detector_id=0, image=image, time_s=0.0)
    cfg = {
        "preprocess": {
            "enable_background_subtraction": True,
            "background_method": "median",
        }
    }
    
    pre = preprocess_frame(raw, calib={}, cfg=cfg)
    
    assert isinstance(pre, PreprocessedFrame)
    assert np.allclose(pre.background, 1.0)
    assert pre.image[5, 5] == 99.0
    
def test_svd_solver():
    from fsglib.attitude.solver import solve_attitude
    from fsglib.common.types import MatchedStar
    
    # Simple exact match (identity rotation)
    stars = [
        MatchedStar(0, 0, 0, np.array([1, 0, 0]), np.array([1, 0, 0])),
        MatchedStar(0, 1, 1, np.array([0, 1, 0]), np.array([0, 1, 0])),
        MatchedStar(0, 2, 2, np.array([0, 0, 1]), np.array([0, 0, 1]))
    ]
    
    cfg = {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": 3
        }
    }
    
    sol = solve_attitude(stars, cfg)
    
    assert sol.valid
    # Identity quaternion [1, 0, 0, 0]
    assert np.allclose(sol.q_ib, [1.0, 0.0, 0.0, 0.0])
    assert sol.residual_rms_arcsec < 1e-2

def test_solver_recovers_nontrivial_rotation():
    from fsglib.attitude.solver import quat_to_dcm, solve_attitude
    from fsglib.common.types import MatchedStar

    c_ib = Rotation.from_euler("zyx", [20.0, -10.0, 5.0], degrees=True).as_matrix()
    inertial = [
        np.array([1.0, 0.0, 0.2]),
        np.array([0.1, 0.9, 0.3]),
        np.array([-0.2, 0.1, 1.0]),
        np.array([0.3, -0.4, 0.8]),
    ]

    stars = []
    for idx, vec in enumerate(inertial):
        vec = vec / np.linalg.norm(vec)
        stars.append(MatchedStar(idx, idx, idx, c_ib @ vec, vec))

    cfg = {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": 4,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 1.0e9,
        },
        "project": {"mode": "init"},
    }

    sol = solve_attitude(stars, cfg)
    assert sol.residual_rms_arcsec < 1e-2
    assert np.allclose(quat_to_dcm(sol.q_ib), c_ib)
