import pytest
import numpy as np

from fsglib.attitude.solver import quat_to_dcm
from fsglib.ephemeris.projector import RealOpticalProjector


def _rotation_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _ideal_pinhole_cfg(mounting_matrix=None) -> dict:
    return {
        "layout": {
            "projection_model": "ideal_pinhole",
            "detectors": [
                {
                    "detector_id": 0,
                    "principal_point_pix": [559.0, 559.0],
                    "resolution": [1119, 1119],
                    "pixel_scale_arcsec_per_pix": 4.83,
                    "mounting_matrix": np.eye(3).tolist() if mounting_matrix is None else mounting_matrix,
                }
            ],
        }
    }


def test_projector_forward_inverse():
    cfg = {
        "layout": {
            "pixel_size_mm": 0.0065,
            "distortion": {
                "a1": 0.134636212,
                "a3": -3.1445e-7,
                "axy2": -8.5532e-8
            },
            "detectors": [
                {
                    "detector_id": 4, # Guide UL
                    "principal_point_pix": [1024.0, 1024.0],
                    "resolution": [2048, 2048],
                    "mounting_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ],
                    "fov_center_mm": [-34.0, 109.565]
                }
            ]
        }
    }
    
    proj = RealOpticalProjector(cfg)
    
    # Test point (corner of Guide UL from user note)
    x_test_mm = -40.66
    y_test_mm = 116.22
    
    # Manually run forward
    u_deg, v_deg = proj._image_to_field_deg(x_test_mm, y_test_mm)
    assert np.isfinite(u_deg)
    assert np.isfinite(v_deg)
    
    # Manually run inverse
    # Need to pass det_id to get the initial guess right
    x_back, y_back = proj._field_deg_to_image(u_deg, v_deg, det_id=4)
    
    assert x_back is not None
    assert abs(x_back - x_test_mm) < 1e-10
    assert abs(y_back - y_test_mm) < 1e-10
    
    # Test vector pipeline
    v_body = proj.pixel_to_los_body(detector_id=4, u_pix=0.0, v_pix=0.0)
    assert v_body is not None
    assert np.isclose(np.linalg.norm(v_body), 1.0)
    
    pix_u, pix_v = proj.los_body_to_pixel(detector_id=4, v_body=v_body)
    assert abs(pix_u - 0.0) < 1e-6
    assert abs(pix_v - 0.0) < 1e-6


@pytest.mark.parametrize(
    "projection_model,setup",
    [
        ("distorted_focal_plane", lambda projector: None),
        ("ideal_pinhole", lambda projector: None),
        ("sky_patch_linearized", lambda projector: projector.set_field_center(304.098, 51.433)),
    ],
)
def test_projector_round_trips_center_edges_and_corners_for_supported_models(projection_model, setup):
    cfg = _ideal_pinhole_cfg()
    cfg["layout"]["projection_model"] = projection_model
    if projection_model == "distorted_focal_plane":
        cfg["layout"]["pixel_size_mm"] = 0.0065
        cfg["layout"]["distortion"] = {"a1": 0.134636212, "a3": -3.1445e-7, "axy2": -8.5532e-8}
        cfg["layout"]["detectors"][0]["fov_center_mm"] = [0.0, 0.0]

    projector = RealOpticalProjector(cfg)
    setup(projector)

    for u_pix, v_pix in [
        (559.0, 559.0),
        (0.0, 559.0),
        (1118.0, 559.0),
        (559.0, 0.0),
        (559.0, 1118.0),
        (0.0, 0.0),
        (1118.0, 1118.0),
    ]:
        los_body = projector.pixel_to_los_body(detector_id=0, u_pix=u_pix, v_pix=v_pix)
        assert los_body is not None
        assert np.isclose(np.linalg.norm(los_body), 1.0)
        u_back, v_back = projector.los_body_to_pixel(detector_id=0, v_body=los_body)
        assert np.allclose([u_back, v_back], [u_pix, v_pix], atol=1.0e-6)


def test_mounting_matrix_maps_detector_local_axes_to_body_axes():
    mounting_matrix = _rotation_z(np.pi / 2.0).tolist()
    projector = RealOpticalProjector(_ideal_pinhole_cfg(mounting_matrix=mounting_matrix))

    center_los = projector.pixel_to_los_body(detector_id=0, u_pix=559.0, v_pix=559.0)
    right_los = projector.pixel_to_los_body(detector_id=0, u_pix=560.0, v_pix=559.0)
    down_los = projector.pixel_to_los_body(detector_id=0, u_pix=559.0, v_pix=560.0)

    assert np.allclose(center_los, [0.0, 0.0, 1.0], atol=1.0e-12)
    assert np.dot(right_los - center_los, np.array([0.0, 1.0, 0.0])) > 0.0
    assert np.dot(down_los - center_los, np.array([-1.0, 0.0, 0.0])) > 0.0


def test_project_to_detectors_uses_scalar_first_inertial_to_body_quaternion():
    projector = RealOpticalProjector(_ideal_pinhole_cfg())
    q_ib = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)])
    los_inertial = np.array([1.0e-4, 0.0, 1.0], dtype=np.float64)
    los_inertial /= np.linalg.norm(los_inertial)

    predicted_xy, predicted_valid, visible_det_ids = projector.project_to_detectors(los_inertial, q_ib)
    expected_body = quat_to_dcm(q_ib) @ los_inertial
    expected_xy = projector.los_body_to_pixel(detector_id=0, v_body=expected_body)

    assert predicted_valid[0]
    assert visible_det_ids == [0]
    assert np.allclose(predicted_xy[0], expected_xy)


def test_ideal_pinhole_roundtrip():
    cfg = {
        "layout": {
            "projection_model": "ideal_pinhole",
            "detectors": [
                {
                    "detector_id": 0,
                    "principal_point_pix": [559.0, 559.0],
                    "resolution": [1119, 1119],
                    "pixel_scale_arcsec_per_pix": 4.83,
                    "mounting_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ],
        }
    }

    proj = RealOpticalProjector(cfg)
    los = proj.pixel_to_los_body(detector_id=0, u_pix=600.0, v_pix=520.0)
    assert los is not None
    assert np.isclose(np.linalg.norm(los), 1.0)

    u_back, v_back = proj.los_body_to_pixel(detector_id=0, v_body=los)
    assert abs(u_back - 600.0) < 1e-6
    assert abs(v_back - 520.0) < 1e-6


def test_sky_patch_linearized_roundtrip():
    cfg = {
        "layout": {
            "projection_model": "sky_patch_linearized",
            "detectors": [
                {
                    "detector_id": 0,
                    "principal_point_pix": [559.0, 559.0],
                    "resolution": [1119, 1119],
                    "pixel_scale_arcsec_per_pix": 4.83,
                    "mounting_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ],
        }
    }

    proj = RealOpticalProjector(cfg)
    proj.set_field_center(304.098, 51.433)

    los = proj.pixel_to_los_body(detector_id=0, u_pix=600.0, v_pix=520.0)
    assert los is not None
    assert np.isclose(np.linalg.norm(los), 1.0)

    u_back, v_back = proj.los_body_to_pixel(detector_id=0, v_body=los)
    assert abs(u_back - 600.0) < 1e-6
    assert abs(v_back - 520.0) < 1e-6


def test_sky_patch_linearized_roundtrip_with_field_offset():
    cfg = {
        "layout": {
            "projection_model": "sky_patch_linearized",
            "detectors": [
                {
                    "detector_id": 0,
                    "principal_point_pix": [559.0, 559.0],
                    "resolution": [1119, 1119],
                    "pixel_scale_arcsec_per_pix": 4.83,
                    "visibility_margin_pix": 0.5,
                    "mounting_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ],
        }
    }

    proj = RealOpticalProjector(cfg)
    proj.set_field_center(304.098, 51.433, field_offset_x_pix=0.25, field_offset_y_pix=-0.40)

    center_los = proj.pixel_to_los_body(detector_id=0, u_pix=559.25, v_pix=558.60)
    assert center_los is not None
    assert np.isclose(np.linalg.norm(center_los), 1.0)

    u_back, v_back = proj.los_body_to_pixel(detector_id=0, v_body=center_los)
    assert abs(u_back - 559.25) < 1e-6
    assert abs(v_back - 558.60) < 1e-6
