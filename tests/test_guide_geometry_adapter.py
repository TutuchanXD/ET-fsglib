from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fsglib.ephemeris.guide_geometry import (
    build_exact_focalplane_geometry_adapter,
    et_field_angles_to_body_vector,
)


class FakeDetector:
    pixel_width = 100.0
    pixel_height = 80.0


class FakeRegistry:
    def __init__(self):
        self.detectors = {"G1": FakeDetector()}

    def get_detector(self, detector_id):
        return self.detectors[detector_id]


class FakeTransformer:
    def __init__(self, rotation_body_from_eq):
        self.rotation_body_from_eq = np.asarray(rotation_body_from_eq, dtype=np.float64)
        self.missing_pixel_sky = False

    @staticmethod
    def _field_angles(xpix, ypix):
        return 0.02 * (float(xpix) - 50.0), 0.03 * (float(ypix) - 40.0)

    def pixel_to_focal(self, detector_id, xpix, ypix):
        field_x_deg, field_y_deg = self._field_angles(xpix, ypix)
        return SimpleNamespace(
            status="ok",
            detector_id=detector_id,
            xpix=float(xpix),
            ypix=float(ypix),
            x_mm=(float(xpix) - 50.0) * 0.0065,
            y_mm=(float(ypix) - 40.0) * 0.0065,
            field_x_deg=field_x_deg,
            field_y_deg=field_y_deg,
        )

    def focal_to_sky(self, detector_id, x_mm, y_mm, *, frame="equatorial"):
        xpix = float(x_mm) / 0.0065 + 50.0
        ypix = float(y_mm) / 0.0065 + 40.0
        field_x_deg, field_y_deg = self._field_angles(xpix, ypix)
        body = et_field_angles_to_body_vector(field_x_deg, field_y_deg)
        inertial = self.rotation_body_from_eq.T @ body
        return SimpleNamespace(status="ok", frame=frame, vector_xyz=tuple(inertial))

    def pixel_to_sky(self, detector_id, xpix, ypix, *, frame="equatorial"):
        if self.missing_pixel_sky:
            return SimpleNamespace(status="error", frame=frame, vector_xyz=None)
        focal = self.pixel_to_focal(detector_id, xpix, ypix)
        return self.focal_to_sky(detector_id, focal.x_mm, focal.y_mm, frame=frame)


def _cfg(los_geometry_mode="exact_et_focalplane"):
    return {
        "guide_init": {
            "los_geometry_mode": los_geometry_mode,
            "frame_alignment_grid_size": 5,
            "detector_batches": [{"detector_id": "G1", "batch_name": "unused"}],
        }
    }


def test_exact_adapter_recovers_alignment_and_projects_pixels_to_body_los():
    rotation_body_from_eq = Rotation.from_euler("zy", [12.0, -5.0], degrees=True).as_matrix()
    adapter = build_exact_focalplane_geometry_adapter(
        _cfg(),
        FakeRegistry(),
        FakeTransformer(rotation_body_from_eq),
    )

    expected = et_field_angles_to_body_vector(*FakeTransformer._field_angles(63.0, 52.0))
    actual = adapter.pixel_to_body_los("G1", 63.0, 52.0)

    assert np.allclose(actual, expected, atol=1.0e-12)
    assert adapter.frame_alignment_fit_rms_arcsec < 1.0e-2

    payload = adapter.serialize()
    assert payload["mode"] == "exact_et_focalplane"
    assert payload["frame_alignment_grid_size"] == 5
    assert "coeffs" not in payload
    assert "body_model_proxy" not in repr(payload)


def test_exact_adapter_rejects_body_model_proxy_mode():
    with pytest.raises(ValueError, match="body_model_proxy.*removed"):
        build_exact_focalplane_geometry_adapter(
            _cfg(los_geometry_mode="body_model_proxy"),
            FakeRegistry(),
            FakeTransformer(np.eye(3)),
        )


def test_exact_adapter_fails_when_pixel_to_sky_has_no_equatorial_vector():
    transformer = FakeTransformer(np.eye(3))
    adapter = build_exact_focalplane_geometry_adapter(_cfg(), FakeRegistry(), transformer)
    transformer.missing_pixel_sky = True

    with pytest.raises(ValueError, match="Missing equatorial vector"):
        adapter.pixel_to_body_los("G1", 50.0, 40.0)
