import json

import numpy as np
import pytest

from fsglib.common.types import PreprocessedFrame
from fsglib.extract.pipeline import extract_stars


def _frame_from_image(image: np.ndarray, noise_level: float = 1.0) -> PreprocessedFrame:
    image = np.asarray(image, dtype=np.float64)
    return PreprocessedFrame(
        detector_id=0,
        image=image,
        background=0.0,
        noise_map=np.full_like(image, noise_level, dtype=np.float64),
        valid_mask=np.isfinite(image),
        preprocess_meta={},
    )


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extract_cfg(method: str, window_size: int = 5) -> dict:
    return {
        "extract": {
            "seed_threshold_sigma": 5.0,
            "grow_threshold_sigma": 3.0,
            "min_area": 1,
            "max_area": 25,
            "centroid_method": method,
            "centroid_window": {"center": "peak", "size": window_size},
            "bbox_expand": 0,
            "reject_edge_margin": 0,
            "max_ellipticity": 1.0,
            "bias_correction": {"enabled": False},
        },
        "bias_profiles": {"profiles": {}},
    }


def test_extract_stars_weighted_centroid_uses_segment_pixels_only():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 1] = 6.0
    image[2, 2] = 10.0
    image[2, 3] = 3.0

    frame = _frame_from_image(image)
    candidates = extract_stars(frame, cfg=_extract_cfg("weighted_centroid"))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert np.isclose(candidate.x, (1.0 * 6.0 + 2.0 * 10.0) / 16.0)
    assert np.isclose(candidate.y, 2.0)
    assert candidate.flags["centroid_method"] == "weighted_centroid"
    assert candidate.flags["centroid_bias_corrected"] is False


def test_extract_stars_grows_connected_pixels_above_grow_threshold():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0
    image[1, 1] = 4.0
    image[0, 4] = 4.5
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"seed_threshold_sigma": 5.0, "grow_threshold_sigma": 3.0}},
    )

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.area == 2
    assert np.isclose(candidate.flux, 14.0)
    assert np.isclose(candidate.snr, 14.0 / np.sqrt(2.0))
    assert candidate.bbox == (1, 1, 2, 2)
    assert np.isclose(candidate.x, (2.0 * 10.0 + 1.0 * 4.0) / 14.0)
    assert candidate.flags["segmentation_mode"] == "hysteresis"
    assert candidate.flags["seed_threshold_sigma"] == 5.0
    assert candidate.flags["grow_threshold_sigma"] == 3.0
    assert candidate.flags["num_seed_pixels"] == 1


def test_extract_stars_grow_equal_seed_reproduces_seed_only_segmentation():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0
    image[2, 3] = 4.0
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"seed_threshold_sigma": 5.0, "grow_threshold_sigma": 5.0}},
    )

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert len(candidates) == 1
    assert candidates[0].area == 1
    assert np.isclose(candidates[0].flux, 10.0)
    assert np.isclose(candidates[0].x, 2.0)


def test_extract_stars_rejects_invalid_hysteresis_thresholds():
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"seed_threshold_sigma": 5.0, "grow_threshold_sigma": 6.0}},
    )

    with pytest.raises(ValueError, match="grow_threshold_sigma"):
        extract_stars(_frame_from_image(np.ones((5, 5), dtype=np.float64)), cfg=cfg)


def test_extract_stars_fixed_window_first_moment_matches_full_window_definition():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 1] = 6.0
    image[2, 2] = 10.0
    image[2, 3] = 3.0

    frame = _frame_from_image(image)
    candidates = extract_stars(
        frame, cfg=_extract_cfg("fixed_window_first_moment", window_size=5)
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    expected_x = (1.0 * 6.0 + 2.0 * 10.0 + 3.0 * 3.0) / 19.0
    assert np.isclose(candidate.x, expected_x)
    assert np.isclose(candidate.y, 2.0)
    assert candidate.bbox == (0, 0, 4, 4)
    assert candidate.flags["centroid_method"] == "fixed_window_first_moment"


def test_extract_stars_populates_shape_metrics_for_round_source():
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 3] = 10.0
    image[2, 3] = 4.0
    image[4, 3] = 4.0
    image[3, 2] = 4.0
    image[3, 4] = 4.0

    candidates = extract_stars(_frame_from_image(image), cfg=_extract_cfg("weighted_centroid"))

    assert len(candidates) == 1
    shape = candidates[0].shape
    assert shape["shape_degenerate"] is False
    assert shape["ellipticity"] < 0.1
    assert shape["fwhm_pix"] > 0.0
    assert shape["sharpness"] > 0.0
    assert candidates[0].flags["shape_filter_passed"] is True
    assert np.isclose(candidates[0].flags["shape_ellipticity"], shape["ellipticity"])


def test_extract_stars_rejects_sources_above_max_ellipticity():
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 2] = 6.0
    image[3, 3] = 10.0
    image[3, 4] = 6.0
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"max_ellipticity": 0.5}},
    )

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert candidates == []


def test_extract_stars_keeps_single_pixel_degenerate_shape():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0

    candidates = extract_stars(_frame_from_image(image), cfg=_extract_cfg("weighted_centroid"))

    assert len(candidates) == 1
    assert candidates[0].shape["shape_degenerate"] is True
    assert candidates[0].shape["ellipticity"] == 0.0
    assert candidates[0].flags["shape_filter_passed"] is True


def test_extract_stars_applies_bias_correction_from_profile(tmp_path):
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 1] = 6.0
    image[2, 2] = 10.0
    image[2, 3] = 3.0
    raw_x = (1.0 * 6.0 + 2.0 * 10.0) / 16.0
    raw_y = 2.0

    profile_path = tmp_path / "bias_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_rows": [
                    {
                        "fsg_x_pix": raw_x,
                        "fsg_y_pix": raw_y,
                        "fsg_dx_err_pix": 0.25,
                        "fsg_dy_err_pix": -0.10,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = _deep_update(
        _extract_cfg("weighted_centroid", window_size=5),
        {
            "psf": {"active_model_key": "unit_test_psf"},
            "extract": {
                "bias_correction": {
                    "enabled": True,
                    "profile": None,
                    "strict_centroid_check": True,
                    "idw_k": 1,
                    "idw_power": 2.0,
                }
            },
            "bias_profiles": {
                "by_psf_model": {
                    "unit_test_psf": {
                        "bias_table_path": str(profile_path),
                        "calibration_key": "fsg",
                        "centroid_method": "weighted_centroid",
                    }
                }
            },
        },
    )

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert np.isclose(candidate.flags["raw_centroid_x_pix"], raw_x)
    assert np.isclose(candidate.flags["raw_centroid_y_pix"], raw_y)
    assert np.isclose(candidate.flags["predicted_bias_x_pix"], 0.25)
    assert np.isclose(candidate.flags["predicted_bias_y_pix"], -0.10)
    assert np.isclose(candidate.x, raw_x - 0.25)
    assert np.isclose(candidate.y, raw_y + 0.10)
    assert candidate.flags["centroid_bias_corrected"] is True
    assert candidate.flags["bias_profile"] == "unit_test_psf"


def test_extract_stars_rejects_mismatched_bias_profile_metadata(tmp_path):
    profile_path = tmp_path / "bias_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_rows": [
                    {
                        "fsg_x_pix": 2.0,
                        "fsg_y_pix": 2.0,
                        "fsg_dx_err_pix": 0.1,
                        "fsg_dy_err_pix": 0.1,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = _deep_update(
        _extract_cfg("weighted_centroid", window_size=5),
        {
            "psf": {"active_model_key": "unit_test_psf"},
            "extract": {
                "bias_correction": {
                    "enabled": True,
                    "profile": None,
                    "strict_centroid_check": True,
                }
            },
            "bias_profiles": {
                "by_psf_model": {
                    "unit_test_psf": {
                        "bias_table_path": str(profile_path),
                        "calibration_key": "fsg",
                        "centroid_method": "fixed_window_first_moment",
                        "window_size": 5,
                    }
                }
            },
        },
    )

    with pytest.raises(ValueError, match="centroid_method mismatch"):
        extract_stars(_frame_from_image(np.eye(5, dtype=np.float64) * 10.0), cfg=cfg)
