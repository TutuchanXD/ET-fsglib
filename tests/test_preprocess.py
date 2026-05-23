import numpy as np
import pytest

from fsglib.common.types import PreprocessedFrame, RawFrame
from fsglib.extract.pipeline import extract_stars
from fsglib.models.mock import build_models
from fsglib.preprocess.calibration import load_calibration_products
from fsglib.preprocess.pipeline import preprocess_frame


def _preprocess_cfg(**overrides):
    cfg = {
        "enable_bias_subtraction": False,
        "enable_dark_subtraction": False,
        "enable_flat_field": False,
        "enable_bad_pixel_mask": False,
        "enable_fpn_subtraction": False,
        "enable_background_subtraction": False,
    }
    cfg.update(overrides)
    return {"preprocess": cfg}


def test_preprocess_applies_ordered_detector_calibration_chain():
    raw = RawFrame(
        detector_id=2,
        image=np.full((3, 3), 20.0, dtype=np.float64),
        time_s=10.0,
        cadence_s=0.5,
        unit="electron_or_adu",
    )
    bad_pixel_mask = np.zeros((3, 3), dtype=bool)
    bad_pixel_mask[1, 1] = True
    calib = {
        "bias": np.full((3, 3), 1.0, dtype=np.float64),
        "dark": np.full((3, 3), 2.0, dtype=np.float64),
        "flat": np.full((3, 3), 2.0, dtype=np.float64),
        "bad_pixel_mask": bad_pixel_mask,
        "fpn_residual": np.full((3, 3), 0.5, dtype=np.float64),
        "meta": {
            "bias": {"path": "/assets/bias.npy"},
            "dark": {"path": "/assets/dark.npy"},
            "flat": {"path": "/assets/flat.npy"},
            "bad_pixel_mask": {"path": "/assets/bad.npy"},
            "fpn_residual": {"path": "/assets/fpn.npy"},
        },
    }
    cfg = _preprocess_cfg(
        enable_bias_subtraction=True,
        enable_dark_subtraction=True,
        enable_flat_field=True,
        enable_bad_pixel_mask=True,
        enable_fpn_subtraction=True,
        gain_e_per_dn=1.0,
    )

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)

    expected_value = (20.0 - 1.0 - 2.0 * 0.5 - 0.5) / 2.0
    expected = np.full((3, 3), expected_value, dtype=np.float64)
    expected[1, 1] = 0.0
    assert isinstance(pre, PreprocessedFrame)
    assert np.allclose(pre.image, expected)
    expected_valid = np.array(
        [[True, True, True], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    assert pre.valid_mask.dtype == bool
    assert np.array_equal(pre.valid_mask, expected_valid)
    assert pre.variance_map is not None
    assert np.allclose(pre.variance_map, pre.noise_map**2)
    assert pre.preprocess_meta["input_unit"] == "electron_or_adu"
    assert pre.preprocess_meta["calibration_order"] == [
        "finite_mask",
        "adc_clip",
        "saturation_guard",
        "bias_subtraction",
        "fpn_subtraction",
        "adu_to_electron_conversion",
        "dark_subtraction",
        "flat_field",
        "bad_pixel_mask",
        "background_subtraction",
        "noise_estimation",
    ]
    for name in ["bias", "dark", "flat", "bad_pixel_mask", "fpn_residual"]:
        assert pre.preprocess_meta["calibration"][name]["applied"] is True
        assert "path" in pre.preprocess_meta["calibration"][name]
    assert pre.preprocess_meta["num_bad_pixels"] == 1
    assert pre.preprocess_meta["num_invalid_pixels"] == 1


def test_preprocess_converts_adu_to_electrons_with_global_gain():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[2.2, 4.4, 6.6]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = _preprocess_cfg(
        convert_to_electrons=True,
        gain_e_per_dn=1.0 / 2.2,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.image, [[1.0, 2.0, 3.0]])
    assert pre.preprocess_meta["input_unit"] == "adu"
    assert pre.preprocess_meta["output_unit"] == "electron"
    assert pre.preprocess_meta["noise_unit"] == "electron"
    assert pre.preprocess_meta["variance_unit"] == "electron^2"
    assert pre.preprocess_meta["adu_to_electron_conversion"] == {
        "enabled": True,
        "applied": True,
        "input_unit": "adu",
        "input_unit_effective": "adu",
        "output_unit": "electron",
        "gain_e_per_dn": 1.0 / 2.2,
        "reason": "input_unit_adu",
    }


def test_preprocess_defaults_missing_unit_to_adu_for_electron_conversion():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[2.0, 4.0]], dtype=np.float64),
        time_s=0.0,
        unit=None,
    )
    cfg = _preprocess_cfg(
        convert_to_electrons=True,
        gain_e_per_dn=0.5,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.image, [[1.0, 2.0]])
    conversion = pre.preprocess_meta["adu_to_electron_conversion"]
    assert conversion["applied"] is True
    assert conversion["input_unit"] is None
    assert conversion["input_unit_effective"] == "adu"
    assert conversion["reason"] == "missing_unit_defaulted_to_adu"


def test_preprocess_treats_legacy_electron_or_adu_as_adu_for_conversion():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[2.0, 4.0]], dtype=np.float64),
        time_s=0.0,
        unit="electron_or_adu",
    )
    cfg = _preprocess_cfg(
        convert_to_electrons=True,
        gain_e_per_dn=0.5,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.image, [[1.0, 2.0]])
    conversion = pre.preprocess_meta["adu_to_electron_conversion"]
    assert conversion["applied"] is True
    assert conversion["input_unit_effective"] == "adu"
    assert conversion["reason"] == "legacy_electron_or_adu_treated_as_adu"


def test_preprocess_keeps_electron_inputs_in_electron_space():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[1.0, 2.0]], dtype=np.float64),
        time_s=0.0,
        unit="electron",
    )
    cfg = _preprocess_cfg(convert_to_electrons=True)

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.image, [[1.0, 2.0]])
    assert pre.preprocess_meta["output_unit"] == "electron"
    conversion = pre.preprocess_meta["adu_to_electron_conversion"]
    assert conversion["applied"] is False
    assert conversion["input_unit_effective"] == "electron"
    assert conversion["reason"] == "input_already_electron"
    assert "gain_e_per_dn" not in conversion


def test_preprocess_rejects_unknown_unit_when_conversion_is_enabled():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[1.0]], dtype=np.float64),
        time_s=0.0,
        unit="counts",
    )
    cfg = _preprocess_cfg(
        convert_to_electrons=True,
        gain_e_per_dn=1.0,
    )

    with pytest.raises(ValueError, match="raw.unit"):
        preprocess_frame(raw, calib={}, cfg=cfg)


def test_preprocess_converts_after_adu_bias_and_fpn_before_electron_dark_and_flat():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[24.0]], dtype=np.float64),
        time_s=0.0,
        cadence_s=2.0,
        unit="adu",
    )
    calib = {
        "bias": np.array([[2.0]], dtype=np.float64),
        "fpn_residual": np.array([[1.0]], dtype=np.float64),
        "dark": np.array([[3.0]], dtype=np.float64),
        "flat": np.array([[2.0]], dtype=np.float64),
    }
    cfg = _preprocess_cfg(
        enable_bias_subtraction=True,
        enable_fpn_subtraction=True,
        convert_to_electrons=True,
        gain_e_per_dn=0.5,
        enable_dark_subtraction=True,
        enable_flat_field=True,
    )

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)

    assert np.allclose(pre.image, [[2.25]])
    assert pre.preprocess_meta["output_unit"] == "electron"
    assert pre.preprocess_meta["calibration"]["dark"]["unit"] == "electron"


def test_preprocess_adc_clip_clips_to_configured_bit_depth_without_masking():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[-2.0, 4094.25, 4096.0, 62142.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = {
        **_preprocess_cfg(
            enable_adc_clip=True,
            enable_saturation_guard=False,
        ),
        "detector": {"adc_bit_depth": 12},
    }

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.image, [[0.0, 4094.25, 4095.0, 4095.0]])
    assert np.all(pre.valid_mask)
    assert pre.preprocess_meta["adc_clip"]["enabled"] is True
    assert pre.preprocess_meta["adc_clip"]["max_value"] == 4095.0
    assert pre.preprocess_meta["adc_clip"]["num_clipped_high_pixels"] == 2
    assert pre.preprocess_meta["adc_clip"]["num_clipped_low_pixels"] == 1
    assert pre.preprocess_meta["artifact_counts"]["saturated"] == 2


def test_preprocess_saturation_guard_masks_saturated_pixels_by_default():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[10.0, 4096.0, 5000.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = {
        **_preprocess_cfg(enable_adc_clip=True),
        "detector": {"adc_bit_depth": 12},
    }

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.image, [[10.0, 0.0, 0.0]])
    assert np.array_equal(pre.valid_mask, [[True, False, False]])
    assert np.array_equal(pre.artifact_masks["saturated"], [[False, True, True]])
    assert np.array_equal(pre.artifact_masks["saturation_guard"], [[False, True, True]])
    assert pre.preprocess_meta["artifact_policy"]["saturation_guard_applied"] is True
    assert pre.preprocess_meta["num_invalid_pixels"] == 2


def test_preprocess_rejects_noninteger_saturation_guard_margin():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[10.0, 4096.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = {
        **_preprocess_cfg(
            enable_adc_clip=True,
            saturation_mask_dilation_pix=1.9,
        ),
        "detector": {"adc_bit_depth": 12},
    }

    with pytest.raises(ValueError, match="preprocess.saturation_mask_dilation_pix"):
        preprocess_frame(raw, calib={}, cfg=cfg)


def test_preprocess_reports_adc_bit_depth_when_derived_max_is_invalid():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[1.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = {
        **_preprocess_cfg(enable_adc_clip=True),
        "detector": {
            "adc_min_value": 10.0,
            "adc_bit_depth": 2,
            "saturation_value": None,
        },
    }

    with pytest.raises(ValueError, match="adc_bit_depth.*adc_min_value"):
        preprocess_frame(raw, calib={}, cfg=cfg)


@pytest.mark.parametrize("bit_depth", [0, -1, 12.5, np.nan, "twelve", True])
def test_preprocess_rejects_invalid_adc_bit_depth(bit_depth):
    raw = RawFrame(
        detector_id=0,
        image=np.array([[1.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = {
        **_preprocess_cfg(enable_adc_clip=True),
        "detector": {
            "adc_bit_depth": bit_depth,
            "saturation_value": None,
        },
    }

    with pytest.raises(ValueError, match="detector.adc_bit_depth"):
        preprocess_frame(raw, calib={}, cfg=cfg)


def test_preprocess_raises_when_enabled_calibration_product_is_missing():
    raw = RawFrame(
        detector_id=0,
        image=np.ones((2, 2), dtype=np.float64),
        time_s=0.0,
    )
    cfg = _preprocess_cfg(enable_bias_subtraction=True)

    with pytest.raises(ValueError, match="preprocess.enable_bias_subtraction"):
        preprocess_frame(raw, calib={}, cfg=cfg)


def test_preprocess_requires_cadence_for_dark_subtraction():
    raw = RawFrame(
        detector_id=0,
        image=np.ones((2, 2), dtype=np.float64),
        time_s=0.0,
        cadence_s=None,
    )
    calib = {"dark": np.ones((2, 2), dtype=np.float64)}
    cfg = _preprocess_cfg(enable_dark_subtraction=True)

    with pytest.raises(ValueError, match="raw.cadence_s"):
        preprocess_frame(raw, calib=calib, cfg=cfg)


def test_preprocess_rejects_calibration_shape_mismatch():
    raw = RawFrame(
        detector_id=0,
        image=np.ones((2, 2), dtype=np.float64),
        time_s=0.0,
    )
    calib = {"bias": np.ones((3, 3), dtype=np.float64)}
    cfg = _preprocess_cfg(enable_bias_subtraction=True)

    with pytest.raises(ValueError, match="bias calibration shape"):
        preprocess_frame(raw, calib=calib, cfg=cfg)


def test_preprocess_marks_nonpositive_flat_pixels_invalid():
    raw = RawFrame(
        detector_id=0,
        image=np.full((2, 2), 10.0, dtype=np.float64),
        time_s=0.0,
    )
    flat = np.array([[1.0, 0.0], [2.0, np.nan]], dtype=np.float64)
    calib = {"flat": flat}
    cfg = _preprocess_cfg(enable_flat_field=True)

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)

    assert np.allclose(pre.image, np.array([[10.0, 0.0], [5.0, 0.0]]))
    assert np.array_equal(pre.valid_mask, np.array([[True, False], [True, False]]))
    assert pre.preprocess_meta["calibration"]["flat"]["num_invalid_flat_pixels"] == 2


def test_preprocess_calibration_preserves_extractable_star_flux_and_centroid():
    image = np.full((7, 7), 10.0, dtype=np.float64)
    image[3, 3] = 110.0
    raw = RawFrame(detector_id=0, image=image, time_s=0.0)
    calib = {"bias": np.full((7, 7), 5.0, dtype=np.float64)}
    cfg = {
        **_preprocess_cfg(
            enable_bias_subtraction=True,
            enable_background_subtraction=True,
        ),
        "extract": {
            "seed_threshold_sigma": 5.0,
            "min_area": 1,
            "max_area": 9,
            "centroid_method": "weighted_centroid",
            "bbox_expand": 0,
            "reject_edge_margin": 0,
        },
    }

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)
    candidates = extract_stars(pre, cfg=cfg)

    assert len(candidates) == 1
    assert np.isclose(candidates[0].x, 3.0)
    assert np.isclose(candidates[0].y, 3.0)
    assert np.isclose(candidates[0].flux, 100.0)


def test_preprocess_records_configured_and_effective_background_method():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[1.0, 2.0], [3.0, 100.0]], dtype=np.float64),
        time_s=0.0,
    )
    cfg = _preprocess_cfg(
        enable_background_subtraction=True,
        background_method="sigma_clip_global",
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert pre.preprocess_meta["background_method_configured"] == "sigma_clip_global"
    assert pre.preprocess_meta["background_method_effective"] == "sigma_clip_global"
    assert pre.preprocess_meta["background_method"] == "sigma_clip_global"
    assert "background_rms" in pre.preprocess_meta


def test_preprocess_records_documented_default_background_method_when_absent():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[1.0, 2.0], [3.0, 100.0]], dtype=np.float64),
        time_s=0.0,
    )
    cfg = _preprocess_cfg(enable_background_subtraction=True)
    cfg["preprocess"].pop("background_method", None)

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert pre.preprocess_meta["background_method_configured"] == "sigma_clip_global"
    assert pre.preprocess_meta["background_method_effective"] == "sigma_clip_global"


def test_sigma_clip_global_background_rejects_bright_outlier():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[0.0, 0.0, 10.0], [10.0, 1000.0, 10.0]], dtype=np.float64),
        time_s=0.0,
    )
    cfg = _preprocess_cfg(
        enable_background_subtraction=True,
        background_method="sigma_clip_global",
        sigma_clip_k=3.0,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.isclose(pre.background, 10.0)
    assert pre.preprocess_meta["background_method_effective"] == "sigma_clip_global"
    assert pre.preprocess_meta["background_num_clipped_pixels"] == 1


def test_mesh_median_background_tracks_spatial_gradient():
    _, xx = np.indices((32, 32), dtype=np.float64)
    image = 10.0 + 0.5 * xx
    image[8, 8] += 500.0
    image[24, 24] += 500.0
    raw = RawFrame(detector_id=0, image=image, time_s=0.0)
    cfg = _preprocess_cfg(
        enable_background_subtraction=True,
        background_method="mesh_median",
        background_mesh_size=8,
        sigma_clip_k=3.0,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert isinstance(pre.background, np.ndarray)
    assert pre.background.shape == image.shape
    assert float(pre.background[16, 28] - pre.background[16, 4]) > 8.0
    assert np.nanmedian(np.abs(pre.image[pre.valid_mask])) < 2.0
    assert pre.preprocess_meta["background_method_effective"] == "mesh_median"
    assert pre.preprocess_meta["background_mesh_size"] == 8


def test_empirical_robust_noise_uses_outlier_resistant_variance():
    image = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 1000.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )
    raw = RawFrame(detector_id=0, image=image, time_s=0.0)
    cfg = _preprocess_cfg(
        enable_background_subtraction=False,
        variance_model="empirical_robust",
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.all(pre.noise_map < 5.0)
    assert np.allclose(pre.variance_map, pre.noise_map**2)
    assert pre.preprocess_meta["variance_model_effective"] == "empirical_robust"


def test_poisson_read_noise_variance_model_converts_back_to_input_units():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0, 200.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = _preprocess_cfg(
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        gain_e_per_dn=2.0,
        read_noise_e=4.0,
        quantization_noise_e=1.0,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    expected_variance = np.array([[54.25, 104.25]], dtype=np.float64)
    assert np.allclose(pre.variance_map, expected_variance)
    assert np.allclose(pre.noise_map, np.sqrt(expected_variance))
    assert pre.preprocess_meta["variance_unit"] == "adu^2"
    assert pre.preprocess_meta["variance_model_effective"] == "poisson_read_noise"


def test_poisson_variance_model_adds_dark_shot_noise_from_cadence():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0]], dtype=np.float64),
        time_s=0.0,
        cadence_s=2.0,
        unit="adu",
    )
    calib = {"dark": np.array([[3.0]], dtype=np.float64)}
    cfg = _preprocess_cfg(
        enable_dark_subtraction=True,
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        gain_e_per_dn=2.0,
        read_noise_e=4.0,
        quantization_noise_e=0.0,
    )

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)

    assert np.allclose(pre.image, [[97.0]])
    assert np.allclose(pre.variance_map, [[54.0]])
    assert pre.preprocess_meta["variance_components"]["dark_current_source"] == "calib.dark"


def test_poisson_variance_model_uses_electron_space_after_gain_conversion():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0]], dtype=np.float64),
        time_s=0.0,
        cadence_s=2.0,
        unit="adu",
    )
    calib = {"dark": np.array([[3.0]], dtype=np.float64)}
    cfg = _preprocess_cfg(
        convert_to_electrons=True,
        gain_e_per_dn=2.0,
        enable_dark_subtraction=True,
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        read_noise_e=4.0,
        quantization_noise_e=0.0,
    )

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)

    assert np.allclose(pre.image, [[194.0]])
    assert np.allclose(pre.variance_map, [[216.0]])
    assert pre.preprocess_meta["variance_unit"] == "electron^2"
    assert pre.preprocess_meta["variance_components"]["gain_e_per_output_unit"] == 1.0
    assert pre.preprocess_meta["variance_components"]["dark_current_source"] == "calib.dark"


def test_poisson_variance_model_propagates_flat_response():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[200.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    calib = {"flat": np.array([[2.0]], dtype=np.float64)}
    cfg = _preprocess_cfg(
        enable_flat_field=True,
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        gain_e_per_dn=2.0,
        read_noise_e=0.0,
        quantization_noise_e=0.0,
    )

    pre = preprocess_frame(raw, calib=calib, cfg=cfg)

    assert np.allclose(pre.image, [[100.0]])
    assert np.allclose(pre.variance_map, [[25.0]])
    assert pre.preprocess_meta["variance_components"]["flat_response_propagated"] is True


def test_scalar_dark_current_is_not_double_counted_when_not_subtracted():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0]], dtype=np.float64),
        time_s=0.0,
        cadence_s=1.0,
        unit="adu",
    )
    cfg = _preprocess_cfg(
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        gain_e_per_dn=2.0,
        read_noise_e=0.0,
        quantization_noise_e=0.0,
        dark_current_e_per_s=10.0,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.variance_map, [[50.0]])
    assert pre.preprocess_meta["variance_components"]["dark_current_source"] == "none"


def test_poisson_variance_model_requires_gain_for_dn_inputs():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0]], dtype=np.float64),
        time_s=0.0,
        unit="adu",
    )
    cfg = _preprocess_cfg(
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        read_noise_e=4.0,
    )

    with pytest.raises(ValueError, match="preprocess.gain_e_per_dn"):
        preprocess_frame(raw, calib={}, cfg=cfg)


def test_poisson_variance_model_allows_electron_inputs_without_gain():
    raw = RawFrame(
        detector_id=0,
        image=np.array([[100.0]], dtype=np.float64),
        time_s=0.0,
        unit="electron",
    )
    cfg = _preprocess_cfg(
        enable_background_subtraction=False,
        variance_model="poisson_read_noise",
        read_noise_e=4.0,
    )

    pre = preprocess_frame(raw, calib={}, cfg=cfg)

    assert np.allclose(pre.variance_map, [[116.0]])
    assert pre.preprocess_meta["variance_unit"] == "electron^2"


def test_extract_snr_uses_preprocess_poisson_noise_map():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 100.0
    raw = RawFrame(detector_id=0, image=image, time_s=0.0, unit="adu")
    cfg = {
        **_preprocess_cfg(
            enable_background_subtraction=False,
            variance_model="poisson_read_noise",
            gain_e_per_dn=2.0,
            read_noise_e=0.0,
        ),
        "extract": {
            "seed_threshold_sigma": 5.0,
            "min_area": 1,
            "max_area": 9,
            "centroid_method": "weighted_centroid",
            "bbox_expand": 0,
            "reject_edge_margin": 0,
        },
    }

    pre = preprocess_frame(raw, calib={}, cfg=cfg)
    candidates = extract_stars(pre, cfg=cfg)

    assert len(candidates) == 1
    assert np.isclose(candidates[0].snr, 100.0 / np.sqrt(50.0))


def test_load_calibration_products_from_yaml_paths(tmp_path):
    paths = {
        "bias_frame_path": tmp_path / "bias.npy",
        "dark_current_path": tmp_path / "dark.npy",
        "flat_field_path": tmp_path / "flat.npy",
        "bad_pixel_mask_path": tmp_path / "bad.npy",
        "fpn_residual_map_path": tmp_path / "fpn.npy",
    }
    np.save(paths["bias_frame_path"], np.full((2, 2), 1.0, dtype=np.float64))
    np.save(paths["dark_current_path"], np.full((2, 2), 0.25, dtype=np.float64))
    np.save(paths["flat_field_path"], np.ones((2, 2), dtype=np.float64))
    np.save(paths["bad_pixel_mask_path"], np.array([[False, True], [False, False]]))
    np.save(paths["fpn_residual_map_path"], np.full((2, 2), 0.1, dtype=np.float64))
    cfg = _preprocess_cfg(
        enable_bias_subtraction=True,
        enable_dark_subtraction=True,
        enable_flat_field=True,
        enable_bad_pixel_mask=True,
        enable_fpn_subtraction=True,
        **{name: str(path) for name, path in paths.items()},
    )

    calib = load_calibration_products(cfg)

    assert set(calib) >= {
        "bias",
        "dark",
        "flat",
        "bad_pixel_mask",
        "fpn_residual",
        "meta",
    }
    assert calib["bad_pixel_mask"].dtype == bool
    assert calib["meta"]["bias"]["path"] == str(paths["bias_frame_path"])
    assert calib["meta"]["bad_pixel_mask"]["shape"] == (2, 2)
    assert calib["meta"]["flat"]["format"] == "npy"


def test_load_calibration_products_requires_configured_path_for_enabled_product():
    cfg = _preprocess_cfg(enable_bias_subtraction=True, bias_frame_path=None)

    with pytest.raises(ValueError, match="preprocess.bias_frame_path"):
        load_calibration_products(cfg)


def test_load_calibration_products_rejects_directory_path(tmp_path):
    cfg = _preprocess_cfg(enable_bias_subtraction=True, bias_frame_path=str(tmp_path))

    with pytest.raises(ValueError, match="must be a file"):
        load_calibration_products(cfg)


def test_load_calibration_products_accepts_npz_data_key(tmp_path):
    bias_path = tmp_path / "bias.npz"
    np.savez_compressed(bias_path, data=np.full((2, 2), 3.0, dtype=np.float32))
    cfg = _preprocess_cfg(enable_bias_subtraction=True, bias_frame_path=str(bias_path))

    calib = load_calibration_products(cfg)

    assert np.array_equal(calib["bias"], np.full((2, 2), 3.0))
    assert calib["meta"]["bias"]["format"] == "npz"
    assert calib["meta"]["bias"]["array_key"] == "data"


def test_load_calibration_products_warns_for_fake_assets(tmp_path):
    fake_dir = tmp_path / "fsglib-data" / "calibration" / "pr09_fake" / "2049x2049"
    fake_dir.mkdir(parents=True)
    bias_path = fake_dir / "bias_frame.npz"
    np.savez_compressed(bias_path, data=np.zeros((2, 2), dtype=np.float32))
    cfg = _preprocess_cfg(enable_bias_subtraction=True, bias_frame_path=str(bias_path))

    with pytest.warns(RuntimeWarning, match="fake PR9 calibration assets"):
        calib = load_calibration_products(cfg)

    assert calib["meta"]["uses_fake_calibration_assets"] is True
    assert calib["meta"]["bias"]["asset_kind"] == "fake"


def test_build_models_loads_configured_calibration_products(tmp_path):
    bias_path = tmp_path / "bias.npy"
    np.save(bias_path, np.ones((2, 2), dtype=np.float64))
    cfg = {
        **_preprocess_cfg(
            enable_bias_subtraction=True,
            bias_frame_path=str(bias_path),
        ),
        "ephemeris": {"gaia_root_dir": str(tmp_path), "mag_limit": 15.0},
        "layout": {
            "detectors": [
                {
                    "detector_id": 0,
                    "mounting_matrix": None,
                    "principal_point_pix": [0.0, 0.0],
                    "resolution": [2, 2],
                }
            ]
        },
    }

    models = build_models(cfg)

    assert np.array_equal(models["calib"]["bias"], np.ones((2, 2)))
    assert models["calib"]["meta"]["bias"]["path"] == str(bias_path)
