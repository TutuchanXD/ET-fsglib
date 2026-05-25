import numpy as np
import pytest

from fsglib.common.types import PreprocessedFrame
from fsglib.extract import pipeline as extract_pipeline
from fsglib.extract.pipeline import extract_stars


def _frame_from_image(
    image: np.ndarray,
    noise_level: float = 1.0,
    artifact_masks: dict[str, np.ndarray] | None = None,
) -> PreprocessedFrame:
    image = np.asarray(image, dtype=np.float64)
    return PreprocessedFrame(
        detector_id=0,
        image=image,
        background=0.0,
        noise_map=np.full_like(image, noise_level, dtype=np.float64),
        valid_mask=np.isfinite(image),
        preprocess_meta={},
        artifact_masks={} if artifact_masks is None else artifact_masks,
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
            "centroid_covariance": {"min_sigma_pix": 0.0},
            "deblend": {"enabled": True, "policy": "flag_only"},
        },
    }


def test_hysteresis_segments_keeps_compact_label_records():
    snr_map = np.zeros((9, 9), dtype=np.float64)
    for y, x in ((1, 1), (1, 7), (7, 1), (7, 7)):
        snr_map[y, x] = 6.0
    valid_mask = np.ones_like(snr_map, dtype=bool)

    labeled, segments = extract_pipeline._hysteresis_segments(
        snr_map,
        valid_mask,
        seed_th=5.0,
        grow_th=5.0,
    )

    assert labeled.shape == snr_map.shape
    assert len(segments) == 4
    label_id, num_seed_pixels, slices = segments[0]
    assert isinstance(label_id, int)
    assert num_seed_pixels == 1
    assert all(isinstance(item, slice) for item in slices)


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
    assert candidate.centroid_cov_pix.shape == (2, 2)
    assert candidate.flags["centroid_covariance_source"] == "noise_propagation"
    assert candidate.flags["centroid_sigma_x_pix"] > 0.0


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
    assert candidate.centroid_cov_pix.shape == (2, 2)


def test_extract_stars_adaptive_moment_centroid_is_explicitly_selectable():
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 2] = 4.0
    image[3, 3] = 10.0
    image[3, 4] = 5.0
    cfg = _extract_cfg("adaptive_moment_centroid")

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert len(candidates) == 1
    assert candidates[0].flags["centroid_method"] == "adaptive_moment_centroid"
    assert candidates[0].centroid_cov_pix.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(candidates[0].centroid_cov_pix) >= 0.0)


def test_extract_stars_rejects_psf_template_fit_until_followup_implementation():
    cfg = _extract_cfg("psf_template_fit")

    with pytest.raises(ValueError, match="psf.template_bundle_path"):
        extract_stars(_frame_from_image(np.eye(5, dtype=np.float64) * 10.0), cfg=cfg)

    cfg["psf"] = {"template_bundle_path": "/tmp/psf.pkl"}
    with pytest.raises(NotImplementedError, match="#89"):
        extract_stars(_frame_from_image(np.eye(5, dtype=np.float64) * 10.0), cfg=cfg)


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


def test_extract_stars_rejects_degenerate_sources_when_configured():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"reject_degenerate_sources": True}},
    )

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert candidates == []


def test_extract_stars_rejects_candidates_overlapping_artifact_masks():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0
    image[2, 3] = 4.0
    artifact = np.zeros_like(image, dtype=bool)
    artifact[2, 2] = True
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {
            "extract": {
                "reject_artifact_mask_overlap": True,
                "artifact_mask_margin_pix": 0,
            }
        },
    )

    candidates = extract_stars(
        _frame_from_image(image, artifact_masks={"saturated": artifact}),
        cfg=cfg,
    )

    assert candidates == []


@pytest.mark.parametrize("margin", [-1, 1.9, "bad"])
def test_extract_stars_rejects_invalid_artifact_mask_margin(margin):
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0
    artifact = np.zeros_like(image, dtype=bool)
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {
            "extract": {
                "reject_artifact_mask_overlap": True,
                "artifact_mask_margin_pix": margin,
            }
        },
    )

    with pytest.raises(ValueError, match="extract.artifact_mask_margin_pix"):
        extract_stars(
            _frame_from_image(image, artifact_masks={"saturated": artifact}),
            cfg=cfg,
        )


def test_extract_stars_flags_multi_peak_blends_by_default():
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 2] = 10.0
    image[3, 3] = 4.0
    image[3, 4] = 9.0

    candidates = extract_stars(_frame_from_image(image), cfg=_extract_cfg("weighted_centroid"))

    assert len(candidates) == 1
    assert candidates[0].flags["blend_flag"] is True
    assert candidates[0].flags["num_local_peaks"] == 2


def test_extract_stars_can_reject_multi_peak_blends():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 10.0
    image[2, 3] = 4.0
    image[2, 4] = 9.0
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"deblend": {"enabled": True, "policy": "reject"}}},
    )

    candidates = extract_stars(_frame_from_image(image), cfg=cfg)

    assert candidates == []


def test_extract_stars_applies_centroid_covariance_floor():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 1] = 6.0
    image[2, 2] = 10.0
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"centroid_covariance": {"min_sigma_pix": 0.2}}},
    )

    candidates = extract_stars(_frame_from_image(image, noise_level=0.0), cfg=cfg)

    assert len(candidates) == 1
    assert np.all(np.linalg.eigvalsh(candidates[0].centroid_cov_pix) >= 0.2**2)


def test_extract_stars_rejects_invalid_deblend_policy():
    cfg = _deep_update(
        _extract_cfg("weighted_centroid"),
        {"extract": {"deblend": {"policy": "split"}}},
    )

    with pytest.raises(ValueError, match="extract.deblend.policy"):
        extract_stars(_frame_from_image(np.eye(5, dtype=np.float64) * 10.0), cfg=cfg)
