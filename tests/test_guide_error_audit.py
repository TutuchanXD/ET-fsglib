import numpy as np

from fsglib.pipeline.guide_error_audit import _vector_stats


def test_vector_stats_keeps_existing_radial_schema_and_adds_spread_metrics():
    stats = _vector_stats([3.0, 0.0], [4.0, 0.0])

    assert stats["count"] == 2
    assert np.isclose(stats["median_radial"], 2.5)
    assert np.isclose(stats["std_dx"], 1.5)
    assert np.isclose(stats["std_dy"], 2.0)
    assert np.isclose(stats["std_radial"], 2.5)
    assert np.isclose(stats["mean_abs_radial_deviation"], 2.5)
    assert np.isclose(stats["rms_radial_deviation"], 2.5)


def test_vector_stats_empty_result_has_same_spread_keys():
    stats = _vector_stats([], [])

    assert stats["count"] == 0
    assert stats["median_radial"] is None
    assert stats["std_dx"] is None
    assert stats["std_dy"] is None
    assert stats["std_radial"] is None
    assert stats["mean_abs_radial_deviation"] is None
    assert stats["rms_radial_deviation"] is None
