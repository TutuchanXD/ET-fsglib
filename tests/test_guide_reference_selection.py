import sys
import types

import numpy as np
import pytest

from fsglib.pipeline.run_guide_init import _build_reference_stars


pd = pytest.importorskip("pandas")


class DummyGaiaSourceFilter:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)


def _cfg(**guide_overrides):
    guide_init = {
        "detector_batches": [{"detector_id": "G1", "batch_name": "unused"}],
        "catalog_g_mag_max": 12.0,
        "reference_topk_per_detector": 2,
    }
    guide_init.update(guide_overrides)
    return {"guide_init": guide_init}


def _install_fake_et_coord(monkeypatch, frame):
    def query_detector_sources(registry, catalog, detector_id, filters, include_coords, target_epoch):
        assert detector_id == "G1"
        assert include_coords == ("pixel",)
        assert target_epoch == 2000.0
        return frame

    monkeypatch.setitem(sys.modules, "et_coord", types.SimpleNamespace(query_detector_sources=query_detector_sources))


def test_build_reference_stars_default_selection_preserves_topk_behavior(monkeypatch):
    DummyGaiaSourceFilter.calls = []
    frame = pd.DataFrame(
        [
            {"source_id": 1, "ra_deg": 0.0, "dec_deg": 0.0, "g_mean_mag": 10.0, "xpix": 10.0, "ypix": 10.0},
            {"source_id": 2, "ra_deg": 1.0, "dec_deg": 0.0, "g_mean_mag": 9.0, "xpix": 20.0, "ypix": 20.0},
            {"source_id": 3, "ra_deg": 2.0, "dec_deg": 0.0, "g_mean_mag": 11.0, "xpix": 30.0, "ypix": 30.0},
        ]
    )
    _install_fake_et_coord(monkeypatch, frame)

    reference, stats = _build_reference_stars(_cfg(), object(), object(), DummyGaiaSourceFilter)

    assert [star.catalog_id for star in reference] == [2, 1]
    assert np.isclose(reference[0].weight_hint, 10.0 ** (-0.4 * 9.0))
    assert reference[0].meta["weight_source"] == "gaia_g"
    assert reference[0].meta["target_epoch"] == 2000.0
    assert DummyGaiaSourceFilter.calls == [{"g_mean_mag_max": 12.0}]
    assert stats["G1"]["num_reference_stars"] == 2
    assert stats["G1"]["num_reference_preselected"] == 2
    assert stats["G1"]["num_reference_isolated"] == 2
    assert stats["G1"]["catalog_g_mag_min"] is None
    assert stats["G1"]["preselect_topk"] == 2
    assert stats["G1"]["isolation_radius_pix"] is None


def test_build_reference_stars_can_preselect_and_filter_non_isolated_sources(monkeypatch):
    DummyGaiaSourceFilter.calls = []
    frame = pd.DataFrame(
        [
            {"source_id": 1, "ra_deg": 0.0, "dec_deg": 0.0, "g_mean_mag": 8.0, "xpix": 0.0, "ypix": 0.0},
            {"source_id": 2, "ra_deg": 1.0, "dec_deg": 0.0, "g_mean_mag": 9.0, "xpix": 1.0, "ypix": 0.0},
            {"source_id": 3, "ra_deg": 2.0, "dec_deg": 0.0, "g_mean_mag": 10.0, "xpix": 10.0, "ypix": 0.0},
            {"source_id": 4, "ra_deg": 3.0, "dec_deg": 0.0, "g_mean_mag": 11.0, "xpix": 20.0, "ypix": 0.0},
        ]
    )
    _install_fake_et_coord(monkeypatch, frame)

    reference, stats = _build_reference_stars(
        _cfg(
            catalog_g_mag_min=7.5,
            reference_preselect_topk_per_detector=4,
            reference_isolation_radius_pix=2.0,
        ),
        object(),
        object(),
        DummyGaiaSourceFilter,
    )

    assert [star.catalog_id for star in reference] == [3, 4]
    assert DummyGaiaSourceFilter.calls == [{"g_mean_mag_max": 12.0, "g_mean_mag_min": 7.5}]
    assert stats["G1"]["num_reference_stars"] == 2
    assert stats["G1"]["num_reference_preselected"] == 4
    assert stats["G1"]["num_reference_isolated"] == 2
    assert stats["G1"]["catalog_g_mag_min"] == 7.5
    assert stats["G1"]["preselect_topk"] == 4
    assert stats["G1"]["isolation_radius_pix"] == 2.0
