import json

import numpy as np
import pytest
from setuptools import find_packages

from fsglib.attitude.solver import quat_to_dcm
from fsglib.common.types import ObservedStar
from fsglib.match.lost_in_space import (
    LostInSpaceMatcher,
    _rms_ratio_is_ambiguous,
    build_lis_index_from_arrays,
    load_lis_index,
    query_pairs_by_angle,
    save_lis_index,
)
from fsglib.tools.build_lis_index import build_lis_index_from_gaia_csv


def test_tools_package_is_discovered_for_distribution():
    assert "fsglib.tools" in find_packages()


def _unit(x: float, y: float, z: float) -> np.ndarray:
    vec = np.array([x, y, z], dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _synthetic_vectors() -> np.ndarray:
    return np.asarray(
        [
            _unit(1.0, 0.0, 0.0),
            _unit(0.0, 1.0, 0.0),
            _unit(0.0, 0.0, 1.0),
            _unit(1.0, 1.0, 0.0),
            _unit(1.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )


def _random_quat(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def _spread_catalog_vectors() -> np.ndarray:
    raw = np.asarray(
        [
            [1.0, 0.1, 0.2],
            [0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0],
            [-0.9, 0.3, 0.2],
            [0.4, -0.8, 0.3],
            [0.3, 0.2, -0.9],
            [-0.5, -0.6, 0.4],
            [0.6, -0.2, -0.7],
        ],
        dtype=np.float64,
    )
    return raw / np.linalg.norm(raw, axis=1)[:, None]


def _observed_from_catalog(index, catalog_positions, q_ib, *, source_offset=1000):
    c_ib = quat_to_dcm(q_ib)
    observed = []
    for obs_pos, catalog_pos in enumerate(catalog_positions):
        body = c_ib @ index.catalog_vectors[catalog_pos]
        observed.append(
            ObservedStar(
                detector_id=0,
                source_id=source_offset + obs_pos,
                x=float("nan"),
                y=float("nan"),
                los_body=body,
                flux=1000.0 - obs_pos,
                snr=50.0 - obs_pos,
            )
        )
    return observed


def _perturb_los(vec: np.ndarray, arcsec: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unit_vec = np.asarray(vec, dtype=np.float64)
    unit_vec = unit_vec / np.linalg.norm(unit_vec)
    axis = rng.normal(size=3)
    axis -= axis.dot(unit_vec) * unit_vec
    axis /= np.linalg.norm(axis)
    return _unit(*(unit_vec + np.deg2rad(arcsec / 3600.0) * axis))


def _lis_test_cfg(**overrides):
    cfg = {
        "match": {
            "validate_min_support": 4,
            "lost_in_space": {
                "max_observed_stars": 20,
                "pair_angle_tolerance_arcsec": 1.0,
                "seed_residual_gate_arcsec": 1.0,
                "expand_residual_gate_arcsec": 1.0,
                "max_seed_candidates": 200,
                "ambiguity_ratio": 0.98,
            },
        },
        "attitude": {
            "min_stars_mathematical": 3,
            "min_stars_operational": 4,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 5.0,
        },
    }
    cfg["match"]["lost_in_space"].update(overrides)
    return cfg


def test_build_lis_index_from_arrays_is_deterministic_and_sorted():
    index = build_lis_index_from_arrays(
        catalog_ids=np.array([50, 20, 30, 10, 40], dtype=np.int64),
        vectors=_synthetic_vectors(),
        magnitudes=np.array([11.0, 9.5, 10.5, 9.5, 12.0], dtype=np.float64),
        config_snapshot={
            "epoch": 2026.0,
            "bandpass": "gaia_g",
            "filters": {"mag_limit": 12.0},
            "source": "unit-test",
        },
    )

    assert index.catalog_ids.tolist() == [10, 20, 30, 50, 40]
    assert index.catalog_mags.tolist() == [9.5, 9.5, 10.5, 11.0, 12.0]
    assert index.catalog_vectors.shape == (5, 3)
    assert np.allclose(np.linalg.norm(index.catalog_vectors, axis=1), 1.0)
    assert index.pair_indices.shape == (10, 2)
    assert np.all(np.diff(index.pair_angles_rad) >= 0.0)
    assert index.metadata["epoch"] == 2026.0
    assert index.metadata["bandpass"] == "gaia_g"
    assert index.metadata["filters"] == {"mag_limit": 12.0}
    assert index.metadata["config_snapshot"]["source"] == "unit-test"
    assert index.metadata["num_catalog_stars"] == 5
    assert index.metadata["num_pairs"] == 10
    assert len(index.checksum) == 64


def test_lost_in_space_matcher_recovers_catalog_ids_without_prior_or_pixels():
    index = build_lis_index_from_arrays(
        catalog_ids=np.arange(100, 108, dtype=np.int64),
        vectors=_spread_catalog_vectors(),
        magnitudes=np.linspace(9.0, 12.5, 8),
        config_snapshot={"epoch": 2026.0, "bandpass": "synthetic", "filters": {}},
    )
    q_ib = _random_quat()
    observed = _observed_from_catalog(index, [0, 2, 3, 5, 6], q_ib)

    result = LostInSpaceMatcher(index, _lis_test_cfg()).match(observed)

    assert result.success is True
    assert result.mode == "lost_in_space"
    assert [m.catalog_id for m in result.matched] == index.catalog_ids[[0, 2, 3, 5, 6]].tolist()
    assert all(m.flags["match_mode"] == "lost_in_space" for m in result.matched)
    assert result.debug["lost_in_space"]["used_prior_attitude"] is False
    assert result.debug["lost_in_space"]["used_predicted_pixels"] is False
    assert result.debug["lost_in_space"]["num_pair_queries"] > 0
    assert result.debug["lost_in_space"]["num_seed_maps"] > 0


def test_lost_in_space_matcher_handles_noise_missing_and_false_stars():
    index = build_lis_index_from_arrays(
        catalog_ids=np.arange(200, 212, dtype=np.int64),
        vectors=np.vstack([_spread_catalog_vectors(), -_spread_catalog_vectors()[:4]]),
        magnitudes=np.linspace(9.0, 13.0, 12),
        config_snapshot={"epoch": 2026.0, "bandpass": "synthetic", "filters": {}},
    )
    q_ib = _random_quat(seed=11)
    observed = _observed_from_catalog(index, [0, 1, 2, 4, 6, 8], q_ib)
    for i, obs in enumerate(observed):
        obs.los_body = _perturb_los(obs.los_body, arcsec=0.4, seed=30 + i)
    observed.append(
        ObservedStar(
            detector_id=0,
            source_id=9999,
            x=float("nan"),
            y=float("nan"),
            los_body=_unit(0.17, -0.41, 0.89),
            flux=10.0,
            snr=2.0,
        )
    )

    result = LostInSpaceMatcher(
        index,
        _lis_test_cfg(pair_angle_tolerance_arcsec=4.0, expand_residual_gate_arcsec=3.0),
    ).match(observed)

    assert result.success is True
    assert len(result.matched) >= 5
    assert 9999 in result.unmatched_observed_ids
    expected_catalog_ids = set(index.catalog_ids[[0, 1, 2, 4, 6, 8]].tolist())
    assert set(m.catalog_id for m in result.matched).issubset(expected_catalog_ids)
    assert result.debug["lost_in_space"]["best_rms_arcsec"] < 2.0


def test_lost_in_space_matcher_marks_ambiguous_repeated_geometry_unsuccessful():
    base_vectors = _spread_catalog_vectors()[:4]
    rotation = quat_to_dcm(_random_quat(seed=99))
    vectors = np.vstack([base_vectors, (rotation @ base_vectors.T).T])
    index = build_lis_index_from_arrays(
        catalog_ids=np.arange(300, 308, dtype=np.int64),
        vectors=vectors,
        magnitudes=np.linspace(9.0, 12.0, 8),
        config_snapshot={"epoch": 2026.0, "bandpass": "synthetic", "filters": {}},
    )
    observed = _observed_from_catalog(index, [0, 1, 2, 3], _random_quat(seed=17))

    result = LostInSpaceMatcher(index, _lis_test_cfg(ambiguity_ratio=1.0)).match(observed)

    assert result.success is False
    assert result.debug["lost_in_space"]["failure_reason"] == "ambiguous_solution"
    assert result.debug["lost_in_space"]["num_ambiguous_candidates"] >= 1


def test_lis_ambiguity_ratio_uses_relative_rms_margin_with_epsilon():
    assert _rms_ratio_is_ambiguous(
        best_rms_arcsec=10.0,
        candidate_rms_arcsec=10.1,
        ambiguity_ratio=0.98,
        epsilon_arcsec=1.0e-6,
    )
    assert not _rms_ratio_is_ambiguous(
        best_rms_arcsec=0.01,
        candidate_rms_arcsec=0.5,
        ambiguity_ratio=0.98,
        epsilon_arcsec=1.0e-6,
    )
    assert _rms_ratio_is_ambiguous(
        best_rms_arcsec=0.0,
        candidate_rms_arcsec=5.0e-7,
        ambiguity_ratio=1.0,
        epsilon_arcsec=1.0e-6,
    )
    assert not _rms_ratio_is_ambiguous(
        best_rms_arcsec=0.0,
        candidate_rms_arcsec=2.0e-6,
        ambiguity_ratio=1.0,
        epsilon_arcsec=1.0e-6,
    )


def test_lost_in_space_seed_generation_passes_remaining_candidate_budget(monkeypatch):
    index = build_lis_index_from_arrays(
        catalog_ids=np.arange(400, 408, dtype=np.int64),
        vectors=_spread_catalog_vectors(),
        magnitudes=np.linspace(9.0, 12.5, 8),
        config_snapshot={"epoch": 2026.0, "bandpass": "synthetic", "filters": {}},
    )
    observed = _observed_from_catalog(index, [0, 1, 2, 3, 4], _random_quat(seed=22))
    matcher = LostInSpaceMatcher(index, _lis_test_cfg(max_seed_candidates=3))
    received_budgets = []

    def fake_maps_from_pair_candidates(obs_positions, pair_candidates, max_candidates):
        received_budgets.append(max_candidates)
        return [
            {obs_positions[0]: 0, obs_positions[1]: 1, obs_positions[2]: 2, obs_positions[3]: 3}
            for _ in range(min(2, max_candidates))
        ]

    monkeypatch.setattr(matcher, "_maps_from_pair_candidates", fake_maps_from_pair_candidates)

    maps, debug = matcher._candidate_catalog_maps(observed)

    assert len(maps) == 3
    assert received_budgets[:2] == [3, 1]
    assert debug["seed_candidate_limit_hit"] is True


def test_lis_index_round_trip_and_angle_query(tmp_path):
    index = build_lis_index_from_arrays(
        catalog_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        vectors=_synthetic_vectors()[:4],
        magnitudes=np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
        config_snapshot={"epoch": 2026.0, "bandpass": "gaia_g", "filters": {}},
    )
    output_path = tmp_path / "fixture.lis_index.npz"

    save_lis_index(index, output_path)
    loaded = load_lis_index(output_path)

    assert loaded.checksum == index.checksum
    assert loaded.metadata == index.metadata
    assert loaded.catalog_ids.tolist() == index.catalog_ids.tolist()
    assert np.allclose(loaded.catalog_vectors, index.catalog_vectors)
    assert np.allclose(loaded.pair_angles_rad, index.pair_angles_rad)

    angle_90 = np.pi / 2.0
    pairs = query_pairs_by_angle(loaded, angle_90, tolerance_rad=1.0e-12)
    pair_catalog_ids = {
        tuple(loaded.catalog_ids[pair].tolist())
        for pair in pairs
    }
    assert pair_catalog_ids == {(1, 2), (1, 3), (2, 3), (3, 4)}


def test_lis_index_load_rejects_checksum_mismatch(tmp_path):
    index = build_lis_index_from_arrays(
        catalog_ids=np.array([1, 2, 3], dtype=np.int64),
        vectors=_synthetic_vectors()[:3],
        magnitudes=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        config_snapshot={"epoch": 2026.0, "bandpass": "gaia_g", "filters": {}},
    )
    good_path = tmp_path / "good.lis_index.npz"
    bad_path = tmp_path / "bad.lis_index.npz"
    save_lis_index(index, good_path)

    with np.load(good_path, allow_pickle=False) as data:
        np.savez_compressed(
            bad_path,
            catalog_ids=data["catalog_ids"],
            catalog_vectors=data["catalog_vectors"],
            catalog_mags=data["catalog_mags"],
            pair_indices=data["pair_indices"],
            pair_angles_rad=data["pair_angles_rad"] + 1.0e-4,
            k_m=data["k_m"],
            k_b=data["k_b"],
            k_vec=data["k_vec"],
            metadata_json=data["metadata_json"],
            checksum=data["checksum"],
        )

    with pytest.raises(ValueError, match="checksum"):
        load_lis_index(bad_path)

    loaded_without_verification = load_lis_index(bad_path, verify_checksum=False)
    assert loaded_without_verification.checksum == index.checksum


def test_build_lis_index_from_arrays_enforces_catalog_size_guard():
    with pytest.raises(ValueError, match="max_catalog_stars"):
        build_lis_index_from_arrays(
            catalog_ids=np.array([1, 2, 3], dtype=np.int64),
            vectors=_synthetic_vectors()[:3],
            magnitudes=np.array([10.0, 11.0, 12.0], dtype=np.float64),
            config_snapshot={"epoch": 2026.0, "bandpass": "gaia_g", "filters": {}},
            max_catalog_stars=2,
        )


def test_build_lis_index_from_gaia_csv_filters_and_records_provenance(tmp_path):
    gaia_root = tmp_path / "gaia"
    gaia_root.mkdir()
    (gaia_root / "healpix_n05_nested_00001.csv").write_text(
        "\n".join(
            [
                "source_id,ra,dec,g_mean_mag",
                "100,0.0,0.0,9.0",
                "200,0.00001,0.0,10.0",
                "300,90.0,0.0,11.0",
            ]
        ),
        encoding="utf-8",
    )
    (gaia_root / "healpix_n05_nested_00002.csv").write_text(
        "\n".join(
            [
                "source_id,ra,dec,g_mean_mag",
                "400,0.0,90.0,12.5",
                "500,180.0,0.0,15.5",
            ]
        ),
        encoding="utf-8",
    )

    index = build_lis_index_from_gaia_csv(
        gaia_root=gaia_root,
        mag_limit=13.0,
        epoch=2026.0,
        bandpass="gaia_g",
        isolation_radius_arcsec=1.0,
        max_files=None,
    )

    assert index.catalog_ids.tolist() == [100, 300, 400]
    assert index.metadata["gaia_root"] == str(gaia_root)
    assert index.metadata["epoch"] == 2026.0
    assert index.metadata["bandpass"] == "gaia_g"
    assert index.metadata["filters"] == {
        "mag_limit": 13.0,
        "isolation_radius_arcsec": 1.0,
        "neighbor_policy": "drop_fainter",
    }
    assert index.metadata["num_catalog_stars"] == 3
    assert index.metadata["num_pairs"] == 3
    assert json.loads(index.metadata["config_snapshot_json"])["bandpass"] == "gaia_g"


def test_build_lis_index_from_gaia_csv_validates_input_root(tmp_path):
    missing_root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Gaia root"):
        build_lis_index_from_gaia_csv(
            gaia_root=missing_root,
            mag_limit=13.0,
            epoch=2026.0,
            bandpass="gaia_g",
            isolation_radius_arcsec=1.0,
        )

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="No Gaia HEALPix CSV"):
        build_lis_index_from_gaia_csv(
            gaia_root=empty_root,
            mag_limit=13.0,
            epoch=2026.0,
            bandpass="gaia_g",
            isolation_radius_arcsec=1.0,
        )


def test_build_lis_index_from_gaia_csv_enforces_catalog_size_guard(tmp_path):
    gaia_root = tmp_path / "gaia"
    gaia_root.mkdir()
    (gaia_root / "healpix_n05_nested_00001.csv").write_text(
        "\n".join(
            [
                "source_id,ra,dec,g_mean_mag",
                "100,0.0,0.0,9.0",
                "200,90.0,0.0,10.0",
                "300,0.0,90.0,11.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_catalog_stars"):
        build_lis_index_from_gaia_csv(
            gaia_root=gaia_root,
            mag_limit=13.0,
            epoch=2026.0,
            bandpass="gaia_g",
            isolation_radius_arcsec=0.0,
            max_catalog_stars=2,
        )
