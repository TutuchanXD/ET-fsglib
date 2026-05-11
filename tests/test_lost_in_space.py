import json

import numpy as np

from fsglib.match.lost_in_space import (
    build_lis_index_from_arrays,
    load_lis_index,
    query_pairs_by_angle,
    save_lis_index,
)
from fsglib.tools.build_lis_index import build_lis_index_from_gaia_csv


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
