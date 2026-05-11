from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LISIndex:
    catalog_ids: np.ndarray
    catalog_vectors: np.ndarray
    catalog_mags: np.ndarray
    pair_indices: np.ndarray
    pair_angles_rad: np.ndarray
    k_m: float
    k_b: float
    k_vec: np.ndarray
    metadata: dict[str, Any]
    checksum: str


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("vectors must be non-zero")
    return arr / norms[:, None]


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metadata_with_counts(
    config_snapshot: dict[str, Any],
    num_catalog_stars: int,
    num_pairs: int,
) -> dict[str, Any]:
    metadata = dict(config_snapshot)
    metadata.setdefault("epoch", config_snapshot.get("epoch"))
    metadata.setdefault("bandpass", config_snapshot.get("bandpass", "gaia_g"))
    metadata.setdefault("filters", config_snapshot.get("filters", {}))
    metadata["config_snapshot"] = dict(config_snapshot)
    metadata["num_catalog_stars"] = int(num_catalog_stars)
    metadata["num_pairs"] = int(num_pairs)
    return metadata


def _build_pairs(catalog_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    angles: list[float] = []
    for i in range(len(catalog_vectors)):
        for j in range(i + 1, len(catalog_vectors)):
            dot = float(np.dot(catalog_vectors[i], catalog_vectors[j]))
            angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
            pairs.append((i, j))
            angles.append(angle)

    if not pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    pair_indices = np.asarray(pairs, dtype=np.int64)
    pair_angles = np.asarray(angles, dtype=np.float64)
    order = np.lexsort((pair_indices[:, 1], pair_indices[:, 0], pair_angles))
    return pair_indices[order], pair_angles[order]


def generate_kvector(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    sorted_vals = np.asarray(values, dtype=np.float64)
    n = int(sorted_vals.size)
    if n == 0:
        return 0.0, 0.0, np.zeros(1, dtype=np.int64)

    y_min = float(sorted_vals[0])
    y_max = float(sorted_vals[-1])
    if np.isclose(y_min, y_max):
        return 0.0, 0.0, np.arange(n + 1, dtype=np.int64)

    k_m = float((n - 1) / (y_max - y_min))
    k_b = float(-k_m * y_min)
    k_vec = np.zeros(n + 1, dtype=np.int64)
    for i, value in enumerate(sorted_vals):
        k = int(np.floor(k_m * float(value) + k_b))
        k = max(0, min(k, n - 1))
        if k_vec[k] == 0 and k != 0:
            k_vec[k] = i
    for i in range(1, len(k_vec)):
        if k_vec[i] == 0:
            k_vec[i] = k_vec[i - 1]
    k_vec[-1] = n
    return k_m, k_b, k_vec


def _checksum_payload(
    catalog_ids: np.ndarray,
    catalog_vectors: np.ndarray,
    catalog_mags: np.ndarray,
    pair_indices: np.ndarray,
    pair_angles_rad: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(catalog_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(catalog_vectors, dtype=np.float64).tobytes())
    digest.update(np.asarray(catalog_mags, dtype=np.float64).tobytes())
    digest.update(np.asarray(pair_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(pair_angles_rad, dtype=np.float64).tobytes())
    digest.update(_canonical_json(metadata).encode("utf-8"))
    return digest.hexdigest()


def build_lis_index_from_arrays(
    *,
    catalog_ids: np.ndarray,
    vectors: np.ndarray,
    magnitudes: np.ndarray,
    config_snapshot: dict[str, Any],
    max_catalog_stars: int | None = None,
) -> LISIndex:
    ids = np.asarray(catalog_ids, dtype=np.int64)
    mags = np.asarray(magnitudes, dtype=np.float64)
    unit_vectors = _normalize_vectors(vectors)
    if ids.ndim != 1 or mags.ndim != 1 or len(ids) != len(unit_vectors) or len(mags) != len(unit_vectors):
        raise ValueError("catalog_ids, vectors, and magnitudes must have matching length")
    if max_catalog_stars is not None and len(ids) > int(max_catalog_stars):
        raise ValueError(
            f"LIS index selected {len(ids)} catalog stars, exceeding max_catalog_stars={int(max_catalog_stars)}"
        )

    order = np.lexsort((ids, mags))
    sorted_ids = ids[order]
    sorted_vectors = unit_vectors[order]
    sorted_mags = mags[order]
    pair_indices, pair_angles_rad = _build_pairs(sorted_vectors)
    k_m, k_b, k_vec = generate_kvector(pair_angles_rad)
    metadata = _metadata_with_counts(config_snapshot, len(sorted_ids), len(pair_indices))
    checksum = _checksum_payload(
        sorted_ids,
        sorted_vectors,
        sorted_mags,
        pair_indices,
        pair_angles_rad,
        metadata,
    )
    return LISIndex(
        catalog_ids=sorted_ids,
        catalog_vectors=sorted_vectors,
        catalog_mags=sorted_mags,
        pair_indices=pair_indices,
        pair_angles_rad=pair_angles_rad,
        k_m=k_m,
        k_b=k_b,
        k_vec=k_vec,
        metadata=metadata,
        checksum=checksum,
    )


def save_lis_index(index: LISIndex, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        catalog_ids=index.catalog_ids,
        catalog_vectors=index.catalog_vectors,
        catalog_mags=index.catalog_mags,
        pair_indices=index.pair_indices,
        pair_angles_rad=index.pair_angles_rad,
        k_m=np.asarray([index.k_m], dtype=np.float64),
        k_b=np.asarray([index.k_b], dtype=np.float64),
        k_vec=index.k_vec,
        metadata_json=np.asarray(_canonical_json(index.metadata)),
        checksum=np.asarray(index.checksum),
    )


def load_lis_index(input_path: str | Path, *, verify_checksum: bool = True) -> LISIndex:
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        catalog_ids = np.asarray(data["catalog_ids"], dtype=np.int64)
        catalog_vectors = np.asarray(data["catalog_vectors"], dtype=np.float64)
        catalog_mags = np.asarray(data["catalog_mags"], dtype=np.float64)
        pair_indices = np.asarray(data["pair_indices"], dtype=np.int64)
        pair_angles_rad = np.asarray(data["pair_angles_rad"], dtype=np.float64)
        checksum = str(np.asarray(data["checksum"]).item())
        if verify_checksum:
            expected_checksum = _checksum_payload(
                catalog_ids,
                catalog_vectors,
                catalog_mags,
                pair_indices,
                pair_angles_rad,
                metadata,
            )
            if checksum != expected_checksum:
                raise ValueError(
                    f"LIS index checksum mismatch for {input_path}: stored={checksum} computed={expected_checksum}"
                )
        return LISIndex(
            catalog_ids=catalog_ids,
            catalog_vectors=catalog_vectors,
            catalog_mags=catalog_mags,
            pair_indices=pair_indices,
            pair_angles_rad=pair_angles_rad,
            k_m=float(np.asarray(data["k_m"])[0]),
            k_b=float(np.asarray(data["k_b"])[0]),
            k_vec=np.asarray(data["k_vec"], dtype=np.int64),
            metadata=metadata,
            checksum=checksum,
        )


def query_pairs_by_angle(index: LISIndex, angle_rad: float, tolerance_rad: float) -> np.ndarray:
    if len(index.pair_angles_rad) == 0:
        return np.empty((0, 2), dtype=np.int64)

    lo = max(0.0, float(angle_rad) - float(tolerance_rad))
    hi = float(angle_rad) + float(tolerance_rad)
    if index.k_m == 0.0:
        start = 0
        stop = len(index.pair_angles_rad)
    else:
        n = len(index.k_vec) - 1
        k_min = int(np.floor(index.k_m * lo + index.k_b)) - 2
        k_max = int(np.ceil(index.k_m * hi + index.k_b)) + 2
        k_min = max(0, min(k_min, n))
        k_max = max(0, min(k_max, n))
        start = int(index.k_vec[k_min])
        stop = int(index.k_vec[k_max])

    angle_window = index.pair_angles_rad[start:stop]
    exact_start = start + int(np.searchsorted(angle_window, lo, side="left"))
    exact_stop = start + int(np.searchsorted(angle_window, hi, side="right"))
    return index.pair_indices[exact_start:exact_stop]
