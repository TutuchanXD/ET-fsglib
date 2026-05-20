from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from fsglib.attitude.solver import compute_residuals, quat_to_dcm, solve_quest
from fsglib.common.types import MatchedStar, MatchingResult, ObservedStar


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


def _arcsec_to_rad(value: float) -> float:
    return np.deg2rad(float(value) / 3600.0)


def _rms_ratio_is_ambiguous(
    *,
    best_rms_arcsec: float,
    candidate_rms_arcsec: float,
    ambiguity_ratio: float,
    epsilon_arcsec: float,
) -> bool:
    ratio = min(max(float(ambiguity_ratio), np.finfo(np.float64).eps), 1.0)
    epsilon = max(float(epsilon_arcsec), 0.0)
    best_rms = max(float(best_rms_arcsec), 0.0)
    candidate_rms = max(float(candidate_rms_arcsec), 0.0)
    if best_rms <= epsilon:
        return candidate_rms <= epsilon
    return candidate_rms <= (best_rms / ratio) + epsilon


class LostInSpaceMatcher:
    def __init__(self, index: LISIndex, cfg: dict):
        self.index = index
        self.cfg = cfg
        self.match_cfg = cfg.get("match", {})
        self.lis_cfg = self.match_cfg.get("lost_in_space", {})

    def _failure(self, observed: list[ObservedStar], reason: str, debug: dict | None = None) -> MatchingResult:
        payload = {
            "failure_reason": reason,
            "used_prior_attitude": False,
            "used_predicted_pixels": False,
            "num_observed_input": len(observed),
            "num_catalog_stars": int(len(self.index.catalog_ids)),
        }
        if debug:
            payload.update(debug)
        return MatchingResult(
            matched=[],
            unmatched_observed_ids=[obs.source_id for obs in observed],
            unmatched_catalog_ids=self.index.catalog_ids.tolist(),
            mode="lost_in_space",
            success=False,
            score=0.0,
            debug={"lost_in_space": payload},
        )

    def _selected_observed(self, observed: list[ObservedStar]) -> list[ObservedStar]:
        limit = int(self.lis_cfg.get("max_observed_stars", 20))
        ranked = sorted(
            observed,
            key=lambda obs: (-float(obs.snr), -float(obs.flux), str(obs.source_id)),
        )
        return ranked[:limit]

    def _candidate_catalog_maps(self, observed_subset: list[ObservedStar]) -> tuple[list[dict[int, int]], dict]:
        tol_rad = _arcsec_to_rad(self.lis_cfg.get("pair_angle_tolerance_arcsec", 120.0))
        max_candidates = int(self.lis_cfg.get("max_seed_candidates", 500))
        candidate_maps: list[dict[int, int]] = []
        debug = {
            "num_observed_seeds_tested": 0,
            "num_pair_queries": 0,
            "num_pair_candidate_edges": 0,
            "num_seed_maps": 0,
            "seed_candidate_limit_hit": False,
        }
        for obs_positions in combinations(range(len(observed_subset)), 4):
            debug["num_observed_seeds_tested"] += 1
            seed_obs = [observed_subset[pos] for pos in obs_positions]
            pair_candidates = []
            for left, right in combinations(range(4), 2):
                left_los = np.asarray(seed_obs[left].los_body, dtype=np.float64)
                right_los = np.asarray(seed_obs[right].los_body, dtype=np.float64)
                left_los = left_los / np.linalg.norm(left_los)
                right_los = right_los / np.linalg.norm(right_los)
                angle = float(np.arccos(np.clip(float(left_los @ right_los), -1.0, 1.0)))
                pairs = query_pairs_by_angle(self.index, angle, tol_rad)
                debug["num_pair_queries"] += 1
                debug["num_pair_candidate_edges"] += int(len(pairs))
                if len(pairs) == 0:
                    pair_candidates = []
                    break
                pair_candidates.append((left, right, pairs))
            if not pair_candidates:
                continue
            remaining_candidates = max_candidates - len(candidate_maps)
            if remaining_candidates <= 0:
                debug["seed_candidate_limit_hit"] = True
                return candidate_maps[:max_candidates], debug
            candidate_maps.extend(
                self._maps_from_pair_candidates(obs_positions, pair_candidates, remaining_candidates)
            )
            debug["num_seed_maps"] = int(len(candidate_maps))
            if len(candidate_maps) >= max_candidates:
                debug["seed_candidate_limit_hit"] = True
                return candidate_maps[:max_candidates], debug
        debug["num_seed_maps"] = int(len(candidate_maps))
        return candidate_maps, debug

    def _maps_from_pair_candidates(
        self,
        obs_positions: tuple[int, int, int, int],
        pair_candidates: list[tuple[int, int, np.ndarray]],
        max_candidates: int,
    ) -> list[dict[int, int]]:
        maps: list[dict[int, int]] = []

        def add_pair(
            mapping: dict[int, int],
            obs_left: int,
            obs_right: int,
            cat_left: int,
            cat_right: int,
        ) -> dict[int, int] | None:
            next_mapping = dict(mapping)
            for obs_pos, cat_pos in ((obs_left, int(cat_left)), (obs_right, int(cat_right))):
                if obs_pos in next_mapping and next_mapping[obs_pos] != cat_pos:
                    return None
                if cat_pos in next_mapping.values() and next_mapping.get(obs_pos) != cat_pos:
                    return None
                next_mapping[obs_pos] = cat_pos
            return next_mapping

        def recurse(edge_index: int, mapping: dict[int, int]) -> None:
            if len(maps) >= max_candidates:
                return
            if edge_index == len(pair_candidates):
                if len(mapping) == 4:
                    maps.append(mapping)
                return
            obs_left_local, obs_right_local, pairs = pair_candidates[edge_index]
            obs_left = obs_positions[obs_left_local]
            obs_right = obs_positions[obs_right_local]
            for cat_left, cat_right in pairs:
                direct = add_pair(mapping, obs_left, obs_right, int(cat_left), int(cat_right))
                if direct is not None:
                    recurse(edge_index + 1, direct)
                swapped = add_pair(mapping, obs_left, obs_right, int(cat_right), int(cat_left))
                if swapped is not None:
                    recurse(edge_index + 1, swapped)

        recurse(0, {})
        return maps

    def _seed_matches(
        self,
        observed_subset: list[ObservedStar],
        mapping: dict[int, int],
    ) -> list[MatchedStar]:
        return [
            MatchedStar(
                detector_id=observed_subset[obs_pos].detector_id,
                source_id=observed_subset[obs_pos].source_id,
                catalog_id=int(self.index.catalog_ids[cat_pos]),
                los_body=observed_subset[obs_pos].los_body,
                los_inertial=self.index.catalog_vectors[cat_pos],
                weight=max(float(observed_subset[obs_pos].weight), 1e-6),
                flags={
                    "match_mode": "lost_in_space",
                    "seed_match": True,
                    "sigma_angle_arcsec": observed_subset[obs_pos].sigma_angle_arcsec,
                    "weight_source": observed_subset[obs_pos].flags.get("weight_source"),
                    "weight_mode": observed_subset[obs_pos].flags.get("weight_mode"),
                },
            )
            for obs_pos, cat_pos in sorted(mapping.items())
        ]

    def _expand_candidate(self, observed_subset: list[ObservedStar], c_ib: np.ndarray) -> list[MatchedStar]:
        gate_arcsec = float(self.lis_cfg.get("expand_residual_gate_arcsec", 240.0))
        rotated_catalog = (c_ib @ self.index.catalog_vectors.T).T
        costs = np.empty((len(observed_subset), len(rotated_catalog)), dtype=np.float64)
        for obs_i, obs in enumerate(observed_subset):
            obs_vec = np.asarray(obs.los_body, dtype=np.float64)
            obs_vec = obs_vec / np.linalg.norm(obs_vec)
            dots = np.clip(rotated_catalog @ obs_vec, -1.0, 1.0)
            costs[obs_i, :] = np.rad2deg(np.arccos(dots)) * 3600.0

        row_indices, col_indices = linear_sum_assignment(costs)
        matched: list[MatchedStar] = []
        for row, col in zip(row_indices, col_indices):
            residual = float(costs[int(row), int(col)])
            if residual > gate_arcsec:
                continue
            obs = observed_subset[int(row)]
            matched.append(
                MatchedStar(
                    detector_id=obs.detector_id,
                    source_id=obs.source_id,
                    catalog_id=int(self.index.catalog_ids[int(col)]),
                    los_body=obs.los_body,
                    los_inertial=self.index.catalog_vectors[int(col)],
                    residual_arcsec=residual,
                    weight=max(float(obs.weight), 1e-6),
                    match_score=1.0 / (1.0 + residual),
                    flags={
                        "match_mode": "lost_in_space",
                        "residual_arcsec": residual,
                        "sigma_angle_arcsec": obs.sigma_angle_arcsec,
                        "weight_source": obs.flags.get("weight_source"),
                        "weight_mode": obs.flags.get("weight_mode"),
                    },
                )
            )
        matched.sort(key=lambda star: str(star.source_id))
        return matched

    def match(self, observed: list[ObservedStar]) -> MatchingResult:
        selected = self._selected_observed(observed)
        min_support = int(self.lis_cfg.get("min_support", self.match_cfg.get("validate_min_support", 3)))
        if len(selected) < max(4, min_support):
            return self._failure(observed, "not_enough_observations", {"num_observed_used": len(selected)})

        seed_gate_arcsec = float(self.lis_cfg.get("seed_residual_gate_arcsec", 180.0))
        candidates = []
        candidate_maps, seed_debug = self._candidate_catalog_maps(selected)
        seed_rejected_by_residual = 0
        expanded_rejected_by_support = 0
        for mapping in candidate_maps:
            seed_matches = self._seed_matches(selected, mapping)
            q_ib = solve_quest(seed_matches, self.cfg)
            c_ib = quat_to_dcm(q_ib)
            seed_residuals = compute_residuals(c_ib, seed_matches)
            if float(np.max(seed_residuals)) > seed_gate_arcsec:
                seed_rejected_by_residual += 1
                continue
            expanded = self._expand_candidate(selected, c_ib)
            if len(expanded) < min_support:
                expanded_rejected_by_support += 1
                continue
            residuals = compute_residuals(c_ib, expanded)
            rms = float(np.sqrt(np.mean(np.square(residuals))))
            rmax = float(np.max(residuals))
            candidates.append((len(expanded), -rms, -rmax, q_ib, c_ib, expanded, rms, rmax, mapping))

        if not candidates:
            reason = "no_pair_candidates" if seed_debug["num_seed_maps"] == 0 else "no_valid_seed"
            return self._failure(
                observed,
                reason,
                {
                    "num_observed_used": len(selected),
                    **seed_debug,
                    "num_seed_candidates_tested": int(len(candidate_maps)),
                    "num_seed_rejected_by_residual": seed_rejected_by_residual,
                    "num_expanded_rejected_by_support": expanded_rejected_by_support,
                },
            )

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best = candidates[0]
        ambiguity_ratio = float(self.lis_cfg.get("ambiguity_ratio", 0.98))
        ambiguity_epsilon_arcsec = float(self.lis_cfg.get("ambiguity_rms_epsilon_arcsec", 1.0e-6))
        ambiguous_candidate = None
        best_catalog_ids = {m.catalog_id for m in best[5]}
        for candidate in candidates[1:]:
            same_support = candidate[0] == best[0]
            different_catalogs = {m.catalog_id for m in candidate[5]} != best_catalog_ids
            close_rms = _rms_ratio_is_ambiguous(
                best_rms_arcsec=best[6],
                candidate_rms_arcsec=candidate[6],
                ambiguity_ratio=ambiguity_ratio,
                epsilon_arcsec=ambiguity_epsilon_arcsec,
            )
            if same_support and different_catalogs and close_rms:
                ambiguous_candidate = candidate
                break
        if ambiguous_candidate is not None:
            return self._failure(
                observed,
                "ambiguous_solution",
                {
                    "num_observed_used": len(selected),
                    **seed_debug,
                    "num_candidates_scored": len(candidates),
                    "num_seed_candidates_tested": int(len(candidate_maps)),
                    "num_ambiguous_candidates": 1,
                    "ambiguity_ratio": ambiguity_ratio,
                    "ambiguity_rms_epsilon_arcsec": ambiguity_epsilon_arcsec,
                    "best_rms_arcsec": best[6],
                    "second_best_rms_arcsec": ambiguous_candidate[6],
                },
            )

        support, _, _, q_ib, _c_ib, matched, rms, rmax, mapping = best
        score_scale = float(self.lis_cfg.get("score_residual_scale_arcsec", 1000.0))
        score = float(support) - rms / score_scale
        matched_source_ids = {star.source_id for star in matched}
        matched_catalog_ids = {star.catalog_id for star in matched}
        return MatchingResult(
            matched=matched,
            unmatched_observed_ids=[obs.source_id for obs in observed if obs.source_id not in matched_source_ids],
            unmatched_catalog_ids=[
                int(catalog_id) for catalog_id in self.index.catalog_ids if int(catalog_id) not in matched_catalog_ids
            ],
            mode="lost_in_space",
            success=True,
            score=score,
            debug={
                "lost_in_space": {
                    "failure_reason": None,
                    "used_prior_attitude": False,
                    "used_predicted_pixels": False,
                    "num_observed_input": len(observed),
                    "num_observed_used": len(selected),
                    "num_catalog_stars": int(len(self.index.catalog_ids)),
                    **seed_debug,
                    "num_seed_candidates_tested": int(len(candidate_maps)),
                    "num_seed_rejected_by_residual": seed_rejected_by_residual,
                    "num_expanded_rejected_by_support": expanded_rejected_by_support,
                    "num_candidates_scored": len(candidates),
                    "best_seed_observed_positions": sorted(mapping.keys()),
                    "best_seed_catalog_positions": [mapping[key] for key in sorted(mapping.keys())],
                    "best_q_ib": q_ib.tolist(),
                    "best_rms_arcsec": rms,
                    "best_max_residual_arcsec": rmax,
                    "num_matches": len(matched),
                }
            },
        )
