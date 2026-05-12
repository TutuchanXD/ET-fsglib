from __future__ import annotations

import itertools
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from fsglib.attitude.solver import dcm_to_quat
from fsglib.common.types import MatchedStar, MatchingResult, ObservedStar
from fsglib.ephemeris.types import ReferenceStar


ARCSEC_PER_RAD = 206264.80624709636


@dataclass(frozen=True)
class ReferencePair:
    i: int
    j: int
    angle_rad: float


@dataclass
class LocalPairIndex:
    pairs: list[ReferencePair]
    angles_rad: np.ndarray
    angle_by_key: dict[tuple[int, int], float]


def _reference_cache_key(reference_stars: list[ReferenceStar]) -> tuple:
    return tuple(
        (
            int(ref.catalog_id),
            float(ref.time_s),
            tuple(round(float(value), 15) for value in np.asarray(ref.los_inertial, dtype=np.float64)),
            tuple(sorted(ref.detector_ids_visible, key=str)),
        )
        for ref in reference_stars
    )


@dataclass
class LocalPyramidCache:
    max_pair_indices: int = 8
    max_query_entries: int = 4096
    pair_index_by_key: OrderedDict[tuple, LocalPairIndex] = field(default_factory=OrderedDict)
    query_pairs_by_key: OrderedDict[tuple[int, int, float], list[ReferencePair]] = field(default_factory=OrderedDict)
    pair_index_hits: int = 0
    pair_index_misses: int = 0
    pair_index_build_time_s: float = 0.0
    query_hits: int = 0
    query_misses: int = 0
    query_time_s: float = 0.0

    def reset_stats(self) -> None:
        self.pair_index_hits = 0
        self.pair_index_misses = 0
        self.pair_index_build_time_s = 0.0
        self.query_hits = 0
        self.query_misses = 0
        self.query_time_s = 0.0

    def clear(self) -> None:
        self.pair_index_by_key.clear()
        self.query_pairs_by_key.clear()
        self.reset_stats()

    def _drop_query_entries_for_pair_index(self, pair_index: LocalPairIndex) -> None:
        pair_index_id = id(pair_index)
        stale_keys = [
            key
            for key in self.query_pairs_by_key
            if key[0] == pair_index_id
        ]
        for key in stale_keys:
            self.query_pairs_by_key.pop(key, None)

    def _enforce_pair_index_limit(self) -> None:
        if self.max_pair_indices <= 0:
            self.clear()
            return
        while len(self.pair_index_by_key) > self.max_pair_indices:
            _, evicted_pair_index = self.pair_index_by_key.popitem(last=False)
            self._drop_query_entries_for_pair_index(evicted_pair_index)

    def _enforce_query_limit(self) -> None:
        if self.max_query_entries <= 0:
            self.query_pairs_by_key.clear()
            return
        while len(self.query_pairs_by_key) > self.max_query_entries:
            self.query_pairs_by_key.popitem(last=False)

    def get_pair_index(self, reference_stars: list[ReferenceStar]) -> LocalPairIndex:
        key = _reference_cache_key(reference_stars)
        if key in self.pair_index_by_key:
            self.pair_index_hits += 1
            self.pair_index_by_key.move_to_end(key)
            return self.pair_index_by_key[key]

        self.pair_index_misses += 1
        start = perf_counter()
        pair_index = _build_local_pair_index(reference_stars)
        self.pair_index_build_time_s += perf_counter() - start
        if self.max_pair_indices > 0:
            self.pair_index_by_key[key] = pair_index
            self._enforce_pair_index_limit()
        return pair_index

    def query_pairs(self, pair_index: LocalPairIndex, angle_rad: float, tolerance_rad: float) -> list[ReferencePair]:
        start = perf_counter()
        bin_width = max(float(tolerance_rad) / 4.0, 1e-15)
        bin_index = int(np.floor(float(angle_rad) / bin_width))
        rounded_tolerance = round(float(tolerance_rad), 15)
        key = (id(pair_index), bin_index, rounded_tolerance)
        if key in self.query_pairs_by_key:
            self.query_hits += 1
            self.query_pairs_by_key.move_to_end(key)
            candidates = self.query_pairs_by_key[key]
        else:
            self.query_misses += 1
            bin_start = float(bin_index) * bin_width
            bin_end = bin_start + bin_width
            lo = max(0.0, bin_start - float(tolerance_rad))
            hi = bin_end + float(tolerance_rad)
            start_index = int(np.searchsorted(pair_index.angles_rad, lo, side="left"))
            stop_index = int(np.searchsorted(pair_index.angles_rad, hi, side="right"))
            candidates = pair_index.pairs[start_index:stop_index]
            if self.max_query_entries > 0:
                self.query_pairs_by_key[key] = candidates
                self._enforce_query_limit()

        exact = [
            pair
            for pair in candidates
            if abs(float(pair.angle_rad) - float(angle_rad)) <= float(tolerance_rad)
        ]
        self.query_time_s += perf_counter() - start
        return exact

    def pair_index_debug(self) -> dict[str, float | int]:
        return {
            "hits": self.pair_index_hits,
            "misses": self.pair_index_misses,
            "build_time_s": self.pair_index_build_time_s,
        }

    def angle_query_debug(self) -> dict[str, float | int]:
        return {
            "hits": self.query_hits,
            "misses": self.query_misses,
            "query_time_s": self.query_time_s,
        }


@dataclass
class SeedSolution:
    observed_indices: tuple[int, int, int, int]
    reference_indices: tuple[int, int, int, int]
    c_ib: np.ndarray
    q_ib: np.ndarray
    seed_rms_arcsec: float
    seed_max_arcsec: float
    pair_angle_residuals_arcsec: list[float]
    seed_scope: str
    detector_ids: tuple[Any, ...]


ExpansionEdge = tuple[float, int, int, float, float | None, tuple[float, float] | None, float]
HypothesisPayload = tuple[float, tuple[int, float, float], SeedSolution, list[MatchedStar], dict[str, Any]]


def _cfg_value(cfg: dict, name: str, default, pyramid_mode: str | None = None):
    match_cfg = cfg.get("match", {})
    pyramid_cfg = match_cfg.get("local_pyramid", {})
    if pyramid_mode is not None and isinstance(pyramid_cfg, dict):
        mode_name = f"{pyramid_mode}_{name}"
        if mode_name in pyramid_cfg:
            return pyramid_cfg[mode_name]
    if isinstance(pyramid_cfg, dict) and name in pyramid_cfg:
        return pyramid_cfg[name]
    if pyramid_mode is not None:
        flat_mode_name = f"pyramid_{pyramid_mode}_{name}"
        if flat_mode_name in match_cfg:
            return match_cfg[flat_mode_name]
    flat_name = f"pyramid_{name}"
    if flat_name in match_cfg:
        return match_cfg[flat_name]
    return default


def _arcsec_to_rad(value: float) -> float:
    return float(value) / ARCSEC_PER_RAD


def _angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    av = np.array(a, dtype=np.float64, copy=True)
    bv = np.array(b, dtype=np.float64, copy=True)
    av /= np.linalg.norm(av)
    bv /= np.linalg.norm(bv)
    return float(np.arccos(np.clip(float(av @ bv), -1.0, 1.0)))


def _angle_arcsec(a: np.ndarray, b: np.ndarray) -> float:
    return _angle_rad(a, b) * ARCSEC_PER_RAD


def _solve_wahba_svd(body_vectors: list[np.ndarray], inertial_vectors: list[np.ndarray]) -> np.ndarray:
    B = np.zeros((3, 3), dtype=np.float64)
    for body, inertial in zip(body_vectors, inertial_vectors):
        w = np.array(body, dtype=np.float64, copy=True)
        v = np.array(inertial, dtype=np.float64, copy=True)
        w /= np.linalg.norm(w)
        v /= np.linalg.norm(v)
        B += np.outer(w, v)
    U, _, Vt = np.linalg.svd(B)
    c_ib = U @ Vt
    if np.linalg.det(c_ib) < 0:
        U[:, -1] *= -1.0
        c_ib = U @ Vt
    return c_ib


def _select_observed(
    observed_stars: list[ObservedStar],
    cfg: dict,
    pyramid_mode: str | None = None,
) -> tuple[list[ObservedStar], list[int]]:
    max_observed = int(_cfg_value(cfg, "max_observed_stars", 40, pyramid_mode) or 0)
    indexed = list(enumerate(observed_stars))
    indexed.sort(
        key=lambda item: (
            float(item[1].snr),
            float(item[1].flux),
            -float(item[0]),
        ),
        reverse=True,
    )
    if max_observed > 0:
        indexed = indexed[:max_observed]
    indexed.sort(key=lambda item: item[0])
    return [star for _, star in indexed], [index for index, _ in indexed]


def _reference_sort_key(item: tuple[int, ReferenceStar]) -> tuple[float, float, int]:
    index, star = item
    mag = np.inf if star.mag_g is None else float(star.mag_g)
    return mag, -float(star.weight_hint), int(index)


def _observed_brightness_ranks(observed_stars: list[ObservedStar]) -> dict[int, int]:
    indexed = list(enumerate(observed_stars))
    indexed.sort(
        key=lambda item: (
            float(item[1].snr),
            float(item[1].flux),
            -float(item[0]),
        ),
        reverse=True,
    )
    return {index: rank for rank, (index, _) in enumerate(indexed)}


def _reference_brightness_ranks(reference_stars: list[ReferenceStar]) -> dict[int, int]:
    indexed = list(enumerate(reference_stars))
    indexed.sort(key=_reference_sort_key)
    return {index: rank for rank, (index, _) in enumerate(indexed)}


def _photometric_rank_penalty(
    obs_index: int,
    ref_index: int,
    observed_ranks: dict[int, int],
    reference_ranks: dict[int, int],
    rank_count: int,
) -> float:
    if rank_count <= 1:
        return 0.0
    obs_rank = observed_ranks[obs_index]
    ref_rank = reference_ranks[ref_index]
    return float(abs(obs_rank - ref_rank) / float(rank_count - 1))


def _select_reference(
    reference_stars: list[ReferenceStar],
    cfg: dict,
    pyramid_mode: str | None = None,
) -> tuple[list[ReferenceStar], list[int]]:
    max_reference = int(_cfg_value(cfg, "max_reference_stars", 300, pyramid_mode) or 0)
    indexed = list(enumerate(reference_stars))
    indexed.sort(key=_reference_sort_key)
    if max_reference > 0:
        indexed = indexed[:max_reference]
    indexed.sort(key=lambda item: item[0])
    return [star for _, star in indexed], [index for index, _ in indexed]


def _build_local_pair_index(reference_stars: list[ReferenceStar]) -> LocalPairIndex:
    pairs: list[ReferencePair] = []
    for i, j in itertools.combinations(range(len(reference_stars)), 2):
        angle = _angle_rad(reference_stars[i].los_inertial, reference_stars[j].los_inertial)
        pairs.append(ReferencePair(i=i, j=j, angle_rad=angle))
    pairs.sort(key=lambda pair: pair.angle_rad)
    return LocalPairIndex(
        pairs=pairs,
        angles_rad=np.asarray([pair.angle_rad for pair in pairs], dtype=np.float64),
        angle_by_key={(pair.i, pair.j): pair.angle_rad for pair in pairs},
    )


def _query_pairs(pair_index: LocalPairIndex, angle_rad: float, tolerance_rad: float) -> list[ReferencePair]:
    lo = max(0.0, float(angle_rad) - float(tolerance_rad))
    hi = float(angle_rad) + float(tolerance_rad)
    start = int(np.searchsorted(pair_index.angles_rad, lo, side="left"))
    stop = int(np.searchsorted(pair_index.angles_rad, hi, side="right"))
    return pair_index.pairs[start:stop]


def _iter_observed_pyramids(
    observed_stars: list[ObservedStar],
    scope: str,
    max_pyramids: int,
):
    yielded = 0
    if scope == "single_detector":
        by_detector: dict[Any, list[int]] = defaultdict(list)
        for index, star in enumerate(observed_stars):
            by_detector[star.detector_id].append(index)
        for detector_indices in by_detector.values():
            if len(detector_indices) < 4:
                continue
            for combo in itertools.combinations(detector_indices, 4):
                yield combo
                yielded += 1
                if max_pyramids > 0 and yielded >= max_pyramids:
                    return
        return

    for combo in itertools.combinations(range(len(observed_stars)), 4):
        if len({observed_stars[index].detector_id for index in combo}) == 1:
            continue
        yield combo
        yielded += 1
        if max_pyramids > 0 and yielded >= max_pyramids:
            return


def _seed_pair_angles(stars: list[ObservedStar], seed: tuple[int, int, int, int]) -> dict[tuple[int, int], float]:
    angles: dict[tuple[int, int], float] = {}
    for local_i, local_j in itertools.combinations(range(4), 2):
        obs_i = seed[local_i]
        obs_j = seed[local_j]
        angles[(local_i, local_j)] = _angle_rad(stars[obs_i].los_body, stars[obs_j].los_body)
    return angles


def _seed_edges_within_limits(
    pair_angles: dict[tuple[int, int], float],
    cfg: dict,
    pyramid_mode: str | None = None,
) -> bool:
    min_edge = _arcsec_to_rad(float(_cfg_value(cfg, "min_edge_arcsec", 0.0, pyramid_mode)))
    max_edge_deg = float(_cfg_value(cfg, "max_edge_deg", np.inf, pyramid_mode))
    max_edge = np.deg2rad(max_edge_deg) if np.isfinite(max_edge_deg) else np.inf
    return all(min_edge <= angle <= max_edge for angle in pair_angles.values())


def _adjacency(candidate_pairs: list[ReferencePair]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for pair in candidate_pairs:
        adj[pair.i].add(pair.j)
        adj[pair.j].add(pair.i)
    return adj


def _find_reference_pyramid_candidates(
    pair_index: LocalPairIndex,
    pair_angles: dict[tuple[int, int], float],
    tolerance_rad: float,
    max_candidates: int,
    cache: LocalPyramidCache | None = None,
) -> list[tuple[tuple[int, int, int, int], list[float]]]:
    edge_candidates = {
        edge: _query_pairs(pair_index, angle, tolerance_rad)
        if cache is None
        else cache.query_pairs(pair_index, angle, tolerance_rad)
        for edge, angle in pair_angles.items()
    }
    if any(len(candidates) == 0 for candidates in edge_candidates.values()):
        return []

    adj_02 = _adjacency(edge_candidates[(0, 2)])
    adj_03 = _adjacency(edge_candidates[(0, 3)])
    adj_12 = _adjacency(edge_candidates[(1, 2)])
    adj_13 = _adjacency(edge_candidates[(1, 3)])
    adj_23 = _adjacency(edge_candidates[(2, 3)])

    candidates: list[tuple[tuple[int, int, int, int], list[float]]] = []
    for pair_01 in edge_candidates[(0, 1)]:
        oriented_01 = ((pair_01.i, pair_01.j), (pair_01.j, pair_01.i))
        for r0, r1 in oriented_01:
            possible_r2 = adj_02.get(r0, set()).intersection(adj_12.get(r1, set()))
            for r2 in possible_r2:
                if r2 in {r0, r1}:
                    continue
                possible_r3 = (
                    adj_03.get(r0, set())
                    .intersection(adj_13.get(r1, set()))
                    .intersection(adj_23.get(r2, set()))
                )
                for r3 in possible_r3:
                    if r3 in {r0, r1, r2}:
                        continue
                    ref_seed = (r0, r1, r2, r3)
                    residuals = []
                    for local_i, local_j in itertools.combinations(range(4), 2):
                        ri = ref_seed[local_i]
                        rj = ref_seed[local_j]
                        ref_angle = _reference_pair_angle(pair_index, ri, rj)
                        residuals.append(abs(ref_angle - pair_angles[(local_i, local_j)]) * ARCSEC_PER_RAD)
                    candidates.append((ref_seed, residuals))
                    if max_candidates > 0 and len(candidates) >= max_candidates:
                        return candidates
    return candidates


def _reference_pair_angle(pair_index: LocalPairIndex, i: int, j: int) -> float:
    if i == j:
        return 0.0
    lo, hi = (i, j) if i < j else (j, i)
    return pair_index.angle_by_key[(lo, hi)]


def _score_seed(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    obs_seed: tuple[int, int, int, int],
    ref_seed: tuple[int, int, int, int],
    pair_residuals_arcsec: list[float],
    scope: str,
) -> SeedSolution:
    body_vectors = [observed_stars[obs_index].los_body for obs_index in obs_seed]
    inertial_vectors = [reference_stars[ref_index].los_inertial for ref_index in ref_seed]
    c_ib = _solve_wahba_svd(body_vectors, inertial_vectors)
    q_ib = dcm_to_quat(c_ib)
    residuals = [
        _angle_arcsec(c_ib @ np.asarray(inertial, dtype=np.float64), body)
        for body, inertial in zip(body_vectors, inertial_vectors)
    ]
    detector_ids = tuple(sorted({observed_stars[index].detector_id for index in obs_seed}, key=str))
    return SeedSolution(
        observed_indices=obs_seed,
        reference_indices=ref_seed,
        c_ib=c_ib,
        q_ib=q_ib,
        seed_rms_arcsec=float(np.sqrt(np.mean(np.square(residuals)))),
        seed_max_arcsec=float(np.max(residuals)),
        pair_angle_residuals_arcsec=pair_residuals_arcsec,
        seed_scope=scope,
        detector_ids=detector_ids,
    )


def _predicted_pixel_residual(
    obs: ObservedStar,
    ref: ReferenceStar,
) -> tuple[float | None, tuple[float, float] | None]:
    if obs.detector_id not in ref.predicted_xy:
        return None, None
    if not ref.predicted_valid.get(obs.detector_id, False):
        return None, None
    predicted_xy = ref.predicted_xy[obs.detector_id]
    dx = float(obs.x) - float(predicted_xy[0])
    dy = float(obs.y) - float(predicted_xy[1])
    return float(np.hypot(dx, dy)), predicted_xy


def _build_expansion_edges(
    seed: SeedSolution,
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
    pyramid_mode: str | None = None,
) -> tuple[list[ExpansionEdge], dict[str, Any]]:
    angular_gate = float(_cfg_value(cfg, "expand_angular_gate_arcsec", 120.0, pyramid_mode))
    pixel_gate = float(
        _cfg_value(
            cfg,
            "expand_pixel_gate_pix",
            cfg.get("match", {}).get("validate_max_residual_pix", 25.0),
            pyramid_mode,
        )
    )
    expansion_policy = str(_cfg_value(cfg, "expansion_policy", "predicted_xy", pyramid_mode))
    geometry_only_allowed = bool(_cfg_value(cfg, "geometry_only_allowed", False, pyramid_mode))
    if expansion_policy == "seed_attitude_only" and not geometry_only_allowed:
        expansion_policy = "predicted_xy"
    angular_sigma = max(angular_gate, 1.0)
    pixel_sigma = max(pixel_gate, 1.0)
    seed_consistency_penalty = float(_cfg_value(cfg, "seed_consistency_penalty", 1.0e-6, pyramid_mode))
    seed_pairs = set(zip(seed.observed_indices, seed.reference_indices))
    photometric_rank_weight = float(_cfg_value(cfg, "photometric_rank_weight", 0.0, pyramid_mode))
    observed_ranks = _observed_brightness_ranks(observed_stars)
    reference_ranks = _reference_brightness_ranks(reference_stars)
    rank_count = max(len(observed_stars), len(reference_stars))
    audit: dict[str, Any] = {
        "expansion_policy": expansion_policy,
        "geometry_only_allowed": geometry_only_allowed,
        "pixel_gate_pix": pixel_gate,
        "angular_gate_arcsec": angular_gate,
        "seed_consistency_penalty": seed_consistency_penalty,
        "photometric_rank_weight": photometric_rank_weight,
        "num_missing_predicted_xy": 0,
        "num_pixel_gate_rejects": 0,
        "num_angular_gate_rejects": 0,
        "num_edges_before_assignment": 0,
        "num_edges_after_assignment": 0,
    }

    edges: list[ExpansionEdge] = []
    for obs_index, obs in enumerate(observed_stars):
        for ref_index, ref in enumerate(reference_stars):
            pixel_residual = None
            predicted_xy = None
            if expansion_policy == "predicted_xy":
                pixel_residual, predicted_xy = _predicted_pixel_residual(obs, ref)
                if pixel_residual is None:
                    audit["num_missing_predicted_xy"] += 1
                    continue
                if pixel_residual > pixel_gate:
                    audit["num_pixel_gate_rejects"] += 1
                    continue

            ref_body = seed.c_ib @ np.asarray(ref.los_inertial, dtype=np.float64)
            angular_residual = _angle_arcsec(ref_body, obs.los_body)
            if angular_residual > angular_gate:
                audit["num_angular_gate_rejects"] += 1
                continue

            if expansion_policy == "predicted_xy":
                cost = (angular_residual / angular_sigma) + (pixel_residual / pixel_sigma)
            elif expansion_policy == "seed_attitude_only":
                cost = angular_residual / angular_sigma
            else:
                raise ValueError(f"unsupported local pyramid expansion_policy: {expansion_policy}")
            photometric_penalty = _photometric_rank_penalty(
                obs_index,
                ref_index,
                observed_ranks,
                reference_ranks,
                rank_count,
            )
            cost += photometric_rank_weight * photometric_penalty
            if (obs_index, ref_index) not in seed_pairs:
                cost += seed_consistency_penalty
            edges.append(
                (
                    float(cost),
                    obs_index,
                    ref_index,
                    float(angular_residual),
                    pixel_residual,
                    predicted_xy,
                    photometric_penalty,
                )
            )
    audit["num_edges_before_assignment"] = len(edges)
    return edges, audit


def _assign_expansion_edges(
    edges: list[ExpansionEdge],
    observed_stars: list[ObservedStar],
) -> list[ExpansionEdge]:
    by_detector: dict[Any, list[ExpansionEdge]] = defaultdict(list)
    for edge in edges:
        by_detector[observed_stars[edge[1]].detector_id].append(edge)

    selected: list[ExpansionEdge] = []
    for detector_edges in by_detector.values():
        obs_indices = sorted({edge[1] for edge in detector_edges})
        ref_indices = sorted({edge[2] for edge in detector_edges})
        obs_pos = {obs_index: pos for pos, obs_index in enumerate(obs_indices)}
        ref_pos = {ref_index: pos for pos, ref_index in enumerate(ref_indices)}
        sentinel = max(edge[0] for edge in detector_edges) + 1.0
        cost = np.full((len(obs_indices), len(ref_indices)), sentinel, dtype=np.float64)
        edge_by_pos: dict[tuple[int, int], ExpansionEdge] = {}
        for edge in detector_edges:
            row = obs_pos[edge[1]]
            col = ref_pos[edge[2]]
            if edge[0] < cost[row, col]:
                cost[row, col] = edge[0]
                edge_by_pos[(row, col)] = edge
        row_indices, col_indices = linear_sum_assignment(cost)
        for row, col in zip(row_indices, col_indices):
            edge = edge_by_pos.get((int(row), int(col)))
            if edge is not None:
                selected.append(edge)

    kept: list[ExpansionEdge] = []
    used_observed: set[int] = set()
    for edge in sorted(selected, key=lambda item: item[0]):
        obs_index = edge[1]
        if obs_index in used_observed:
            continue
        used_observed.add(obs_index)
        kept.append(edge)

    kept.sort(key=lambda item: item[1])
    return kept


def _build_matched_stars(
    seed: SeedSolution,
    assigned_edges: list[ExpansionEdge],
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
) -> list[MatchedStar]:
    matched: list[MatchedStar] = []
    for cost, obs_index, ref_index, angular_residual, pixel_residual, predicted_xy, photometric_penalty in assigned_edges:
        obs = observed_stars[obs_index]
        ref = reference_stars[ref_index]
        matched.append(
            MatchedStar(
                detector_id=obs.detector_id,
                source_id=obs.source_id,
                catalog_id=ref.catalog_id,
                los_body=obs.los_body,
                los_inertial=ref.los_inertial,
                residual_arcsec=angular_residual,
                weight=max(float(obs.snr), 1.0),
                match_score=1.0 / (1.0 + float(cost)),
                flags={
                    "match_mode": "local_pyramid",
                    "seed_scope": seed.seed_scope,
                    "seed_detector_ids": seed.detector_ids,
                    "seed_rms_arcsec": seed.seed_rms_arcsec,
                    "residual_arcsec": angular_residual,
                    "residual_pix": pixel_residual,
                    "assignment_cost": cost,
                    "photometric_rank_penalty": photometric_penalty,
                    "observed_xy": (obs.x, obs.y),
                    "predicted_xy": predicted_xy,
                },
            )
        )
    return matched


def _per_detector_residuals(
    matched: list[MatchedStar],
    cfg: dict | None = None,
    pyramid_mode: str | None = None,
) -> dict[str, dict[str, float | int | str]]:
    residuals: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for star in matched:
        predicted_xy = star.flags.get("predicted_xy")
        observed_xy = star.flags.get("observed_xy")
        if predicted_xy is None or observed_xy is None:
            continue
        dx = float(observed_xy[0]) - float(predicted_xy[0])
        dy = float(observed_xy[1]) - float(predicted_xy[1])
        residuals[str(star.detector_id)].append((dx, dy, float(np.hypot(dx, dy))))

    payload: dict[str, dict[str, float | int | str]] = {}
    warn_gate = np.inf
    mean_reject_gate = np.inf
    rms_reject_gate = np.inf
    max_reject_gate = np.inf
    if cfg is not None:
        warn_gate = float(_cfg_value(cfg, "detector_mean_warn_pix", np.inf, pyramid_mode))
        mean_reject_gate = float(_cfg_value(cfg, "detector_mean_reject_pix", np.inf, pyramid_mode))
        rms_reject_gate = float(_cfg_value(cfg, "detector_rms_reject_pix", np.inf, pyramid_mode))
        max_reject_gate = float(_cfg_value(cfg, "detector_max_reject_pix", np.inf, pyramid_mode))

    for detector_id, values in residuals.items():
        arr = np.asarray(values, dtype=np.float64)
        mean_dx = float(np.mean(arr[:, 0]))
        mean_dy = float(np.mean(arr[:, 1]))
        mean_norm = float(np.hypot(mean_dx, mean_dy))
        rms = float(np.sqrt(np.mean(np.square(arr[:, 2]))))
        max_residual = float(np.max(arr[:, 2]))
        status = "ok"
        if mean_norm > mean_reject_gate or rms > rms_reject_gate or max_residual > max_reject_gate:
            status = "reject"
        elif mean_norm > warn_gate:
            status = "warn"
        payload[detector_id] = {
            "num_matches": int(arr.shape[0]),
            "mean_dx_pix": mean_dx,
            "mean_dy_pix": mean_dy,
            "mean_norm_pix": mean_norm,
            "rms_pix": rms,
            "max_pix": max_residual,
            "status": status,
        }
    return payload


def _detector_residual_rejection(
    per_detector_residuals: dict[str, dict[str, float | int | str]],
    cfg: dict,
    seed_scope: str,
    pyramid_mode: str | None = None,
) -> tuple[bool, list[str]]:
    reject_ids = [
        detector_id
        for detector_id, payload in per_detector_residuals.items()
        if payload.get("status") == "reject"
    ]
    if reject_ids:
        return True, reject_ids

    reject_mixed_warning = bool(_cfg_value(cfg, "mixed_detector_reject_on_detector_warning", False, pyramid_mode))
    if seed_scope == "mixed_detector" and reject_mixed_warning:
        warn_ids = [
            detector_id
            for detector_id, payload in per_detector_residuals.items()
            if payload.get("status") == "warn"
        ]
        if warn_ids:
            return True, warn_ids

    return False, []


def _build_seed_debug(
    seed: SeedSolution,
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
) -> dict[str, Any]:
    return {
        "scope": seed.seed_scope,
        "detector_ids": list(seed.detector_ids),
        "observed_indices": list(seed.observed_indices),
        "reference_indices": list(seed.reference_indices),
        "observed_source_ids": [
            observed_stars[index].source_id
            for index in seed.observed_indices
        ],
        "reference_catalog_ids": [
            reference_stars[index].catalog_id
            for index in seed.reference_indices
        ],
        "pair_angle_residuals_arcsec": list(seed.pair_angle_residuals_arcsec),
        "rms_arcsec": seed.seed_rms_arcsec,
        "max_arcsec": seed.seed_max_arcsec,
    }


def _hypothesis_mapping_key(matched: list[MatchedStar]) -> tuple[tuple[Any, Any], ...]:
    return tuple((match.source_id, match.catalog_id) for match in matched)


def _mean_assignment_cost(matched: list[MatchedStar]) -> float:
    costs = [
        float(match.flags["assignment_cost"])
        for match in matched
        if match.flags.get("assignment_cost") is not None
    ]
    return float(np.mean(costs)) if costs else 0.0


def _hypothesis_score(matched: list[MatchedStar], seed: SeedSolution, seed_rms_gate: float) -> float:
    seed_scale = max(float(seed_rms_gate), 1.0)
    return (float(len(matched)) * 1000.0) - _mean_assignment_cost(matched) - (seed.seed_rms_arcsec / seed_scale)


def _build_result(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    matched: list[MatchedStar],
    cfg: dict,
    debug: dict,
) -> MatchingResult:
    matched_source_ids = {star.source_id for star in matched}
    matched_catalog_ids = {star.catalog_id for star in matched}
    min_support = int(cfg.get("match", {}).get("validate_min_support", 3))
    residuals_pix = [
        float(star.flags["residual_pix"])
        for star in matched
        if star.flags.get("residual_pix") is not None
    ]
    per_detector_residuals = debug.get("best_per_detector_residuals")
    if per_detector_residuals is None:
        per_detector_residuals = _per_detector_residuals(matched, cfg)
    debug = {
        **debug,
        "num_matched": len(matched),
        "expanded_mean_residual_pix": float(np.mean(residuals_pix)) if residuals_pix else None,
        "expanded_rms_residual_pix": float(np.sqrt(np.mean(np.square(residuals_pix)))) if residuals_pix else None,
        "per_detector_residuals": per_detector_residuals,
        "stars_per_detector": {
            detector_id: int(payload["num_matches"])
            for detector_id, payload in per_detector_residuals.items()
        },
    }
    return MatchingResult(
        matched=matched,
        unmatched_observed_ids=[
            obs.source_id for obs in observed_stars if obs.source_id not in matched_source_ids
        ],
        unmatched_catalog_ids=[
            ref.catalog_id for ref in reference_stars if ref.catalog_id not in matched_catalog_ids
        ],
        mode=cfg.get("project", {}).get("mode", "init"),
        success=len(matched) >= min_support,
        score=float(len(matched)),
        debug=debug,
    )


def match_local_pyramid(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
    cache: LocalPyramidCache | None = None,
    pyramid_mode: str | None = None,
) -> MatchingResult:
    if pyramid_mode is None:
        pyramid_mode = cfg.get("match", {}).get("mode", cfg.get("project", {}).get("mode", "init"))
    if cache is not None:
        cache.reset_stats()
    empty_pair_index_cache = {"hits": 0, "misses": 0, "build_time_s": 0.0}
    empty_angle_query_cache = {"hits": 0, "misses": 0, "query_time_s": 0.0}
    debug = {
        "algorithm": cfg.get("match", {}).get("algorithm", "local_pyramid"),
        "selected_strategy": "local_pyramid",
        "pyramid_mode": pyramid_mode,
        "pyramid_enabled": True,
        "num_observed_input": len(observed_stars),
        "num_reference_input": len(reference_stars),
        "num_observed_pyramids_tested": 0,
        "num_reference_pyramid_candidates": 0,
        "num_seed_attitudes_scored": 0,
        "best_seed_scope": None,
        "best_seed_detector_ids": None,
        "best_seed_rms_arcsec": None,
        "best_seed": None,
        "best_expansion": None,
        "best_expanded_matches": 0,
        "num_valid_seed_hypotheses": 0,
        "ambiguity_margin": None,
        "ambiguous": False,
        "second_best_seed": None,
        "seed_rejection_counters": {
            "edge_limit": 0,
            "seed_rms_gate": 0,
            "seed_max_gate": 0,
            "under_min_expanded": 0,
            "detector_residual": 0,
        },
        "rejection_reason": None,
        "fallback_strategy": None,
        "pair_index_cache": empty_pair_index_cache,
        "angle_query_cache": empty_angle_query_cache,
    }

    if len(observed_stars) < 4:
        debug["rejection_reason"] = "not_enough_observed_stars"
        return _build_result(observed_stars, reference_stars, [], cfg, debug)
    if len(reference_stars) < 4:
        debug["rejection_reason"] = "not_enough_reference_stars"
        return _build_result(observed_stars, reference_stars, [], cfg, debug)

    selected_observed, selected_observed_indices = _select_observed(observed_stars, cfg, pyramid_mode)
    selected_reference, selected_reference_indices = _select_reference(reference_stars, cfg, pyramid_mode)
    debug["num_observed_used"] = len(selected_observed)
    debug["num_reference_used"] = len(selected_reference)

    if cache is None:
        pair_index_start = perf_counter()
        pair_index = _build_local_pair_index(selected_reference)
        debug["pair_index_cache"] = {
            "hits": 0,
            "misses": 1,
            "build_time_s": perf_counter() - pair_index_start,
        }
    else:
        pair_index = cache.get_pair_index(selected_reference)
        debug["pair_index_cache"] = cache.pair_index_debug()
    debug["num_reference_pairs"] = len(pair_index.pairs)

    scopes = list(_cfg_value(cfg, "seed_scopes", ["single_detector", "mixed_detector"], pyramid_mode))
    debug["pyramid_seed_scope_order"] = scopes
    max_observed_pyramids = int(_cfg_value(cfg, "max_observed_pyramids", 5000, pyramid_mode) or 0)
    max_candidates_per_seed = int(_cfg_value(cfg, "max_candidates_per_observed_seed", 200, pyramid_mode) or 0)
    max_seed_attitudes = int(_cfg_value(cfg, "max_seed_attitudes", 2000, pyramid_mode) or 0)
    seed_rms_gate = float(_cfg_value(cfg, "seed_rms_gate_arcsec", 60.0, pyramid_mode))
    seed_max_gate = float(_cfg_value(cfg, "seed_max_gate_arcsec", 180.0, pyramid_mode))
    min_expanded = int(_cfg_value(cfg, "min_expanded_matches", cfg.get("match", {}).get("validate_min_support", 3), pyramid_mode))
    ambiguity_min_score_margin = float(_cfg_value(cfg, "ambiguity_min_score_margin", 1.0, pyramid_mode))

    best_payload = None
    valid_hypotheses: list[HypothesisPayload] = []
    best_rejected_detector_payload: HypothesisPayload | None = None
    best_rejected_detector_residuals: dict[str, dict[str, float | int | str]] | None = None
    best_rejected_detector_ids: list[str] = []
    for scope in scopes:
        tol_key = "pair_angle_tol_arcsec_mixed_detector" if scope == "mixed_detector" else "pair_angle_tol_arcsec_single_detector"
        tolerance = _arcsec_to_rad(float(_cfg_value(cfg, tol_key, 300.0 if scope == "mixed_detector" else 120.0, pyramid_mode)))
        scope_hypotheses_by_mapping: dict[tuple[tuple[int, int], ...], HypothesisPayload] = {}
        for obs_seed in _iter_observed_pyramids(selected_observed, scope, max_observed_pyramids):
            debug["num_observed_pyramids_tested"] += 1
            pair_angles = _seed_pair_angles(selected_observed, obs_seed)
            if not _seed_edges_within_limits(pair_angles, cfg, pyramid_mode):
                debug["seed_rejection_counters"]["edge_limit"] += 1
                continue
            ref_candidates = _find_reference_pyramid_candidates(
                pair_index,
                pair_angles,
                tolerance,
                max_candidates_per_seed,
                cache=cache,
            )
            debug["num_reference_pyramid_candidates"] += len(ref_candidates)
            for selected_ref_seed, pair_residuals in ref_candidates:
                if max_seed_attitudes > 0 and debug["num_seed_attitudes_scored"] >= max_seed_attitudes:
                    break
                debug["num_seed_attitudes_scored"] += 1
                original_ref_seed = tuple(selected_reference_indices[index] for index in selected_ref_seed)
                seed = _score_seed(
                    selected_observed,
                    selected_reference,
                    obs_seed,
                    selected_ref_seed,
                    pair_residuals,
                    scope,
                )
                if seed.seed_rms_arcsec > seed_rms_gate:
                    debug["seed_rejection_counters"]["seed_rms_gate"] += 1
                    continue
                if seed.seed_max_arcsec > seed_max_gate:
                    debug["seed_rejection_counters"]["seed_max_gate"] += 1
                    continue
                seed.observed_indices = tuple(selected_observed_indices[index] for index in obs_seed)
                seed.reference_indices = original_ref_seed
                edges, expansion_debug = _build_expansion_edges(seed, observed_stars, reference_stars, cfg, pyramid_mode)
                assigned = _assign_expansion_edges(edges, observed_stars)
                expansion_debug["num_edges_after_assignment"] = len(assigned)
                matched = _build_matched_stars(seed, assigned, observed_stars, reference_stars)
                if len(matched) < min_expanded:
                    debug["seed_rejection_counters"]["under_min_expanded"] += 1
                    continue
                per_detector_residuals = _per_detector_residuals(matched, cfg, pyramid_mode)
                detector_rejected, detector_ids = _detector_residual_rejection(
                    per_detector_residuals,
                    cfg,
                    seed.seed_scope,
                    pyramid_mode,
                )
                expansion_debug["per_detector_residuals"] = per_detector_residuals
                expansion_debug["detector_residual_reject_ids"] = detector_ids
                residuals = [
                    float(star.flags["residual_pix"])
                    for star in matched
                    if star.flags.get("residual_pix") is not None
                ]
                rms_pix = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
                rank = (len(matched), -rms_pix, -seed.seed_rms_arcsec)
                score = _hypothesis_score(matched, seed, seed_rms_gate)
                payload = (score, rank, seed, matched, expansion_debug)
                if detector_rejected:
                    debug["seed_rejection_counters"]["detector_residual"] += 1
                    if best_rejected_detector_payload is None or (score, rank) > (
                        best_rejected_detector_payload[0],
                        best_rejected_detector_payload[1],
                    ):
                        best_rejected_detector_payload = payload
                        best_rejected_detector_residuals = per_detector_residuals
                        best_rejected_detector_ids = detector_ids
                    continue
                mapping_key = _hypothesis_mapping_key(matched)
                existing = scope_hypotheses_by_mapping.get(mapping_key)
                if existing is None or (score, rank) > (existing[0], existing[1]):
                    scope_hypotheses_by_mapping[mapping_key] = payload
            if max_seed_attitudes > 0 and debug["num_seed_attitudes_scored"] >= max_seed_attitudes:
                break
        if scope_hypotheses_by_mapping:
            valid_hypotheses = sorted(
                scope_hypotheses_by_mapping.values(),
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
            best_payload = valid_hypotheses[0]
            break

    if best_payload is None:
        if best_rejected_detector_payload is not None:
            _, _, rejected_seed, rejected_matched, rejected_expansion = best_rejected_detector_payload
            debug["rejection_reason"] = "detector_residual_reject"
            debug["best_seed_scope"] = rejected_seed.seed_scope
            debug["best_seed_detector_ids"] = list(rejected_seed.detector_ids)
            debug["best_seed_rms_arcsec"] = rejected_seed.seed_rms_arcsec
            debug["best_seed"] = _build_seed_debug(rejected_seed, observed_stars, reference_stars)
            debug["best_expansion"] = rejected_expansion
            debug["best_expanded_matches"] = len(rejected_matched)
            debug["best_per_detector_residuals"] = best_rejected_detector_residuals
            debug["detector_residual_reject_ids"] = best_rejected_detector_ids
            if cache is not None:
                debug["angle_query_cache"] = cache.angle_query_debug()
            return _build_result(observed_stars, reference_stars, [], cfg, debug)
        debug["rejection_reason"] = "no_valid_pyramid_seed"
        if cache is not None:
            debug["angle_query_cache"] = cache.angle_query_debug()
        return _build_result(observed_stars, reference_stars, [], cfg, debug)

    debug["num_valid_seed_hypotheses"] = len(valid_hypotheses)
    if len(valid_hypotheses) > 1:
        second_payload = valid_hypotheses[1]
        margin = max(0.0, best_payload[0] - second_payload[0])
        debug["ambiguity_margin"] = margin
        debug["second_best_seed"] = _build_seed_debug(second_payload[2], observed_stars, reference_stars)
        if ambiguity_min_score_margin > 0.0 and margin < ambiguity_min_score_margin:
            debug["ambiguous"] = True
            debug["rejection_reason"] = "ambiguous_seed_hypotheses"
            best_seed = best_payload[2]
            debug["best_seed_scope"] = best_seed.seed_scope
            debug["best_seed_detector_ids"] = list(best_seed.detector_ids)
            debug["best_seed_rms_arcsec"] = best_seed.seed_rms_arcsec
            debug["best_seed"] = _build_seed_debug(best_seed, observed_stars, reference_stars)
            debug["best_expansion"] = best_payload[4]
            debug["best_expanded_matches"] = len(best_payload[3])
            if cache is not None:
                debug["angle_query_cache"] = cache.angle_query_debug()
            return _build_result(observed_stars, reference_stars, [], cfg, debug)

    _, _, best_seed, matched, best_expansion = best_payload
    debug["best_seed_scope"] = best_seed.seed_scope
    debug["best_seed_detector_ids"] = list(best_seed.detector_ids)
    debug["best_seed_rms_arcsec"] = best_seed.seed_rms_arcsec
    debug["best_seed"] = _build_seed_debug(best_seed, observed_stars, reference_stars)
    debug["best_expansion"] = best_expansion
    debug["best_expanded_matches"] = len(matched)
    debug["best_per_detector_residuals"] = best_expansion.get("per_detector_residuals")
    if cache is not None:
        debug["angle_query_cache"] = cache.angle_query_debug()
    return _build_result(observed_stars, reference_stars, matched, cfg, debug)
