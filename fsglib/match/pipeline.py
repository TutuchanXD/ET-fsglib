from scipy.optimize import linear_sum_assignment

from fsglib.common.types import MatchedStar, MatchingContext, MatchingResult, ObservedStar
from fsglib.ephemeris.types import ReferenceStar


CandidateEdge = tuple[float, int, int, ObservedStar, ReferenceStar]


def _build_matched_star(obs: ObservedStar, ref: ReferenceStar, dist2: float) -> MatchedStar:
    residual_pix = float(dist2 ** 0.5)
    return MatchedStar(
        detector_id=obs.detector_id,
        source_id=obs.source_id,
        catalog_id=ref.catalog_id,
        los_body=obs.los_body,
        los_inertial=ref.los_inertial,
        residual_arcsec=None,
        weight=max(obs.snr, 1.0),
        match_score=1.0 / (1.0 + dist2),
        flags={
            "match_mode": "predicted_position",
            "residual_pix": residual_pix,
            "observed_xy": (obs.x, obs.y),
            "predicted_xy": ref.predicted_xy.get(obs.detector_id),
        },
    )


def _collect_candidate_edges(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    max_dist2: float,
) -> list[CandidateEdge]:
    candidate_edges: list[CandidateEdge] = []

    for obs_index, obs in enumerate(observed_stars):
        for ref_index, ref in enumerate(reference_stars):
            if obs.detector_id not in ref.predicted_xy:
                continue
            if not ref.predicted_valid.get(obs.detector_id, False):
                continue

            pred_x, pred_y = ref.predicted_xy[obs.detector_id]
            dx = obs.x - pred_x
            dy = obs.y - pred_y
            dist2 = dx * dx + dy * dy
            if dist2 <= max_dist2:
                candidate_edges.append((dist2, obs_index, ref_index, obs, ref))

    return candidate_edges


def _build_matching_result(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
    matched: list[MatchedStar],
    *,
    unique_assignment_enabled: bool,
    num_candidate_edges: int,
) -> MatchingResult:
    matched_source_ids = {m.source_id for m in matched}
    matched_catalog_ids = {m.catalog_id for m in matched}
    stars_per_detector: dict[str, int] = {}
    residuals_pix: list[float] = []
    for matched_star in matched:
        key = str(matched_star.detector_id)
        stars_per_detector[key] = stars_per_detector.get(key, 0) + 1
        residual_pix = matched_star.flags.get("residual_pix")
        if residual_pix is not None:
            residuals_pix.append(float(residual_pix))

    mean_residual_pix = float(sum(residuals_pix) / len(residuals_pix)) if residuals_pix else None

    return MatchingResult(
        matched=matched,
        unmatched_observed_ids=[
            obs.source_id for obs in observed_stars if obs.source_id not in matched_source_ids
        ],
        unmatched_catalog_ids=[
            ref.catalog_id for ref in reference_stars if ref.catalog_id not in matched_catalog_ids
        ],
        mode="tracking",
        success=len(matched) >= int(cfg["match"].get("validate_min_support", 3)),
        score=float(len(matched)),
        debug={
            "selected_strategy": "predicted_position",
            "num_reference_stars": len(reference_stars),
            "num_candidate_edges": num_candidate_edges,
            "unique_assignment_enabled": unique_assignment_enabled,
            "num_unique_matches": len(matched) if unique_assignment_enabled else None,
            "mean_residual_pix": mean_residual_pix,
            "stars_per_detector": stars_per_detector,
        },
    )


def _associate_nearest_unique_by_distance(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> MatchingResult:
    max_dist2 = float(cfg["match"].get("validate_max_residual_pix", 25.0)) ** 2
    candidate_edges = _collect_candidate_edges(observed_stars, reference_stars, max_dist2)
    edges_by_detector: dict[object, list[CandidateEdge]] = {}
    for edge in candidate_edges:
        obs = edge[3]
        edges_by_detector.setdefault(obs.detector_id, []).append(edge)

    selected_edges: list[CandidateEdge] = []
    for detector_edges in edges_by_detector.values():
        obs_indices = sorted({edge[1] for edge in detector_edges})
        ref_indices = sorted({edge[2] for edge in detector_edges})
        obs_pos = {obs_index: pos for pos, obs_index in enumerate(obs_indices)}
        ref_pos = {ref_index: pos for pos, ref_index in enumerate(ref_indices)}
        sentinel = max_dist2 + 1.0
        cost = [[sentinel for _ in ref_indices] for _ in obs_indices]
        edge_by_pos: dict[tuple[int, int], CandidateEdge] = {}

        for edge in detector_edges:
            dist2, obs_index, ref_index, _, _ = edge
            row = obs_pos[obs_index]
            col = ref_pos[ref_index]
            if dist2 < cost[row][col]:
                cost[row][col] = dist2
                edge_by_pos[(row, col)] = edge

        row_indices, col_indices = linear_sum_assignment(cost)
        for row, col in zip(row_indices, col_indices):
            edge = edge_by_pos.get((int(row), int(col)))
            if edge is not None:
                selected_edges.append(edge)

    selected_edges.sort(key=lambda edge: edge[1])
    matched = [_build_matched_star(obs, ref, dist2) for dist2, _, _, obs, ref in selected_edges]
    return _build_matching_result(
        observed_stars,
        reference_stars,
        cfg,
        matched,
        unique_assignment_enabled=True,
        num_candidate_edges=len(candidate_edges),
    )


def associate_nearest(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> MatchingResult:
    if cfg["match"].get("enforce_unique_assignment", False):
        return _associate_nearest_unique_by_distance(observed_stars, reference_stars, cfg)

    matched: list[MatchedStar] = []
    num_candidate_edges = 0
    max_dist2 = float(cfg["match"].get("validate_max_residual_pix", 25.0)) ** 2

    for obs in observed_stars:
        best = None
        best_dist = None

        for ref in reference_stars:
            if obs.detector_id not in ref.predicted_xy:
                continue
            if not ref.predicted_valid.get(obs.detector_id, False):
                continue

            pred_x, pred_y = ref.predicted_xy[obs.detector_id]
            dx = obs.x - pred_x
            dy = obs.y - pred_y
            dist2 = dx * dx + dy * dy

            if dist2 > max_dist2:
                continue

            num_candidate_edges += 1
            if best_dist is None or dist2 < best_dist:
                best_dist = dist2
                best = ref

        if best is None:
            continue

        matched.append(_build_matched_star(obs, best, best_dist))

    return _build_matching_result(
        observed_stars,
        reference_stars,
        cfg,
        matched=matched,
        unique_assignment_enabled=False,
        num_candidate_edges=num_candidate_edges,
    )


def _match_with_triangle(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> list[MatchedStar]:
    if len(observed_stars) < 3:
        return []

    from fsglib.match.triangle import TriangleMatcher

    gsc_path = cfg["match"].get("triangle_gsc_path")
    if not gsc_path:
        return []

    matcher = TriangleMatcher(
        gsc_path=gsc_path,
        angle_tol_deg=cfg["match"].get("triangle_tolerance_deg", 0.005),
        max_stars=cfg["match"].get("triangle_max_stars", 15),
    )
    matched = matcher.match(observed_stars)
    if not matched:
        return []

    if reference_stars:
        allowed_catalog_ids = {ref.catalog_id for ref in reference_stars}
        matched = [m for m in matched if m.catalog_id in allowed_catalog_ids]

    return matched


def match_stars(
    ctx: MatchingContext,
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> MatchingResult:
    algorithm = cfg["match"].get("algorithm", "local_triangle")
    local_result = associate_nearest(ctx.observed_stars, reference_stars, cfg)
    local_matches = local_result.matched
    triangle_matches: list[MatchedStar] = []

    if algorithm in {"triangle", "local_triangle"}:
        triangle_matches = _match_with_triangle(ctx.observed_stars, reference_stars, cfg)

    matched = local_matches
    selected_strategy = "predicted_position"
    if len(triangle_matches) > len(local_matches):
        matched = triangle_matches
        selected_strategy = "triangle"

    matched_source_ids = {m.source_id for m in matched}
    matched_catalog_ids = {m.catalog_id for m in matched}

    result = MatchingResult(
        matched=matched,
        unmatched_observed_ids=[
            obs.source_id for obs in ctx.observed_stars if obs.source_id not in matched_source_ids
        ],
        unmatched_catalog_ids=[
            ref.catalog_id for ref in reference_stars if ref.catalog_id not in matched_catalog_ids
        ],
        mode=ctx.mode,
        success=len(matched) >= int(cfg["match"].get("validate_min_support", 3)),
        score=float(len(matched)),
        debug={
            "algorithm": algorithm,
            "selected_strategy": selected_strategy,
            "num_matched": len(matched),
            "num_local_matches": len(local_matches),
            "num_triangle_matches": len(triangle_matches),
            "num_reference_stars": len(reference_stars),
            "num_candidate_edges": local_result.debug.get("num_candidate_edges", 0),
            "unique_assignment_enabled": local_result.debug.get("unique_assignment_enabled", False),
            "num_unique_matches": local_result.debug.get("num_unique_matches"),
            "mean_residual_pix": local_result.debug.get("mean_residual_pix"),
            "stars_per_detector": local_result.debug.get("stars_per_detector", {}),
        },
    )
    return result


def validate_match_hypothesis(
    matching: MatchingResult,
    solution,
    cfg: dict,
    attitude_delta_arcsec: float | None = None,
) -> tuple[bool, dict]:
    min_support = int(cfg["match"].get("validate_min_support", 3))
    rms_gate = float(cfg["attitude"].get("outlier_max_residual_arcsec", float("inf")))
    jump_gate = float(cfg["tracking"].get("max_attitude_jump_arcsec", float("inf")))

    reason = "ok"
    valid = True
    if len(matching.matched) < min_support:
        valid = False
        reason = "not_enough_matches"
    elif not solution.valid:
        valid = False
        reason = "attitude_invalid"
    elif solution.residual_rms_arcsec > rms_gate:
        valid = False
        reason = "residual_gate"
    elif attitude_delta_arcsec is not None and attitude_delta_arcsec > jump_gate:
        valid = False
        reason = "attitude_jump"

    debug = {
        "reason": reason,
        "min_support": min_support,
        "residual_gate_arcsec": rms_gate,
        "attitude_delta_arcsec": attitude_delta_arcsec,
        "attitude_jump_gate_arcsec": jump_gate,
    }
    return valid, debug


def match_stars_init(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> list[MatchedStar]:
    ctx = MatchingContext(
        mode="init",
        time_s=0.0,
        observed_stars=observed_stars,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=cfg.get("match", {}),
        reference_stars=reference_stars,
    )
    return match_stars(ctx, reference_stars, cfg).matched
