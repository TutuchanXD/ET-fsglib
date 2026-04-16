from fsglib.common.types import MatchedStar, MatchingContext, MatchingResult, ObservedStar
from fsglib.ephemeris.types import ReferenceStar


def associate_nearest(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> MatchingResult:
    matched: list[MatchedStar] = []
    residuals_pix: list[float] = []
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

            if best_dist is None or dist2 < best_dist:
                best_dist = dist2
                best = ref

        if best is None:
            continue

        residual_pix = float(best_dist ** 0.5) if best_dist is not None else 0.0
        residuals_pix.append(residual_pix)

        matched.append(
            MatchedStar(
                detector_id=obs.detector_id,
                source_id=obs.source_id,
                catalog_id=best.catalog_id,
                los_body=obs.los_body,
                los_inertial=best.los_inertial,
                residual_arcsec=None,
                weight=max(obs.snr, 1.0),
                match_score=1.0 if best_dist is None else 1.0 / (1.0 + best_dist),
                flags={
                    "match_mode": "predicted_position",
                    "residual_pix": residual_pix,
                    "observed_xy": (obs.x, obs.y),
                    "predicted_xy": best.predicted_xy.get(obs.detector_id),
                },
            )
        )

    matched_source_ids = {m.source_id for m in matched}
    matched_catalog_ids = {m.catalog_id for m in matched}
    stars_per_detector: dict[str, int] = {}
    for matched_star in matched:
        key = str(matched_star.detector_id)
        stars_per_detector[key] = stars_per_detector.get(key, 0) + 1

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
            "mean_residual_pix": mean_residual_pix,
            "stars_per_detector": stars_per_detector,
        },
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
            "num_local_matches": len(local_matches),
            "num_triangle_matches": len(triangle_matches),
            "num_reference_stars": len(reference_stars),
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
