import numpy as np

from fsglib.common.types import MatchingContext, ObservedStar
from fsglib.ephemeris.types import ReferenceStar
from fsglib.match.pipeline import match_stars
from fsglib.match.pyramid import match_local_pyramid


def _unit(x: float, y: float, z: float = 1.0) -> np.ndarray:
    vec = np.array([x, y, z], dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _rotation_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _cfg() -> dict:
    return {
        "match": {
            "algorithm": "local_pyramid",
            "validate_min_support": 4,
            "validate_max_residual_pix": 2.0,
            "local_pyramid": {
                "max_observed_stars": 10,
                "max_reference_stars": 10,
                "seed_scopes": ["single_detector", "mixed_detector"],
                "pair_angle_tol_arcsec_single_detector": 1200.0,
                "pair_angle_tol_arcsec_mixed_detector": 2400.0,
                "seed_rms_gate_arcsec": 5.0,
                "seed_max_gate_arcsec": 10.0,
                "expand_angular_gate_arcsec": 10.0,
                "expand_pixel_gate_pix": 2.0,
                "min_expanded_matches": 4,
            },
        },
        "attitude": {"outlier_max_residual_arcsec": 30.0},
        "tracking": {"max_attitude_jump_arcsec": 100.0},
    }


def _reference_stars(detector_id=0) -> list[ReferenceStar]:
    vectors = [
        _unit(-0.030, -0.020),
        _unit(0.024, -0.018),
        _unit(0.018, 0.032),
        _unit(-0.026, 0.027),
        _unit(0.043, 0.021),
    ]
    refs = []
    for index, vector in enumerate(vectors):
        x = 100.0 + 25.0 * index
        y = 200.0 + 17.0 * index
        refs.append(
            ReferenceStar(
                catalog_id=1000 + index,
                time_s=0.0,
                los_inertial=vector,
                mag_g=9.0 + index,
                detector_ids_visible=[detector_id],
                predicted_xy={detector_id: (x, y)},
                predicted_valid={detector_id: True},
                weight_hint=1.0,
            )
        )
    return refs


def _observed_from_refs(refs: list[ReferenceStar], c_ib: np.ndarray | None = None) -> list[ObservedStar]:
    if c_ib is None:
        c_ib = np.eye(3, dtype=np.float64)
    observed = []
    for index, ref in enumerate(refs):
        detector_id = ref.detector_ids_visible[0]
        x, y = ref.predicted_xy[detector_id]
        observed.append(
            ObservedStar(
                detector_id=detector_id,
                source_id=2000 + index,
                x=x,
                y=y,
                los_body=c_ib @ ref.los_inertial,
                flux=1000.0 - index,
                snr=100.0 - index,
            )
        )
    return observed


def test_local_pyramid_matches_rotated_reference_stars():
    refs = _reference_stars()
    observed = _observed_from_refs(refs, _rotation_z(np.deg2rad(0.2)))

    result = match_local_pyramid(observed, refs, _cfg())

    assert result.success
    assert [match.catalog_id for match in result.matched] == [ref.catalog_id for ref in refs]
    assert len({match.catalog_id for match in result.matched}) == len(result.matched)
    assert result.debug["best_seed_scope"] == "single_detector"
    assert result.debug["best_expanded_matches"] == 5


def test_local_pyramid_reuses_pair_index_cache_for_same_reference_geometry():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cache = LocalPyramidCache()

    first = match_local_pyramid(observed, refs, cfg, cache=cache)
    second = match_local_pyramid(observed, refs, cfg, cache=cache)

    assert first.success
    assert second.success
    assert [match.catalog_id for match in second.matched] == [ref.catalog_id for ref in refs]
    assert first.debug["pair_index_cache"]["misses"] == 1
    assert first.debug["pair_index_cache"]["hits"] == 0
    assert second.debug["pair_index_cache"]["hits"] == 1
    assert second.debug["pair_index_cache"]["misses"] == 0
    assert second.debug["pair_index_cache"]["build_time_s"] == 0.0


def test_local_pyramid_pair_index_cache_misses_when_reference_los_changes():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cache = LocalPyramidCache()

    first = match_local_pyramid(observed, refs, cfg, cache=cache)
    shifted_refs = _reference_stars()
    shifted_refs[0].los_inertial = _unit(-0.05, -0.02)
    second = match_local_pyramid(observed, shifted_refs, cfg, cache=cache)

    assert first.success
    assert second.debug["pair_index_cache"]["misses"] == 1
    assert second.debug["pair_index_cache"]["hits"] == 0


def test_local_pyramid_reuses_angle_query_cache_without_changing_matches():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cache = LocalPyramidCache()

    baseline = match_local_pyramid(observed, refs, cfg)
    first = match_local_pyramid(observed, refs, cfg, cache=cache)
    second = match_local_pyramid(observed, refs, cfg, cache=cache)

    assert [match.catalog_id for match in first.matched] == [match.catalog_id for match in baseline.matched]
    assert [match.catalog_id for match in second.matched] == [match.catalog_id for match in baseline.matched]
    assert first.debug["angle_query_cache"]["misses"] > 0
    assert second.debug["angle_query_cache"]["hits"] > 0


def test_local_pyramid_cached_early_return_reports_cache_debug():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)[:3]

    result = match_local_pyramid(observed, refs, _cfg(), cache=LocalPyramidCache())

    assert not result.success
    assert result.debug["rejection_reason"] == "not_enough_observed_stars"
    assert result.debug["pair_index_cache"] == {"hits": 0, "misses": 0, "build_time_s": 0.0}
    assert result.debug["angle_query_cache"] == {"hits": 0, "misses": 0, "query_time_s": 0.0}


def test_local_pyramid_cache_clear_removes_pair_and_query_entries():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cache = LocalPyramidCache()

    result = match_local_pyramid(observed, refs, _cfg(), cache=cache)

    assert result.success
    assert len(cache.pair_index_by_key) == 1
    assert len(cache.query_pairs_by_key) > 0

    cache.clear()

    assert cache.pair_index_by_key == {}
    assert cache.query_pairs_by_key == {}
    assert cache.pair_index_hits == 0
    assert cache.query_hits == 0


def test_local_pyramid_cache_evicts_pair_indices_and_related_queries():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cache = LocalPyramidCache(max_pair_indices=1, max_query_entries=2)

    first = match_local_pyramid(observed, refs, _cfg(), cache=cache)
    old_pair_index_ids = {id(pair_index) for pair_index in cache.pair_index_by_key.values()}

    shifted_refs = _reference_stars()
    for index, ref in enumerate(shifted_refs):
        ref.catalog_id = 3000 + index
    second = match_local_pyramid(observed, shifted_refs, _cfg(), cache=cache)

    assert first.success
    assert len(cache.pair_index_by_key) == 1
    assert all(id(pair_index) not in old_pair_index_ids for pair_index in cache.pair_index_by_key.values())
    assert all(pair_index_id not in old_pair_index_ids for pair_index_id, _, _ in cache.query_pairs_by_key)
    assert len(cache.query_pairs_by_key) <= cache.max_query_entries
    assert second.debug["pair_index_cache"]["misses"] == 1


def test_local_pyramid_reports_seed_and_expansion_audit():
    refs = _reference_stars()
    observed = _observed_from_refs(refs, _rotation_z(np.deg2rad(0.2)))

    result = match_local_pyramid(observed, refs, _cfg())

    assert result.success
    seed_debug = result.debug["best_seed"]
    assert seed_debug["scope"] == "single_detector"
    assert seed_debug["detector_ids"] == [0]
    assert seed_debug["observed_source_ids"] == [2000, 2001, 2002, 2003]
    assert seed_debug["reference_catalog_ids"] == [1000, 1001, 1002, 1003]
    assert len(seed_debug["pair_angle_residuals_arcsec"]) == 6
    assert seed_debug["rms_arcsec"] == result.debug["best_seed_rms_arcsec"]

    expansion_debug = result.debug["best_expansion"]
    assert expansion_debug["num_edges_before_assignment"] >= len(result.matched)
    assert expansion_debug["num_edges_after_assignment"] == len(result.matched)
    assert expansion_debug["num_pixel_gate_rejects"] > 0
    assert "num_angular_gate_rejects" in expansion_debug


def test_local_pyramid_rejects_false_observed_star_during_expansion():
    refs = _reference_stars()[:4]
    observed = _observed_from_refs(refs)
    observed.append(
        ObservedStar(
            detector_id=0,
            source_id=9999,
            x=900.0,
            y=900.0,
            los_body=_unit(0.20, -0.10),
            flux=5000.0,
            snr=500.0,
        )
    )

    result = match_local_pyramid(observed, refs, _cfg())

    assert result.success
    assert len(result.matched) == 4
    assert 9999 in result.unmatched_observed_ids
    assert [match.catalog_id for match in result.matched] == [ref.catalog_id for ref in refs]


def test_match_stars_uses_local_pyramid_when_configured():
    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    ctx = MatchingContext(
        mode="init",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=_cfg()["match"],
        reference_stars=refs,
    )

    result = match_stars(ctx, refs, _cfg())

    assert result.success
    assert result.mode == "init"
    assert result.debug["selected_strategy"] == "local_pyramid"
    assert result.debug["num_pyramid_matches"] == 5
    assert result.debug["num_predicted_position_matches"] == 5
    assert result.debug["num_local_pyramid_matches"] == 5
    assert result.debug["num_triangle_matches"] == 0


def test_match_stars_passes_context_cache_to_local_pyramid():
    from fsglib.match.pyramid import LocalPyramidCache

    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cache = LocalPyramidCache()
    ctx = MatchingContext(
        mode="tracking",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=cfg["match"],
        reference_stars=refs,
        match_cache=cache,
    )

    first = match_stars(ctx, refs, cfg)
    second = match_stars(ctx, refs, cfg)

    assert first.success
    assert second.success
    assert first.debug["pyramid_debug"]["pair_index_cache"]["misses"] == 1
    assert second.debug["pyramid_debug"]["pair_index_cache"]["hits"] == 1


def test_match_stars_hybrid_keeps_predicted_position_on_equal_support():
    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cfg["match"]["algorithm"] = "predicted_position_and_local_pyramid"
    ctx = MatchingContext(
        mode="init",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=cfg["match"],
        reference_stars=refs,
    )

    result = match_stars(ctx, refs, cfg)

    assert result.success
    assert result.debug["selected_strategy"] == "predicted_position"
    assert result.debug["num_predicted_position_matches"] == 5
    assert result.debug["num_local_pyramid_matches"] == 5
    assert [match.flags["match_mode"] for match in result.matched] == ["predicted_position"] * 5
    assert result.debug["nearest_vs_pyramid"]["nearest_matched_count"] == 5
    assert result.debug["nearest_vs_pyramid"]["pyramid_matched_count"] == 5
    assert result.debug["nearest_vs_pyramid"]["same_catalog_id_mapping"] is True
    assert result.debug["nearest_vs_pyramid"]["catalog_id_disagreements"] == []


def test_match_stars_reports_pyramid_recovery_when_nearest_gate_misses_shifted_predictions():
    refs = _reference_stars()
    for ref in refs:
        detector_id = ref.detector_ids_visible[0]
        pred_x, pred_y = ref.predicted_xy[detector_id]
        ref.predicted_xy[detector_id] = (pred_x + 8.0, pred_y - 4.0)

    observed = _observed_from_refs(_reference_stars())
    cfg = _cfg()
    cfg["match"]["algorithm"] = "local_pyramid"
    cfg["match"]["validate_max_residual_pix"] = 2.0
    cfg["match"]["local_pyramid"]["expand_pixel_gate_pix"] = 12.0
    ctx = MatchingContext(
        mode="tracking",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=cfg["match"],
        reference_stars=refs,
    )

    result = match_stars(ctx, refs, cfg)

    assert result.success
    assert result.debug["selected_strategy"] == "local_pyramid"
    assert result.debug["num_predicted_position_matches"] == 0
    assert result.debug["num_local_pyramid_matches"] == 5
    assert [match.catalog_id for match in result.matched] == [ref.catalog_id for ref in refs]

    comparison = result.debug["nearest_vs_pyramid"]
    assert comparison["nearest_matched_count"] == 0
    assert comparison["pyramid_matched_count"] == 5
    assert comparison["same_catalog_id_mapping"] is False
    assert comparison["catalog_id_disagreements"] == [
        {"source_id": 2000, "nearest_catalog_id": None, "pyramid_catalog_id": 1000},
        {"source_id": 2001, "nearest_catalog_id": None, "pyramid_catalog_id": 1001},
        {"source_id": 2002, "nearest_catalog_id": None, "pyramid_catalog_id": 1002},
        {"source_id": 2003, "nearest_catalog_id": None, "pyramid_catalog_id": 1003},
        {"source_id": 2004, "nearest_catalog_id": None, "pyramid_catalog_id": 1004},
    ]


def test_reacquire_uses_geometry_only_expansion_when_predictions_are_stale():
    refs = _reference_stars()
    stale_refs = _reference_stars()
    for ref in stale_refs:
        detector_id = ref.detector_ids_visible[0]
        pred_x, pred_y = ref.predicted_xy[detector_id]
        ref.predicted_xy[detector_id] = (pred_x + 200.0, pred_y - 150.0)

    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cfg["match"]["algorithm"] = "predicted_position_with_pyramid_reacquire"
    cfg["match"]["validate_max_residual_pix"] = 2.0
    cfg["match"]["local_pyramid"]["expand_pixel_gate_pix"] = 2.0
    cfg["match"]["local_pyramid"]["reacquire_expansion_policy"] = "seed_attitude_only"
    cfg["match"]["local_pyramid"]["reacquire_geometry_only_allowed"] = True
    ctx = MatchingContext(
        mode="tracking",
        time_s=0.0,
        observed_stars=observed,
        prior_attitude_q=None,
        detector_layout={},
        optical_model={},
        matching_cfg=cfg["match"],
        reference_stars=stale_refs,
    )

    result = match_stars(ctx, stale_refs, cfg)

    assert result.success
    assert result.debug["selected_strategy"] == "local_pyramid"
    assert result.debug["num_predicted_position_matches"] == 0
    assert [match.catalog_id for match in result.matched] == [ref.catalog_id for ref in refs]
    assert all(match.flags["residual_pix"] is None for match in result.matched)
    assert result.debug["pyramid_debug"]["pyramid_mode"] == "reacquire"
    assert result.debug["pyramid_debug"]["best_expansion"]["expansion_policy"] == "seed_attitude_only"


def test_local_pyramid_rejects_ambiguous_geometry_only_hypotheses():
    refs = _reference_stars()[:4]
    duplicate_refs = []
    for index, ref in enumerate(refs):
        duplicate_refs.append(
            ReferenceStar(
                catalog_id=3000 + index,
                time_s=ref.time_s,
                los_inertial=np.array(ref.los_inertial, copy=True),
                mag_g=ref.mag_g,
                detector_ids_visible=list(ref.detector_ids_visible),
                predicted_xy={},
                predicted_valid={},
                weight_hint=ref.weight_hint,
            )
        )

    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cfg["match"]["local_pyramid"]["seed_scopes"] = ["single_detector"]
    cfg["match"]["local_pyramid"]["expansion_policy"] = "seed_attitude_only"
    cfg["match"]["local_pyramid"]["geometry_only_allowed"] = True
    cfg["match"]["local_pyramid"]["ambiguity_min_score_margin"] = 1.0

    result = match_local_pyramid(observed, refs + duplicate_refs, cfg, pyramid_mode="reacquire")

    assert not result.success
    assert result.matched == []
    assert result.debug["rejection_reason"] == "ambiguous_seed_hypotheses"
    assert result.debug["ambiguous"] is True
    assert result.debug["num_valid_seed_hypotheses"] > 1
    assert result.debug["ambiguity_margin"] == 0.0
    assert result.debug["second_best_seed"] is not None


def test_mixed_detector_seed_rejects_coherent_detector_offset_when_configured():
    refs = _reference_stars(detector_id="guide_left")[:4]
    for index, ref in enumerate(refs):
        detector_id = "guide_left" if index < 2 else "guide_right"
        x, y = ref.predicted_xy["guide_left"]
        ref.detector_ids_visible = [detector_id]
        ref.predicted_xy = {detector_id: (x, y)}
        ref.predicted_valid = {detector_id: True}

    observed = _observed_from_refs(refs)
    offset_refs = []
    for ref in refs:
        detector_id = ref.detector_ids_visible[0]
        x, y = ref.predicted_xy[detector_id]
        if detector_id == "guide_right":
            x += 20.0
        offset_refs.append(
            ReferenceStar(
                catalog_id=ref.catalog_id,
                time_s=ref.time_s,
                los_inertial=np.array(ref.los_inertial, copy=True),
                mag_g=ref.mag_g,
                detector_ids_visible=list(ref.detector_ids_visible),
                predicted_xy={detector_id: (x, y)},
                predicted_valid={detector_id: True},
                weight_hint=ref.weight_hint,
            )
        )

    cfg = _cfg()
    cfg["match"]["local_pyramid"]["seed_scopes"] = ["mixed_detector"]
    cfg["match"]["local_pyramid"]["expand_pixel_gate_pix"] = 50.0
    cfg["match"]["local_pyramid"]["detector_mean_warn_pix"] = 5.0
    cfg["match"]["local_pyramid"]["mixed_detector_reject_on_detector_warning"] = True

    result = match_local_pyramid(observed, offset_refs, cfg)

    assert not result.success
    assert result.debug["rejection_reason"] == "detector_residual_reject"
    assert result.debug["best_per_detector_residuals"]["guide_right"]["status"] == "warn"
    assert result.debug["best_per_detector_residuals"]["guide_right"]["mean_norm_pix"] == 20.0


def test_photometric_rank_penalty_breaks_geometry_only_tie_when_enabled():
    refs = _reference_stars()[:4]
    duplicate_refs = []
    for index, ref in enumerate(refs):
        duplicate_refs.append(
            ReferenceStar(
                catalog_id=3000 + index,
                time_s=ref.time_s,
                los_inertial=np.array(ref.los_inertial, copy=True),
                mag_g=20.0 + index,
                detector_ids_visible=list(ref.detector_ids_visible),
                predicted_xy={},
                predicted_valid={},
                weight_hint=ref.weight_hint,
            )
        )

    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cfg["match"]["local_pyramid"]["seed_scopes"] = ["single_detector"]
    cfg["match"]["local_pyramid"]["expansion_policy"] = "seed_attitude_only"
    cfg["match"]["local_pyramid"]["geometry_only_allowed"] = True
    cfg["match"]["local_pyramid"]["photometric_rank_weight"] = 25.0
    cfg["match"]["local_pyramid"]["ambiguity_min_score_margin"] = 0.1

    result = match_local_pyramid(observed, refs + duplicate_refs, cfg, pyramid_mode="reacquire")

    assert result.success
    assert result.debug["ambiguous"] is False
    assert [match.catalog_id for match in result.matched] == [ref.catalog_id for ref in refs]
    assert all(match.flags["photometric_rank_penalty"] == 0.0 for match in result.matched)
    assert result.debug["best_expansion"]["photometric_rank_weight"] == 25.0


def test_local_pyramid_prefers_detector_local_seed_before_mixed_seed():
    refs = _reference_stars(detector_id="guide_left")
    observed = _observed_from_refs(refs)
    extra_ref = ReferenceStar(
        catalog_id=3000,
        time_s=0.0,
        los_inertial=_unit(0.07, -0.03),
        mag_g=14.0,
        detector_ids_visible=["guide_right"],
        predicted_xy={"guide_right": (700.0, 800.0)},
        predicted_valid={"guide_right": True},
        weight_hint=1.0,
    )
    refs.append(extra_ref)
    observed.append(
        ObservedStar(
            detector_id="guide_right",
            source_id=4000,
            x=700.0,
            y=800.0,
            los_body=extra_ref.los_inertial,
            flux=10.0,
            snr=10.0,
        )
    )

    result = match_local_pyramid(observed, refs, _cfg())

    assert result.success
    assert result.debug["best_seed_scope"] == "single_detector"
    assert result.debug["best_seed_detector_ids"] == ["guide_left"]
    assert len(result.matched) == 6


def test_local_pyramid_allows_same_catalog_on_different_detectors():
    refs = _reference_stars(detector_id="guide_left")
    shared_ref = refs[0]
    shared_ref.detector_ids_visible.append("guide_right")
    shared_ref.predicted_xy["guide_right"] = (700.0, 800.0)
    shared_ref.predicted_valid["guide_right"] = True
    observed = _observed_from_refs(refs)
    observed.append(
        ObservedStar(
            detector_id="guide_right",
            source_id=5000,
            x=700.0,
            y=800.0,
            los_body=shared_ref.los_inertial,
            flux=900.0,
            snr=90.0,
        )
    )

    result = match_local_pyramid(observed, refs, _cfg())

    assert result.success
    assert len(result.matched) == 6
    repeated = [
        match
        for match in result.matched
        if match.catalog_id == shared_ref.catalog_id
    ]
    assert [match.detector_id for match in repeated] == ["guide_left", "guide_right"]


def test_local_pyramid_reports_clean_failure_when_under_supported():
    refs = _reference_stars()[:3]
    observed = _observed_from_refs(refs)

    result = match_local_pyramid(observed, refs, _cfg())

    assert not result.success
    assert result.matched == []
    assert result.debug["rejection_reason"] == "not_enough_observed_stars"
