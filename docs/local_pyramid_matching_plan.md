# Local Pyramid Matching Implementation Plan

## Status

This document is the historical design record for replacing the obsolete
triangle-index matching path with a local pyramid matcher. The implementation
is now in `fsglib.match.pyramid` and is selected through `match.algorithm`;
current configuration semantics live in `docs/yaml_configuration_reference.md`.

Local cleanup captured by this design:

- The ignored local `data/catalogs/` directory is obsolete and has no retention value.
- The old triangle matcher and GSC builder remain in the repository only as
  deprecated compatibility paths.

## Problem Statement

The current matching stack has two different catalog paths:

1. The active guide simulation and guide initialization paths use `et_focalplane` / `et_coord` with the local Gaia catalog at `/home/cxgao/gaia_dr3_19mag`.
2. The triangle matcher uses an old local NPZ index through `match.triangle_gsc_path`, historically `data/catalogs/gsc_8mag.npz`.

This breaks the most important invariant for simulated guide-star validation:

> The simulator-side star field and the guide-side reference catalog must come from the same catalog source, coordinate model, target epoch, magnitude filters, and detector geometry.

The old triangle index violates this invariant. It also has a shallower and inconsistent magnitude range, does not use the same `ReferenceStar` list, and cannot be trusted for the current `et_focalplane` pipeline.

The next matcher should therefore be a local pyramid matcher built from the `ReferenceStar` objects already produced by the active guide pipeline.

## Goals

- Implement local pyramid matching using only the current frame's `reference_stars`.
- Avoid any global precomputed GSC/NPZ index in this phase.
- Preserve the existing predicted-position nearest-neighbor matcher as the baseline and fallback.
- Use four-star pyramid geometry for robust initial association or reacquisition.
- Expand a validated pyramid seed into a full matched-star set.
- Keep simulator, guide reference building, and matching catalog identities consistent.
- Support the four-guide-detector layout without assuming perfect detector assembly alignment.
- Produce clear debug output for match quality, candidate pruning, and detector-level residuals.

## Non-Goals For This Phase

- No lost-in-space global matching.
- No global all-sky pyramid index.
- No replacement of `et_focalplane` catalog querying.
- No automatic detector-geometry calibration as part of the first implementation.
- No removal of tracked `triangle.py`, `build_gsc.py`, or `tests/test_triangle.py` until the new path is validated.

## External Practice Review

The implementation should follow the pattern used in flight guidance systems: local sensor measurements feed attitude determination, but calibration and residual checks keep sensor boundaries explicit.

Findings from public references:

- Hubble uses multiple Fine Guidance Sensors for pointing. NASA describes three FGS units, with two required for pointing and holding a target steady. This is a joint spacecraft pointing function built from multiple guide sensors, not an all-sensors-are-one-camera assumption.
  - Source: https://science.nasa.gov/mission/hubble/observatory/design/fine-guidance-sensors/
- JWST FGS has two guider channels. STScI documents guide-star identification as a detector image compared to a catalog with a pattern-matching algorithm; the measured guide-star position is then sent to ACS. JWST fine guiding uses a single guide star in one FGS field for a visit, while roll is handled separately by star trackers. This argues for detector/channel-specific acquisition and health checks even when the result feeds the common ACS.
  - Source: https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-fine-guidance-sensor
  - Source: https://jwst-docs.stsci.edu/jwst-observatory-characteristics-and-performance/jwst-pointing-performance/jwst-guide-stars
- Euclid's FGS feeds guide-star data into the spacecraft Attitude and Orbit Control System, and an in-flight software patch addressed false signals that interfered with resolving star patterns. This is evidence that pattern matching must be robust to false detections and detector-specific artifacts.
  - Source: https://www.esa.int/Science_Exploration/Space_Science/Euclid/Guide_stars_found_as_Euclid_s_navigation_fine_tuned
- NASA MMS attitude ground-system documentation describes definitive attitude generation from multiple star-tracker heads, with alignment transformations applied per head and repeated calibration for relative shifts. This directly supports preserving detector/head boundaries and validating relative alignment rather than blindly mixing measurements.
  - Source: https://ntrs.nasa.gov/api/citations/20150019870/downloads/20150019870.pdf

Engineering conclusion for `fsglib`:

- The final attitude solution should be joint across all available guide detectors.
- Pyramid seed generation and validation should be detector-aware.
- Cross-detector pyramids are useful after geometry is trusted, but detector-local pyramids should be the default robust seed source.
- Per-detector residual summaries are required. A joint solution with one detector showing a coherent residual offset should be treated as a geometry/alignment issue, not merely a matching failure.

## Concern: Relative Detector Assembly Offsets

This concern is real.

If the four detector frames have small relative mounting errors, a mixed-detector pyramid can fail even when each detector is internally consistent:

- A pure small rigid rotation/translation error for one detector mostly preserves pair angles among stars inside that detector.
- The same error changes cross-detector angular separations between stars on different detectors.
- A pyramid whose four stars span multiple detectors can therefore be more sensitive to relative detector alignment than a pyramid contained within one detector.

The matcher should handle this by design:

1. Generate detector-local pyramid seeds first when a detector has at least four usable stars.
2. Allow mixed-detector seeds only as a secondary strategy with a looser angular gate and explicit per-detector residual validation.
3. After a seed attitude is found, expand matches jointly across all detectors using predicted detector positions and one-to-one assignment.
4. Report detector-level residual mean, RMS, and coherent offset. Do not hide detector-specific failure inside a single global RMS.
5. Leave detector alignment correction as a later calibration layer, not part of first local pyramid matching.

## Proposed Matching Modes

### `predicted_position`

Existing nearest-neighbor predicted detector position matching.

Use cases:

- Normal tracking with a good prior attitude.
- Guide first-frame runs where `et_focalplane` predicted positions are already accurate.
- Fallback when pyramid support is insufficient.

### `local_pyramid`

Run local pyramid matching using current `reference_stars`; select pyramid result only if it passes validation. This mode should not call the old triangle matcher.

Use cases:

- Initialization with moderate uncertainty.
- Reacquisition when predicted positions are not tight enough for direct nearest-neighbor matching.
- Debugging catalog/geometry consistency.

### `predicted_position_with_pyramid_reacquire`

Attempt predicted-position matching first. If it fails support or residual gates, attempt local pyramid. If pyramid succeeds, expand and return pyramid matches.

Use cases:

- Future tracking state-machine reacquire path.

### `predicted_position_and_local_pyramid`

Always run both predicted-position matching and local pyramid matching, then use
the result with more matched stars. If both strategies return the same number of
matches, keep the predicted-position result as the conservative tie-breaker.

Use cases:

- Comparing local pyramid performance against the current guide-star baseline.
- Transition period where both matchers should be visible in debug output.

## High-Level Algorithm

Inputs:

- `observed_stars: list[ObservedStar]`
- `reference_stars: list[ReferenceStar]`
- `cfg: dict`

Outputs:

- `MatchingResult`
- `matched: list[MatchedStar]`
- detailed debug payload

Steps:

1. Preselect observed stars.
2. Preselect reference stars.
3. Build local reference pair-angle index.
4. Generate observed pyramid seeds.
5. Find reference pyramid candidates using six pair-angle constraints.
6. Solve candidate attitude from the four matched vector pairs.
7. Score candidate by angular residual and detector predicted-position consistency.
8. Expand the best candidate into full-frame matches.
9. Enforce one-to-one assignment.
10. Validate support, residuals, and detector-level consistency.
11. Return a `MatchingResult`.

## Data Model

### `PyramidSeed`

Fields:

- `observed_indices: tuple[int, int, int, int]`
- `reference_indices: tuple[int, int, int, int]`
- `pair_angle_residuals_arcsec: list[float]`
- `candidate_attitude_q: np.ndarray`
- `candidate_c_ib: np.ndarray`
- `seed_rms_arcsec: float`
- `detector_ids: tuple[object, ...]`
- `seed_scope: str`
  - `single_detector`
  - `mixed_detector`

### `LocalPairIndex`

Fields:

- `pairs_by_bin: dict[int, list[ReferencePair]]`
- `pair_angles_rad: np.ndarray`
- `pair_ref_indices: np.ndarray`
- `bin_width_rad: float`
- `reference_count: int`

Reference pair:

- `i: int`
- `j: int`
- `angle_rad: float`

The first implementation can use sorted arrays and `np.searchsorted`, not a K-vector. The local reference count is small enough that clarity is more important than an approximate custom index.

### Debug Payload

Required debug fields:

- `algorithm`
- `selected_strategy`
- `pyramid_enabled`
- `pyramid_seed_scope_order`
- `num_observed_input`
- `num_observed_used`
- `num_reference_input`
- `num_reference_used`
- `num_predicted_position_matches`
- `num_local_pyramid_matches`
- `num_reference_pairs`
- `num_observed_pyramids_tested`
- `num_reference_pyramid_candidates`
- `num_seed_attitudes_scored`
- `best_seed_scope`
- `best_seed_detector_ids`
- `best_seed_rms_arcsec`
- `best_expanded_matches`
- `expanded_mean_residual_pix`
- `expanded_rms_residual_pix`
- `per_detector_residuals`
- `rejection_reason`
- `fallback_strategy`

## Preselection

Observed stars should be sorted by:

1. Higher SNR.
2. Higher flux.
3. Lower shape/edge warning if available.

Config:

```yaml
match:
  algorithm: local_pyramid
  pyramid_max_observed_stars: 40
  pyramid_max_reference_stars: 300
  pyramid_min_seed_stars: 4
  pyramid_seed_scopes: ["single_detector", "mixed_detector"]
```

Reference stars should be sorted by:

1. Lower `mag_g` when available.
2. Higher `weight_hint`.
3. Valid predicted detector coordinates.

For guide runs, `reference_topk_per_detector` remains controlled by `guide_init`.

## Reference Pair Index

For each pair of selected reference stars:

1. Compute angular separation between `los_inertial` vectors.
2. Store `(angle_rad, ref_i, ref_j)`.
3. Sort by `angle_rad`.

Query function:

```python
def query_pairs(angle_rad: float, tolerance_rad: float) -> list[ReferencePair]:
    lo = angle_rad - tolerance_rad
    hi = angle_rad + tolerance_rad
    start = np.searchsorted(pair_angles_rad, lo, side="left")
    stop = np.searchsorted(pair_angles_rad, hi, side="right")
    return pairs[start:stop]
```

Use exact sorted-array bounds. Do not reintroduce the old approximate K-vector logic.

## Pyramid Candidate Generation

For each observed four-star combination:

1. Compute the six observed edge angles.
2. Query reference candidate pairs for each observed edge.
3. Build candidate reference four-tuples that satisfy all six pair constraints.
4. Respect observed-to-reference vertex consistency.
5. Reject duplicate catalog IDs.
6. Reject geometrically degenerate pyramids.

Degeneracy checks:

- Minimum edge angle.
- Maximum edge angle.
- Minimum tetrahedral spread.
- Condition number of the Wahba/QUEST input matrix.

Suggested initial config:

```yaml
match:
  pyramid_pair_angle_tol_arcsec_single_detector: 120.0
  pyramid_pair_angle_tol_arcsec_mixed_detector: 300.0
  pyramid_min_edge_arcsec: 30.0
  pyramid_max_edge_deg: 20.0
  pyramid_min_volume: 1.0e-9
  pyramid_max_candidates_per_observed_seed: 200
  pyramid_max_seed_attitudes: 2000
```

The mixed-detector tolerance is intentionally looser because detector relative alignment errors affect cross-detector pair angles.

## Candidate Attitude Solve

For each reference four-tuple:

1. Build four vector correspondences:
   - body: observed `los_body`
   - inertial: reference `los_inertial`
2. Solve attitude with existing QUEST/SVD machinery.
3. Compute angular residuals for the four seed pairs.
4. Reject if residual RMS or max residual exceeds gates.

Config:

```yaml
match:
  pyramid_seed_rms_gate_arcsec: 60.0
  pyramid_seed_max_gate_arcsec: 180.0
```

The first implementation can use the existing `solve_quest` or a small SVD Wahba helper shared with the attitude solver. Avoid duplicating numerical code where possible.

## Expansion To Full Matches

After a seed attitude is accepted:

1. Rotate every selected reference inertial vector into body frame.
2. Compare to observed body vectors by angular separation.
3. Optionally project reference stars to detector predicted positions if available.
4. Build candidate edges between observed and reference stars.
5. Score each edge with a weighted combination:
   - angular residual in body coordinates
   - detector predicted-position residual in pixels, when valid
   - magnitude/SNR prior, weak weight only
6. Run one-to-one assignment per detector first.
7. Optionally run a final global one-to-one catalog uniqueness pass.

Suggested scoring:

```text
cost = angular_residual_arcsec / angular_sigma_arcsec
     + predicted_residual_pix / pixel_sigma_pix
     + weak_magnitude_penalty
```

Config:

```yaml
match:
  pyramid_expand_angular_gate_arcsec: 120.0
  pyramid_expand_pixel_gate_pix: 25.0
  pyramid_expand_min_matches: 5
  pyramid_enforce_unique_assignment: true
```

## Detector-Aware Validation

The local pyramid matcher should produce both global and per-detector validation.

Global gates:

- `num_matches >= validate_min_support`
- attitude residual RMS below gate
- one-to-one assignment success
- no duplicated catalog IDs

Per-detector gates:

- minimum support per active detector, when the detector participates
- detector RMS below gate
- detector mean residual vector below warning threshold
- detector max residual below reject threshold

Debug structure:

```json
{
  "per_detector_residuals": {
    "guide_left": {
      "num_matches": 95,
      "mean_dx_pix": 0.12,
      "mean_dy_pix": -0.08,
      "rms_pix": 0.42,
      "max_pix": 1.6,
      "status": "ok"
    }
  }
}
```

If one detector has a coherent offset while others are clean, the result should be marked as:

- `success=true` only if the attitude remains valid and enough other detectors support it.
- `debug.detector_alignment_warning=true`
- `debug.detector_alignment_warning_ids=[...]`

This warning is important for diagnosing assembly/calibration issues.

## Single-Detector Versus Mixed-Detector Strategy

Default strategy:

1. Try `single_detector` pyramid seeds first.
2. Score each detector independently.
3. Use the best detector-local seed to expand jointly across all detectors.
4. If no detector-local seed succeeds, try mixed-detector seeds.
5. If mixed-detector succeeds, require stronger per-detector residual validation.

Rationale:

- Detector-local seeds are less sensitive to relative mounting errors.
- Joint expansion still uses all available guide information for final attitude.
- Mixed-detector seeds remain useful when each detector has fewer than four usable stars.

Config:

```yaml
match:
  pyramid_seed_scopes:
    - single_detector
    - mixed_detector
  pyramid_min_stars_per_seed_detector: 4
  pyramid_mixed_detector_enable: true
```

## Integration Plan

### Phase 1: Documentation Only

Files:

- `docs/local_pyramid_matching_plan.md`

Actions:

- Document the old triangle-index problem.
- Define local pyramid design.
- Define detector-aware strategy.
- Define config, debug, test, and migration plan.

### Phase 2: Deprecate Old Triangle Path

Files:

- `fsglib/match/triangle.py`
- `fsglib/tools/build_gsc.py`
- `tests/test_triangle.py`
- `configs/base.yaml`
- `docs/README_API.md`

Actions:

- Mark `triangle.py` and `build_gsc.py` as deprecated.
- Remove `triangle_gsc_path` from default configs.
- Change default algorithm from `local_triangle` to `predicted_position` or `local_pyramid` only after `local_pyramid` exists.
- Keep tests passing by quarantining old triangle tests or marking them legacy.

### Phase 3: Implement `fsglib/match/pyramid.py`

New module:

- `fsglib/match/pyramid.py`

Core public API:

```python
def match_local_pyramid(
    observed_stars: list[ObservedStar],
    reference_stars: list[ReferenceStar],
    cfg: dict,
) -> MatchingResult:
    ...
```

Internal helpers:

- `_select_observed_for_pyramid`
- `_select_reference_for_pyramid`
- `_build_local_pair_index`
- `_generate_observed_pyramids`
- `_find_reference_pyramid_candidates`
- `_solve_seed_attitude`
- `_score_seed`
- `_expand_matches`
- `_build_pyramid_matching_result`

### Phase 4: Pipeline Integration

Files:

- `fsglib/match/pipeline.py`

Actions:

- Add support for `algorithm == "local_pyramid"`.
- Add support for `algorithm == "predicted_position_with_pyramid_reacquire"`.
- Do not call `TriangleMatcher` for new algorithms.
- Preserve `associate_nearest` behavior.

### Phase 5: Tests

New tests:

- `tests/test_pyramid.py`

Required cases:

1. Perfect local four-star pyramid matches exactly.
2. Rotated/noisy observed vectors recover catalog IDs.
3. Extra false observed stars are rejected.
4. Duplicate catalog assignment is impossible.
5. Single-detector seed expands to multi-detector matches.
6. Mixed-detector seed uses looser tolerance.
7. Detector relative offset causes detector warning, not silent success.
8. Empty or under-supported inputs return clean failure.
9. `match_stars()` selects local pyramid only when configured.
10. Legacy triangle path is not invoked by local pyramid config.

### Phase 6: Guide-Run Validation

Run current guide scenarios:

- `examples/run_guide_first_frame.py`
- `examples/run_guide_first_frame_truth_noise.py`
- `examples/run_guide_first_frame_truth_noise_exact.py`
- `examples/run_microlens_guide_first_frame.py`

Acceptance:

- Local pyramid finds a valid seed in synthetic truth-noise mode.
- Expanded matches have zero incorrect catalog IDs in truth-backed audits.
- Predicted-position baseline remains unchanged.
- Detector residual debug identifies any detector-specific bias.

## Proposed Config Migration

Old:

```yaml
match:
  algorithm: local_triangle
  triangle_tolerance_deg: 0.005
  triangle_max_stars: 15
  triangle_gsc_path: /home/cxgao/ET/FSG/fsglib/data/catalogs/gsc_8mag.npz
```

New:

```yaml
match:
  algorithm: predicted_position
  validate_max_residual_pix: 10.0
  validate_min_support: 5
  enforce_unique_assignment: true

  local_pyramid:
    enabled: false
    max_observed_stars: 40
    max_reference_stars: 300
    seed_scopes: ["single_detector", "mixed_detector"]
    pair_angle_tol_arcsec_single_detector: 120.0
    pair_angle_tol_arcsec_mixed_detector: 300.0
    seed_rms_gate_arcsec: 60.0
    seed_max_gate_arcsec: 180.0
    expand_angular_gate_arcsec: 120.0
    expand_pixel_gate_pix: 25.0
    min_expanded_matches: 5
    enforce_unique_assignment: true
```

The exact nesting can be adjusted during implementation. The important point is to remove global triangle index configuration from defaults.

## Failure Modes And Handling

### Not Enough Stars

Return failure with:

- `reason="not_enough_observed_stars"` or `reason="not_enough_reference_stars"`

### Too Many Ambiguous Candidates

Return failure or fallback with:

- `reason="too_many_pyramid_candidates"`

Mitigation:

- reduce observed/reference preselection
- prefer isolated stars
- use magnitude ordering as weak prior

### Seed Found But Expansion Fails

Return failure or fallback with:

- `reason="seed_expansion_failed"`

Include seed debug so the geometry can be inspected.

### Detector Alignment Warning

Return success only if global attitude is valid; include warning fields.

### Catalog Inconsistency

This should be rare for local pyramid because references come from the same active catalog path. If truth audit detects mismatched Source IDs, treat as a pipeline/catalog issue, not a pyramid algorithm issue.

## Acceptance Criteria

Design acceptance:

- The plan avoids old `data/catalogs` indices.
- The plan uses only `reference_stars` for local matching.
- The plan explicitly handles detector-local and mixed-detector seeds.
- The plan includes detector-relative offset mitigation.
- The plan defines tests and debug outputs.

Implementation acceptance:

- `pytest -q` passes.
- Current predicted-position behavior is unchanged unless algorithm config opts into pyramid.
- Local pyramid succeeds on synthetic controlled tests.
- Local pyramid succeeds on truth-noise guide first-frame runs.
- No old `gsc_*.npz` files are required.
- Debug output explains why pyramid succeeded or failed.

## Open Decisions

1. Default algorithm after implementation:
   - Conservative: keep `predicted_position`.
   - Experimental: use `predicted_position_with_pyramid_reacquire`.
2. Whether mixed-detector seeds should be enabled by default.
   - Recommendation: enabled, but tried after detector-local seeds.
3. Whether detector alignment warning should invalidate the result.
   - Recommendation: warn first; invalidate only if residual gates fail.
4. Whether future detector calibration should estimate only 2D pixel offsets or a full small rotation per detector.
   - Recommendation: defer until local pyramid diagnostics show the dominant error mode.
