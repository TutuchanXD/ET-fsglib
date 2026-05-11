# Local Pyramid Reacquire Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the local-pyramid matcher so it can serve as a local reacquire matcher with geometry-only expansion, ambiguity rejection, detector residual validation, and optional magnitude/SNR consistency scoring.

**Architecture:** Keep PR4 inside the matching layer. `match_stars()` selects a pyramid operation mode, while `match_local_pyramid()` resolves mode-specific gates and scores seed hypotheses. Expansion supports the existing predicted-pixel policy plus a first reacquire policy that uses seed-attitude angular geometry only; projector-backed seed-attitude pixel reprojection remains a follow-up issue.

**Tech Stack:** Python dataclasses, NumPy, SciPy linear assignment, pytest regression tests, YAML config defaults, GitHub issue tracking.

---

### Task 1: Geometry-Only Reacquire Expansion

**Files:**
- Modify: `fsglib/match/pipeline.py`
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] Add a test where `predicted_position_with_pyramid_reacquire` fails the nearest pixel gate because `predicted_xy` is stale, then succeeds through local pyramid using `reacquire_expansion_policy: seed_attitude_only`.
- [ ] Verify the test fails because `_build_expansion_edges()` rejects missing or stale predicted pixels before angular geometry is evaluated.
- [ ] Add `pyramid_mode` plumbing from `match_stars()` into `match_local_pyramid()`, using `reacquire` only for `predicted_position_with_pyramid_reacquire` fallback.
- [ ] Add mode-aware config resolution so `reacquire_*` local-pyramid keys override default pyramid keys.
- [ ] Add `expansion_policy` support:
  - `predicted_xy`: existing behavior.
  - `seed_attitude_only`: angular residual only; pixel residual and predicted XY may be `None`.
- [ ] Record `pyramid_mode`, `expansion_policy`, and `geometry_only_allowed` in debug.

### Task 2: Ambiguity Detection

**Files:**
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] Add a symmetric/repeated-geometry test where two different expanded catalog mappings score equivalently under geometry-only expansion.
- [ ] Verify the test fails because the matcher silently returns the first/best hypothesis.
- [ ] Track valid seed hypotheses by expanded source-to-catalog mapping.
- [ ] Compute and expose:
  - `num_valid_seed_hypotheses`
  - `ambiguity_margin`
  - `second_best_seed`
  - `ambiguous`
- [ ] Reject ambiguous results by returning no matches with `rejection_reason: ambiguous_seed_hypotheses` when the best and second-best score margin is below `ambiguity_min_score_margin`.

### Task 3: Detector Residual Validation

**Files:**
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] Add a mixed-detector coherent-offset test where one detector's predicted pixels are offset while angular geometry still produces a seed.
- [ ] Verify the test fails because per-detector residuals only report `status: ok`.
- [ ] Add detector residual warning/reject thresholds:
  - `detector_mean_warn_pix`
  - `detector_mean_reject_pix`
  - `detector_rms_reject_pix`
  - `detector_max_reject_pix`
  - `mixed_detector_reject_on_detector_warning`
- [ ] Mark per-detector residual statuses as `ok`, `warn`, or `reject`.
- [ ] Reject mixed-detector seed results when configured warning/reject thresholds indicate coherent detector offset.

### Task 4: Optional Magnitude/SNR Rank Penalty

**Files:**
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] Add a geometry-ambiguous brightness-order test with `photometric_rank_weight > 0`.
- [ ] Verify it fails because expansion cost does not use observed flux/SNR or reference magnitude rank.
- [ ] Add rank-based soft penalty to expansion edge cost:
  - observed rank from SNR, flux, source order.
  - reference rank from `mag_g`, `weight_hint`, catalog order.
  - penalty `abs(obs_rank - ref_rank) / max(rank_count - 1, 1)`.
- [ ] Keep `photometric_rank_weight` default at `0.0`.
- [ ] Include per-edge `photometric_rank_penalty` in matched flags and aggregate penalty debug.

### Task 5: Config, Docs, Verification, Publish

**Files:**
- Modify: `configs/base.yaml`
- Modify: `docs/yaml_configuration_reference.md`
- Possibly create: follow-up GitHub issue for projector-backed seed-attitude expansion

- [ ] Add conservative default config keys for PR4 behavior.
- [ ] Document the new keys and explain that `seed_attitude_projected` is not implemented in PR4.
- [ ] Search GitHub issues for an existing projector-backed seed-attitude expansion issue; create one if absent.
- [ ] Run focused tests:
  - `pytest tests/test_pyramid.py -q`
  - `pytest tests/test_match.py tests/test_tracking.py -q`
- [ ] Run full verification:
  - `pytest -q`
  - `git diff --check`
- [ ] Commit, push `match/pr04-reacquire-gates`, and open a PR against `main`.
