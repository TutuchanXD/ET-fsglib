# ET-fsglib Guide Algorithm Maintenance Roadmap

Repository: `TutuchanXD/ET-fsglib`

Source planning note: the local Chinese DOCX roadmap under `plan/` (planning material; not committed)

Date: 2026-05-11

Status snapshot updated: 2026-05-27

Goal: move ET-fsglib from an algorithm validation and workflow test library toward an auditable, regressible, and extensible high-precision guide-star algorithm chain.

## Executive Summary

Use 24 code PRs plus one housekeeping action to close the temporary test issue #22. Do not split the backlog into one PR per issue, and do not compress it into a few large PRs. PR boundaries should follow error-propagation boundaries and independently verifiable algorithm capabilities.

The guide-star error chain crosses module boundaries. Preprocessing noise affects extraction, centroids affect line-of-sight vectors, LOS vectors affect matching, matching affects attitude, and attitude feeds tracking and reference prediction. Each PR therefore needs a clear boundary, tests, acceptance criteria, and structured debug output.

## Maintenance Principles

- Fix operability before performance, and performance before precision. For example, tracking must honor `match.algorithm` before local pyramid or lost-in-space work can be validated in the real tracking path.
- Keep local reacquire and lost-in-space separate. Both can use angular invariants, but their inputs, indices, validation gates, and failure policies are different.
- Every gate should have a physical basis. Fixed constants are acceptable as a first implementation only when the PR states the assumption and leaves a path to covariance-, projection-, catalog-, or alignment-derived gates.
- Every algorithmic capability needs adversarial tests. Nominal examples are not enough; cover prior error, missing stars, false stars, close neighbors, and multi-detector offsets.
- Structured debug is part of the algorithm. The chain should explain why each source was accepted or rejected and why each attitude solution was valid or invalid.
- PRs do not need to be perfect in one step, but every merged PR must improve regressibility.

## Milestones

| Milestone | Goal | PRs | Notes |
| --- | --- | --- | --- |
| M0 | Cleanup and baseline | PR0 | Close #22 without a code PR. |
| M1 | Matching and state-machine baseline | PR1-PR5 | Fix configuration routing, prove local pyramid behavior, add caches, harden reacquire gates, and define operating modes. |
| M2 | Image, geometry, catalog, and LIS foundation | PR6-PR18 | Build lost-in-space foundations while advancing preprocessing, centroiding, optics, and ephemeris. |
| M3 | Attitude precision and validation closure | PR19-PR22 | Add covariance-weighted attitude solving, robust outlier rejection, error budget, Monte Carlo, and ET validation scenarios. |
| M4 | Performance and maintainability | PR23-PR24 | Add runtime budgets, typed config, structured debug artifacts, and design documentation. |

## Current Status Snapshot

Status checked against merged PRs and open issues on 2026-05-27.

| Roadmap PR | Status | Evidence and remaining work |
| --- | --- | --- |
| PR0 | Completed | Temporary issue #22 is closed. |
| PR1 | Completed | Merged as GitHub PR #66. |
| PR2 | Completed | Merged as GitHub PR #70. |
| PR3 | Completed | Merged as GitHub PR #73. Runtime-budget generalization remains PR23/#54. |
| PR4 | Completed with follow-ups | Merged as GitHub PR #76. Covariance-derived gates remain in PR19/#31; projector-backed expansion remains #75. |
| PR5 | Completed | Merged as GitHub PR #78. |
| PR6 | Completed | Merged as GitHub PR #71. |
| PR7 | Completed | Merged as GitHub PR #79. |
| PR8 | Completed | Merged as GitHub PR #80. |
| PR9 | Completed | Merged as GitHub PR #82. |
| PR10 | Completed with follow-ups | Merged as GitHub PR #85. Real detector noise assets and flat-field uncertainty remain #83/#84. |
| PR11 | Completed with follow-up | Merged as GitHub PR #88. Future unmasked cosmic-ray detection remains #87. |
| PR12 | Completed | Merged as GitHub PR #86. |
| PR13 | Partially completed | Merged GitHub PR #90 for centroid covariance propagation. Follow-ups remain for PSF-template fitting (#89/#38) and catalog-aware deblending (#40). |
| PR14 | Completed | Merged as GitHub PR #72. |
| PR15 | Completed with follow-up | Merged as GitHub PR #74. Broader multi-detector validation remains #57/PR22. |
| PR16 | Pending | Open issue #43. |
| PR17 | Completed | Merged as GitHub PR #77. |
| PR18 | Completed | Merged as GitHub PR #81 for the generic HEALPix provider/cache/provenance path. |
| PR19 | Partially completed | Merged GitHub PR #92 for attitude covariance quality metrics and closed #62. Covariance-weighted Wahba/QUEST remains open in #47, and covariance-derived gates remain #31. |
| PR20 | Completed | Merged GitHub PR #94 and closed #48. Additional robust-solver evaluation is tracked separately in #91/#93. |
| PR21 | Completed with follow-ups | Merged GitHub PR #98 and closed #50. Error-budget prior/dashboard follow-ups remain #95/#96/#97. |
| PR22 | Pending | Open issues #51, #52, #53, #57, and #65. |
| PR23 | Partially covered, not complete | Local-pyramid timing/cache work exists from PR3/#73; full stage profiler and CI budgets remain #54. |
| PR24 | Partially covered, not complete | Incremental docs/debug work exists, but YAML audit, typed config, structured artifacts, logging policy, and design docs remain #69/#55/#58/#59/#60. |

## Current Issue Map

Historical issues:

| Issue | Title | Owner PR | Handling |
| --- | --- | --- | --- |
| #15 | match: validate local pyramid independence before readying PR #13 | PR2 | Local pyramid independence, seed/expansion debug, controlled failure case. |
| #16 | config: clarify match.local_pyramid.enabled semantics | PR1 / PR24 | Runtime semantics in PR1; typed schema and docs in PR24. |
| #17 | match: validate detector-aware mixed-detector pyramid seeds | PR4 | Single-detector and mixed-detector seed gates with detector residual audit. |
| #18 | match: add robustness tests and gate review for local pyramid | PR4 / PR22 | Matcher gates in PR4; Monte Carlo and scenario validation in PR22. |
| #19 | perf: benchmark local pyramid search on guide frames | PR3 / PR23 | Local cache in PR3; CI runtime budget in PR23. |
| #20 | catalog: audit guide catalog and geometry consistency | PR18 | Catalog root, registry, epoch, bandpass, and magnitude provenance. |
| #21 | match: develop lost-in-space pyramid matching after local pyramid | PR6 / PR7 / PR8 | Split into offline index, all-sky matcher, and pipeline integration. |
| #22 | Test issue: verify ChatGPT GitHub issue creation | Housekeeping | Close without a code PR. |

Backlog issues:

| Issue | Title | Owner PR |
| --- | --- | --- |
| #23 | PIPE: Honor configured matching algorithm in tracking pipeline | PR1 |
| #24 | PIPE: Add explicit INIT/TRACKING/REACQUIRE/LOST_IN_SPACE state machine | PR5 |
| #25 | LIS: Implement true all-sky lost-in-space recognizer using k-vector pair index | PR7 |
| #26 | LIS: Build offline all-sky guide-star catalog/index generation tool | PR6 |
| #27 | MATCH: Add LocalPairIndex cache for local pyramid matcher | PR3 |
| #28 | MATCH: Add angle-query cache and binned lookup for local pyramid | PR3 |
| #29 | MATCH: Report pyramid rejection diagnostics and nearest-match comparison | PR2 |
| #30 | MATCH: Support geometry-only pyramid expansion for reacquire mode | PR4 |
| #31 | MATCH: Derive matching gates from covariance and mode instead of fixed constants | PR4 / PR19 |
| #32 | PRE: Implement detector calibration chain: bias, dark, flat, bad-pixel mask | PR9 |
| #33 | PRE: Replace global median/std with robust local background and variance model | PR10 |
| #34 | PRE: Add unit-aware electron/DN conversion and Poisson variance propagation | PR10 |
| #35 | PRE: Add cosmic-ray, hot-pixel, and transient artifact rejection | PR11 |
| #36 | EXT: Implement hysteresis segmentation using grow_threshold_sigma | PR12 |
| #37 | EXT: Compute and enforce PSF shape metrics, ellipticity, FWHM, and sharpness | PR12 |
| #38 | EXT: Add PSF-fit or maximum-likelihood centroid with covariance output | PR13 follow-up |
| #39 | EXT: Propagate centroid covariance to ObservedStar LOS covariance and match weights | Completed by PR13/#90 |
| #40 | EXT: Add catalog-aware deblending for close or partially overlapping stars | PR13 follow-up |
| #41 | OPT: Formalize camera-frame conventions and round-trip projection tests for every detector | PR14 |
| #42 | OPT: Replace body_model_proxy with calibrated exact focal-plane adapter for production | PR15 |
| #43 | OPT: Add rolling-shutter and exposure-midpoint timing correction | PR16 |
| #44 | EPH: Apply Gaia proper motion, parallax, radial velocity, and epoch propagation | PR17 |
| #45 | EPH: Use ET/Kepler bandpass magnitude and flux-derived weights consistently | PR17 |
| #46 | EPH: Cache Gaia HEALPix partitions and catalog region queries | PR18 |
| #47 | ATT: Use covariance-weighted Wahba/QUEST instead of SNR-only weights | PR19 follow-up |
| #48 | ATT: Replace single-pass outlier rejection with robust iterative validation | Completed by PR20/#94 |
| #49 | ATT: Add golden quaternion/DCM convention tests and documentation | Completed by PR14/PR19 |
| #50 | VAL: Add full error-budget ledger from detector noise to attitude error | Completed by PR21/#98 |
| #51 | VAL: Add Monte Carlo regression suite across SNR, PSF, background, attitude prior, and artifacts | PR22 |
| #52 | VAL: Implement TE/HSFE/LSFE-style validation scenarios for ET guide requirements | PR22 |
| #53 | VAL: Strengthen catalog-ID truth verification and false-match metrics | PR22 |
| #54 | PERF: Add runtime profiler and CI performance budgets by stage | PR23 |
| #55 | CONFIG: Add typed config schema and validation for missing, unused, and mode-specific keys | PR24 |
| #56 | TEST: Broaden local-pyramid unit and integration tests | PR2 / PR4 |
| #57 | TEST: Add multi-detector end-to-end guide tests with cross-detector geometry | PR15 / PR22 |
| #58 | OBS: Persist structured debug artifacts for every stage and every frame | PR24 |
| #59 | QA: Replace print/silent catalog errors with structured logging and failure policy | PR24 |
| #60 | DOC: Add algorithm-chain design document mapped to star-sensor textbook chapters | PR24 |
| #61 | EXT: Add saturation, nonlinearity, residual-image, and crosstalk guards | PR11 |
| #62 | ATT: Estimate attitude covariance and expose guide-control quality metrics | Completed by PR19/#92 |
| #63 | MATCH: Add ambiguity detection for repeated local geometry and close catalog neighbors | Completed by PR4 |
| #64 | MATCH: Add magnitude/SNR consistency check in matching validation | Completed by PR4 |
| #65 | VAL: Add adversarial tests for prior-attitude errors and reacquire thresholds | PR22 |

## Dependency Path

Core matching, state machine, and lost-in-space path:

```text
PR1 -> PR2 -> PR4 -> PR5 -> PR8
        ^
PR3 ----/
PR6 -> PR7 ----------------/
```

Detector-to-attitude precision path:

```text
PR9 -> PR10 -> PR12 -> PR13 -> PR19 -> PR20 -> PR21 -> PR22
          \-> PR11 -----------------------------/
```

Geometry, catalog, performance, and maintainability support:

```text
PR14 -> PR15 -> PR16
PR17 -> PR18
PR3  -> PR23
PR1  -> PR24
```

## PR Execution Cards

### PR0. housekeeping: close temporary test issue #22 (completed: #22 closed)

| Field | Value |
| --- | --- |
| Phase | M0 / housekeeping |
| Dependencies | None |
| Issues | #22 |
| Parallelism | Execute immediately |
| Risk | No code risk |
| Goal | Close the temporary connector verification issue so backlog statistics stay clean. |

Scope: close #22 only. Do not modify code and do not include this in a release milestone.

Acceptance: #22 is closed and no PR is produced.

### PR1. pipeline: route tracking through configured matcher (completed: GitHub PR #66)

| Field | Value |
| --- | --- |
| Phase | M1 / matcher baseline |
| Dependencies | None |
| Issues | #23, #16 |
| Parallelism | Serial starting point |
| Risk | High, because all tracking results pass through this path |
| Goal | Make tracking honor `match.algorithm` so it can run local pyramid, future lost-in-space, and predicted-position fallback paths. |

Scope:

- In the tracking frame builder, call `match_stars()` after constructing `MatchingContext` instead of calling `associate_nearest()` directly.
- Clarify `match.local_pyramid.enabled` versus `match.algorithm` semantics. The first stage should treat `algorithm` as authoritative; if `enabled=false` conflicts with a pyramid algorithm, emit a warning or config validation failure.
- Preserve predicted-position behavior and existing nearest-neighbor tests.

Key files: `fsglib/pipeline/run_tracking.py`, `fsglib/match/pipeline.py`, `configs/*.yaml`, `tests/test_match.py`, `tests/test_tracking.py`.

Implementation notes:

- Replace `_build_tracking_frame()` matching from `associate_nearest(obs, ref, cfg)` to `match_stars(match_ctx, ref, cfg)`.
- Ensure tracking debug includes `algorithm`, `selected_strategy`, and `requested_mode`.
- Add a regression test with `algorithm=local_pyramid` and a monkeypatched `match_local_pyramid()` to prove routing.

Acceptance:

- Tracking can exercise every `match.algorithm` path.
- The previous predicted-position behavior does not regress.
- Debug payloads explain the requested and selected strategy.

### PR2. match: add local-pyramid diagnostics and independence tests (completed: GitHub PR #70)

| Field | Value |
| --- | --- |
| Phase | M1 / local pyramid evidence |
| Dependencies | PR1 |
| Issues | #29, #56, #15 |
| Parallelism | PR3 can run in parallel |
| Risk | Medium, mostly tests and debug |
| Goal | Explain and verify whether local pyramid has independent reacquire value, rather than merely reproducing predicted-position matching. |

Scope:

- Add seed audit: observed seed IDs, reference seed IDs, pair residuals, seed RMS, and detector scope.
- Add expansion audit: pixel gate rejects, angular gate rejects, and edge counts before and after assignment.
- Add nearest-vs-pyramid comparison: matched count, catalog-ID mapping equality, and attitude residual comparison.
- Build a controlled case where prior or predicted positions are perturbed so nearest matching fails or drops matches while pyramid can recover.

Key files: `fsglib/match/pyramid.py`, `fsglib/match/pipeline.py`, `tests/test_match.py`, `tests/test_tracking.py`, `tests/test_local_pyramid.py`.

Acceptance:

- The nominal case explains why pyramid and nearest agree when they do.
- A perturbed case proves independent pyramid recovery, or the limitation is explicitly recorded.
- The existing local-pyramid PR can be readied or kept draft based on evidence.

### PR3. perf: cache local-pyramid pair index and angle queries (completed: GitHub PR #73)

| Field | Value |
| --- | --- |
| Phase | M1 / local pyramid performance |
| Dependencies | PR2 optional, recommended after PR2 |
| Issues | #27, #28, #19 |
| Parallelism | Can overlap with early PR4 work |
| Risk | Medium, because cache inconsistency can cause wrong matches |
| Goal | Reduce local pyramid pair-index and repeated angle-query cost so it can be used as a reacquire module. |

Scope:

- Add `LocalPyramidCache`.
- Cache `LocalPairIndex` by catalog-ID tuple and relevant epoch or reference-vector hash.
- Cache pair queries by angle bin and tolerance.
- Report cache hit/miss, pair-index build time, and query time in debug.

Key files: `fsglib/match/pyramid.py`, `fsglib/match/pipeline.py`, `tests/test_match.py`, `tests/test_performance.py`.

Acceptance:

- Repeated calls with the same reference list do not rebuild the pair index.
- Cached and uncached results are identical.
- Default local-pyramid runtime has a measurable decrease.

### PR4. match: harden local-pyramid gates for reacquire (completed: GitHub PR #76; follow-ups remain)

| Field | Value |
| --- | --- |
| Phase | M1 / reacquire reliability |
| Dependencies | PR2 |
| Issues | #30, #31, #56, #63, #64, #17, #18 |
| Parallelism | After PR3, or in parallel with careful merge order |
| Risk | High, because gate changes affect false-positive and false-negative behavior |
| Goal | Upgrade local pyramid from a local geometry demo into a robust ET reacquire matcher. |

Scope:

- Separate tracking gates from reacquire gates.
- Use different covariance and gate logic for single-detector and mixed-detector seeds.
- Support geometry-only seed and expansion modes that do not fully depend on predicted positions.
- Add ambiguity detection for repeated local geometry, close catalog neighbors, and multiple equivalent seeds.
- Add magnitude/SNR consistency as a soft score rather than an initial hard gate.

Key files: `fsglib/match/pyramid.py`, `fsglib/match/pipeline.py`, `configs/base.yaml`, `tests/test_match.py`, `tests/test_tracking.py`.

Acceptance:

- Missing-star, false-star, and brightness-ordering changes are covered by tests.
- Mixed-detector coherent offsets are not silently accepted.
- Bad seeds cannot produce high-confidence large-scale mismatches.

### PR5. pipeline: introduce explicit guiding mode state machine (completed: GitHub PR #78)

| Field | Value |
| --- | --- |
| Phase | M1 / operating modes |
| Dependencies | PR1 |
| Issues | #24, #21 |
| Parallelism | PR6 can run in parallel |
| Risk | High, because multi-frame recovery semantics change |
| Goal | Establish flight-like mode control for known-field init, tracking, local reacquire, lost-in-space, and safe lost states. |

Scope:

- Introduce enum-like modes: `INIT_KNOWN_FIELD`, `TRACKING`, `LOCAL_REACQUIRE`, `LOST_IN_SPACE`, `SAFE_LOST`.
- Route repeated tracking failures into local reacquire and repeated reacquire failures into lost-in-space.
- Record transition reason, counters, and requested matcher in state history.
- Stop silently mapping unknown modes to init.

Key files: `fsglib/common/types.py`, `fsglib/pipeline/run_tracking.py`, `fsglib/match/pipeline.py`, `tests/test_tracking.py`.

Acceptance:

- All state transitions are deterministic.
- Sequence results can audit requested mode, selected matcher, and failure reason for every frame.
- Lost-in-space can be reserved as an interface even if it initially returns not implemented.

### PR6. lis: build offline all-sky guide-star index (completed: GitHub PR #71)

| Field | Value |
| --- | --- |
| Phase | M2 / lost-in-space foundation |
| Dependencies | None |
| Issues | #26, #21 |
| Parallelism | Can run beside PR1-PR5 |
| Risk | Medium, because index versioning and catalog provenance must be strict |
| Goal | Provide a reproducible and versioned offline guide-star index for all-sky lost-in-space recognition. |

Scope:

- Add `fsglib/tools/build_lis_index.py`.
- Accept Gaia root, magnitude limits, epoch, isolation radius, and sky tiling.
- Output catalog IDs, inertial vectors, magnitudes, pair arrays, sorted pair angles, k-vector metadata, checksum, and config snapshot.

Key files: `fsglib/tools/build_lis_index.py`, `fsglib/match/lost_in_space.py`, `tests/test_lost_in_space.py`, `docs/*.md`.

Acceptance:

- Index building is deterministic.
- A small fixture verifies pair counts and sorted angles exactly.
- The index load API is usable by the matcher.

### PR7. lis: implement all-sky lost-in-space matcher (completed: GitHub PR #79)

| Field | Value |
| --- | --- |
| Phase | M2 / lost-in-space algorithm |
| Dependencies | PR6 |
| Issues | #25, #21 |
| Parallelism | PR9-PR18 can continue in parallel |
| Risk | High, because all-sky false positives are severe |
| Goal | Implement prior-free all-sky attitude recognition based on angular invariants. |

Scope:

- Query observed star-pair angles.
- Generate triangle or four-star pyramid seeds.
- Score candidates with Wahba/QUEST.
- Handle ambiguity, false stars, and missing stars.
- Return `MatchedStar` and `MatchingResult` without using predicted pixel positions.

Key files: `fsglib/match/lost_in_space.py`, `fsglib/match/pipeline.py`, `tests/test_lost_in_space.py`.

Acceptance:

- Random attitude fixtures with 5-20 stars recover attitude under noise, missing stars, and false stars.
- False-positive rate has an explicit test and threshold.
- A valid solution can be returned without prior attitude.

### PR8. pipeline: integrate lost-in-space recovery mode (completed: GitHub PR #80)

| Field | Value |
| --- | --- |
| Phase | M2 / lost-in-space integration |
| Dependencies | PR5 and PR7 |
| Issues | #21, #24, #25 |
| Parallelism | Waits for PR5 and PR7 |
| Risk | High, because state transitions and matcher validation meet here |
| Goal | Integrate all-sky lost-in-space into sequence tracking so the system can recover from complete loss. |

Scope:

- Implement the `LOST_IN_SPACE` branch in the state machine.
- On success, refresh prior attitude, reference projection, and track table.
- On failure, enter `SAFE_LOST` or the configured retry policy.
- Save LIS debug artifacts to frame metadata.

Key files: `fsglib/pipeline/run_tracking.py`, `fsglib/match/pipeline.py`, `fsglib/common/types.py`, `tests/test_tracking.py`, `tests/test_lost_in_space.py`.

Acceptance:

- End-to-end test covers tracking failure, reacquire failure, LIS success, and return to tracking.
- LIS failure does not pollute track state.

### PR9. preprocess: implement detector calibration chain (completed: GitHub PR #82)

| Field | Value |
| --- | --- |
| Phase | M2 / detector calibration |
| Dependencies | None |
| Issues | #32 |
| Parallelism | Can run beside matching and LIS work |
| Risk | Medium, because extraction inputs change |
| Goal | Upgrade preprocessing from simulation convenience to a detector-aware calibration chain. |

Scope:

- Bias subtraction.
- Dark subtraction with temperature, exposure, or time scaling.
- Flat-field and PRNU correction.
- Bad-pixel mask and FPN residual map.
- Metadata recording for every applied correction.

Key files: `fsglib/preprocess/pipeline.py`, `fsglib/common/types.py`, `configs/base.yaml`, `tests/test_preprocess.py`.

Acceptance:

- Each correction has unit tests.
- Missing calibration data preserves backward-compatible behavior.
- Preprocess metadata records correction provenance.

### PR10. preprocess: add robust background and noise variance model (completed: GitHub PR #85; follow-ups remain)

| Field | Value |
| --- | --- |
| Phase | M2 / noise model |
| Dependencies | PR9 |
| Issues | #33, #34 |
| Parallelism | PR12 can start once the interface is stable |
| Risk | High, because SNR, thresholds, and weights change |
| Goal | Replace global background/std estimation with a CMOS-appropriate local background and variance model. |

Scope:

- Add mesh median, sigma clipping, and optional polynomial local background.
- Model variance from Poisson noise, read noise, dark current, and flat uncertainty.
- Support DN/electron conversion.
- Output `noise_map` and, where needed, `variance_map`.

Key files: `fsglib/preprocess/pipeline.py`, `fsglib/common/types.py`, `configs/base.yaml`, `tests/test_preprocess.py`.

Acceptance:

- Local background beats global median on synthetic background gradients.
- Poisson and read-noise variance match analytical values.
- Extraction SNR has a clear definition.

### PR11. preprocess: reject detector artifacts (completed: GitHub PR #88; follow-up remains)

| Field | Value |
| --- | --- |
| Phase | M2 / detector artifact guards |
| Dependencies | PR9-PR10 |
| Issues | #35, #61 |
| Parallelism | Safer after PR12 |
| Risk | Medium, because over-rejection can remove real stars |
| Goal | Prevent cosmic rays, hot pixels, residual images, nonlinearity, saturation, and crosstalk from silently entering extraction and matching. |

Scope:

- Hot-pixel maps and temporal hot-pixel detection.
- Cosmic-ray sharpness and morphology rejection.
- Saturation and nonlinearity flags.
- Residual-image and crosstalk candidate flags.
- Propagate artifact flags to candidate and observed-star stages.

Key files: `fsglib/preprocess/pipeline.py`, `fsglib/extract/pipeline.py`, `configs/base.yaml`, `tests/test_preprocess.py`, `tests/test_extract.py`.

Acceptance:

- A synthetic cosmic ray is not treated as a valid star.
- Saturated and nonlinear stars are flagged and excluded from high-precision attitude by default.
- Residual-image and crosstalk fixtures are covered.

### PR12. extract: implement hysteresis segmentation and PSF shape filters (completed: GitHub PR #86)

| Field | Value |
| --- | --- |
| Phase | M2 / star extraction |
| Dependencies | PR10 |
| Issues | #36, #37 |
| Parallelism | PR11 can run in parallel |
| Risk | Medium, because source counts change |
| Goal | Make `seed_threshold_sigma` and `grow_threshold_sigma` produce real star regions instead of truncated PSF wings and biased centroids. |

Scope:

- Seed mask above seed threshold.
- Grow mask above grow threshold.
- Connected expansion from seed regions into grow regions.
- Second moments, ellipticity, FWHM, sharpness, and area metrics.
- Shape filters such as `max_ellipticity`.

Key files: `fsglib/extract/pipeline.py`, `fsglib/common/types.py`, `configs/base.yaml`, `tests/test_extract.py`.

Acceptance:

- Changing grow threshold changes segmented area.
- Elongated, trailed, and cosmic-ray-like sources are classified by shape filters.
- Centroid bias decreases on synthetic PSF fixtures.

### PR13. extract: add precision centroiding with covariance and deblending (partially completed: GitHub PR #90; follow-ups remain)

| Field | Value |
| --- | --- |
| Phase | M2 / centroid precision |
| Dependencies | PR12 |
| Issues | #38, #39, #40 |
| Parallelism | PR14 and PR17 can run in parallel |
| Risk | High, because centroid precision is a core module |
| Goal | Add precision centroiding and covariance output to prepare for covariance-weighted attitude solving. |

Scope:

- PSF-fit or maximum-likelihood centroid.
- Centroid covariance derived from noise variance, PSF, and flux.
- Catalog-aware deblending, or at least a close-source flag.
- Pass covariance into `ObservedStar` fields or flags.

Key files: `fsglib/extract/pipeline.py`, `fsglib/pipeline/convert.py`, `fsglib/common/types.py`, `tests/test_extract.py`.

Acceptance:

- Synthetic PSF centroid error is lower than first-moment centroid error.
- Output covariance is comparable to Monte Carlo scatter.
- Blended close sources are not silently treated as high-quality single stars.

### PR14. optics: lock camera-frame conventions with round-trip and golden quaternion tests (completed: GitHub PR #72)

| Field | Value |
| --- | --- |
| Phase | M2 / geometry convention |
| Dependencies | None |
| Issues | #41, #49 |
| Parallelism | Should be done early |
| Risk | High, because coordinate mistakes contaminate every module |
| Goal | Lock pixel-to-LOS, body-to-inertial, DCM, and quaternion conventions with immutable tests. |

Scope:

- Pixel-to-LOS-to-pixel round trips for every detector.
- Center, corner, and edge test vectors.
- Golden tests for `q_ib`, `C_ib`, and scalar-first quaternion conversion.
- Documentation for image axes, focal axes, body axes, and inertial axes.

Key files: `fsglib/ephemeris/projector.py`, `fsglib/attitude/solver.py`, `tests/test_optical.py`, `tests/test_attitude.py`, `docs/*.md`.

Acceptance:

- All detector round-trip residuals are within threshold.
- Quaternion and DCM direction are unambiguous.
- Later PRs can rely on these golden tests.

### PR15. optics: replace body-model proxy with calibrated exact focal-plane adapter (completed: GitHub PR #74; follow-up remains)

| Field | Value |
| --- | --- |
| Phase | M2 / ET geometry adapter |
| Dependencies | PR14 |
| Issues | #42, #57 |
| Parallelism | PR17 and PR18 can run in parallel |
| Risk | High, because guide first-frame results can change |
| Goal | Move production guiding from the body-model proxy toward exact ET focal-plane geometry and quantify proxy residuals. |

Scope:

- Define exact adapter API.
- Keep proxy only as explicit fallback or debug path.
- Add multi-detector end-to-end guide tests.
- Record adapter provenance and fit residuals.

Key files: `fsglib/pipeline/run_guide_init.py`, `fsglib/ephemeris/projector.py`, `configs/*.yaml`, `tests/test_optical.py`, `tests/test_guide_reference_selection.py`.

Acceptance:

- Exact adapter passes synthetic and ET-coordinate fixtures.
- Proxy versus exact differences are quantified.
- Multi-detector tests cover cross-detector consistency.

### PR16. timing: add exposure midpoint and rolling-shutter correction

| Field | Value |
| --- | --- |
| Phase | M2 / timing model |
| Dependencies | PR14 |
| Issues | #43 |
| Parallelism | Safer after PR15 |
| Risk | Medium, because timestamp semantics must be explicit |
| Goal | Upgrade frame time from a scalar into an exposure midpoint and optional rolling-shutter timing model. |

Scope:

- Add exposure start, stop, or midpoint to `RawFrame`.
- Support row-dependent time for rolling shutter.
- Allow projection and reference prediction to use exposure midpoint.
- Record timing model in debug.

Key files: `fsglib/common/types.py`, `fsglib/common/io.py`, `fsglib/ephemeris/pipeline.py`, `fsglib/ephemeris/projector.py`, `configs/base.yaml`, `tests/test_io.py`.

Acceptance:

- Time fields parse clearly.
- A synthetic angular-rate plus rolling-shutter case shows measurable before/after residual change.
- Global-shutter midpoint behavior remains compatible.

### PR17. ephemeris: apply Gaia epoch propagation and bandpass weighting (completed: GitHub PR #77)

| Field | Value |
| --- | --- |
| Phase | M2 / catalog physics |
| Dependencies | Independent, recommended after PR14 |
| Issues | #44, #45 |
| Parallelism | Can run beside PR9-PR16 |
| Risk | Medium, because catalog vectors and weights change |
| Goal | Add Gaia epoch propagation and ET/Kepler bandpass weighting so reference vectors and magnitude weights are physically consistent. |

Scope:

- Proper motion, parallax, radial velocity, and target epoch.
- Gaia G to ET/Kepler magnitude conversion.
- Tie `weight_hint` to expected flux or SNR.
- Record epoch and bandpass in config and metadata.

Key files: `fsglib/ephemeris/catalog.py`, `fsglib/ephemeris/pipeline.py`, `fsglib/common/coords.py`, `configs/base.yaml`, `tests/test_ephemeris.py`.

Acceptance:

- A high-proper-motion fixture matches an Astropy reference.
- Bandpass weight is auditable.
- Reference ranking no longer depends only on Gaia G.

### PR18. catalog: cache Gaia region queries and expose catalog/geometry provenance (completed: GitHub PR #81)

| Field | Value |
| --- | --- |
| Phase | M2 / catalog performance and audit |
| Dependencies | PR17 optional |
| Issues | #46, #20 |
| Parallelism | Can run beside PR15 |
| Risk | Low to medium, with cache consistency risk |
| Goal | Make catalog queries faster and auditable, avoiding silent partition failures and unclear provenance. |

Scope:

- HEALPix partition dataframe cache.
- Region query cache.
- Structured logging instead of `print`.
- Debug provenance for Gaia root, registry, epoch, magnitude limits, bandpass, and missing partitions.

Key files: `fsglib/ephemeris/catalog.py`, `fsglib/ephemeris/pipeline.py`, `fsglib/pipeline/run_guide_init.py`, `tests/test_ephemeris.py`.

Acceptance:

- Repeated queries are faster.
- Catalog provenance is fully traceable in outputs.
- Missing partitions produce structured warnings or errors.

### PR19. attitude: implement covariance-weighted Wahba/QUEST and attitude covariance (partially completed: GitHub PR #92)

| Field | Value |
| --- | --- |
| Phase | M3 / attitude precision |
| Dependencies | PR13 and PR14 |
| Issues | #31, #39, #47, #62 |
| Parallelism | PR20 follows |
| Risk | High, because this is a core numerical path |
| Goal | Move from SNR-only scalar weights to covariance-aware attitude solving and guide-control quality metrics. |

Scope:

- Support LOS covariance or angular variance.
- Derive weights from covariance.
- Output attitude covariance or a local small-angle covariance approximation.
- Keep QUEST or SVD fallback paths.

Key files: `fsglib/attitude/solver.py`, `fsglib/common/types.py`, `fsglib/pipeline/convert.py`, `tests/test_attitude.py`.

Acceptance:

- High-SNR or low-covariance stars receive higher weight.
- Monte Carlo attitude scatter is comparable to predicted covariance.
- Existing attitude tests do not regress.

### PR20. attitude: robust iterative outlier and false-match rejection (completed: GitHub PR #94)

| Field | Value |
| --- | --- |
| Phase | M3 / robust solve |
| Dependencies | PR19 |
| Issues | #48, #63 |
| Parallelism | Complete before PR21 |
| Risk | High, because over-rejection can cause loss of lock |
| Goal | Replace single-pass residual gates with iterative attitude validation for false stars, wrong matches, and repeated-geometry ambiguity. |

Scope:

- Iterative residual rejection.
- Sigma clipping plus hard gate.
- Minimum support and detector diversity.
- False-match quality flags.
- Ambiguity margin passed from matcher into attitude validation.

Key files: `fsglib/attitude/solver.py`, `fsglib/match/pipeline.py`, `tests/test_attitude.py`, `tests/test_match.py`.

Acceptance:

- One wrong match does not contaminate final attitude.
- Multiple wrong matches produce invalid rather than confidently wrong attitude.
- Debug explains every rejection.

### PR21. validation: add detector-to-attitude error-budget ledger (completed: GitHub PR #98; follow-ups remain)

| Field | Value |
| --- | --- |
| Phase | M3 / error closure |
| Dependencies | Core interfaces from PR9-PR20 |
| Issues | #50 |
| Parallelism | PR22 follows |
| Risk | Medium, because units and assumptions must be consistent |
| Goal | Create a traceable error budget from detector noise through attitude error for ET guide precision analysis. |

Scope:

- Detector noise: read, dark, shot, FPN, and PRNU.
- Centroid variance and bias.
- Optical distortion and detector alignment.
- Catalog and ephemeris terms.
- Matching and attitude residual terms.
- JSON or CSV ledger output.

Key files: `fsglib/pipeline/guide_error_audit.py`, `fsglib/pipeline/evaluate.py`, `fsglib/common/types.py`, `tests/test_guide_error_audit.py`.

Acceptance:

- A single frame can output a complete ledger.
- Every term has unit, source, value, and assumption.
- The ledger helps explain final non-roll and roll error.

### PR22. validation: add Monte Carlo and ET TE/HSFE/LSFE scenarios

| Field | Value |
| --- | --- |
| Phase | M3 / validation suite |
| Dependencies | PR21 |
| Issues | #51, #52, #53, #65, #18 |
| Parallelism | PR23 can run in parallel |
| Risk | Medium, mostly test runtime control |
| Goal | Move validation from a few examples to statistical and ET-style performance scenarios. |

Scope:

- Monte Carlo over SNR, PSF, background, artifacts, prior error, false stars, and missing stars.
- TE instantaneous random-error scenarios.
- HSFE pixel phase and local spatial-error scenarios.
- LSFE thermal drift, distortion, and long-term low-frequency scenarios.
- Catalog-ID truth verification and false-match metrics.

Key files: `tests/test_validation_*.py`, `fsglib/pipeline/evaluate.py`, `scripts/*.py`, `configs/*.yaml`.

Acceptance:

- Fixed random-seed regressions exist.
- TE, HSFE, and LSFE each have distinct metrics.
- False-match validation checks catalog-ID correctness, not only match count.

### PR23. perf: add runtime profiler and CI performance budgets by stage (partially covered, not complete)

| Field | Value |
| --- | --- |
| Phase | M4 / performance control |
| Dependencies | PR3; other stages can be enrolled incrementally |
| Issues | #54, #19 |
| Parallelism | Can run beside PR22 |
| Risk | Low to medium, because CI thresholds must be conservative |
| Goal | Establish maintainable runtime budgets for preprocessing, extraction, projection, catalog, matching, attitude, and validation. |

Scope:

- Stage timing context manager.
- Per-frame and per-sequence p50, p95, and p99.
- Cache hit/miss metrics.
- CI performance thresholds.
- Local pyramid and LIS benchmark commands.

Key files: `fsglib/common/debug.py`, `fsglib/pipeline/run_tracking.py`, `fsglib/match/pyramid.py`, `tests/test_performance.py`, `docs/*.md`.

Acceptance:

- Each major stage has timing.
- Local pyramid cache hits are visible.
- CI can catch order-of-magnitude regressions.

### PR24. config/docs/debug: typed config schema and structured artifacts (partially covered, not complete)

| Field | Value |
| --- | --- |
| Phase | M4 / maintainability |
| Dependencies | Can start after PR1 and continue incrementally |
| Issues | #55, #58, #59, #60, #16 |
| Parallelism | Can be interleaved |
| Risk | Low to medium, because config validation affects users |
| Goal | Add long-term maintainability through typed config, unused-key checks, structured debug artifacts, and algorithm-chain design docs. |

Scope:

- Pydantic or dataclass config schema.
- Missing, unused, and mode-specific key validation.
- Structured per-frame artifacts for preprocess, extract, match, attitude, and evaluation.
- Replace print and silent errors with logging and explicit failure policy.
- Design document mapping star-sensor textbook concepts to ET-fsglib modules.

Key files: `configs/*.yaml`, `fsglib/common/debug.py`, `fsglib/ephemeris/catalog.py`, `docs/*.md`, `tests/test_config.py`.

Acceptance:

- Invalid config is reported clearly.
- Unused keys can be audited.
- Per-frame debug artifacts can locate chain-stage failures.
- Developers can understand the full algorithm chain from documentation.

## Recommended Execution Cadence

First execute PR1-PR5. Do not keep more than two active PRs open in this phase. PR1 is the route into the whole roadmap, PR2-PR4 close out the local pyramid branch, and PR5 provides the control plane for lost-in-space integration.

In the second phase, at most four lines can run in parallel: the lost-in-space line (PR6-PR8), the image and centroid line (PR9-PR13), the optics and catalog line (PR14-PR18), and early support work from PR23-PR24.

In the third phase, reduce concurrency to one or two active PRs. PR19-PR22 merge centroid covariance, matching, attitude, and validation into one closed loop and should be reviewed carefully.

## Universal Quality Gates

- Every PR must include at least one unit or integration test proving new behavior.
- Existing tests must keep passing unless the PR explicitly changes behavior and updates acceptance criteria.
- Every PR must add or preserve debug payloads so results remain auditable.
- Production paths must not use silent fallback. Fallback must have an explicit reason.
- Gate and threshold PRs must state their physical basis or temporary assumption.
- Matching PRs must not report match count only; they must report catalog-ID correctness or residual quality.
- Attitude PRs must report RMS, maximum residual, support count, and detector diversity.
- Performance PRs must report before/after runtime or at least stage timing.

## Risk Register

| Risk | Related PRs | Consequence | Control |
| --- | --- | --- | --- |
| Tracking bypasses configured matcher | PR1 | Matcher work cannot be validated end to end | Fix PR1 first. |
| Local pyramid merely reproduces nearest matching | PR2 / PR4 | Reacquire capability is overestimated | Controlled prior-error case and nearest comparison. |
| Local pyramid is too slow | PR3 / PR23 | Reacquire exceeds runtime budget | Pair-index/query cache and runtime budget. |
| Coordinate or quaternion convention is wrong | PR14 | Residuals may look plausible while attitude is physically wrong | Golden DCM/quaternion tests and round-trip projection tests. |
| Catalog epoch or bandpass is inconsistent | PR17 / PR18 | LIS or matching false positives | Provenance, epoch propagation, and bandpass weights. |
| Preprocessing noise model is wrong | PR9 / PR10 | SNR, thresholds, and weights drift | Unit-aware variance model. |
| Centroid covariance is missing | PR13 / PR19 | Attitude weights cannot be physically explained | Propagate centroid covariance to LOS covariance. |
| LIS false positive | PR7 / PR8 / PR22 | Completely wrong attitude can be accepted | Ambiguity margin and Monte Carlo false-match metrics. |
| Validation set is too narrow | PR22 | Examples pass but engineering cases fail | Monte Carlo, TE/HSFE/LSFE, adversarial tests. |

## Branch and Naming Guidance

- PR branch naming: `prXX-short-topic`, for example `pr01-match-routing` or `pr03-pyramid-cache`.
- Issue labels should continue to use `area:*`, `priority:*`, and `type:*`.
- PR title prefixes should use module names: `pipeline:`, `match:`, `lis:`, `preprocess:`, `extract:`, `optics:`, `ephemeris:`, `attitude:`, `validation:`, `perf:`, and `config/docs/debug:`.
- Every PR description should include scope, not-in-scope, linked issues, tests, before/after behavior, risks, and rollback.

## Recommended PR Template

```markdown
## Problem

...

## Scope

- ...

## Not In Scope

- ...

## Linked Issues

- Closes #...

## Algorithm / Physics Rationale

- ...

## Tests

- [ ] Unit tests
- [ ] Integration tests
- [ ] Adversarial / regression case

## Debug / Observability

- ...

## Risk and Rollback

- ...
```

## Shortest Mergeable Path

The shortest route to merge the current local-pyramid work safely is PR1-PR4:

- PR1 fixes tracking matcher routing.
- PR2 proves local pyramid independence.
- PR3 addresses baseline performance.
- PR4 finishes reacquire gates and robustness.

After these four PRs, local pyramid is suitable as the foundation for `LOCAL_REACQUIRE`.

## Final Acceptance Targets

- Normal tracking: predicted-position matching is fast, low-residual, and has high support.
- Local reacquire: when prior or predicted positions have moderate error, local pyramid can recover matching and attitude.
- Lost-in-space: when no prior attitude is available, all-sky matching can return absolute attitude with controlled false-positive rate.
- Image chain: preprocessing, extraction, and centroiding output physically meaningful SNR, variance, and centroid covariance.
- Attitude chain: Wahba/QUEST uses covariance weights and outputs residuals plus attitude covariance.
- Validation chain: TE/HSFE/LSFE, Monte Carlo, adversarial prior-error, false-star, and missing-star scenarios are covered.
- Maintainability: every frame has structured debug, every config key has schema validation, and every runtime stage has a budget.
