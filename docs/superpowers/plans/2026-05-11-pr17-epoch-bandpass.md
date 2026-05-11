# PR17 Epoch Propagation and Bandpass Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generic reference-star generation epoch-aware and give reference stars auditable bandpass/flux-derived weights.

**Architecture:** Keep PR17 scoped to ephemeris/catalog/reference construction. `HealpixCatalogProvider` parses Gaia provenance fields, `fsglib.ephemeris.pipeline` owns astrometric propagation and reference weighting, and guide `et_coord` workflows continue delegating epoch propagation to `query_detector_sources(target_epoch=...)` while recording compatible metadata.

**Tech Stack:** Python 3.10+, NumPy, Astropy `SkyCoord.apply_space_motion`, pytest.

---

### Task 1: Catalog Provenance Fields

**Files:**
- Modify: `fsglib/ephemeris/types.py`
- Modify: `fsglib/ephemeris/catalog.py`
- Test: `tests/test_ephemeris.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that `HealpixCatalogProvider` maps `ref_epoch` into `CatalogStar.ref_epoch` and leaves missing radial velocity as `None` instead of `0.0`.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_ephemeris.py -q`
Expected: FAIL because `CatalogStar.ref_epoch` does not exist and catalog parsing still uses `rv_km_s=0.0`.

- [ ] **Step 3: Implement catalog parsing**

Add optional `ref_epoch` to `CatalogStar`; parse `ref_epoch` and optional RV columns in `HealpixCatalogProvider`.

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/test_ephemeris.py -q`
Expected: PASS.

### Task 2: Epoch-Aware Reference LOS

**Files:**
- Modify: `fsglib/ephemeris/pipeline.py`
- Test: `tests/test_ephemeris.py`

- [ ] **Step 1: Write failing tests**

Add tests for a high proper-motion `CatalogStar` where `build_reference_stars()` returns a propagated LOS and metadata for original/propagated coordinates, target epoch, and correction status.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_ephemeris.py -q`
Expected: FAIL because `build_reference_stars()` still uses static RA/Dec.

- [ ] **Step 3: Implement propagation helper**

Add `catalog_star_to_unit_vector(star, cfg, target_epoch=None)` using Astropy when `ephemeris.enable_proper_motion` is true. Use `ephemeris.target_epoch`, default `2000.0`; use `star.ref_epoch`, default `2016.0`; omit RV when missing/non-finite.

- [ ] **Step 4: Wire reference generation**

Use the helper in `build_reference_stars()` and populate `ReferenceStar.meta` with astrometry provenance.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/test_ephemeris.py -q`
Expected: PASS.

### Task 3: Bandpass/Flux-Derived Weighting

**Files:**
- Modify: `fsglib/ephemeris/pipeline.py`
- Modify: `fsglib/pipeline/run_guide_init.py`
- Test: `tests/test_ephemeris.py`
- Test: `tests/test_guide_reference_selection.py`

- [ ] **Step 1: Write failing tests**

Assert that generic references derive `weight_hint` from converted Kepler magnitude when available, falling back to Gaia G, and that guide references expose the same weight/provenance shape without reimplementing `et_coord` epoch propagation.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_ephemeris.py tests/test_guide_reference_selection.py -q`
Expected: FAIL because all reference weights are currently `1.0`.

- [ ] **Step 3: Implement weighting helpers**

Add `magnitude_to_flux_weight()` and `reference_weight_from_magnitudes()` in `fsglib.ephemeris.pipeline`.

- [ ] **Step 4: Apply generic and guide weights**

Set `ReferenceStar.weight_hint` from bandpass/flux proxy and record `weight_source`, `weight_magnitude`, `flux_weight`, and `target_epoch` in metadata.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/test_ephemeris.py tests/test_guide_reference_selection.py -q`
Expected: PASS.

### Task 4: Config and Documentation

**Files:**
- Modify: `configs/base.yaml`
- Modify: `docs/yaml_configuration_reference.md`

- [ ] **Step 1: Update config**

Add `ephemeris.target_epoch: 2000.0` and `ephemeris.reference_epoch_default: 2016.0`.

- [ ] **Step 2: Update docs**

Mark `ephemeris.enable_proper_motion` active for generic `HealpixCatalogProvider` reference generation and document the new epoch keys plus weight provenance.

- [ ] **Step 3: Verify docs/config parse**

Run: `python -m py_compile fsglib/ephemeris/pipeline.py fsglib/ephemeris/catalog.py`
Expected: PASS.

### Task 5: Final Verification and Publish

**Files:**
- All changed files.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_ephemeris.py tests/test_guide_reference_selection.py -q`
Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 3: Check whitespace**

Run: `git diff --check`
Expected: PASS.

- [ ] **Step 4: Commit and push**

Commit message: `ephemeris: apply epoch propagation and bandpass weights`

- [ ] **Step 5: Open PR**

Open PR from `ephemeris/pr17-epoch-bandpass` to `main` with scope, behavior changes, tests, and PR4 conflict note.
