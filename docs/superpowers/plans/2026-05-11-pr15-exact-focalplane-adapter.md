# PR15 Exact Focal-Plane Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace guide-chain body-model proxy geometry with an exact-only ET focal-plane adapter that fails loudly when exact geometry is unavailable.

**Architecture:** Add a small shared adapter in `fsglib/ephemeris/guide_geometry.py` and route `run_guide_init`, `run_guide_truth_noise`, and guide error audit through it. The adapter uses `et_coord` exact pixel-to-sky vectors, fits only a frame-alignment rotation from exact field-angle samples, and rejects `body_model_proxy`.

**Tech Stack:** Python, NumPy, SciPy Rotation/SVD alignment, pytest, local `et_coord` transformer APIs.

---

### Task 1: Add Exact Adapter Red Tests

**Files:**
- Create: `tests/test_guide_geometry_adapter.py`
- Modify: `tests/test_guide_outputs.py`

- [x] **Step 1: Write failing adapter tests**

Add tests for exact-only mode, hard failures when `pixel_to_sky()` lacks an equatorial vector, and serialized payloads that do not expose body-model proxy fields.

- [x] **Step 2: Run red tests**

Run:

```bash
pytest tests/test_guide_geometry_adapter.py tests/test_guide_outputs.py::test_run_guide_first_frame_init_debug_context_is_opt_in -q
```

Expected: FAIL because `fsglib.ephemeris.guide_geometry` and `build_exact_focalplane_geometry_adapter` do not exist yet.

### Task 2: Implement Exact Geometry Adapter

**Files:**
- Create: `fsglib/ephemeris/guide_geometry.py`
- Test: `tests/test_guide_geometry_adapter.py`

- [x] **Step 1: Add `et_field_angles_to_body_vector()`**

Use the existing ET convention: ET field `+X` maps to fsglib body `-X`, ET field `+Y` maps to body `+Y`, and optical axis is `+Z`.

- [x] **Step 2: Add `ExactFocalPlaneGeometryAdapter`**

Expose:
- `pixel_to_focal(detector_id, x_pix, y_pix)`
- `pixel_to_body_los(detector_id, x_pix, y_pix)`
- `serialize()`

- [x] **Step 3: Add exact-only builder**

`build_exact_focalplane_geometry_adapter(cfg, registry, transformer, guide_section="guide_init")` must reject any `los_geometry_mode` other than `exact_et_focalplane`.

- [x] **Step 4: Run adapter tests**

Run:

```bash
pytest tests/test_guide_geometry_adapter.py -q
```

Expected: PASS.

### Task 3: Refactor Guide Workflows

**Files:**
- Modify: `fsglib/pipeline/run_guide_init.py`
- Modify: `fsglib/pipeline/run_guide_truth_noise.py`
- Modify: `tests/test_guide_outputs.py`

- [x] **Step 1: Replace local geometry functions**

Remove `_fit_focal_body_model`, `_focal_mm_to_body_vector`, local exact/proxy branching, and duplicate serializer code from guide workflows.

- [x] **Step 2: Route observed-star LOS through adapter**

Use `geometry_adapter.pixel_to_focal()` for metadata and `geometry_adapter.pixel_to_body_los()` for `ObservedStar.los_body`.

- [x] **Step 3: Replace output payload**

Return the `geometry_adapter` serialized payload. Do not return `geometry_model` or `body_model`.

- [x] **Step 4: Run workflow tests**

Run:

```bash
pytest tests/test_guide_outputs.py tests/test_et_coord_config.py tests/test_guide_reference_selection.py -q
```

Expected: PASS.

### Task 4: Refactor Guide Error Audit

**Files:**
- Modify: `fsglib/pipeline/guide_error_audit.py`

- [x] **Step 1: Use adapter for truth model LOS**

Replace direct `body_model` exact/proxy branching with `geometry_adapter.pixel_to_body_los()`.

- [x] **Step 2: Preserve audit field names**

Keep `truth_model_los_body`, `truth_exact_body`, and existing summary keys stable, but source model LOS from exact adapter only.

- [x] **Step 3: Run audit tests**

Run:

```bash
pytest tests/test_guide_error_audit.py tests/test_guide_outputs.py -q
```

Expected: PASS.

### Task 5: Docs and Verification

**Files:**
- Modify: `docs/yaml_configuration_reference.md`
- Modify: `docs/frame_conventions.md`
- Modify: guide configs if proxy-only keys become stale.

- [x] **Step 1: Document exact-only geometry**

State that `guide_init.los_geometry_mode` and `guide_truth_noise.los_geometry_mode` support only `exact_et_focalplane`; `body_model_proxy` has been removed.

- [x] **Step 2: Run focused and full tests**

Run:

```bash
pytest tests/test_guide_geometry_adapter.py tests/test_guide_outputs.py tests/test_guide_error_audit.py tests/test_optical.py -q
pytest -q
git diff --check
```

Expected: PASS.

- [x] **Step 3: Run real-data smoke tests**

Run guide first-frame scripts/configs for at least:
- `configs/guide_truth_noise_0065pix_exact_etcoord.yaml`
- `configs/guide_microlens_v1_noise_psf_etcoord.yaml`

Expected: commands complete without falling back to a proxy geometry path.
