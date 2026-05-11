# PR14 Frame Conventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock ET-fsglib image, focal-plane, body-frame, inertial-frame, DCM, and quaternion conventions with deterministic tests and documentation.

**Architecture:** PR14 adds convention-focused tests around existing projector and attitude solver behavior, plus small helper functions where needed to make SciPy quaternion conversion explicit. It documents the conventions in a dedicated frame-conventions document. Large geometry changes, exact ET focal-plane adapter replacement, and tracking integration remain out of scope.

**Tech Stack:** Python, NumPy, SciPy Rotation, pytest, Markdown docs.

---

### Task 1: Lock Quaternion and DCM Conventions

**Files:**
- Modify: `tests/test_attitude.py`
- Modify: `fsglib/attitude/solver.py`

- [x] **Step 1: Write failing golden quaternion tests**

Add tests that assert:
- `q_ib` is scalar-first `[w, x, y, z]`.
- `quat_to_dcm([cos(45deg), 0, 0, sin(45deg)])` rotates inertial `+X` into body `+Y`.
- `dcm_to_quat()` returns the same scalar-first sign-normalized quaternion.
- a helper named `scalar_first_quat_to_scipy_xyzw()` converts `[w, x, y, z]` to SciPy `[x, y, z, w]`.

- [x] **Step 2: Run the targeted test and verify the helper is missing**

Run: `pytest tests/test_attitude.py::test_scalar_first_quaternion_convention_matches_scipy_rotation -q`
Expected: FAIL because `scalar_first_quat_to_scipy_xyzw` is not defined.

- [x] **Step 3: Implement minimal helper and use it in attitude solver**

Add `scalar_first_quat_to_scipy_xyzw()` and `scipy_xyzw_to_scalar_first_quat()` to `fsglib/attitude/solver.py`, then use the helper in `quat_to_dcm()`.

- [x] **Step 4: Run attitude tests**

Run: `pytest tests/test_attitude.py -q`
Expected: PASS.

### Task 2: Lock Projector Pixel/LOS Round Trips

**Files:**
- Modify: `tests/test_optical.py`
- Modify: `fsglib/ephemeris/projector.py`

- [x] **Step 1: Write failing convention tests**

Add tests that assert:
- every projection model round-trips center, edge, and corner pixels.
- a non-identity detector mounting matrix rotates detector-local `+Z` into the expected body vector at principal point.
- `project_to_detectors()` uses scalar-first `q_ib` and inertial-to-body convention.

- [x] **Step 2: Run the targeted optical tests**

Run: `pytest tests/test_optical.py -q`
Expected: FAIL if project_to_detectors does not expose shared quaternion conversion helper or if a convention mismatch exists.

- [x] **Step 3: Implement minimal projector cleanup if required**

Use the attitude helper in `project_to_detectors()` and correct only small comment/name mismatches unless a test exposes an actual sign or direction bug.

- [x] **Step 4: Run optical tests**

Run: `pytest tests/test_optical.py -q`
Expected: PASS.

### Task 3: Document Frame Conventions

**Files:**
- Create: `docs/frame_conventions.md`

- [x] **Step 1: Write documentation**

Document:
- image pixel `u/x` increases right and `v/y` increases down.
- detector-local optical axis is `+Z`.
- detector-local `+X/+Y` follows the projector pixel-offset formula.
- `mounting_matrix` maps detector-local vectors into body frame.
- `C_ib` and `q_ib` map inertial vectors into body vectors.
- quaternions are stored scalar-first `[w, x, y, z]`; SciPy APIs require `[x, y, z, w]`.

- [x] **Step 2: Run final verification**

Run:
- `pytest tests/test_optical.py tests/test_attitude.py -q`
- `pytest -q`
- `git diff --check`

Expected: focused and full tests pass with no whitespace errors.
