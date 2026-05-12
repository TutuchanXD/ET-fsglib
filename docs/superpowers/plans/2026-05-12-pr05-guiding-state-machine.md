# Guiding State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce explicit guide operating modes so tracking, local reacquire, lost-in-space, and safe-lost transitions are deterministic and auditable.

**Architecture:** Add enum-like mode constants while preserving string serialization. `run_sequence_tracking()` dispatches by explicit mode instead of collapsing unknown modes to init. Tracking frames keep the predicted-position policy, local reacquire frames use the PR4 `predicted_position_with_pyramid_reacquire` policy, and lost-in-space is a deliberate not-implemented placeholder that can transition to safe lost.

**Tech Stack:** Python dataclasses/strings, pytest table-driven state tests, existing frame/matcher pipeline, GitHub PR workflow.

---

### Task 1: Mode Constants And State Payload

**Files:**
- Modify: `fsglib/common/types.py`
- Test: `tests/test_tracking.py`

- [x] Add failing tests for canonical mode names: `init_known_field`, `tracking`, `local_reacquire`, `lost_in_space`, `safe_lost`.
- [x] Add state fields for `requested_mode`, `requested_match_algorithm`, `selected_match_strategy`, and `validation_reason`.
- [x] Preserve compatibility with existing `SolveStateMachine(mode="init")` and `mode="tracking"` call sites.

### Task 2: Table-Driven Transitions

**Files:**
- Modify: `fsglib/pipeline/run_tracking.py`
- Test: `tests/test_tracking.py`

- [x] Add failing table-driven transition tests for:
  - init-known-field success -> tracking.
  - tracking failure below threshold -> tracking.
  - repeated tracking failure -> local_reacquire.
  - local_reacquire success -> tracking.
  - repeated local_reacquire failure -> lost_in_space.
  - lost_in_space failure -> safe_lost.
  - safe_lost remains safe_lost.
- [x] Implement `normalize_solve_mode()` and explicit transition logic.
- [x] Replace old `reacquire_init` / `lost_after_init_failures` reasons with explicit reasons while preserving summary compatibility where useful.

### Task 3: Per-Mode Frame Dispatch

**Files:**
- Modify: `fsglib/pipeline/run_tracking.py`
- Test: `tests/test_tracking.py`

- [x] Add failing tests proving `run_sequence_tracking()` does not silently map non-init/tracking modes back to init.
- [x] Add `_build_local_reacquire_frame()` as a small wrapper around the tracking frame builder that overrides matching algorithm to `predicted_position_with_pyramid_reacquire`.
- [x] Add `_build_lost_in_space_frame()` placeholder that returns an invalid frame with reason `lost_in_space_not_implemented`.
- [x] Record `requested_mode`, `requested_match_algorithm`, `selected_match_strategy`, and `validation_reason` on each frame meta and copied state history entry.

### Task 4: Matcher Policy And Config

**Files:**
- Modify: `configs/base.yaml`
- Modify: `docs/yaml_configuration_reference.md`
- Test: `tests/test_tracking.py`

- [x] Add explicit tracking config keys:
  - `reacquire_after_tracking_failures`
  - `lost_in_space_after_reacquire_failures`
  - `safe_lost_after_lis_failures`
  - `tracking_match_algorithm`
  - `local_reacquire_match_algorithm`
  - `lost_in_space_match_algorithm`
- [x] Keep `reacquire_after_failures` and `lost_after_init_failures` as compatibility aliases.
- [x] Document the explicit mode names and PR5 lost-in-space placeholder.

### Task 5: Verification And Publish

**Files:**
- All modified files

- [x] Run focused tests:
  - `pytest tests/test_tracking.py tests/test_match.py tests/test_pyramid.py -q`
- [x] Run full verification:
  - `pytest -q`
  - `git diff --check`
- [ ] Commit, push `pipeline/pr05-guiding-state-machine`, and open a PR against `main`.
