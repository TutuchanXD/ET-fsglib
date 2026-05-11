# Local Pyramid Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable local-pyramid cache so repeated guide frames do not rebuild the O(Nref^2) pair index and can optionally reuse exact angle-window query results.

**Architecture:** `LocalPyramidCache` lives in `fsglib/match/pyramid.py` beside `LocalPairIndex`. `match_local_pyramid(..., cache=None)` uses it for pair-index and query lookup while keeping the no-cache path behavior identical. `MatchingContext` carries an optional cache so `match_stars()`, init, and tracking can reuse one cache stored in `models["match_cache"]`.

**Tech Stack:** Python dataclasses, NumPy arrays, pytest regression tests, existing `MatchingResult.debug` payloads.

---

### Task 1: Pair Index Cache

**Files:**
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] **Step 1: Write failing cache reuse test**

Add this import:

```python
from fsglib.match.pyramid import LocalPyramidCache, match_local_pyramid
```

Add this test:

```python
def test_local_pyramid_reuses_pair_index_cache_for_same_reference_geometry():
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
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_pyramid.py::test_local_pyramid_reuses_pair_index_cache_for_same_reference_geometry -q`

Expected: FAIL because `LocalPyramidCache` cannot be imported or `match_local_pyramid()` does not accept `cache`.

- [ ] **Step 3: Implement minimal pair-index cache**

Add `LocalPyramidCache` with:

```python
@dataclass
class LocalPyramidCache:
    pair_index_by_key: dict[tuple, LocalPairIndex] = field(default_factory=dict)
    pair_index_hits: int = 0
    pair_index_misses: int = 0
    pair_index_build_time_s: float = 0.0

    def reset_stats(self) -> None:
        self.pair_index_hits = 0
        self.pair_index_misses = 0
        self.pair_index_build_time_s = 0.0

    def get_pair_index(self, reference_stars: list[ReferenceStar]) -> LocalPairIndex:
        key = _reference_cache_key(reference_stars)
        if key in self.pair_index_by_key:
            self.pair_index_hits += 1
            return self.pair_index_by_key[key]
        self.pair_index_misses += 1
        start = perf_counter()
        pair_index = _build_local_pair_index(reference_stars)
        self.pair_index_build_time_s += perf_counter() - start
        self.pair_index_by_key[key] = pair_index
        return pair_index
```

Use it in `match_local_pyramid(..., cache=None)`. The no-cache path should build directly and report one miss plus build time.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_pyramid.py::test_local_pyramid_reuses_pair_index_cache_for_same_reference_geometry -q`

Expected: PASS.

### Task 2: Cache Key Invalidation

**Files:**
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] **Step 1: Write failing invalidation test**

```python
def test_local_pyramid_pair_index_cache_misses_when_reference_los_changes():
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
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_pyramid.py::test_local_pyramid_pair_index_cache_misses_when_reference_los_changes -q`

Expected: FAIL if the key only includes catalog IDs.

- [ ] **Step 3: Include geometry in cache key**

Implement `_reference_cache_key(reference_stars)` as a tuple of per-star entries:

```python
(
    int(ref.catalog_id),
    float(ref.time_s),
    tuple(round(float(value), 15) for value in np.asarray(ref.los_inertial, dtype=np.float64)),
    tuple(sorted(ref.detector_ids_visible, key=str)),
)
```

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_pyramid.py::test_local_pyramid_pair_index_cache_misses_when_reference_los_changes -q`

Expected: PASS.

### Task 3: Query Cache

**Files:**
- Modify: `fsglib/match/pyramid.py`
- Test: `tests/test_pyramid.py`

- [ ] **Step 1: Write failing query-cache test**

```python
def test_local_pyramid_reuses_angle_query_cache_without_changing_matches():
    refs = _reference_stars()
    observed = _observed_from_refs(refs)
    cfg = _cfg()
    cache = LocalPyramidCache()

    baseline = match_local_pyramid(observed, refs, cfg)
    first = match_local_pyramid(observed, refs, cfg, cache=cache)
    second = match_local_pyramid(observed, refs, cfg, cache=cache)

    assert [match.catalog_id for match in first.matched] == [match.catalog_id for match in baseline.matched]
    assert [match.catalog_id for match in second.matched] == [match.catalog_id for match in baseline.matched]
    assert second.debug["angle_query_cache"]["hits"] > 0
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_pyramid.py::test_local_pyramid_reuses_angle_query_cache_without_changing_matches -q`

Expected: FAIL because angle-query cache debug is missing.

- [ ] **Step 3: Implement cached query lookup**

Add `LocalPyramidCache.query_pairs(pair_index, angle_rad, tolerance_rad)`. The key uses `id(pair_index)`, `round(angle_rad / bin_width)`, and `round(tolerance_rad, 15)` where `bin_width = max(tolerance_rad / 4.0, 1e-15)`. After retrieving a cached bin, apply exact `abs(pair.angle_rad - angle_rad) <= tolerance_rad` filtering before returning.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_pyramid.py::test_local_pyramid_reuses_angle_query_cache_without_changing_matches -q`

Expected: PASS.

### Task 4: Pipeline Integration

**Files:**
- Modify: `fsglib/common/types.py`
- Modify: `fsglib/match/pipeline.py`
- Modify: `fsglib/pipeline/run_init.py`
- Modify: `fsglib/pipeline/run_tracking.py`
- Modify: `fsglib/models/mock.py`
- Test: `tests/test_match.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write failing match_stars cache propagation test**

```python
def test_match_stars_passes_context_cache_to_local_pyramid():
    from fsglib.match.pyramid import LocalPyramidCache
    from tests.test_pyramid import _cfg, _observed_from_refs, _reference_stars

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

    match_stars(ctx, refs, cfg)
    result = match_stars(ctx, refs, cfg)

    assert result.debug["pyramid_debug"]["pair_index_cache"]["hits"] == 1
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_match.py::test_match_stars_passes_context_cache_to_local_pyramid -q`

Expected: FAIL because `MatchingContext` has no `match_cache` field.

- [ ] **Step 3: Implement cache propagation**

Add `match_cache: Any | None = None` to `MatchingContext`. Pass `ctx.match_cache` into `match_local_pyramid()`. In `run_init.py` and `run_tracking.py`, create or reuse `models["match_cache"]`. In `models/mock.py`, include `"match_cache": LocalPyramidCache()`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_match.py::test_match_stars_passes_context_cache_to_local_pyramid -q`

Expected: PASS.

### Task 5: Full Verification

**Files:**
- All modified files

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/test_pyramid.py tests/test_match.py tests/test_tracking.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Run full suite**

Run: `pytest -q`

Expected: full suite passes.

- [ ] **Step 3: Run whitespace check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Inspect final diff**

Run: `git diff --stat`

Expected: changed files are limited to PR3 plan, matcher cache implementation, context/pipeline cache propagation, and tests.
