# PR6 LIS Index Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic offline all-sky guide-star index builder and loader for future lost-in-space matching.

**Architecture:** PR6 adds a new `fsglib.match.lost_in_space` module for index data structures, NPZ serialization, k-vector generation/query, checksum, and deterministic pair construction. A thin `fsglib.tools.build_lis_index` CLI reads Gaia-style CSV partitions, filters stars, applies the first-version close-neighbor policy, and writes the NPZ bundle. PR7 will consume this index for actual matching; PR6 only provides build/load/query primitives and tests.

**Tech Stack:** Python, NumPy, pandas for CSV ingestion, pytest, existing fsglib package layout.

---

### Task 1: Add Deterministic In-Memory Index Tests

**Files:**
- Create: `tests/test_lost_in_space.py`
- Create: `fsglib/match/lost_in_space.py`

- [ ] **Step 1: Write failing tests for deterministic pair generation**

Add tests that construct five synthetic stars with known unit vectors, call `build_lis_index_from_arrays(...)`, and assert:
- catalog IDs are sorted by magnitude then catalog ID before pair generation.
- pair indices are sorted by angle.
- pair angles are in radians.
- metadata records `epoch`, `bandpass`, `filters`, and a config snapshot.

- [ ] **Step 2: Run the new tests to verify missing module failure**

Run: `pytest tests/test_lost_in_space.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fsglib.match.lost_in_space'`.

- [ ] **Step 3: Implement minimal index data structures and deterministic builder**

Implement `LISIndex`, `build_lis_index_from_arrays(...)`, vector normalization, pair generation, sorted pair arrays, and JSON metadata.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lost_in_space.py -q`
Expected: PASS for deterministic in-memory index tests.

### Task 2: Add K-Vector Query and NPZ Round-Trip

**Files:**
- Modify: `tests/test_lost_in_space.py`
- Modify: `fsglib/match/lost_in_space.py`

- [ ] **Step 1: Write failing tests for query and NPZ round-trip**

Add tests that save an index to `tmp_path / "lis_index.npz"`, load it with `load_lis_index(...)`, and assert:
- catalog arrays, pair arrays, metadata, and checksum survive the round trip.
- `query_pairs_by_angle(...)` returns the exact pair for a known synthetic angle and tolerance.

- [ ] **Step 2: Run tests to verify missing functions fail**

Run: `pytest tests/test_lost_in_space.py -q`
Expected: FAIL with missing `save_lis_index`, `load_lis_index`, or `query_pairs_by_angle`.

- [ ] **Step 3: Implement NPZ save/load and exact k-vector-backed query**

Implement `save_lis_index(...)`, `load_lis_index(...)`, `generate_kvector(...)`, and `query_pairs_by_angle(...)`. Store metadata as `metadata_json`, checksum as `checksum`, and pair angles as `pair_angles_rad`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lost_in_space.py -q`
Expected: PASS for all LIS index tests.

### Task 3: Add Gaia CSV Builder CLI and Ignore Local Index Assets

**Files:**
- Create: `fsglib/tools/build_lis_index.py`
- Modify: `.gitignore`
- Modify: `tests/test_lost_in_space.py`

- [ ] **Step 1: Write failing tests for CSV ingestion and close-neighbor filtering**

Add a test that writes two Gaia-style CSV partitions under `tmp_path`, calls `build_lis_index_from_gaia_csv(...)`, and asserts:
- stars fainter than `mag_limit` are filtered.
- close-neighbor policy keeps the brighter star and drops the fainter neighbor.
- metadata records Gaia root, mag limit, epoch, bandpass, isolation radius, and checksum.

- [ ] **Step 2: Run tests to verify missing builder failure**

Run: `pytest tests/test_lost_in_space.py -q`
Expected: FAIL with missing `build_lis_index_from_gaia_csv`.

- [ ] **Step 3: Implement CSV ingestion and CLI wrapper**

Implement `build_lis_index_from_gaia_csv(...)` in `fsglib.tools.build_lis_index`, plus an `argparse` CLI with `--gaia-root`, `--out`, `--mag-limit`, `--epoch`, `--bandpass`, `--isolation-radius-arcsec`, and optional `--max-files`.

- [ ] **Step 4: Ignore generated LIS NPZ assets**

Add ignore patterns for local/generated LIS index files, including `*.lis_index.npz` and `lis_indexes/`.

- [ ] **Step 5: Run focused and full verification**

Run:
- `pytest tests/test_lost_in_space.py -q`
- `pytest -q`
- `git diff --check`

Expected:
- LIS tests pass.
- Full suite passes with the existing skipped test count.
- No whitespace errors.
