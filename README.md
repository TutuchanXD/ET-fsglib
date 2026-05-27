# fsglib

Fine Star Guiding Library for ET algorithm verification.

`fsglib` is the Python validation package for the ET fine-guidance algorithm
chain. It connects simulated guide-detector frames to calibration,
preprocessing, star extraction, detector-to-line-of-sight geometry, catalog
matching, QUEST attitude solving, tracking-mode recovery, and structured error
audits.

The maintained path is the ET four-guide-detector workflow. Both ET payload
families are supported:

- transit telescope guide detectors;
- microlensing telescope guide detectors.

The active detector family is selected by YAML configuration. Transit guide
runs use the default `et_focalplane` registry data. Microlensing guide runs use
the microlensing registry data through `ETCoordConfig.microlens_guide_only()`.

## What This Package Does

The guide pipeline currently supports:

1. Loading simulated NPZ frames and simulation truth metadata.
2. Applying detector calibration products and image preprocessing.
3. Extracting star candidates, centroids, shape metrics, and centroid
   covariance estimates.
4. Converting detector pixels to body-frame LOS vectors with exact
   `et_focalplane` geometry.
5. Building Gaia-derived per-detector reference stars.
6. Matching observations with predicted-position, local-pyramid, or
   lost-in-space strategies.
7. Solving inertial-to-body attitude with QUEST and iterative outlier
   rejection.
8. Writing guide error audits, error-budget ledgers, debug bundles, and overlay
   plots.

Truth-noise workflows are also maintained. They bypass real image centroid
extraction, inject controlled detector-pixel centroid noise into truth
positions, and run matching plus attitude solving. Use them to separate
geometry, matching, and QUEST behavior from image-level centroid errors.

## Repository Layout

- `fsglib/`: package source code.
- `configs/`: base YAML plus workflow overlays.
- `examples/`: executable guide, truth-noise, smoke, and generic demos.
- `docs/`: maintained API, debug, frame-convention, configuration, and roadmap
  documents.
- `data/`: small local catalog/index fixtures tracked for tests and examples.
- `tests/`: unit tests for extraction, calibration, matching, attitude,
  tracking, audits, and configuration behavior.
- `requirements.txt`: lightweight local runtime/test dependencies.
- `pyproject.toml`: package metadata and pytest configuration.

Large simulation outputs, long reports, notebooks, and generated debug products
are intentionally kept outside the package or ignored by git.

## Package Architecture

```text
fsglib/
|-- attitude/
|   `-- solver.py                  # QUEST, quaternion/DCM helpers, robust rejection
|-- common/
|   |-- coords.py                  # RA/Dec and unit-vector helpers
|   |-- debug.py                   # debug bundle serialization
|   |-- io.py                      # NPZ frames, batches, truth metadata
|   `-- types.py                   # shared dataclasses and solve states
|-- ephemeris/
|   |-- catalog.py                 # catalog providers
|   |-- guide_geometry.py          # exact ET focal-plane guide adapter
|   |-- pipeline.py                # reference-star orchestration
|   |-- projector.py               # generic projection models
|   `-- types.py                   # catalog/reference/ephemeris types
|-- extract/
|   `-- pipeline.py                # hysteresis segments, centroids, shape checks
|-- match/
|   |-- cache.py                   # local-pyramid cache lifecycle
|   |-- lost_in_space.py           # all-sky LIS index and matcher
|   |-- pipeline.py                # matcher dispatcher and result selection
|   |-- pyramid.py                 # local-pyramid init/reacquire matcher
|   `-- triangle.py                # legacy triangle matcher
|-- models/
|   `-- mock.py                    # local model builder for generic demos/tests
|-- pipeline/
|   |-- centroid_audit.py          # centroid-step audit helpers
|   |-- convert.py                 # candidates -> observed stars
|   |-- error_budget.py            # detector-to-attitude ledger
|   |-- evaluate.py                # frame and sequence metrics
|   |-- guide_error_audit.py       # per-star guide error decomposition
|   |-- guide_outputs.py           # guide output directories and overlays
|   |-- run_guide_init.py          # real-centroid four-guide init
|   |-- run_guide_truth_noise.py   # synthetic truth-noise guide init
|   |-- run_init.py                # generic single-frame API
|   `-- run_tracking.py            # tracking/reacquire/LIS state machine
|-- preprocess/
|   |-- calibration.py             # calibration-product loading
|   `-- pipeline.py                # calibration, background, variance, masks
`-- tools/
    |-- build_gsc.py               # legacy GSC helper
    `-- build_lis_index.py         # offline lost-in-space Gaia index builder
```

## Data Flow

```text
NPZ frames + run metadata
        |
        v
load_npz_frame() -> RawFrame
        |
        v
preprocess_frame() -> PreprocessedFrame
        |
        v
extract_stars() -> list[StarCandidate]
        |
        v
exact ET focal-plane adapter -> list[ObservedStar]
        |
        v
Gaia/et_coord reference query -> list[ReferenceStar]
        |
        v
match_stars() -> MatchingResult
        |
        v
solve_attitude() -> AttitudeSolution
        |
        v
guide_error_audit + error_budget + debug outputs
```

Truth-noise runs start from truth detector coordinates, inject synthetic
centroid offsets, create `ObservedStar` records directly, and then rejoin the
same reference, matching, attitude, and audit stages.

## Main Workflows

### Transit Guide First Frame With Real Centroids

```bash
python examples/run_guide_first_frame.py
```

This merges:

- `configs/base.yaml`
- `configs/guide_v1_noise_psf_etcoord.yaml`

It reads four transit guide-detector batches, extracts image centroids, builds
exact ET focal-plane LOS vectors, matches Gaia reference stars, solves attitude,
and writes compact JSON outputs plus error-budget files under `outputs/debug/`.

### Microlensing Guide First Frame With Real Centroids

```bash
python examples/run_microlens_guide_first_frame.py
```

This merges:

- `configs/base.yaml`
- `configs/guide_microlens_v1_noise_psf_etcoord.yaml`

The microlensing overlay selects:

```yaml
et_coord:
  data_dir: /home/cxgao/ET/et_focalplane/data_microlens
  config_factory: microlens_guide_only
```

When `project.output_dir` is left at the default and the configured dataset
root exists, guide output helpers write to a sibling results tree:

```text
<dataset_root>_fsg-results/frameXXXXXX/debug/
<dataset_root>_fsg-results/frameXXXXXX/figures/
```

### Truth-Noise Guide Runs

```bash
python examples/run_guide_first_frame_truth_noise.py
python examples/run_guide_first_frame_truth_noise_exact.py
```

The first script writes compact debug JSONs. The exact script writes a fuller
run bundle under `project.output_dir`, including config snapshots, matching
records, detector summaries, geometry metadata, validation ledgers, and
matching overlays.

Current truth-noise overlays:

- `configs/guide_truth_noise_0065pix.yaml`
- `configs/guide_truth_noise_0065pix_exact_etcoord.yaml`

### Full Transit Truth-Noise Batch

```bash
python examples/run_transit_full_truth_noise_exact_parallel.py
```

This runs the exact transit truth-noise chain across many frames with
per-frame output directories, status files, optional worker parallelism, and
resume behavior for completed frames. It is intended for larger validation
runs on local or SSHFS result roots.

### Error-Budget Smoke

```bash
python examples/run_pr21_error_budget_smoke.py
```

This is a resource-limited smoke entry point for exercising the detector to
attitude error-budget ledger. It defaults to the truth-noise exact mode and can
also exercise a real-image no-calibration mode via environment variables.

### Generic APIs

The older generic APIs are still present:

- `fsglib.pipeline.run_init.run_single_frame_init`
- `fsglib.pipeline.run_tracking.run_sequence_tracking`

They require externally supplied projection and catalog models. The guide
workflows using `et_coord` are the primary maintained ET validation path.

## Configuration Model

The standard pattern is:

1. Load `configs/base.yaml`.
2. Recursively overlay a workflow-specific YAML file.
3. Let scalar values override base values and lists replace base lists.

`configs/base.yaml` defines common defaults for:

- project output behavior;
- NPZ and truth metadata conventions;
- detector calibration, ADC clipping, and saturation masks;
- background and variance models;
- star extraction, shape filtering, deblending, and centroid covariance;
- predicted-position, local-pyramid, and lost-in-space matching;
- tracking state-machine thresholds;
- Gaia/ephemeris defaults;
- QUEST weights, covariance, and robust rejection;
- guide audits, centroid audits, and error-budget output.

Workflow overlays currently include:

- `configs/guide_v1_noise_psf_etcoord.yaml`: transit real-centroid guide init.
- `configs/guide_microlens_v1_noise_psf_etcoord.yaml`: microlensing
  real-centroid guide init.
- `configs/guide_truth_noise_0065pix.yaml`: compact truth-noise guide init.
- `configs/guide_truth_noise_0065pix_exact_etcoord.yaml`: exact truth-noise
  guide init with full output bundle.
- `configs/main_sim_v2.yaml`: generic single-frame simulation overlay.
- `configs/detector_layout.yaml`: legacy generic detector layout.

For the detailed key-by-key reference, see
`docs/yaml_configuration_reference.md`.

## Matching And Tracking

`match.algorithm` controls guide matching strategy. Active values are:

- `predicted_position`
- `local_pyramid`
- `predicted_position_and_local_pyramid`
- `predicted_position_with_pyramid_reacquire`

Deprecated compatibility values are `triangle` and `local_triangle`.

The local-pyramid matcher supports single-detector and mixed-detector seed
searches, pair-index/query caches, geometry-only reacquire expansion, ambiguity
rejection, and detector-level residual diagnostics.

The tracking state machine uses explicit solve modes:

- `init_known_field`
- `tracking`
- `local_reacquire`
- `lost_in_space`
- `safe_lost`

`lost_in_space` uses a prebuilt LIS index supplied through runtime models. Build
one from Gaia HEALPix CSV partitions with:

```bash
python -m fsglib.tools.build_lis_index \
  --gaia-root /path/to/gaia_root \
  --out /path/to/guide_stars.lis_index.npz \
  --mag-limit 12.5 \
  --epoch 2000.0 \
  --bandpass gaia_g \
  --isolation-radius-arcsec 0.0
```

Small tracked fixtures are available in `data/guide_catalog/` and `data/index/`.

## External Data Dependencies

Guide workflows expect local workstation data roots. The exact paths in the
checked-in YAML files are environment-specific and should be edited or overlaid
when running elsewhere.

### ET focal-plane geometry

```yaml
et_coord:
  src_dir: /home/cxgao/ET/et_focalplane/src
  data_dir: /home/cxgao/ET/et_focalplane/data
```

Microlensing guide detectors use:

```yaml
et_coord:
  src_dir: /home/cxgao/ET/et_focalplane/src
  data_dir: /home/cxgao/ET/et_focalplane/data_microlens
  config_factory: microlens_guide_only
```

### Gaia catalog shards

```yaml
et_coord:
  gaia_root_dir: /home/cxgao/gaia_dr3_19mag
```

Guide configs then constrain catalog depth and selection through fields such as
`catalog_g_mag_min`, `catalog_g_mag_max`, `reference_topk_per_detector`,
`reference_preselect_topk_per_detector`, and
`reference_isolation_radius_pix`.

### Simulated guide frames

Guide dataset roots contain one batch directory per selected detector. Each
batch should include:

- `frames/*.npz`
- `run_meta.json`
- `stars.ecsv`

Detector/batch pairs are configured in `guide_init.detector_batches` or
`guide_truth_noise.detector_batches`.

## Outputs

Compact examples write top-level JSON products such as:

- `*_result.json`
- `*_error_audit.json`
- `*_error_budget.json`
- `*_error_budget_terms.csv`

Full debug bundles may include:

- raw, preprocessed, background, variance, and mask arrays;
- extracted candidates and truth stars;
- reference stars, observed stars, matched stars, and matching debug metadata;
- attitude solution summaries, covariance, and robust-rejection audit;
- guide error audit and error-budget ledger;
- config snapshots and run metadata;
- matching overlays and residual vector plots.

See `docs/README_debug.md` for the bundle layout and field notes.

## Installation

Use Python 3.10 or newer.

```bash
cd /home/cxgao/ET/FSG/fsglib
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Some workflows also need local packages or data outside `requirements.txt`,
including:

- `astropy`, for reading simulation truth tables;
- `pandas`, for building LIS indexes from Gaia CSV partitions;
- `et_coord` from `et_focalplane`;
- local Gaia catalog shards;
- simulated guide-detector frame batches from Photosim runs.

## Documentation Index

- `docs/README_API.md`: API and pipeline notes.
- `docs/README_debug.md`: debug-output layout.
- `docs/frame_conventions.md`: image, detector, ET focal-plane, and attitude
  frame conventions.
- `docs/yaml_configuration_reference.md`: maintained YAML reference.
- `docs/local_pyramid_matching_plan.md`: local-pyramid design and migration
  notes.
- `docs/roadmaps/guide_algorithm_maintenance_roadmap.md`: guide algorithm
  maintenance roadmap.

## Running Tests

```bash
pytest -q
```

The test suite covers core math and data-flow components, including:

- preprocessing calibration, variance, ADC, and saturation behavior;
- star extraction, shape filters, blend flags, and centroid covariance;
- predicted-position, local-pyramid, cache, triangle, and lost-in-space
  matching;
- QUEST attitude solving, covariance, and robust rejection;
- guide geometry adapter and ET-coordinate configuration selection;
- guide outputs, audits, error-budget ledgers, and tracking state transitions.

## Development Notes

- Keep detector-family selection in YAML configuration. Do not hard-code
  transit or microlensing detector assumptions in shared solver code.
- Prefer exact `et_focalplane` geometry for guide-detector validation.
- Use truth-noise runs to isolate algorithmic effects before interpreting
  residuals from real image centroid extraction.
- Keep in-repository docs operational and concise. Long reports, notebooks,
  generated outputs, and exploratory artifacts should stay outside the package
  or under ignored paths.
