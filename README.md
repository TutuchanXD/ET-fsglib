# fsglib

Fine Star Guiding Library for ET algorithm verification.

`fsglib` is a Python package for prototyping, validating, and auditing the ET
fine-guidance attitude initialization chain. It connects simulated guide
detector images to star extraction, detector-to-line-of-sight geometry,
catalog matching, QUEST attitude solving, and error decomposition.

The current repository is focused on first-frame guide-star initialization for
the ET payloads. ET has two science payload families:

- the transit telescope guide detectors;
- the microlensing telescope guide detectors.

Both payloads use four guide detectors. The active detector family is selected
by configuration, not by hard-coded detector names in the solver. Transit guide
workflows use the default `et_focalplane` registry data, while microlensing
guide workflows use the microlensing registry data and the
`ETCoordConfig.microlens_guide_only()` factory.

## What This Package Does

The package provides a complete guide-initialization chain:

1. Load one simulated frame from each guide detector.
2. Preprocess images and estimate background/noise.
3. Extract star candidates and compute real centroids.
4. Convert detector pixels to body-frame line-of-sight vectors using
   `et_focalplane` geometry.
5. Query Gaia-derived reference stars for each detector.
6. Match observed stars to reference stars.
7. Solve the initial attitude with QUEST.
8. Reject outliers, evaluate residuals, and write debug/audit products.

It also contains truth-noise workflows that bypass real centroid extraction and
inject controlled centroid noise into truth positions. Those workflows are used
to isolate geometry, matching, and attitude-solving errors from image-level
centroid errors.

## Repository Layout

- `fsglib/`: package source code.
- `configs/`: YAML configuration files for common runs.
- `examples/`: executable scripts for transit, microlensing, truth-noise, and
  generic single-frame demos.
- `docs/`: compact maintained documentation. Longer reports and notebook
  materials are intentionally kept outside this repository.
- `tests/`: unit tests for extraction, matching, attitude solving,
  ET-coordinate configuration, and evaluation helpers.
- `requirements.txt`: lightweight runtime/test dependencies used by the local
  development environment.
- `pyproject.toml`: package metadata and pytest configuration.

## Package Architecture

```
fsglib/
├── attitude/              # Attitude solving (QUEST algorithm)
│   ├── solver.py          #   solve_quest, solve_attitude, reject_outliers
│   └── __init__.py
├── common/                # Shared infrastructure
│   ├── coords.py          #   RA/Dec ↔ unit vector conversions
│   ├── io.py              #   NPZ frame loading, dataset batch I/O, truth table parsing
│   ├── types.py           #   20+ dataclasses covering the full pipeline data model
│   └── debug.py           #   Debug helpers
├── ephemeris/             # Ephemeris and reference star management
│   ├── catalog.py         #   Star catalog query interface
│   ├── pipeline.py        #   Ephemeris pipeline orchestration
│   ├── projector.py       #   Sky-to-detector projection models
│   └── types.py           #   CatalogStar, ReferenceStar, EphemerisContext
├── extract/               # Star candidate extraction
│   ├── pipeline.py        #   extract_stars: island detection + centroid computation
│   └── bias.py            #   Centroid bias prediction and correction
├── match/                 # Star matching
│   ├── pipeline.py        #   match_stars: predicted-position + triangle matching
│   └── triangle.py        #   TriangleMatcher implementation
├── models/                # Optical and detector models
│   └── mock.py            #   Mock models for testing
├── pipeline/              # High-level orchestration (core business logic)
│   ├── run_guide_init.py          # Guide first-frame init main entry point
│   ├── run_guide_truth_noise.py   # Truth-noise workflow (bypasses real centroid extraction)
│   ├── run_init.py                # Generic single-frame initialization API
│   ├── run_tracking.py            # Multi-frame sequence tracking
│   ├── evaluate.py                # Frame evaluation and sequence summary metrics
│   ├── guide_error_audit.py       # Per-star error decomposition and counterfactual analysis
│   ├── centroid_audit.py          # Centroid step-by-step audit
│   └── convert.py                 # Format conversion utilities
├── preprocess/            # Image preprocessing
│   └── pipeline.py        #   preprocess_frame: background subtraction + noise estimation
└── tools/                 # Build and maintenance utilities
    └── build_gsc.py       #   GSC star catalog builder
```

### Module Responsibilities

| Module | Role | Key Functions |
|--------|------|---------------|
| `common` | Data types, coordinate math, file I/O | `load_npz_frame`, `radec_to_unit_vector`, 20 dataclasses |
| `preprocess` | Image conditioning | `preprocess_frame` — median background subtraction, std-dev noise map |
| `extract` | Star detection and centroiding | `extract_stars` — SNR-threshold island detection, weighted-centroid / fixed-window first-moment |
| `ephemeris` | Reference star query and projection | Gaia catalog queries, sky-to-detector projection, proper-motion correction |
| `match` | Observed-to-reference star association | Predicted-position nearest-neighbour, Hungarian unique assignment, triangle matching |
| `attitude` | QUEST attitude solving | `solve_quest`, `solve_attitude` with iterative outlier rejection, degraded-level detection |
| `pipeline` | End-to-end workflow orchestration | Guide first-frame init, truth-noise runs, sequence tracking, error audit |

### Data Flow

```
Simulated NPZ frames
        │
        ▼
  [common.io]  load_npz_frame()  ──►  RawFrame
        │
        ▼
  [preprocess] preprocess_frame()  ──►  PreprocessedFrame
        │
        ▼
  [extract]   extract_stars()  ──►  list[StarCandidate]
        │
        ▼
  [pipeline]  sim→detector coord bridge  ──►  list[ObservedStar]
              (offset or affine transform)
        │
        ▼
  [ephemeris] Gaia catalog query  ──►  list[ReferenceStar]
        │
        ▼
  [match]     match_stars()  ──►  MatchingResult
        │
        ▼
  [attitude]  solve_attitude()  ──►  AttitudeSolution
        │
        ▼
  [pipeline]  compute_guide_error_audit()  ──►  error audit JSON
```

The pipeline supports two detector families — **transit** and **microlensing** — each with
four guide detectors. The active family is selected via `et_coord.config_factory` in the
YAML configuration, not by hard-coded detector names in the solver.

## Main Workflows

### Transit Guide First Frame With Real Centroids

Use this workflow when the simulated images were produced for the transit guide
detectors and centroid extraction should be part of the chain.

```bash
python examples/run_guide_first_frame.py
```

The script merges:

- `configs/base.yaml`
- `configs/guide_v1_noise_psf_etcoord.yaml`

It reads four guide-detector simulation batches, extracts centroids, maps them
through the exact ET focal-plane geometry, matches them against Gaia reference
stars, solves attitude, and writes:

- `outputs/debug/guide_first_frame_v1_noise_psf_result.json`
- `outputs/debug/guide_first_frame_v1_noise_psf_error_audit.json`

### Microlensing Guide First Frame With Real Centroids

Use this workflow when the images were generated for the microlensing guide
detectors. The geometry must come from the microlensing `et_focalplane`
configuration.

```bash
python examples/run_microlens_guide_first_frame.py
```

The script merges:

- `configs/base.yaml`
- `configs/guide_microlens_v1_noise_psf_etcoord.yaml`

The microlensing configuration selects:

```yaml
et_coord:
  data_dir: /home/cxgao/ET/et_focalplane/data_microlens
  config_factory: microlens_guide_only
```

This is the key distinction from the transit guide workflow. It ensures that
the four guide detectors are interpreted as the microlensing guide detectors
rather than the transit guide detectors.

The script writes:

- `outputs/debug/microlens_guide_first_frame_v1_noise_psf_result.json`
- `outputs/debug/microlens_guide_first_frame_v1_noise_psf_error_audit.json`

### Truth-Noise Guide Runs

Truth-noise runs are used for controlled attitude-chain verification. They read
truth star positions from simulation metadata, inject prescribed detector-pixel
noise, and then run matching and attitude solving.

Typical entry points:

```bash
python examples/run_guide_first_frame_truth_noise.py
python examples/run_guide_first_frame_truth_noise_exact.py
```

Use these scripts when you want to separate the effects of image-level centroid
measurement from geometry, matching, and QUEST residual behavior.

### Generic Single-Frame and Tracking APIs

The repository also contains lower-level generic APIs:

- `fsglib.pipeline.run_init.run_single_frame_init`
- `fsglib.pipeline.run_tracking.run_sequence_tracking`

These interfaces depend on externally supplied projection and catalog models.
The guide-specific ET-coordinate workflows are currently the primary maintained
path for ET guide-detector validation.

## Installation

Use Python 3.10 or newer.

```bash
cd /home/cxgao/ET/FSG/fsglib
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Some guide workflows also require packages and local modules that are provided
by neighboring ET repositories or the workstation environment, most notably:

- `astropy`, for reading simulation truth tables;
- `et_coord` from `et_focalplane`;
- local Gaia catalog shards;
- simulated guide-detector frame batches from Photosim6/Photosim7.

The example configurations contain workstation paths used in the current ET
development environment. Update those paths if you run the package elsewhere.

## External Data Dependencies

The guide workflows expect several external data roots.

### ET focal-plane geometry

`run_guide_first_frame_init` loads `et_coord` from:

```yaml
et_coord:
  src_dir: /home/cxgao/ET/et_focalplane/src
  data_dir: /home/cxgao/ET/et_focalplane/data
```

For microlensing guide detectors, use:

```yaml
et_coord:
  src_dir: /home/cxgao/ET/et_focalplane/src
  data_dir: /home/cxgao/ET/et_focalplane/data_microlens
  config_factory: microlens_guide_only
```

The optional `config_factory` field is resolved as an `ETCoordConfig` class
method and passed into `load_registry`. This keeps the detector-family choice
explicit in the configuration file.

### Gaia catalog

Reference-star construction uses:

```yaml
et_coord:
  gaia_root_dir: /home/cxgao/gaia_dr3_19mag
```

The guide configs then limit the effective catalog depth, for example with
`guide_init.catalog_g_mag_max`. They can also constrain the bright end with
`guide_init.catalog_g_mag_min`, preselect a larger bright-star pool before
final truncation with `guide_init.reference_preselect_topk_per_detector`, and
discard crowded reference stars with `guide_init.reference_isolation_radius_pix`.

### Simulated guide frames

The guide first-frame scripts expect a dataset root containing per-detector
batch directories. Each selected batch must include:

- `frames/*.npz`
- `run_meta.json`
- `stars.ecsv`

The selected detector/batch pairs live under `guide_init.detector_batches`.
The transit and microlensing configs intentionally use different detector
orders and dataset roots.

## Configuration Model

`configs/base.yaml` defines the common processing stack:

- image keys and dataset conventions;
- detector dimensions;
- preprocessing switches;
- extraction thresholds and centroid method;
- matching settings;
- attitude solver settings;
- evaluation and debug-output settings.

Payload-specific configs override only the fields that differ for a run:

- `guide_init.dataset_root`
- `guide_init.detector_batches`
- `guide_init.body_model_initial_f_mm`
- `guide_init.catalog_g_mag_min`
- `guide_init.catalog_g_mag_max`
- `guide_init.reference_preselect_topk_per_detector`
- `guide_init.reference_isolation_radius_pix`
- `detector.image_height`
- `detector.image_width`
- `detector.pixel_size_um`
- `et_coord.data_dir`
- `et_coord.config_factory`

Configuration files are merged in the example scripts with a recursive
dictionary update. Values in the guide-specific YAML override `base.yaml`.
`match.enforce_unique_assignment` defaults to `false`; when enabled, local
predicted-position matching uses a per-detector one-to-one minimum-cost
assignment instead of allowing multiple observations to reuse the same
reference star.

## Pipeline Internals

The guide first-frame path is implemented in:

- `fsglib.pipeline.run_guide_init.run_guide_first_frame_init`

The major internal stages are:

- `fsglib.common.io.load_npz_frame`: load a single image frame.
- `fsglib.preprocess.pipeline.preprocess_frame`: build preprocessed image and
  noise map.
- `fsglib.extract.pipeline.extract_stars`: detect star islands and compute
  centroids.
- `fsglib.pipeline.run_guide_init`: bridge simulation pixel coordinates to
  `et_focalplane` detector coordinates, generate LOS vectors, and build
  per-detector reference stars.
- `fsglib.match.pipeline.match_stars`: run local-triangle or predicted-position
  matching depending on context.
- `fsglib.attitude.solver.solve_attitude`: solve inertial-to-body attitude with
  QUEST and optional outlier rejection.
- `fsglib.pipeline.guide_error_audit.compute_guide_error_audit`: compare truth,
  extracted centroids, predicted detector positions, LOS geometry, and final
  attitude components.

The returned result dictionary contains:

- `solution`
- `matching`
- `observed_count`
- `reference_count`
- `detector_stats`
- `sim_to_detector_map`
- `geometry_model`
- `body_model`
- `error_audit`
- `meta`

## Debug Outputs

Guide examples write compact run-level JSON files under `outputs/debug/`.
When debug bundles are enabled, frame-level products may include:

- raw and preprocessed arrays;
- estimated noise maps;
- extracted candidates;
- reference stars and predicted detector positions;
- matched pairs;
- attitude solution summaries;
- overlay plots and residual-vector plots.

For field definitions and debug-product details, see:

- `docs/README_API.md`
- `docs/README_debug.md`

## Running Tests

```bash
pytest -q
```

The current tests cover core math and data-flow components, including:

- centroid extraction methods and bias-correction checks;
- star matching and triangle matching;
- QUEST attitude solving;
- frame evaluation metrics;
- ET-coordinate config-factory selection;
- guide pipeline configuration behavior.

## Development Notes

- Keep detector-family selection in YAML configuration. Avoid hard-coding
  transit or microlensing detector assumptions inside shared solver code.
- Keep long-form reports, presentation materials, and exploratory notebooks
  outside the package repository. The maintained in-repository docs should stay
  small and operational.
- Prefer exact `et_focalplane` geometry for guide-detector validation whenever
  the required registry data is available.
- Use truth-noise runs to isolate algorithmic effects before interpreting
  residuals from full image-level centroid extraction.
- Preserve mixed-worktree changes carefully when preparing PRs; local
  simulation artifacts and report materials are often intentionally untracked.
