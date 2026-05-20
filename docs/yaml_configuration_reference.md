# fsglib YAML Configuration Reference

## Scope

This document describes the YAML configuration keys currently recognized by
`fsglib`, with emphasis on the guide-detector initialization chain:

- `run_guide_first_frame_init(cfg)`
- `run_guide_first_frame_truth_noise(cfg)`
- `run_single_frame_init(npz_path, cfg, models, dataset_ctx=None)`
- `run_sequence_tracking(npz_paths, cfg, models, dataset_ctx=None)`

The maintained configuration pattern is to load `configs/base.yaml` first and
then overlay a workflow-specific YAML file. The example scripts implement a
recursive dictionary merge: nested dictionaries are merged, scalar values are
overridden, and lists are replaced as a whole.

Status labels used below:

- `active`: the current code reads this key and changes behavior from it.
- `declared`: the key exists in YAML and is useful as metadata, but the current
  code does not yet use it to change behavior.
- `reserved`: the key is intended for future behavior and is currently ignored.
- `deprecated`: the key is still accepted by legacy code but should not be used
  for new guide-chain validation.
- `external`: the key is passed to or interpreted through an external package,
  mainly `et_focalplane` / `et_coord`.

Important caveat: `io.*`, several preprocessing switches, several ephemeris
correction switches, and legacy triangle-matching keys are not equally mature.
They are documented here so configuration files do not hide silent no-ops.

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/base.yaml` | Common defaults for project metadata, extraction, matching, tracking, ephemeris, attitude, evaluation, and debug output. |
| `configs/guide_v1_noise_psf_etcoord.yaml` | Transit guide-detector first-frame initialization with real centroid extraction and `et_focalplane` geometry. |
| `configs/guide_microlens_v1_noise_psf_etcoord.yaml` | Microlensing guide-detector first-frame initialization with real centroid extraction and microlensing `et_focalplane` registry data. |
| `configs/guide_truth_noise_0065pix.yaml` | Truth-position synthetic centroid workflow with detector-pixel Gaussian noise and exact ET focal-plane geometry. |
| `configs/guide_truth_noise_0065pix_exact_etcoord.yaml` | Truth-position synthetic centroid workflow with exact `et_focalplane` LOS geometry. |
| `configs/main_sim_v2.yaml` | Generic single-frame simulated detector layout overlay. |
| `configs/detector_layout.yaml` | Physical detector layout definition for the older generic optical model path. |

## Workflow Selection

### Real Guide-Detector Centroids

Use:

```yaml
base: configs/base.yaml
overlay: configs/guide_v1_noise_psf_etcoord.yaml
```

or for microlensing:

```yaml
base: configs/base.yaml
overlay: configs/guide_microlens_v1_noise_psf_etcoord.yaml
```

Required top-level sections:

- `guide_init`
- `et_coord`
- `match`
- `attitude`
- `evaluation.guide_error_audit` if guide audit output is wanted

### Truth-Noise Guide Workflow

Use:

```yaml
base: configs/base.yaml
overlay: configs/guide_truth_noise_0065pix.yaml
```

or:

```yaml
base: configs/base.yaml
overlay: configs/guide_truth_noise_0065pix_exact_etcoord.yaml
```

Required top-level sections:

- `guide_truth_noise`
- `et_coord`
- `match`
- `attitude`

Internally this workflow maps `guide_truth_noise` into the same helper shape as
`guide_init`, then bypasses real image centroid extraction and creates
`ObservedStar` records directly from truth detector coordinates plus injected
noise.

### Generic Single-Frame Workflow

Use:

```yaml
base: configs/base.yaml
overlay: configs/main_sim_v2.yaml
```

Required top-level sections:

- `layout`
- `extract`
- `match`
- `ephemeris`
- `attitude`

This path depends on `models["projector"]` and `models["catalog"]`. The local
`build_models(cfg)` helper currently creates `RealOpticalProjector` and
`HealpixCatalogProvider`.

## Top-Level Sections

| Section | Status | Used by |
|---------|--------|---------|
| `project` | active | debug output and fallback mode labeling |
| `io` | declared | intended NPZ key mapping; loader currently hard-codes keys |
| `dataset` | active | static truth coordinate interpretation |
| `detector` | declared | detector metadata passed through context |
| `layout` | active | generic optical projector and detector visibility |
| `psf` | partially active | PSF template-fit interface metadata |
| `preprocess` | partially active | background subtraction and noise estimation |
| `extract` | active | star detection, centroiding, covariance, and blend flags |
| `guide_init` | active | real-centroid guide first-frame workflow |
| `guide_truth_noise` | active | truth-noise guide workflow |
| `et_coord` | active/external | `et_focalplane` registry, transformer, Gaia catalog |
| `match` | active | predicted-position, local pyramid, legacy triangle matching |
| `tracking` | partially active | generic sequence state machine |
| `ephemeris` | active | generic Gaia HEALPix reference query |
| `attitude` | active | QUEST solve and outlier rejection |
| `metrics` | reserved | currently not read by code |
| `evaluation` | active | dataset evaluation and audit switches |
| `logging` | partially active | debug bundle array output |
| `corrections` | reserved | passed through ephemeris context, currently no implemented keys |

## `project`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `project.name` | string | `fsglib` | declared | Human-readable run name. Current code does not branch on it. |
| `project.mode` | string | `init` | active | Fallback mode label used when an attitude solve is called without an explicit `AttitudeSolveInput`; also used by local pyramid result metadata. |
| `project.save_debug` | bool | `true` | active | Enables `save_debug_bundle(result, cfg)` output. |
| `project.output_dir` | path string | `outputs/debug` | active | Directory where debug bundles are written. |

## `io`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `io.npz_image_key` | string | `images` | declared | Intended image-array key. Current `load_npz_frame` still requires hard-coded `images`. |
| `io.npz_time_key` | string | `time_s` | declared | Intended time key. Current loader still requires hard-coded `time_s`. |
| `io.npz_variant_key` | string | `variant_ids` | declared | Intended variant-id key. Current loader checks hard-coded `variant_ids`. |
| `io.npz_cadence_key` | string | `cadence_s` | declared | Intended cadence key. Current loader checks hard-coded `cadence_s`. |
| `io.npz_unit_key` | string | `unit` | declared | Intended unit key. Current loader checks hard-coded `unit`. |

Current required NPZ keys are `images` and `time_s`. Current optional NPZ keys
include `variant_ids`, `cadence_s`, `coadd_start`, `coadd_stop`, `unit`, and the
truth arrays handled by `load_npz_frame`.

## `dataset`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `dataset.truth_origin` | string | `centered_pixels` | active | Interprets `stars.ecsv` `x0/y0` as centered detector offsets when set to `centered_pixels`; any other value is treated as already-pixel coordinates. |
| `dataset.truth_y_axis_up` | bool | `false` | active | When `truth_origin: centered_pixels`, flips centered `y0` before converting to image-pixel coordinates. |

`DatasetContext` can also derive field offsets from `run_meta.json` or estimate
them from truth stars for the `sky_patch_linearized` layout model.

## `detector`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `detector.num_detectors` | int | `4` | declared | Detector count metadata. |
| `detector.image_height` | int | `2049` | declared | Image height metadata for the default 2049-pixel guide simulation frames. |
| `detector.image_width` | int | `2049` | declared | Image width metadata for the default 2049-pixel guide simulation frames. |
| `detector.pixel_size_um` | float | `6.5` | active metadata | Default guide-detector pixel pitch in microns. |
| `detector.adc_bit_depth` | int | `12` | active | Digital ADC bit depth used to derive the default maximum DN when `detector.saturation_value` is null. |
| `detector.adc_min_value` | float | `0.0` | active | Lower digital clip bound used when `preprocess.enable_adc_clip=true`. |
| `detector.saturation_value` | float | `4095.0` | active | Upper digital clip bound and saturation mask threshold for the default 12-bit guide-detector images. |
| `detector.bad_pixel_map` | path string or null | `null` | reserved compatibility key | Bad-pixel masking is implemented through `preprocess.enable_bad_pixel_mask` and `preprocess.bad_pixel_mask_path`; this detector-level key is not consumed yet. |

## `layout`

`layout` is required by the generic single-frame and tracking APIs that use
`RealOpticalProjector`. Guide workflows using `et_coord` do not use this section
for detector geometry.

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `layout.frame_name` | string | `fgs_body` | declared | Frame label for layout documentation. |
| `layout.projection_model` | string | `distorted_focal_plane` | active | Projection model. Supported values are `distorted_focal_plane`, `ideal_pinhole`, and `sky_patch_linearized`. |
| `layout.default_detector_id` | int | `0` | active | Detector id used when loading generic NPZ frames. |
| `layout.pixel_size_mm` | float | `1.0` | active for `distorted_focal_plane` | Pixel size in mm for focal-plane conversion. |
| `layout.visibility_margin_pix` | float | `0.0` | active | Default margin used when deciding whether a projected star is visible on a detector. |
| `layout.distortion.a1` | float | `0.0` | active for `distorted_focal_plane` | Linear coefficient from focal-plane mm to field angle. |
| `layout.distortion.a3` | float | `0.0` | active for `distorted_focal_plane` | Cubic radial-like distortion coefficient. |
| `layout.distortion.axy2` | float | `0.0` | active for `distorted_focal_plane` | Cross-term distortion coefficient. |

Each `layout.detectors[]` entry supports:

| Key | Type | Required | Status | Description |
|-----|------|----------|--------|-------------|
| `layout.detectors[].detector_id` | int | yes | active | Numeric detector id used in `ObservedStar`, `ReferenceStar`, and projection dictionaries. |
| `layout.detectors[].name` | string | no | declared | Human-readable detector name. |
| `layout.detectors[].active` | bool | no | declared | Current projector does not filter inactive detectors. Omit inactive detectors instead. |
| `layout.detectors[].resolution` | two-number list | yes | active | Pixel width and height used for visibility checks. |
| `layout.detectors[].principal_point_pix` | two-number list | yes | active | Detector principal point in pixel coordinates. |
| `layout.detectors[].mounting_matrix` | 3x3 number list | yes | active | Detector-local to body-frame rotation matrix. |
| `layout.detectors[].fov_center_mm` | two-number list | no | active for `distorted_focal_plane` | Detector center position in focal-plane mm. Defaults to `[0.0, 0.0]`. |
| `layout.detectors[].pixel_scale_arcsec_per_pix` | float or null | required for `ideal_pinhole` and `sky_patch_linearized` | active | Plate scale used by linearized and ideal-pinhole projection models. |
| `layout.detectors[].visibility_margin_pix` | float | no | active | Per-detector override for `layout.visibility_margin_pix`. |

## `psf`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `psf.active_model_key` | string or null | `null` | declared | Names the active PSF model for audit and future template selection. It does not change the default `weighted_centroid` path. |
| `psf.template_bundle_path` | path string or null | `null` | reserved | Required when `extract.centroid_method=psf_template_fit`; PR13 defines the interface, while actual Photsim7 PSF-template fitting is deferred to #89. |

## `preprocess`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `preprocess.enable_background_subtraction` | bool | `true` | active | If true, subtracts the configured background model from finite pixels after detector calibration. |
| `preprocess.enable_bias_subtraction` | bool | `true` | active | If true, subtracts `preprocess.bias_frame_path` from the raw image before dark/FPN/flat correction. `base.yaml` points to a no-op 2049-pixel PR9 fake asset. |
| `preprocess.bias_frame_path` | path string or null | local fake 2049 asset | active with bias subtraction | `.npy` or `.npz` 2-D finite numeric bias frame. Missing path raises an error when enabled. |
| `preprocess.enable_dark_subtraction` | bool | `true` | active | If true, subtracts `preprocess.dark_current_path * raw.cadence_s`. Missing `raw.cadence_s` raises an error. |
| `preprocess.dark_current_path` | path string or null | local fake 2049 asset | active with dark subtraction | `.npy` or `.npz` 2-D finite numeric dark-current map in image units per second. |
| `preprocess.enable_flat_field` | bool | `true` | active | If true, divides by `preprocess.flat_field_path`; non-finite or non-positive flat pixels are marked invalid. |
| `preprocess.flat_field_path` | path string or null | local fake 2049 asset | active with flat field | `.npy` or `.npz` 2-D numeric flat/PRNU response map. Positive finite pixels are used as divisors. |
| `preprocess.enable_bad_pixel_mask` | bool | `true` | active | If true, applies `preprocess.bad_pixel_mask_path`; `true` or `1` means bad and sets `valid_mask=false`. |
| `preprocess.bad_pixel_mask_path` | path string or null | local fake 2049 asset | active with bad-pixel mask | `.npy` or `.npz` 2-D bool or numeric 0/1 mask matching the raw image shape. |
| `preprocess.enable_fpn_subtraction` | bool | `true` | active | If true, subtracts an additive fixed-pattern residual map before flat-field correction. `base.yaml` points to a no-op 2049-pixel PR9 fake asset. |
| `preprocess.fpn_residual_map_path` | path string or null | local fake 2049 asset | active with FPN subtraction | `.npy` or `.npz` 2-D finite numeric residual map matching the raw image shape. |
| `preprocess.enable_adc_clip` | bool | `true` | active | If true, clips finite raw input pixels into `[detector.adc_min_value, detector.saturation_value]` before calibration/background estimation. |
| `preprocess.enable_saturation_guard` | bool | `true` | active | If true, removes saturated pixels from `valid_mask` and records `artifact_masks["saturation_guard"]`. |
| `preprocess.saturation_mask_dilation_pix` | int | `0` | active | Optional binary dilation radius applied to saturated pixels before masking. |
| `preprocess.background_method` | string | `sigma_clip_global` | active | Supported values: `median`, `sigma_clip_global`, and `mesh_median`. `mesh_median` writes a 2-D background map. |
| `preprocess.sigma_clip_k` | float | `3.0` | active | Rejection threshold for sigma-clipped global and mesh background/noise estimates. |
| `preprocess.sigma_clip_max_iters` | int | `3` | active | Maximum robust sigma-clipping iterations for background/noise estimation. |
| `preprocess.background_mesh_size` | int | `64` | active with `mesh_median` | Mesh cell size in pixels for local median background and empirical local RMS estimates. |
| `preprocess.variance_model` | string | `empirical_robust` | active | Supported values: `empirical_robust` and `poisson_read_noise`. `noise_map` is always `sqrt(variance_map)`. |
| `preprocess.gain_e_per_dn` | float or null | `null` | active with `poisson_read_noise` | Electrons per DN/ADU for non-electron inputs. Required when `variance_model=poisson_read_noise` and the raw unit is not an electron unit. |
| `preprocess.read_noise_e` | float or null | `null` | active with `poisson_read_noise` | Read noise in electrons. Required, and may be zero for analytic/noiseless fixtures. |
| `preprocess.quantization_noise_e` | float | `0.0` | active with `poisson_read_noise` | Optional quantization noise term in electrons. |
| `preprocess.dark_current_e_per_s` | float or null | `null` | active with `poisson_read_noise` | Optional scalar dark-current shot-noise source in electrons per second, used only when the dark mean has been explicitly removed. Loaded dark-current calibration maps are preferred and use `raw.cadence_s`. |
| `preprocess.denoise_method` | string | `none` | reserved | Denoising is not implemented. |

The default `empirical_robust` variance model estimates RMS with a MAD-based
robust sigma after the configured background subtraction; `mesh_median`
produces spatially varying background and noise maps. `poisson_read_noise`
computes variance from photon counts, loaded dark-current maps scaled by
`raw.cadence_s` when present, read noise, and quantization noise. For DN/ADU
inputs the calculation uses `preprocess.gain_e_per_dn` internally and converts
`variance_map` back to the output image unit squared; `PreprocessedFrame.image`,
`background`, and `noise_map` remain in the input image unit. Photon/read/dark
variance is propagated through flat-response division when flat-field correction
is enabled, but flat-field uncertainty itself is not included yet and is reported
in metadata as disabled. Calibration asset paths are loaded by
`build_models(cfg)` into
`models["calib"]`; enabled products with missing paths, missing files, wrong
rank, or shape mismatches raise explicit errors. `.npz` assets must either use
the `data` array key or contain exactly one array. Local calibration products
should live outside the source repository, for example under an external
`fsglib-data/calibration/` asset root, and be referenced by YAML path. The
default `base.yaml` paths use PR9 fake 2049-pixel no-op assets; loading those
assets emits a `RuntimeWarning` so precision runs do not silently use fake
calibration.

PR11 applies ADC clipping before detector calibration as an input guard. The
authoritative detector ADC saturation simulation belongs in Photsim7; `fsglib`
keeps this guard so real or externally simulated inputs are bounded and
saturated pixels are tracked consistently. With the default 12-bit
guide-detector settings, raw pixels above `4095` DN are clipped to `4095`,
negative finite pixels are clipped to `0`, and pixels at the saturation
threshold are recorded in `PreprocessedFrame.artifact_masks`. When
`preprocess.enable_saturation_guard=true`, the saturation guard mask is removed
from `valid_mask` before background and noise estimation. `preprocess_meta`
records `adc_clip`, `artifact_counts`, and `artifact_policy` for audit.

Cosmic-ray injection is intentionally a simulation-side concern, not an
`fsglib` runtime preprocessing step. `fsglib` consumes the resulting image and
artifact masks through `PreprocessedFrame.artifact_masks`; simulation pipelines
should own event-rate sampling, stamp placement, rotation, resampling, and ADC
ordering. Real observation frames are not expected to provide cosmic-ray masks,
so PR11 controls cosmic-ray contamination only through ADC saturation guards,
degenerate-source rejection, sharpness limits, artifact-mask overlap when masks
exist, and downstream fit/match residual checks. A cosmic ray that hits a real
star but remains unsaturated and morphologically star-like is not guaranteed to
be rejected by PR11.

The external cosmic-ray data assets prepared for simulation-side use are now
owned by Photsim7-data:
`/home/cxgao/ET/Photsim7-data/cosmic_ray/dark_test_10um/event_library_10um.npz`
and
`/home/cxgao/ET/Photsim7-data/cosmic_ray/guide_6p5um/event_library_6p5um.npz`.
The source asset is the 10um measured dark-test event library; the 6.5um asset
is a guide-detector derivative stored outside this source repository.

## `extract`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `extract.detection_image` | string | `snr` | declared | Current extractor always thresholds the SNR image. |
| `extract.seed_threshold_sigma` | float | `5.0` | active | High SNR threshold for hysteresis seeds: `image / noise > seed_threshold_sigma`. |
| `extract.grow_threshold_sigma` | float | `3.0` | active | Low SNR threshold for 8-connected hysteresis growth. Must be less than or equal to `seed_threshold_sigma`. |
| `extract.min_area` | int | `3` | active | Rejects grown connected components with fewer pixels. |
| `extract.max_area` | int | `200` | active | Rejects grown connected components with more pixels. |
| `extract.centroid_method` | string | `weighted_centroid` | active | Supported active values: `weighted_centroid`, `adaptive_moment_centroid`, `fixed_window_first_moment`, and `full_window_first_moment`. `psf_template_fit` is reserved and raises until #89. |
| `extract.centroid_window.center` | string | `peak` | declared | Current fixed-window modes always center on the detected peak pixel. |
| `extract.centroid_window.size` | odd int | `31` | active for fixed-window modes | Window size for `fixed_window_first_moment` and `full_window_first_moment`; must be positive and odd. |
| `extract.centroid_covariance.min_sigma_pix` | float | `0.03` | active | Per-axis covariance floor added to extracted centroids so high-SNR stars do not receive unrealistically zero measurement uncertainty. |
| `extract.centroid_covariance.jacobian_step_pix` | float | `0.01` | active | Pixel step used by `candidates_to_observed()` to finite-difference the projector and propagate pixel covariance to LOS/angular sigma. |
| `extract.deblend.enabled` | bool | `true` | active | Enables image-only multi-peak blend detection inside a grown segment. |
| `extract.deblend.policy` | string | `flag_only` | active | Supported values: `flag_only` and `reject`. `flag_only` records blend risk without dropping the candidate. |
| `extract.deblend.peak_threshold_sigma` | float or null | `null` | active | SNR threshold for counting local peaks. Null reuses `extract.seed_threshold_sigma`. |
| `extract.bbox_expand` | int | `2` | active | Expands the stored segmentation bounding box for weighted centroids. |
| `extract.reject_edge_margin` | int | `3` | active | Rejects candidates whose centroid window touches an image edge within this margin. |
| `extract.max_ellipticity` | float | `0.8` | active | Rejects candidates whose measured second-moment ellipticity exceeds this value. |
| `extract.reject_degenerate_sources` | bool | `true` | active | Rejects single-pixel or second-moment-degenerate candidates, which are common hot-pixel/cosmic-ray artifacts. |
| `extract.min_fwhm_pix` | float or null | `null` | active | Optional lower bound on measured second-moment FWHM. Null disables this filter. |
| `extract.max_sharpness` | float or null | `10.0` | active | Optional upper bound on peak divided by grown-segment mean surface brightness. |
| `extract.reject_artifact_mask_overlap` | bool | `true` | active | Rejects candidates whose segmentation bbox overlaps any `PreprocessedFrame.artifact_masks` entry after optional margin expansion. |
| `extract.artifact_mask_margin_pix` | int | `2` | active | Pixel margin used when checking candidate bbox overlap with artifact masks. |

Extraction uses the SNR image for segmentation. Seed and grow masks both use
strict `>` threshold comparisons, and grow pixels must be connected to at least
one seed pixel. Connectivity is 8-connected. If multiple seed pixels fall in the
same grown connected component, the default PR13 behavior still returns one
candidate but records `blend_flag`, `num_local_peaks`, and `local_peaks`; setting
`extract.deblend.policy=reject` drops multi-peak segments. `weighted_centroid`,
`flux`,
`area`, and candidate `snr` are based on the grown segment when
`extract.centroid_method=weighted_centroid`; in fixed-window centroid modes,
`area` and candidate `snr` remain grown-segment measurements, while `flux` and
`StarCandidate.bbox` come from the centroid window. The grown segment bbox is
always preserved separately as `StarCandidate.flags["segment_bbox"]`. Setting
`grow_threshold_sigma` equal to `seed_threshold_sigma` reproduces seed-only
segmentation.

PR13 keeps `weighted_centroid` as the default because it is the flight-oriented
low-latency estimator. It now also propagates pixel noise into
`StarCandidate.centroid_cov_pix` and stores scalar summaries such as
`centroid_sigma_x_pix`, `centroid_sigma_y_pix`, and
`centroid_sigma_radial_pix` in flags. `adaptive_moment_centroid` is an explicit
YAML-selected second-moment weighted centroid variant. It uses the grown segment
to derive an adaptive elliptical kernel, then computes a weighted centroid and
covariance. `psf_template_fit` is intentionally not implemented in PR13 because
ET off-axis PSFs can be strongly non-Gaussian and require a configured external
PSF bundle; selecting it currently requires `psf.template_bundle_path` and then
raises a follow-up implementation error. The dedicated implementation is tracked
in #89.

Shape metrics are measured directly from the candidate pixels; PR12 does not use
an external PSF model. `StarCandidate.shape` includes `sigma_major_pix`,
`sigma_minor_pix`, `theta_rad`, `ellipticity`, `fwhm_pix`, `sharpness`,
`roundness`, and `shape_degenerate`. Ellipticity is defined as
`1 - sqrt(lambda_min / lambda_max)` from the second-moment eigenvalues.
`fwhm_pix` is the major-axis second-moment proxy `2.3548 * sigma_major_pix`;
`sharpness` is peak divided by mean grown-segment surface brightness. PR11 makes
artifact filtering configurable: the default `base.yaml` rejects degenerate
single-pixel sources, overly sharp candidates, and candidates overlapping
preprocess artifact masks such as saturation guards. Accepted candidates also
copy key shape values into `StarCandidate.flags` so they propagate through
existing `ObservedStar.flags` paths.

## `guide_init`

`guide_init` drives real-centroid guide first-frame initialization. It is used by
`run_guide_first_frame_init`.

| Key | Type | Required | Status | Description |
|-----|------|----------|--------|-------------|
| `guide_init.dataset_root` | path string | yes | active | Root directory containing one batch directory per guide detector. |
| `guide_init.frame_index` | int | no | active | Frame index selected from each `batch*/frames/*.npz`; default `0`. |
| `guide_init.max_observed_per_detector` | int or null | no | active | Keeps only the brightest/SNR-best candidates per detector before matching. Null disables the limit. |
| `guide_init.los_geometry_mode` | string | no | active | Must be `exact_et_focalplane`; default `exact_et_focalplane`. |
| `guide_init.frame_alignment_grid_size` | int | no | active | Grid size used to fit the exact ET field-angle to body-frame alignment; default `13`. |
| `guide_init.catalog_g_mag_min` | float or null | no | active | Optional bright-end Gaia G magnitude cut. |
| `guide_init.catalog_g_mag_max` | float | yes | active | Faint-end Gaia G magnitude cut for per-detector reference-star query. |
| `guide_init.reference_topk_per_detector` | int | yes | active | Final number of brightest reference stars retained per detector. |
| `guide_init.reference_preselect_topk_per_detector` | int | no | active | Number of brightest stars considered before optional isolation filtering; defaults to `reference_topk_per_detector`. |
| `guide_init.reference_isolation_radius_pix` | float or null | no | active | If positive, rejects preselected reference stars whose nearest preselected neighbor is within this detector-pixel radius. |
| `guide_init.target_epoch` | float | no | active/external | Epoch passed into `et_coord.query_detector_sources`; default `2000.0`. |
| `guide_init.detector_batches` | list[dict] | yes | active | Per-detector batch mapping. |

Each `guide_init.detector_batches[]` entry supports:

| Key | Type | Required | Status | Description |
|-----|------|----------|--------|-------------|
| `guide_init.detector_batches[].detector_id` | string | yes | active/external | Detector id understood by the active `et_focalplane` registry, for example `guide_left` or `guide_top`. |
| `guide_init.detector_batches[].batch_name` | string | yes | active | Batch directory under `guide_init.dataset_root`. |

## `guide_truth_noise`

`guide_truth_noise` has the same reference-star and detector-batch controls as
`guide_init`, but it bypasses real centroid extraction. It is used by
`run_guide_first_frame_truth_noise`.

| Key | Type | Required | Status | Description |
|-----|------|----------|--------|-------------|
| `guide_truth_noise.dataset_root` | path string | yes | active | Root directory containing one batch directory per guide detector. |
| `guide_truth_noise.frame_index` | int | no | active | Frame index selected from each detector batch; default `0`. |
| `guide_truth_noise.max_observed_per_detector` | int or null | no | active | Keeps only the brightest synthetic truth candidates per detector. |
| `guide_truth_noise.los_geometry_mode` | string | no | active | Must be `exact_et_focalplane`; default `exact_et_focalplane`. |
| `guide_truth_noise.frame_alignment_grid_size` | int | no | active | Grid size used to fit the exact ET field-angle to body-frame alignment; default `13`. |
| `guide_truth_noise.catalog_g_mag_min` | float or null | no | active | Optional bright-end Gaia G magnitude cut. |
| `guide_truth_noise.catalog_g_mag_max` | float | yes | active | Faint-end Gaia G magnitude cut. |
| `guide_truth_noise.reference_topk_per_detector` | int | yes | active | Final reference-star count per detector. |
| `guide_truth_noise.reference_preselect_topk_per_detector` | int | no | active | Pre-isolation reference count; defaults to `reference_topk_per_detector`. |
| `guide_truth_noise.reference_isolation_radius_pix` | float or null | no | active | Optional detector-pixel isolation filter for reference stars. |
| `guide_truth_noise.target_epoch` | float | no | active/external | Epoch passed into `et_coord.query_detector_sources`; default `2000.0`. |
| `guide_truth_noise.centroid_noise_mean_pix` | float | no | active | Mean of injected Gaussian centroid noise in detector pixels; default `0.0`. |
| `guide_truth_noise.centroid_noise_sigma_pix` | float | yes | active | Standard deviation of injected Gaussian centroid noise in detector pixels. |
| `guide_truth_noise.centroid_noise_space` | string | no | declared | Output label for the noise coordinate system. Current injection is detector-pixel noise. |
| `guide_truth_noise.random_seed` | int | no | active | Random generator seed; default `0`. |
| `guide_truth_noise.detector_batches` | list[dict] | yes | active | Same shape as `guide_init.detector_batches`. |
| `guide_truth_noise.detector_batches[].detector_id` | string | yes | active/external | Detector id understood by the active `et_focalplane` registry. |
| `guide_truth_noise.detector_batches[].batch_name` | string | yes | active | Batch directory under `guide_truth_noise.dataset_root`. |

## `et_coord`

`et_coord` is required by guide workflows.

| Key | Type | Required | Status | Description |
|-----|------|----------|--------|-------------|
| `et_coord.src_dir` | path string | yes | active | Path inserted into `sys.path` before importing `et_coord`. |
| `et_coord.data_dir` | path string | yes | active/external | Registry data directory passed to `et_coord.load_registry`. |
| `et_coord.config_factory` | string or null | no | active/external | Optional `ETCoordConfig` factory name, for example `microlens_guide_only`. If omitted, default registry loading is used. |
| `et_coord.gaia_root_dir` | path string | yes | active/external | Local Gaia catalog root passed to `et_coord.GaiaCatalog`. |

For catalog consistency, guide simulations and guide matching should use the
same Gaia catalog root, same `et_coord` registry data, same detector family, and
same target epoch.

## `match`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `match.algorithm` | string | `predicted_position` | active | Matching strategy. Supported active values are `predicted_position`, `local_pyramid`, `predicted_position_with_pyramid_reacquire`, and `predicted_position_and_local_pyramid`. Deprecated values are `triangle` and `local_triangle`. |
| `match.mode` | string | `init` | declared | Matching mode label in YAML. Runtime contexts currently set the mode directly. |
| `match.init_max_catalog_radius_deg` | float | `1.5` | active for generic ephemeris | Search radius for generic HEALPix Gaia query in init mode. Guide workflows query per detector through `et_coord` instead. |
| `match.init_bright_star_topk` | int | `20` | reserved | Not used by current predicted-position or local-pyramid matcher. |
| `match.pair_angle_tol_arcsec` | float | `120.0` | reserved | Not used by current local-pyramid implementation; use the local-pyramid tolerance keys below. |
| `match.hypothesis_topk` | int | `50` | reserved | Not used by current local-pyramid implementation. |
| `match.validate_max_residual_arcsec` | float | `60.0` | reserved | Current validation uses `attitude.outlier_max_residual_arcsec`; local pyramid has separate angular gates. |
| `match.validate_max_residual_pix` | float | `25.0` | active | Pixel gate for predicted-position matching; also default local-pyramid expansion pixel gate. |
| `match.validate_min_support` | int | `5` | active | Minimum matched stars required for `MatchingResult.success`. |
| `match.enforce_unique_assignment` | bool | `true` | active | Uses Hungarian one-to-one assignment per detector for predicted-position matching. |

### Matching Algorithms

| Value | Status | Behavior |
|-------|--------|----------|
| `predicted_position` | active | Match each observed star to catalog references using predicted detector positions and `validate_max_residual_pix`. |
| `local_pyramid` | active | Run only local pyramid matching and return its expanded matches if successful. |
| `predicted_position_with_pyramid_reacquire` | active | Run predicted-position first; if it fails support, attempt local pyramid reacquisition. |
| `predicted_position_and_local_pyramid` | active | Always run both predicted-position and local pyramid, then choose the result with more matched stars; ties keep predicted-position. |
| `triangle` | deprecated | Uses the old local triangle/GSC index path. Do not use for current guide-chain validation. |
| `local_triangle` | deprecated | Alias for the old triangle path. |

### `match.local_pyramid`

The nested `match.local_pyramid.*` keys are preferred. For backward
compatibility, `match.pyramid_<name>` flat aliases are also read by the local
pyramid implementation.

Mode-specific local-pyramid keys can be provided by prefixing the key with
`reacquire_`, for example `reacquire_expansion_policy`. These override the
base key only when the matcher is invoked as a reacquire fallback.

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `match.local_pyramid.enabled` | bool | `false` | declared | Informational/config-completeness flag. Actual execution is controlled by `match.algorithm`. |
| `match.local_pyramid.max_observed_stars` | int | `40` | active | Keeps the highest SNR/flux observed stars for seed generation. `0` means no limit. |
| `match.local_pyramid.max_reference_stars` | int | `300` | active | Keeps the brightest reference stars for pair-index construction. `0` means no limit. |
| `match.local_pyramid.seed_scopes` | list[string] | `["single_detector", "mixed_detector"]` | active | Seed search order. `single_detector` tests four-star seeds within one detector; `mixed_detector` tests cross-detector seeds. |
| `match.local_pyramid.pair_angle_tol_arcsec_single_detector` | float | `120.0` | active | Pair-angle tolerance for single-detector seed matching. |
| `match.local_pyramid.pair_angle_tol_arcsec_mixed_detector` | float | `300.0` | active | Pair-angle tolerance for mixed-detector seed matching. |
| `match.local_pyramid.seed_rms_gate_arcsec` | float | `60.0` | active | Rejects seed attitudes whose seed LOS residual RMS exceeds this value. |
| `match.local_pyramid.seed_max_gate_arcsec` | float | `180.0` | active | Rejects seed attitudes whose max seed LOS residual exceeds this value. |
| `match.local_pyramid.expansion_policy` | string | `predicted_xy` | active | Expansion policy for normal local-pyramid matching. `predicted_xy` keeps the detector-pixel gate behavior. `seed_attitude_only` uses only seed-attitude angular residuals and requires `geometry_only_allowed=true`. |
| `match.local_pyramid.geometry_only_allowed` | bool | `false` | active | Safety switch required before `seed_attitude_only` expansion is honored. |
| `match.local_pyramid.reacquire_expansion_policy` | string | `seed_attitude_only` | active | Reacquire override used by `predicted_position_with_pyramid_reacquire` after predicted-position matching fails. |
| `match.local_pyramid.reacquire_geometry_only_allowed` | bool | `true` | active | Allows reacquire fallback to expand from seed attitude without relying on stale predicted pixels. |
| `match.local_pyramid.expand_angular_gate_arcsec` | float | `120.0` | active | Angular gate used when expanding a valid seed to all observed/reference pairs. |
| `match.local_pyramid.expand_pixel_gate_pix` | float | `match.validate_max_residual_pix` | active | Detector-pixel gate used when expanding a valid seed. |
| `match.local_pyramid.min_expanded_matches` | int | `match.validate_min_support` | active | Minimum expanded matches required to accept a seed. |
| `match.local_pyramid.max_observed_pyramids` | int | `5000` | active | Maximum observed four-star seeds tested per scope. `0` means no limit. |
| `match.local_pyramid.max_candidates_per_observed_seed` | int | `200` | active | Maximum reference seed candidates retained per observed seed. `0` means no limit. |
| `match.local_pyramid.max_seed_attitudes` | int | `2000` | active | Maximum seed attitudes scored across the search. `0` means no limit. |
| `match.local_pyramid.min_edge_arcsec` | float | `0.0` | active | Rejects observed pyramid seeds with any pair angle smaller than this. |
| `match.local_pyramid.max_edge_deg` | float | infinity | active | Rejects observed pyramid seeds with any pair angle larger than this. |
| `match.local_pyramid.ambiguity_min_score_margin` | float | `1.0` | active | Rejects local-pyramid results when the best and runner-up expanded hypotheses are too close in score. `0.0` disables ambiguity rejection. |
| `match.local_pyramid.detector_mean_warn_pix` | float | `5.0` | active | Marks a detector residual summary as `warn` when the coherent mean pixel residual norm exceeds this value. |
| `match.local_pyramid.detector_mean_reject_pix` | float | `25.0` | active | Marks a detector residual summary as `reject` when the coherent mean pixel residual norm exceeds this value. |
| `match.local_pyramid.detector_rms_reject_pix` | float | `25.0` | active | Marks a detector residual summary as `reject` when detector RMS pixel residual exceeds this value. |
| `match.local_pyramid.detector_max_reject_pix` | float | `50.0` | active | Marks a detector residual summary as `reject` when any detector pixel residual exceeds this value. |
| `match.local_pyramid.mixed_detector_reject_on_detector_warning` | bool | `true` | active | Rejects mixed-detector seed results when any participating detector has a warning residual status. |
| `match.local_pyramid.photometric_rank_weight` | float | `0.0` | active | Optional soft cost weight comparing observed SNR/flux rank to reference magnitude rank. Kept off by default until bandpass/flux weighting is formalized. |
| `match.local_pyramid.seed_consistency_penalty` | float | `1.0e-6` | active | Tiny assignment penalty for expansion edges that are not part of the candidate seed, used only to make exact ties deterministic. |

`seed_attitude_projected` expansion is not implemented. PR4 provides
`seed_attitude_only` for local reacquire; projector-backed seed-attitude pixel
reprojection is tracked as follow-up work.

### Deprecated Triangle Keys

| Key | Type | Status | Description |
|-----|------|--------|-------------|
| `match.triangle_gsc_path` | path string | deprecated | Old precomputed local triangle/GSC NPZ index. This index is catalog-inconsistent with the active guide path and should not be used. |
| `match.triangle_tolerance_deg` | float | deprecated | Angular tolerance for old triangle matcher. |
| `match.triangle_max_stars` | int | deprecated | Maximum observed stars used by old triangle matcher. |

## `tracking`

`run_sequence_tracking()` uses explicit mode names:
`init_known_field`, `tracking`, `local_reacquire`, `lost_in_space`, and
`safe_lost`. In PR8, `lost_in_space` invokes the all-sky
`LostInSpaceMatcher` when the runtime models mapping includes
`models["lis_index"]`. If the index is absent, the frame fails cleanly with
reason `lost_in_space_index_missing` and the state machine applies
`tracking.safe_lost_after_lis_failures`. YAML-based LIS index-path loading is
deferred to the configuration audit tracked in #69.

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `tracking.search_radius_pix` | float | `10.0` | reserved | Tracking currently uses catalog prediction plus `match.validate_max_residual_pix`, not this key. |
| `tracking.max_miss_count` | int | `3` | active | Track states remain active until their miss count exceeds this value. |
| `tracking.tracking_match_algorithm` | string | `predicted_position` | active | Matcher policy used while in `tracking`. If absent, runtime falls back to `match.algorithm`. |
| `tracking.local_reacquire_match_algorithm` | string | `predicted_position_with_pyramid_reacquire` | active | Matcher policy used in `local_reacquire`; this invokes predicted-position matching first, then PR4 local-pyramid reacquire on failure. |
| `tracking.lost_in_space_match_algorithm` | string | `lost_in_space` | active | Matcher policy label used in `lost_in_space`; PR8 routes this mode to `LostInSpaceMatcher` with runtime `models["lis_index"]`. |
| `tracking.reacquire_after_tracking_failures` | int | `2` | active | Number of consecutive `tracking` failures before transitioning to `local_reacquire`. |
| `tracking.lost_in_space_after_reacquire_failures` | int | `3` | active | Number of consecutive `local_reacquire` failures before transitioning to `lost_in_space`. |
| `tracking.safe_lost_after_lis_failures` | int | `1` | active | Number of consecutive `lost_in_space` failures before transitioning to `safe_lost`. |
| `tracking.reacquire_after_failures` | int | `2` | active compatibility alias | Legacy alias for `tracking.reacquire_after_tracking_failures`. |
| `tracking.lost_after_init_failures` | int | `3` | active compatibility alias | Legacy init-failure threshold and alias for the local-reacquire lost-in-space threshold when the new key is absent. |
| `tracking.max_attitude_jump_arcsec` | float | `300.0` | active | Rejects tracking hypotheses with an attitude jump larger than this. |

## `ephemeris`

These keys affect the generic `HealpixCatalogProvider` and generic projector
path. Guide workflows use `et_coord` per-detector source queries instead.

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `ephemeris.catalog_backend` | string | `healpix` | declared | Current `build_models` always creates `HealpixCatalogProvider`; no backend switch is implemented. |
| `ephemeris.gaia_root_dir` | path string | `/home/cxgao/gaia_dr3_19mag` | active | Root directory of nested HEALPix Gaia CSV partitions for generic catalog queries. |
| `ephemeris.mag_limit` | float | `15.0` | active | Faint-end Gaia G magnitude limit for generic catalog queries. |
| `ephemeris.gaia_partition_cache_size` | int | `8` | active | Per-provider LRU cache size for generic Gaia HEALPix partition DataFrames. `0` disables caching. |
| `ephemeris.missing_partition_policy` | string enum | `ignore` | active | Handling for missing or unreadable generic Gaia HEALPix partitions: `ignore`, `warn`, or `error`. |
| `ephemeris.tracking_catalog_radius_deg` | float | `2.0` | active | Search radius used by generic tracking catalog query. |
| `ephemeris.reference_selection_mode` | string | `visible_only` | active | `visible_only` keeps projected visible stars; `sim_rect_topk` additionally selects top-k by converted Kepler magnitude for init mode. |
| `ephemeris.reference_topk` | int | `0` | active with `sim_rect_topk` | Maximum number of reference stars retained after visible projection. `0` disables the limit. |
| `ephemeris.gaia_to_kp_poly_path` | path string or null | configured path | active | Optional NumPy polynomial coefficient file for Gaia G to Kepler magnitude conversion. Missing/unreadable files silently disable conversion. |
| `ephemeris.target_epoch` | float | `2000.0` | active | Target Julian year used when propagating generic Gaia catalog stars before projection. |
| `ephemeris.reference_epoch_default` | float | `2016.0` | active | Fallback Gaia reference epoch when a catalog partition does not provide `ref_epoch`. |
| `ephemeris.enable_proper_motion` | bool | `true` | active | Enables generic Gaia proper-motion propagation in `build_reference_stars`; guide workflows delegate epoch handling to `et_coord`. |
| `ephemeris.enable_precession` | bool | `false` | reserved | Not implemented. |
| `ephemeris.enable_nutation` | bool | `false` | reserved | Not implemented. |
| `ephemeris.enable_dva` | bool | `false` | reserved | Not implemented. |
| `ephemeris.enable_relativity` | bool | `false` | reserved | Not implemented. |

Generic `HealpixCatalogProvider.last_query_stats` records candidate pixels,
cache hit/miss/eviction pixels, missing/failed partitions, row counts, and the
number of stars returned for the most recent query. Generic catalog
`CatalogStar.meta` records catalog provenance including provider name, root
directory, source file, HEALPix pixel/nside/order, query radius, and magnitude
limit. Generic `ReferenceStar.meta` then records original and propagated
coordinates, reference/target epoch, proper-motion status, converted Kepler
magnitude when available, and the `weight_source`/`flux_weight` used for
`weight_hint`.

## `attitude`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `attitude.solver` | string | `quest` | declared | Only QUEST/SVD fallback is implemented. The key does not select another solver. |
| `attitude.min_stars_mathematical` | int | `2` | active | Minimum matched stars required to attempt attitude solving. |
| `attitude.min_stars_operational` | int | `4` | active | Minimum matched stars required for a `VALID` attitude solution. |
| `attitude.weight_mode` | string | `variance_snr_hybrid` | active | Controls how `ObservedStar.weight` is populated before matching: `snr`, `centroid_variance`, or `variance_snr_hybrid`. `sigma_angle_arcsec` is recorded regardless of mode when centroid covariance is available. |
| `attitude.outlier_reject_enable` | bool | `true` | active | Enables one-pass residual-gate outlier rejection. |
| `attitude.outlier_max_residual_arcsec` | float | `30.0` | active | Residual gate for outlier rejection and final validity. |
| `attitude.outlier_sigma_clip` | float | `3.0` | reserved | Sigma-clipping outlier rejection is not implemented. |
| `attitude.max_iterations` | int | `2` | reserved | Iterative multi-pass outlier rejection is not implemented. |
| `attitude.quest_tol` | float | `1e-12` | active | Newton tolerance for QUEST characteristic-root solve. Not written in `base.yaml` yet. |
| `attitude.quest_max_iter` | int | `50` | active | Maximum QUEST Newton iterations. Not written in `base.yaml` yet. |

## `metrics`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `metrics.enable_truth_compare` | bool | `true` | reserved | Evaluation currently runs when truth is available, without checking this switch. |
| `metrics.report_centroid_error` | bool | `true` | reserved | Centroid metrics are emitted when evaluation runs. |
| `metrics.report_attitude_error` | bool | `true` | reserved | Attitude metrics are emitted when evaluation runs. |
| `metrics.report_runtime` | bool | `true` | reserved | Timings are collected by pipeline code without checking this switch. |

## `evaluation`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `evaluation.batch_glob` | string | `batch*` | active | Batch directory glob for `evaluate_dataset`. |
| `evaluation.frame_stride` | int | `1` | active | Frame stride for dataset evaluation. |
| `evaluation.max_frames_per_batch` | int or null | `null` | active | Optional cap on evaluated frames per batch. |

### `evaluation.centroid_step_audit`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `evaluation.centroid_step_audit.enabled` | bool | `false` | active | Enables detailed centroid-step audit in generic frame evaluation. |
| `evaluation.centroid_step_audit.stamp_size` | odd int | `31` | active | Stamp size used around each truth star. |
| `evaluation.centroid_step_audit.truth_match_radius_pix` | float | `3.0` | active | Candidate-to-truth matching radius for audit pairing. |
| `evaluation.centroid_step_audit.max_stars` | int or null | `null` | active | Optional cap on audited truth stars, sorted by magnitude. |
| `evaluation.centroid_step_audit.ft_root` | path string | configured path | active | Photosim root used for single-star PSF synthesis. |
| `evaluation.centroid_step_audit.data_dir` | path string | configured path | active | Photosim data root; also assigned to `ET_DATA_DIR`. |
| `evaluation.centroid_step_audit.config_xlsx` | path string | configured path | active | Photosim configuration workbook. |
| `evaluation.centroid_step_audit.psf_field_id` | int | `6` | active | PSF model index selected from the Photosim PSF manager. |
| `evaluation.centroid_step_audit.mag_type` | string | `ET` | active | Magnitude type passed to Photosim star catalog build. |

### `evaluation.guide_error_audit`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `evaluation.guide_error_audit.enabled` | bool | `false` | active | Enables guide-specific per-star/per-detector error audit. |
| `evaluation.guide_error_audit.truth_match_radius_pix` | float | `3.0` | active | Radius for assigning selected candidates to truth detector coordinates. |

## `logging`

| Key | Type | Default | Status | Description |
|-----|------|---------|--------|-------------|
| `logging.level` | string | `INFO` | declared | No logging subsystem currently reads this key. |
| `logging.save_intermediate_arrays` | bool | `true` | active | Controls `raw.npy`, `preprocessed.npy`, `noise_map.npy`, and `artifact_mask_*.npy` in debug bundles. |
| `logging.save_source_catalog` | bool | `true` | declared | Reference-star JSON is currently always written when a debug bundle is saved. |
| `logging.save_match_result` | bool | `true` | declared | Match JSON is currently always written when a debug bundle is saved. |

## Practical Complete Match Block

For guide workflows during the predicted-position to local-pyramid transition,
use an explicit block like this:

```yaml
match:
  algorithm: predicted_position_and_local_pyramid
  mode: init
  init_max_catalog_radius_deg: 1.5
  init_bright_star_topk: 20
  pair_angle_tol_arcsec: 120.0
  hypothesis_topk: 50
  validate_max_residual_arcsec: 60.0
  validate_max_residual_pix: 10.0
  validate_min_support: 5
  enforce_unique_assignment: true
  local_pyramid:
    enabled: false
    max_observed_stars: 40
    max_reference_stars: 300
    seed_scopes: ["single_detector", "mixed_detector"]
    pair_angle_tol_arcsec_single_detector: 120.0
    pair_angle_tol_arcsec_mixed_detector: 300.0
    seed_rms_gate_arcsec: 60.0
    seed_max_gate_arcsec: 180.0
    expansion_policy: predicted_xy
    geometry_only_allowed: false
    reacquire_expansion_policy: seed_attitude_only
    reacquire_geometry_only_allowed: true
    expand_angular_gate_arcsec: 120.0
    expand_pixel_gate_pix: 10.0
    min_expanded_matches: 5
    max_observed_pyramids: 5000
    max_candidates_per_observed_seed: 200
    max_seed_attitudes: 2000
    min_edge_arcsec: 0.0
    max_edge_deg: .inf
    ambiguity_min_score_margin: 1.0
    detector_mean_warn_pix: 5.0
    detector_mean_reject_pix: 25.0
    detector_rms_reject_pix: 25.0
    detector_max_reject_pix: 50.0
    mixed_detector_reject_on_detector_warning: true
    photometric_rank_weight: 0.0
    seed_consistency_penalty: 1.0e-6

tracking:
  tracking_match_algorithm: predicted_position
  local_reacquire_match_algorithm: predicted_position_with_pyramid_reacquire
  lost_in_space_match_algorithm: lost_in_space
  reacquire_after_tracking_failures: 2
  lost_in_space_after_reacquire_failures: 3
  safe_lost_after_lis_failures: 1
```

`local_pyramid.enabled` is intentionally shown for configuration readability,
but the current implementation uses `match.algorithm` to decide whether the
local pyramid matcher runs.

## Known Configuration Debt

The following keys are present in YAML but currently do not change runtime
behavior:

- `io.*`
- `preprocess.denoise_method`
- `extract.detection_image`
- `match.init_bright_star_topk`
- `match.pair_angle_tol_arcsec`
- `match.hypothesis_topk`
- `match.validate_max_residual_arcsec`
- `tracking.search_radius_pix`
- `ephemeris.catalog_backend`
- `ephemeris.enable_precession`
- `ephemeris.enable_nutation`
- `ephemeris.enable_dva`
- `ephemeris.enable_relativity`
- `attitude.solver` beyond `quest`
- `attitude.outlier_sigma_clip`
- `attitude.max_iterations`
- `metrics.*`
- `logging.level`
- `logging.save_source_catalog`
- `logging.save_match_result`

The old triangle configuration keys are supported only by deprecated code and
should be removed after the local pyramid path is validated.
