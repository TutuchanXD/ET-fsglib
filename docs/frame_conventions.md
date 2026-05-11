# Frame Conventions

This document locks the coordinate conventions used by the guide-star optical
projection and attitude solver code.

## Image Pixels

- Pixel coordinates are `(u, v)`, also referred to as image `(x, y)`.
- `u`/image `x` increases to the right.
- `v`/image `y` increases downward.
- Detector `principal_point_pix` is expressed in the same top-left pixel frame.

## Detector-Local Frame

- Detector-local line-of-sight vectors use optical axis `+Z`.
- Positive horizontal pixel offset, `u - principal_point_x > 0`, maps to
  detector-local `+X`.
- Positive vertical pixel offset, `v - principal_point_y > 0`, maps to
  detector-local `+Y`.
- The distorted-focal-plane and ideal-pinhole models both use
  `atan(x / z)` and `atan(y / z)` for local field angles.

## Detector Mounting

- Each detector `mounting_matrix` maps detector-local vectors into the body
  frame:

  ```text
  v_body = mounting_matrix @ v_detector_local
  ```

- The inverse body-to-detector-local transform is the matrix transpose. The
  mounting matrices are therefore expected to be orthonormal rotations.

## Projection Models

- `distorted_focal_plane`: pixel offsets are converted to focal-plane
  millimeters with `pixel_size_mm`, then shifted by each detector
  `fov_center_mm` before the polynomial field distortion is evaluated.
- `ideal_pinhole`: pixel offsets are converted directly to field angles using
  `pixel_scale_arcsec_per_pix`.
- `sky_patch_linearized`: pixel offsets are interpreted around the configured
  inertial field center. Optional `field_offset_x_pix` and
  `field_offset_y_pix` shift the image-space field center before converting to
  RA/Dec offsets.

All three projection models must preserve forward/inverse round trips at the
detector center, representative edges, and representative corners.

## ET Guide Focal-Plane Adapter

- Guide first-frame workflows use `exact_et_focalplane` geometry only.
- The adapter asks `et_coord` for exact equatorial `pixel_to_sky()` vectors and
  then maps them into the fsglib body frame with a fitted
  `rotation_body_from_eq`.
- The fitted frame alignment is derived from exact ET field-angle samples. It
  is not a fallback projection model.
- ET field `+X` maps to fsglib body `-X`; ET field `+Y` maps to fsglib body
  `+Y`; optical axis remains `+Z`.
- If exact ET geometry cannot provide an equatorial vector, guide workflows
  fail loudly instead of falling back to an approximate geometry path.

## Attitude

- `C_ib` is the direction cosine matrix that maps inertial vectors into body
  vectors:

  ```text
  v_body = C_ib @ v_inertial
  ```

- `q_ib` is the matching inertial-to-body quaternion in scalar-first order:

  ```text
  [w, x, y, z]
  ```

- SciPy `Rotation` APIs use `[x, y, z, w]`. Use
  `scalar_first_quat_to_scipy_xyzw()` and
  `scipy_xyzw_to_scalar_first_quat()` rather than reordering quaternion
  components inline.

- `RealOpticalProjector.project_to_detectors()` applies the same convention:

  ```text
  v_body = quat_to_dcm(q_ib) @ los_inertial
  ```
