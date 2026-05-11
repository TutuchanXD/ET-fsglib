from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def et_field_angles_to_body_vector(field_x_deg: float, field_y_deg: float) -> np.ndarray:
    field_x_rad = np.radians(float(field_x_deg))
    field_y_rad = np.radians(float(field_y_deg))
    # et_coord field +X points opposite to the fsglib camera/body +X convention.
    tan_x = -np.tan(field_x_rad)
    tan_y = np.tan(field_y_rad)
    z = 1.0 / np.sqrt(1.0 + tan_x**2 + tan_y**2)
    return np.array([z * tan_x, z * tan_y, z], dtype=np.float64)


def _normalize_vector(value: Any, *, label: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if vec.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{label} must be a finite 3-vector.")
    return vec / norm


def _solve_alignment(body_vectors: list[np.ndarray], inertial_vectors: list[np.ndarray]) -> np.ndarray:
    if len(body_vectors) != len(inertial_vectors) or not body_vectors:
        raise ValueError("Frame alignment requires paired body and inertial vectors.")

    b_mat = np.zeros((3, 3), dtype=np.float64)
    for body, inertial in zip(body_vectors, inertial_vectors):
        b_mat += np.outer(body, inertial)

    u_mat, _, vt_mat = np.linalg.svd(b_mat)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0.0:
        u_mat[:, -1] *= -1.0
        rotation = u_mat @ vt_mat
    return rotation


def _angle_arcsec_between(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = _normalize_vector(lhs, label="lhs")
    rhs = _normalize_vector(rhs, label="rhs")
    dot = np.clip(float(np.dot(lhs, rhs)), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)) * 3600.0)


@dataclass
class ExactFocalPlaneGeometryAdapter:
    transformer: Any
    rotation_body_from_eq: np.ndarray
    frame_alignment_grid_size: int
    frame_alignment_fit_rms_arcsec: float
    frame_alignment_fit_max_arcsec: float
    frame_alignment_num_samples: int
    detector_ids: tuple[str, ...]
    mode: str = "exact_et_focalplane"

    def __post_init__(self) -> None:
        self.rotation_body_from_eq = np.asarray(self.rotation_body_from_eq, dtype=np.float64)
        if self.rotation_body_from_eq.shape != (3, 3):
            raise ValueError("rotation_body_from_eq must be a 3x3 matrix.")

    def pixel_to_focal(self, detector_id: str, x_pix: float, y_pix: float):
        return self.transformer.pixel_to_focal(detector_id, float(x_pix), float(y_pix))

    def inertial_to_body_los(self, los_inertial: np.ndarray) -> np.ndarray:
        los_eq = _normalize_vector(los_inertial, label="los_inertial")
        los_body = self.rotation_body_from_eq @ los_eq
        return _normalize_vector(los_body, label="los_body")

    def pixel_to_body_los(self, detector_id: str, x_pix: float, y_pix: float) -> np.ndarray:
        sky = self.transformer.pixel_to_sky(
            detector_id,
            float(x_pix),
            float(y_pix),
            frame="equatorial",
        )
        vector_xyz = getattr(sky, "vector_xyz", None)
        if vector_xyz is None:
            status = getattr(sky, "status", "unknown")
            raise ValueError(
                f"Missing equatorial vector for detector {detector_id!r} "
                f"at pixel ({float(x_pix)}, {float(y_pix)}); status={status!r}."
            )
        try:
            los_eq = _normalize_vector(vector_xyz, label="equatorial vector")
        except (TypeError, ValueError) as exc:
            status = getattr(sky, "status", "unknown")
            raise ValueError(
                f"Invalid equatorial vector for detector {detector_id!r} "
                f"at pixel ({float(x_pix)}, {float(y_pix)}); status={status!r}: {exc}"
            ) from exc
        los_body = self.rotation_body_from_eq @ los_eq
        return _normalize_vector(los_body, label="los_body")

    def serialize(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "detector_ids": list(self.detector_ids),
            "rotation_body_from_eq": [
                [float(value) for value in row]
                for row in np.asarray(self.rotation_body_from_eq, dtype=np.float64)
            ],
            "frame_alignment_grid_size": int(self.frame_alignment_grid_size),
            "frame_alignment_num_samples": int(self.frame_alignment_num_samples),
            "frame_alignment_fit_rms_arcsec": float(self.frame_alignment_fit_rms_arcsec),
            "frame_alignment_fit_max_arcsec": float(self.frame_alignment_fit_max_arcsec),
        }


def _detector_ids_from_cfg(cfg: dict, guide_section: str) -> tuple[str, ...]:
    guide_cfg = cfg[guide_section]
    detector_ids = tuple(str(entry["detector_id"]) for entry in guide_cfg.get("detector_batches", []))
    if not detector_ids:
        raise ValueError(f"{guide_section}.detector_batches is empty.")
    return detector_ids


def _sample_alignment_vectors(registry, transformer, detector_ids: tuple[str, ...], grid_size: int):
    body_vectors: list[np.ndarray] = []
    inertial_vectors: list[np.ndarray] = []

    for detector_id in detector_ids:
        detector = registry.get_detector(detector_id)
        xs = np.linspace(0.0, float(detector.pixel_width), grid_size)
        ys = np.linspace(0.0, float(detector.pixel_height), grid_size)
        for x_pix in xs:
            for y_pix in ys:
                focal = transformer.pixel_to_focal(detector_id, float(x_pix), float(y_pix))
                field_x_deg = getattr(focal, "field_x_deg", None)
                field_y_deg = getattr(focal, "field_y_deg", None)
                if field_x_deg is None or field_y_deg is None:
                    raise ValueError(
                        f"Missing field angles for detector {detector_id!r} "
                        f"at pixel ({float(x_pix)}, {float(y_pix)})."
                    )
                sky = transformer.focal_to_sky(
                    detector_id,
                    float(focal.x_mm),
                    float(focal.y_mm),
                    frame="equatorial",
                )
                vector_xyz = getattr(sky, "vector_xyz", None)
                if vector_xyz is None:
                    status = getattr(sky, "status", "unknown")
                    raise ValueError(
                        f"Missing equatorial vector for detector {detector_id!r} "
                        f"alignment sample ({float(x_pix)}, {float(y_pix)}); status={status!r}."
                    )

                body_vectors.append(et_field_angles_to_body_vector(field_x_deg, field_y_deg))
                try:
                    inertial_vector = _normalize_vector(vector_xyz, label="alignment vector")
                except (TypeError, ValueError) as exc:
                    status = getattr(sky, "status", "unknown")
                    raise ValueError(
                        f"Invalid equatorial alignment vector for detector {detector_id!r} "
                        f"at pixel ({float(x_pix)}, {float(y_pix)}); status={status!r}: {exc}"
                    ) from exc
                inertial_vectors.append(inertial_vector)

    return body_vectors, inertial_vectors


def build_exact_focalplane_geometry_adapter(
    cfg: dict,
    registry,
    transformer,
    *,
    guide_section: str = "guide_init",
) -> ExactFocalPlaneGeometryAdapter:
    guide_cfg = cfg[guide_section]
    mode = str(guide_cfg.get("los_geometry_mode", "exact_et_focalplane"))
    if mode != "exact_et_focalplane":
        raise ValueError(
            f"Unsupported {guide_section}.los_geometry_mode={mode!r}; "
            "body_model_proxy has been removed and exact_et_focalplane is required."
        )

    grid_size = int(guide_cfg.get("frame_alignment_grid_size", 13))
    if grid_size < 2:
        raise ValueError(f"{guide_section}.frame_alignment_grid_size must be >= 2.")

    detector_ids = _detector_ids_from_cfg(cfg, guide_section)
    body_vectors, inertial_vectors = _sample_alignment_vectors(registry, transformer, detector_ids, grid_size)
    rotation_body_from_eq = _solve_alignment(body_vectors, inertial_vectors)

    residuals = [
        _angle_arcsec_between(body, rotation_body_from_eq @ inertial)
        for body, inertial in zip(body_vectors, inertial_vectors)
    ]
    residuals_arr = np.asarray(residuals, dtype=np.float64)

    return ExactFocalPlaneGeometryAdapter(
        transformer=transformer,
        rotation_body_from_eq=rotation_body_from_eq,
        frame_alignment_grid_size=grid_size,
        frame_alignment_fit_rms_arcsec=float(np.sqrt(np.mean(residuals_arr**2))),
        frame_alignment_fit_max_arcsec=float(np.max(residuals_arr)),
        frame_alignment_num_samples=int(len(residuals)),
        detector_ids=detector_ids,
    )
