from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DEBUG_OUTPUT_DIR = Path("outputs/debug")


def _project_output_dir(cfg: dict) -> Path:
    return Path(cfg.get("project", {}).get("output_dir", str(DEFAULT_DEBUG_OUTPUT_DIR))).expanduser()


def _is_default_output_dir(path: Path) -> bool:
    parts = Path(path).parts
    if parts and parts[0] == ".":
        parts = parts[1:]
    return parts == DEFAULT_DEBUG_OUTPUT_DIR.parts


def _dataset_results_root(cfg: dict, output_dir: Path) -> Path | None:
    dataset_root_value = cfg.get("guide_init", {}).get("dataset_root")
    if dataset_root_value is not None and _is_default_output_dir(output_dir):
        dataset_root = Path(dataset_root_value).expanduser()
        if dataset_root.exists():
            dataset_root = dataset_root.resolve()
            return dataset_root.with_name(f"{dataset_root.name}_fsg-results")
    return None


def _frame_output_dir_name(cfg: dict) -> str:
    frame_index = int(cfg.get("guide_init", {}).get("frame_index", 0))
    return f"frame{frame_index:06d}"


def resolve_fsg_results_root(cfg: dict) -> Path:
    output_dir = _project_output_dir(cfg)
    dataset_root = _dataset_results_root(cfg, output_dir)
    if dataset_root is not None:
        return dataset_root
    return output_dir


def resolve_debug_output_dir(cfg: dict) -> Path:
    output_dir = _project_output_dir(cfg)
    dataset_root = _dataset_results_root(cfg, output_dir)
    if dataset_root is not None:
        return dataset_root / _frame_output_dir_name(cfg) / "debug"
    return output_dir


def resolve_figures_output_dir(cfg: dict) -> Path:
    output_dir = _project_output_dir(cfg)
    dataset_root = _dataset_results_root(cfg, output_dir)
    if dataset_root is not None:
        return dataset_root / _frame_output_dir_name(cfg) / "figures"
    return output_dir / "figures"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_detector_name(detector_id: Any) -> str:
    return str(detector_id).replace("/", "_").replace("\\", "_").replace(":", "_")


def _xy_array(points: list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _plot_points(ax: Any, xy: np.ndarray, *, label: str, color: str, marker: str, size: float, alpha: float = 1.0) -> None:
    if xy.size == 0:
        return
    if marker == "o-open":
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=size,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            alpha=alpha,
            label=label,
        )
    else:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=size,
            c=color,
            marker=marker,
            linewidths=0.9,
            alpha=alpha,
            label=label,
        )


def _image_limits(image: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(image, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.7))
    if vmax <= vmin:
        return None, None
    return vmin, vmax


def save_matching_overlays(result: dict, output_dir: str | Path) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to save matching overlays") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matching = result["matching"]
    debug_context = result.get("debug_context", {})
    detectors = {str(detector_id): context for detector_id, context in debug_context.get("detectors", {}).items()}
    observed_stars = list(debug_context.get("observed_stars", []))
    reference_stars = list(debug_context.get("reference_stars", []))

    observed_by_detector: dict[str, list[Any]] = {}
    for star in observed_stars:
        observed_by_detector.setdefault(str(star.detector_id), []).append(star)

    reference_by_detector: dict[str, list[tuple[Any, Any]]] = {}
    for star in reference_stars:
        for detector_id in star.predicted_xy:
            if star.predicted_valid.get(detector_id, False):
                reference_by_detector.setdefault(str(detector_id), []).append((detector_id, star))

    matched_by_detector: dict[str, list[Any]] = {}
    matched_sources: set[tuple[str, str]] = set()
    matched_catalogs: set[tuple[str, int]] = set()
    for star in matching.matched:
        detector_id = str(star.detector_id)
        matched_by_detector.setdefault(detector_id, []).append(star)
        matched_sources.add((detector_id, str(star.source_id)))
        matched_catalogs.add((detector_id, int(star.catalog_id)))

    all_detector_ids = sorted(set(detectors) | set(observed_by_detector) | set(reference_by_detector) | set(matched_by_detector))
    summary = {
        "output_dir": str(output_path),
        "selected_strategy": matching.debug.get("selected_strategy"),
        "algorithm": matching.debug.get("algorithm"),
        "detectors": {},
    }

    for detector_id in all_detector_ids:
        detector_context = detectors.get(detector_id, {})
        image = np.asarray(detector_context.get("image", np.zeros((1, 1), dtype=np.float64)), dtype=np.float64)
        observed = observed_by_detector.get(detector_id, [])
        reference = reference_by_detector.get(detector_id, [])
        matched = matched_by_detector.get(detector_id, [])

        matched_observed_xy = _xy_array(
            [
                tuple(star.flags["observed_xy"])
                for star in matched
                if star.flags.get("observed_xy") is not None
            ]
        )
        matched_reference_xy = _xy_array(
            [
                tuple(star.flags["predicted_xy"])
                for star in matched
                if star.flags.get("predicted_xy") is not None
            ]
        )
        unmatched_observed_xy = _xy_array(
            [
                (float(star.x), float(star.y))
                for star in observed
                if (detector_id, str(star.source_id)) not in matched_sources
            ]
        )
        unmatched_reference_xy = _xy_array(
            [
                tuple(star.predicted_xy[detector_key])
                for detector_key, star in reference
                if (detector_id, int(star.catalog_id)) not in matched_catalogs
            ]
        )

        fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
        vmin, vmax = _image_limits(image)
        ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
        _plot_points(
            ax,
            unmatched_reference_xy,
            label="unmatched reference",
            color="#46a0ff",
            marker="o-open",
            size=18,
            alpha=0.35,
        )
        _plot_points(
            ax,
            unmatched_observed_xy,
            label="unmatched observed",
            color="#ff4d4d",
            marker="x",
            size=32,
            alpha=0.95,
        )
        _plot_points(
            ax,
            matched_reference_xy,
            label="matched reference",
            color="#ffd166",
            marker="o-open",
            size=36,
            alpha=0.9,
        )
        _plot_points(
            ax,
            matched_observed_xy,
            label="matched observed",
            color="#7CFC00",
            marker="+",
            size=34,
            alpha=0.95,
        )

        if matched_reference_xy.shape == matched_observed_xy.shape and matched_reference_xy.size:
            for ref_xy, obs_xy in zip(matched_reference_xy, matched_observed_xy):
                ax.plot(
                    [ref_xy[0], obs_xy[0]],
                    [ref_xy[1], obs_xy[1]],
                    color="#ffd166",
                    linewidth=0.45,
                    alpha=0.7,
                )

        ax.set_title(
            f"{detector_id} matching overlay\n"
            f"matched={len(matched)} unmatched_obs={len(unmatched_observed_xy)} "
            f"unmatched_ref={len(unmatched_reference_xy)} strategy={matching.debug.get('selected_strategy')}"
        )
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()

        overlay_path = output_path / f"matching_overlay_{_safe_detector_name(detector_id)}.png"
        fig.savefig(overlay_path, bbox_inches="tight")
        plt.close(fig)

        summary["detectors"][detector_id] = {
            "overlay_path": str(overlay_path),
            "selected_strategy": matching.debug.get("selected_strategy"),
            "num_observed": int(len(observed)),
            "num_reference": int(len(reference)),
            "num_matched_observed": int(len(matched)),
            "num_unmatched_observed": int(len(unmatched_observed_xy)),
            "num_matched_reference": int(len(matched_reference_xy)),
            "num_unmatched_reference": int(len(unmatched_reference_xy)),
        }

    (output_path / "matching_overlay_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary
