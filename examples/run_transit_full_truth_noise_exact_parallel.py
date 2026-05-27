"""
Run the transit_full_01 full guide-init exact-etcoord solve for many frames.

Default command:

    python examples/run_transit_full_truth_noise_exact_parallel.py

Outputs are written under:

    /home/cxgao/Results-sshfs/liyang/transit_full_01/Results/frame000000
    ...
    /home/cxgao/Results-sshfs/liyang/transit_full_01/Results/frame000099
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "examples"))

from fsglib.pipeline.run_guide_init import run_guide_first_frame_init
from fsglib.pipeline.guide_outputs import save_matching_overlays
from run_guide_first_frame_truth_noise_exact import (
    _deep_update,
    _solution_payload,
    _write_error_budget_csv,
    _write_json,
)

DEFAULT_DATASET_ROOT = Path("/home/cxgao/Results-sshfs/liyang/transit_full_01")
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_GUIDE_CONFIG = (
    REPO_ROOT / "configs" / "guide_v1_noise_psf_etcoord.yaml"
)
DEFAULT_WORKERS = 2
DEFAULT_WORKER_MEMORY_GB = 10.0
DEFAULT_MEMORY_RESERVE_GB = 8.0


def _load_cfg(base_cfg_path: Path, guide_cfg_path: Path) -> dict:
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    return _deep_update(cfg, yaml.safe_load(guide_cfg_path.read_text(encoding="utf-8")))


def _mem_available_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / (1024.0 * 1024.0)
    return None


def _resolve_worker_count(
    requested_workers: int,
    *,
    worker_memory_gb: float,
    memory_reserve_gb: float,
    disable_memory_cap: bool,
) -> tuple[int, dict[str, Any]]:
    requested_workers = max(1, int(requested_workers))
    info: dict[str, Any] = {
        "requested_workers": requested_workers,
        "worker_memory_gb": float(worker_memory_gb),
        "memory_reserve_gb": float(memory_reserve_gb),
        "memory_cap_enabled": not disable_memory_cap,
        "memory_available_gb": _mem_available_gb(),
        "capped": False,
    }
    if disable_memory_cap or worker_memory_gb <= 0.0:
        return requested_workers, info

    available_gb = info["memory_available_gb"]
    if available_gb is None:
        return requested_workers, info

    usable_gb = max(float(available_gb) - max(float(memory_reserve_gb), 0.0), 0.0)
    max_workers_by_memory = max(1, int(usable_gb // float(worker_memory_gb)))
    info["max_workers_by_memory"] = max_workers_by_memory
    if requested_workers <= max_workers_by_memory:
        return requested_workers, info

    info["capped"] = True
    return max_workers_by_memory, info


def _frame_tag(frame_index: int) -> str:
    return f"frame{frame_index:06d}"


def _validate_dataset(
    dataset_root: Path, cfg: dict, frame_start: int, frame_stop: int
) -> dict[str, int]:
    if frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    if frame_stop <= frame_start:
        raise ValueError("--frame-stop must be greater than --frame-start")

    guide_cfg = cfg["guide_init"]
    counts: dict[str, int] = {}
    for entry in guide_cfg["detector_batches"]:
        detector_id = str(entry["detector_id"])
        batch_path = dataset_root / str(entry["batch_name"])
        frame_count = len(sorted((batch_path / "frames").glob("*.npz")))
        if frame_count == 0:
            raise FileNotFoundError(
                f"No frame files found for {detector_id}: {batch_path / 'frames'}"
            )
        counts[detector_id] = frame_count

    min_count = min(counts.values())
    if frame_stop > min_count:
        raise ValueError(
            f"Requested frame range [{frame_start}, {frame_stop}) exceeds available frames; "
            f"per-detector counts={counts}"
        )
    return counts


def _status_path(output_dir: Path) -> Path:
    return output_dir / "frame_status.json"


def _read_completed_status(output_dir: Path) -> dict | None:
    path = _status_path(output_dir)
    if not path.exists():
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if status.get("status") == "ok" and status.get("pipeline") == "guide_init_full":
        return status
    return None


def _build_frame_cfg(
    *,
    base_cfg_path: Path,
    guide_cfg_path: Path,
    dataset_root: Path,
    output_dir: Path,
    frame_index: int,
) -> dict:
    cfg = _load_cfg(base_cfg_path, guide_cfg_path)
    cfg.setdefault("project", {})["output_dir"] = str(output_dir)
    cfg["guide_init"]["dataset_root"] = str(dataset_root)
    cfg["guide_init"]["frame_index"] = int(frame_index)
    return cfg



def _write_array(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))


def _write_preprocessed_outputs(detector_dir: Path, preprocessed: Any) -> None:
    if preprocessed is None:
        return

    _write_array(detector_dir / "preprocessed_image.npy", preprocessed.image)
    _write_array(detector_dir / "preprocessed_valid_mask.npy", preprocessed.valid_mask)

    scalar_meta: dict[str, Any] = {
        "preprocess_meta": getattr(preprocessed, "preprocess_meta", {}),
    }
    for name in ("background", "noise_map", "variance_map"):
        value = getattr(preprocessed, name, None)
        if value is None:
            scalar_meta[name] = None
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            scalar_meta[name] = array.item()
        else:
            _write_array(detector_dir / f"preprocessed_{name}.npy", array)
            scalar_meta[name] = str(detector_dir / f"preprocessed_{name}.npy")

    artifact_masks = getattr(preprocessed, "artifact_masks", {}) or {}
    if artifact_masks:
        masks_dir = detector_dir / "artifact_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        scalar_meta["artifact_masks"] = {}
        for name, mask in artifact_masks.items():
            mask_path = masks_dir / f"{name}.npy"
            _write_array(mask_path, mask)
            scalar_meta["artifact_masks"][name] = str(mask_path)

    _write_json(detector_dir / "preprocessed_meta.json", scalar_meta)


def _write_guide_init_bundle(
    result: dict,
    cfg: dict,
    *,
    base_cfg_path: Path,
    guide_cfg_path: Path,
    output_dir: Path,
) -> dict:
    solution = result["solution"]
    matching = result["matching"]
    error_audit = result["error_audit"]
    error_budget = result["error_budget"]
    debug_context = result.get("debug_context", {})
    detectors = debug_context.get("detectors", {})
    observed_stars = list(debug_context.get("observed_stars", []))
    reference_stars = list(debug_context.get("reference_stars", []))
    stem = "guide_init_exact_etcoord"

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("config", "detectors", "matching", "audit", "validation", "figures", "geometry"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    error_audit_summary = (
        {
            "enabled": True,
            "truth_match_radius_pix": float(error_audit["truth_match_radius_pix"]),
            "summary": error_audit["summary"],
            "per_detector": error_audit["per_detector"],
            "selected_without_truth_count": int(len(error_audit["selected_without_truth"])),
        }
        if error_audit.get("enabled", False)
        else {"enabled": False}
    )
    payload = {
        "pipeline": "guide_init_full",
        "solution": _solution_payload(solution),
        "matching": {
            "success": bool(matching.success),
            "score": float(matching.score),
            "mode": matching.mode,
            "debug": matching.debug,
        },
        "observed_count": int(result["observed_count"]),
        "reference_count": int(result["reference_count"]),
        "detector_stats": result["detector_stats"],
        "sim_to_detector_map": result["sim_to_detector_map"],
        "geometry_adapter": result["geometry_adapter"],
        "error_audit": error_audit_summary,
        "error_budget": error_budget,
        "meta": {**result["meta"], "config_path": str(guide_cfg_path), "output_dir": str(output_dir)},
    }

    _write_json(output_dir / f"{stem}_result.json", payload)
    _write_json(output_dir / f"{stem}_error_audit.json", error_audit)
    _write_json(output_dir / f"{stem}_error_budget.json", error_budget)
    _write_error_budget_csv(output_dir / f"{stem}_error_budget_terms.csv", error_budget)

    _write_json(output_dir / "run_summary.json", payload)
    _write_json(output_dir / "solution.json", payload["solution"])
    _write_json(output_dir / "geometry" / "geometry_adapter.json", result["geometry_adapter"])
    _write_json(output_dir / "geometry" / "sim_to_detector_map.json", result["sim_to_detector_map"])
    _write_json(output_dir / "matching" / "matching_result.json", payload["matching"])
    _write_json(output_dir / "matching" / "matched_stars.json", matching.matched)
    _write_json(output_dir / "matching" / "observed_stars.json", observed_stars)
    _write_json(output_dir / "matching" / "reference_stars.json", reference_stars)
    _write_json(output_dir / "matching" / "detector_stats.json", result["detector_stats"])
    _write_json(output_dir / "audit" / "guide_error_audit.json", error_audit)
    _write_json(output_dir / "validation" / "error_budget.json", error_budget)
    _write_error_budget_csv(output_dir / "validation" / "error_budget_terms.csv", error_budget)

    (output_dir / "config" / base_cfg_path.name).write_text(
        base_cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "config" / guide_cfg_path.name).write_text(
        guide_cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "config" / "merged_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(output_dir / "config" / "run_meta.json", payload["meta"])

    for detector_id, context in detectors.items():
        detector_dir = output_dir / "detectors" / str(detector_id)
        detector_dir.mkdir(parents=True, exist_ok=True)

        raw = context.get("raw")
        if raw is not None:
            _write_array(detector_dir / "raw_image.npy", raw.image)
            _write_json(detector_dir / "raw_meta.json", raw.meta)
        elif context.get("image") is not None:
            _write_array(detector_dir / "raw_image.npy", context["image"])
            _write_json(detector_dir / "raw_meta.json", {})

        preprocessed = context.get("preprocessed")
        if preprocessed is not None:
            _write_preprocessed_outputs(detector_dir, preprocessed)
        elif context.get("preprocessed_image") is not None:
            _write_array(detector_dir / "preprocessed_image.npy", context["preprocessed_image"])
            _write_json(detector_dir / "preprocessed_meta.json", {})

        _write_json(detector_dir / "all_candidates.json", context.get("all_candidates", []))
        _write_json(detector_dir / "selected_candidates.json", context.get("selected_candidates", []))
        _write_json(detector_dir / "detector_stats.json", result["detector_stats"].get(detector_id, {}))

    overlay_summary = save_matching_overlays(result, output_dir / "figures")
    _write_json(output_dir / "figures" / "matching_overlay_summary.json", overlay_summary)

    readme = f"""# fsglib full guide-init exact-etcoord debug bundle

Configuration: `{guide_cfg_path}`
Dataset root: `{result["meta"]["dataset_root"]}`

## Result

- valid: `{bool(solution.valid)}`
- matched: `{int(solution.num_matched)}` / observed `{int(result["observed_count"])}`
- reference stars: `{int(result["reference_count"])}`
- residual_rms_arcsec: `{float(solution.residual_rms_arcsec):.6f}`
- matching strategy: `{matching.debug.get("selected_strategy")}`
- predicted-position matches: `{matching.debug.get("num_predicted_position_matches")}`
- local-pyramid matches: `{matching.debug.get("num_local_pyramid_matches")}`

## Contents

- `run_summary.json`: full top-level result payload.
- `solution.json`: attitude solution details.
- `matching/`: observed/reference/matched stars and matcher debug.
- `detectors/<detector_id>/`: raw image, preprocessed image/products, extracted candidates.
- `geometry/`: exact ET focal-plane geometry and sim-to-detector bridge.
- `audit/guide_error_audit.json`: guide-specific truth/matching/LOS/attitude audit.
- `validation/error_budget.json` and `validation/error_budget_terms.csv`: error-budget ledger.
- `figures/matching_overlay_*.png`: per-detector matching overlays.
- `config/`: base, guide-init, and merged configuration snapshots.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "result_json": str(output_dir / f"{stem}_result.json"),
        "error_audit_json": str(output_dir / f"{stem}_error_audit.json"),
        "error_budget_json": str(output_dir / f"{stem}_error_budget.json"),
        "error_budget_csv": str(output_dir / f"{stem}_error_budget_terms.csv"),
        "overlay_summary": str(output_dir / "figures" / "matching_overlay_summary.json"),
    }


def _run_one_frame(
    frame_index: int,
    *,
    base_cfg_path: str,
    guide_cfg_path: str,
    dataset_root: str,
    output_root: str,
    overwrite: bool,
) -> dict[str, Any]:
    base_cfg = Path(base_cfg_path)
    guide_cfg = Path(guide_cfg_path)
    root = Path(dataset_root)
    output_dir = Path(output_root) / _frame_tag(frame_index)
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_frame_tag(frame_index)}.log"

    if not overwrite:
        completed = _read_completed_status(output_dir)
        if completed is not None:
            return {
                **completed,
                "status": "skipped",
                "frame_index": int(frame_index),
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }

    with log_path.open("w", encoding="utf-8") as log_handle:
        with (
            contextlib.redirect_stdout(log_handle),
            contextlib.redirect_stderr(log_handle),
        ):
            print(f"frame_index={frame_index}")
            print(f"dataset_root={root}")
            print(f"output_dir={output_dir}")
            try:
                cfg = _build_frame_cfg(
                    base_cfg_path=base_cfg,
                    guide_cfg_path=guide_cfg,
                    dataset_root=root,
                    output_dir=output_dir,
                    frame_index=frame_index,
                )
                result = run_guide_first_frame_init(cfg, include_debug_context=True)
                bundle_paths = _write_guide_init_bundle(
                    result,
                    cfg,
                    base_cfg_path=base_cfg,
                    guide_cfg_path=guide_cfg,
                    output_dir=output_dir,
                )
                solution = result["solution"]
                status = {
                    "status": "ok",
                    "pipeline": "guide_init_full",
                    "frame_index": int(frame_index),
                    "output_dir": str(output_dir),
                    "log_path": str(log_path),
                    "valid": bool(solution.valid),
                    "matched": int(solution.num_matched),
                    "observed": int(result["observed_count"]),
                    "reference": int(result["reference_count"]),
                    "residual_rms_arcsec": float(solution.residual_rms_arcsec),
                    "quality_flag": solution.quality_flag,
                    "degraded_level": solution.degraded_level,
                    "result_json": bundle_paths["result_json"],
                    "error_budget_json": bundle_paths["error_budget_json"],
                    "overlay_summary": bundle_paths["overlay_summary"],
                }
                _write_json(_status_path(output_dir), status)
                print(json.dumps(status, indent=2))
                return status
            except BaseException as exc:
                status = {
                    "status": "failed",
                    "pipeline": "guide_init_full",
                    "frame_index": int(frame_index),
                    "output_dir": str(output_dir),
                    "log_path": str(log_path),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                _write_json(_status_path(output_dir), status)
                print(status["traceback"])
                return status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 100-frame four-detector full guide-init exact-etcoord solves in parallel.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Input dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Default: <dataset-root>/Results/guide_init_full",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=f"Base config path. Default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--guide-config",
        type=Path,
        default=DEFAULT_GUIDE_CONFIG,
        help=f"Guide config path. Default: {DEFAULT_GUIDE_CONFIG}",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Inclusive start frame index. Default: 0",
    )
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=100,
        help="Exclusive stop frame index. Default: 100",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of frame solves to run concurrently. Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--worker-memory-gb",
        type=float,
        default=DEFAULT_WORKER_MEMORY_GB,
        help=(
            "Estimated memory budget per worker for automatic worker capping. "
            f"Default: {DEFAULT_WORKER_MEMORY_GB:g} GiB"
        ),
    )
    parser.add_argument(
        "--memory-reserve-gb",
        type=float,
        default=DEFAULT_MEMORY_RESERVE_GB,
        help=(
            "Memory kept free when automatically capping workers. "
            f"Default: {DEFAULT_MEMORY_RESERVE_GB:g} GiB"
        ),
    )
    parser.add_argument(
        "--disable-memory-cap",
        action="store_true",
        help="Run exactly --workers processes without the automatic memory safety cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run frames even when frame_status.json already reports status=ok.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root / "Results" / "guide_init_full"
    )
    base_cfg_path = args.base_config.expanduser().resolve()
    guide_cfg_path = args.guide_config.expanduser().resolve()
    workers, worker_memory_info = _resolve_worker_count(
        int(args.workers),
        worker_memory_gb=float(args.worker_memory_gb),
        memory_reserve_gb=float(args.memory_reserve_gb),
        disable_memory_cap=bool(args.disable_memory_cap),
    )
    frames = list(range(int(args.frame_start), int(args.frame_stop)))

    cfg = _load_cfg(base_cfg_path, guide_cfg_path)
    counts = _validate_dataset(
        dataset_root, cfg, int(args.frame_start), int(args.frame_stop)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    print("Transit full guide-init exact parallel run")
    print(f"dataset_root: {dataset_root}")
    print(f"output_root:  {output_root}")
    print(
        f"frames:       {args.frame_start}..{args.frame_stop - 1} ({len(frames)} total)"
    )
    print(f"workers:      {workers} requested={worker_memory_info['requested_workers']}")
    if worker_memory_info.get("capped", False):
        print(
            "memory cap:   requested workers capped from {requested} to {actual}; "
            "available={available:.1f} GiB reserve={reserve:.1f} GiB "
            "per_worker={per_worker:.1f} GiB".format(
                requested=worker_memory_info["requested_workers"],
                actual=workers,
                available=worker_memory_info["memory_available_gb"],
                reserve=worker_memory_info["memory_reserve_gb"],
                per_worker=worker_memory_info["worker_memory_gb"],
            )
        )
    print(f"detectors:    {counts}")
    print(f"overwrite:    {bool(args.overwrite)}")

    worker_kwargs = {
        "base_cfg_path": str(base_cfg_path),
        "guide_cfg_path": str(guide_cfg_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "overwrite": bool(args.overwrite),
    }
    results: list[dict[str, Any]] = []
    if workers == 1:
        for frame_index in frames:
            status = _run_one_frame(frame_index, **worker_kwargs)
            results.append(status)
            print(
                "[{status}] {tag}: matched={matched} rms={rms} log={log}".format(
                    status=status["status"],
                    tag=_frame_tag(frame_index),
                    matched=status.get("matched", "-"),
                    rms=status.get("residual_rms_arcsec", "-"),
                    log=status["log_path"],
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_one_frame, frame_index, **worker_kwargs
                ): frame_index
                for frame_index in frames
            }
            for future in as_completed(futures):
                frame_index = futures[future]
                status = future.result()
                results.append(status)
                print(
                    "[{status}] {tag}: matched={matched} rms={rms} log={log}".format(
                        status=status["status"],
                        tag=_frame_tag(frame_index),
                        matched=status.get("matched", "-"),
                        rms=status.get("residual_rms_arcsec", "-"),
                        log=status["log_path"],
                    ),
                    flush=True,
                )

    results = sorted(results, key=lambda item: int(item["frame_index"]))
    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "base_config": str(base_cfg_path),
        "guide_config": str(guide_cfg_path),
        "frame_start": int(args.frame_start),
        "frame_stop": int(args.frame_stop),
        "workers": workers,
        "worker_memory_info": worker_memory_info,
        "overwrite": bool(args.overwrite),
        "counts": {
            "ok": sum(1 for item in results if item["status"] == "ok"),
            "skipped": sum(1 for item in results if item["status"] == "skipped"),
            "failed": sum(1 for item in results if item["status"] == "failed"),
        },
        "frames": results,
    }
    summary_path = output_root / "parallel_run_summary.json"
    _write_json(summary_path, summary)
    print(f"Summary JSON: {summary_path}")

    failed = [item for item in results if item["status"] == "failed"]
    if failed:
        print(f"Failed frames: {[item['frame_index'] for item in failed]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
