"""
Run the transit_full_01 truth-noise exact-etcoord solve for many frames.

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

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "examples"))

from fsglib.pipeline import run_guide_first_frame_truth_noise
from run_guide_first_frame_truth_noise_exact import _deep_update, _write_json, _write_truth_noise_bundle


DEFAULT_DATASET_ROOT = Path("/home/cxgao/Results-sshfs/liyang/transit_full_01")
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_GUIDE_CONFIG = REPO_ROOT / "configs" / "guide_truth_noise_0065pix_exact_etcoord.yaml"


def _load_cfg(base_cfg_path: Path, guide_cfg_path: Path) -> dict:
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    return _deep_update(cfg, yaml.safe_load(guide_cfg_path.read_text(encoding="utf-8")))


def _frame_tag(frame_index: int) -> str:
    return f"frame{frame_index:06d}"


def _validate_dataset(dataset_root: Path, cfg: dict, frame_start: int, frame_stop: int) -> dict[str, int]:
    if frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    if frame_stop <= frame_start:
        raise ValueError("--frame-stop must be greater than --frame-start")

    guide_cfg = cfg["guide_truth_noise"]
    counts: dict[str, int] = {}
    for entry in guide_cfg["detector_batches"]:
        detector_id = str(entry["detector_id"])
        batch_path = dataset_root / str(entry["batch_name"])
        frame_count = len(sorted((batch_path / "frames").glob("*.npz")))
        if frame_count == 0:
            raise FileNotFoundError(f"No frame files found for {detector_id}: {batch_path / 'frames'}")
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
    if status.get("status") == "ok":
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
    cfg["guide_truth_noise"]["dataset_root"] = str(dataset_root)
    cfg["guide_truth_noise"]["frame_index"] = int(frame_index)
    return cfg


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
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
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
                result = run_guide_first_frame_truth_noise(cfg, include_debug_context=True)
                bundle_paths = _write_truth_noise_bundle(
                    result,
                    cfg,
                    base_cfg_path=base_cfg,
                    guide_cfg_path=guide_cfg,
                    output_dir=output_dir,
                )
                solution = result["solution"]
                status = {
                    "status": "ok",
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
        description="Run 100-frame four-detector truth-noise exact-etcoord solves in parallel.",
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
        help="Output root. Default: <dataset-root>/Results",
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
        default=4,
        help="Number of frame solves to run concurrently. Default: 4",
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
        else dataset_root / "Results"
    )
    base_cfg_path = args.base_config.expanduser().resolve()
    guide_cfg_path = args.guide_config.expanduser().resolve()
    workers = max(1, int(args.workers))
    frames = list(range(int(args.frame_start), int(args.frame_stop)))

    cfg = _load_cfg(base_cfg_path, guide_cfg_path)
    counts = _validate_dataset(dataset_root, cfg, int(args.frame_start), int(args.frame_stop))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    print("Transit full truth-noise exact parallel run")
    print(f"dataset_root: {dataset_root}")
    print(f"output_root:  {output_root}")
    print(f"frames:       {args.frame_start}..{args.frame_stop - 1} ({len(frames)} total)")
    print(f"workers:      {workers}")
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
                executor.submit(_run_one_frame, frame_index, **worker_kwargs): frame_index
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
