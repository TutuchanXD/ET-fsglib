from __future__ import annotations

import os
from typing import Any


def _read_mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024**2)
    except OSError:
        return None
    return None


def _memory_limit_gb(default_memory_gb: float) -> dict[str, Any]:
    explicit = os.environ.get("FSGLIB_SMOKE_MAX_MEMORY_GB")
    if explicit is not None:
        return {
            "requested_gb": float(explicit),
            "source": "FSGLIB_SMOKE_MAX_MEMORY_GB",
            "available_gb": _read_mem_available_gb(),
            "reserve_gb": None,
            "fraction": None,
        }

    if default_memory_gb <= 0:
        return {
            "requested_gb": 0.0,
            "source": "disabled_default",
            "available_gb": _read_mem_available_gb(),
            "reserve_gb": None,
            "fraction": None,
        }

    available_gb = _read_mem_available_gb()
    if available_gb is None:
        return {
            "requested_gb": float(default_memory_gb),
            "source": "fallback_default",
            "available_gb": None,
            "reserve_gb": None,
            "fraction": None,
        }

    reserve_gb = float(os.environ.get("FSGLIB_SMOKE_RESERVE_MEMORY_GB", "8.0"))
    fraction = float(os.environ.get("FSGLIB_SMOKE_MEMORY_FRACTION", "0.60"))
    min_gb = float(os.environ.get("FSGLIB_SMOKE_MIN_MEMORY_GB", "4.0"))
    usable_gb = max(available_gb - reserve_gb, 0.0)
    requested_gb = max(min_gb, usable_gb * fraction)
    requested_gb = min(requested_gb, available_gb)
    return {
        "requested_gb": requested_gb,
        "source": "mem_available_fraction",
        "available_gb": available_gb,
        "reserve_gb": reserve_gb,
        "fraction": fraction,
    }


def configure_smoke_process(
    *,
    default_memory_gb: float = 12.0,
    default_cpu_seconds: int = 180,
) -> dict[str, Any]:
    """Apply conservative process limits for local smoke scripts.

    The limits are intentionally process-local. They keep smoke runs from
    exhausting workstation memory while leaving production library code
    unrestricted.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")

    info: dict[str, Any] = {
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "memory_limit_gb": None,
        "memory_limit_source": None,
        "memory_available_gb": None,
        "memory_reserve_gb": None,
        "memory_fraction": None,
        "cpu_limit_seconds": None,
        "resource_module": False,
    }

    try:
        import resource
    except Exception:
        return info

    info["resource_module"] = True
    memory_request = _memory_limit_gb(default_memory_gb)
    memory_gb = float(memory_request["requested_gb"])
    info["memory_limit_source"] = memory_request["source"]
    info["memory_available_gb"] = memory_request["available_gb"]
    info["memory_reserve_gb"] = memory_request["reserve_gb"]
    info["memory_fraction"] = memory_request["fraction"]
    cpu_seconds = int(os.environ.get("FSGLIB_SMOKE_MAX_CPU_SECONDS", default_cpu_seconds))

    if memory_gb > 0:
        requested = int(memory_gb * (1024**3))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        hard_limit = requested if hard in (-1, resource.RLIM_INFINITY) else min(hard, requested)
        soft_limit = requested if soft in (-1, resource.RLIM_INFINITY) else min(soft, requested)
        soft_limit = min(soft_limit, hard_limit)
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
        info["memory_limit_gb"] = soft_limit / (1024**3)

    if cpu_seconds > 0:
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        hard_limit = cpu_seconds if hard in (-1, resource.RLIM_INFINITY) else min(hard, cpu_seconds)
        soft_limit = cpu_seconds if soft in (-1, resource.RLIM_INFINITY) else min(soft, cpu_seconds)
        soft_limit = min(soft_limit, hard_limit)
        resource.setrlimit(resource.RLIMIT_CPU, (soft_limit, hard_limit))
        info["cpu_limit_seconds"] = soft_limit

    return info


def apply_guide_smoke_scale(cfg: dict, *, section: str) -> dict:
    guide_cfg = cfg[section]
    max_observed = int(os.environ.get("FSGLIB_SMOKE_MAX_OBS_PER_DETECTOR", "25"))
    reference_topk = int(os.environ.get("FSGLIB_SMOKE_REFERENCE_TOPK_PER_DETECTOR", "60"))
    catalog_g_mag_max = float(os.environ.get("FSGLIB_SMOKE_CATALOG_G_MAG_MAX", "12.0"))
    guide_cfg["max_observed_per_detector"] = max_observed
    guide_cfg["reference_topk_per_detector"] = reference_topk
    guide_cfg["reference_preselect_topk_per_detector"] = reference_topk
    guide_cfg["catalog_g_mag_max"] = catalog_g_mag_max
    return {
        "section": section,
        "max_observed_per_detector": max_observed,
        "reference_topk_per_detector": reference_topk,
        "reference_preselect_topk_per_detector": reference_topk,
        "catalog_g_mag_max": catalog_g_mag_max,
    }
