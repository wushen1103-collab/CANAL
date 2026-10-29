from __future__ import annotations

import os


MAX_CPU_THREADS = 192
DEFAULT_CPU_THREADS = 32
MAX_GPUS = 4


def cpu_threads(default: int = DEFAULT_CPU_THREADS) -> int:
    raw = os.environ.get("CANAL_CPU_THREADS", str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(1, min(value, MAX_CPU_THREADS))


def gpu_list() -> list[str]:
    raw = os.environ.get("CANAL_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not raw.strip():
        return []
    gpus = [g.strip() for g in raw.split(",") if g.strip()]
    return gpus[:MAX_GPUS]


def apply_thread_env(default: int = DEFAULT_CPU_THREADS) -> int:
    threads = cpu_threads(default)
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = str(threads)
    return threads
