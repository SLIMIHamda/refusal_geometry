"""Reproducibility spine: global seeding, deterministic flags, environment capture.

Importing this module never requires torch/transformers; the GPU-only bits degrade
gracefully so the harness spine stays testable without a GPU. `capture_env()` returns
exactly the columns the `runs` table (and Tab A) need for one-command reproduction.
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
from typing import Any


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we touch and (optionally) force deterministic kernels."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # warn_only: some HF ops lack deterministic kernels; we log rather than crash.
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    out = _git("status", "--porcelain")
    return None if out is None else bool(out)


def gpu_info() -> tuple[str | None, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.device_count()
    except Exception:
        pass
    return None, 0


def capture_env() -> dict[str, Any]:
    """Snapshot the host/software state stamped onto every run row."""
    gpu_type, gpu_count = gpu_info()
    cuda = None
    try:
        import torch

        cuda = torch.version.cuda
    except Exception:
        pass
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "torch_version": _pkg_version("torch"),
        "transformers_version": _pkg_version("transformers"),
        "accelerate_version": _pkg_version("accelerate"),
        "datasets_version": _pkg_version("datasets"),
        "cuda_version": cuda,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
