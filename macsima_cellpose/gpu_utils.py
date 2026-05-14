"""GPU checks and memory cleanup."""

from __future__ import annotations

import logging


def require_gpu(logger: logging.Logger) -> str:
    """Require CUDA and return a concise GPU description."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("PyTorch is required for GPU Cellpose execution.") from exc
    available = bool(torch.cuda.is_available())
    logger.info("CUDA availability: %s", available)
    if not available:
        raise RuntimeError("CUDA GPU is required, but CUDA is not available.")
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    info = f"cuda:{device_index} {props.name} ({props.total_memory / 1024**3:.1f} GiB)"
    logger.info("GPU info: %s", info)
    return info


def empty_cuda_cache() -> None:
    """Release cached CUDA memory when PyTorch is available."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def is_cuda_oom(exc: BaseException) -> bool:
    """Detect CUDA out-of-memory failures across common libraries."""

    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in text and ("out of memory" in text or "oom" in text)
