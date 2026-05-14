"""Cellpose nuclei inference."""

from __future__ import annotations

import logging

import numpy as np

from .gpu_utils import empty_cuda_cache


class NucleiSegmenter:
    """GPU-only Cellpose nuclei segmenter."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        from cellpose import models

        try:
            self.model = models.CellposeModel(gpu=True, model_type="nuclei")
        except TypeError:
            self.model = models.Cellpose(gpu=True, model_type="nuclei")
        self.logger.info("Cellpose model: nuclei")

    def segment_tile(self, image: np.ndarray) -> np.ndarray:
        """Run Cellpose on one 2-D nuclear tile with diameter auto-detection."""

        result = self.model.eval(image, diameter=None)
        masks = result[0] if isinstance(result, tuple) else result
        empty_cuda_cache()
        return np.asarray(masks, dtype=np.uint32)


def normalize_tile_for_cellpose(image: np.ndarray) -> np.ndarray:
    """Prepare one tile for Cellpose while leaving intensity normalization to Cellpose defaults."""

    return np.ascontiguousarray(image)
