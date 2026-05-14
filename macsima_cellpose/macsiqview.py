"""MacsIQView-compatible binary mask export."""

from __future__ import annotations

import numpy as np
import tifffile
from pathlib import Path


def remove_label_touching_borders(labels: np.ndarray) -> np.ndarray:
    """Set pixels touching a different non-zero label in any 8-neighbor direction to zero."""

    source = np.asarray(labels)
    output = source.copy()
    height, width = source.shape
    if height < 3 or width < 3:
        output[output != 0] = 1
        return output.astype(np.uint8, copy=False)
    center = source[1:-1, 1:-1]
    border = np.zeros_like(center, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = source[1 + dy : height - 1 + dy, 1 + dx : width - 1 + dx]
            border |= (center != 0) & (neighbor != 0) & (neighbor != center)
    inner = output[1:-1, 1:-1]
    inner[border] = 0
    output[output != 0] = 1
    return output.astype(np.uint8, copy=False)


def write_macsiqview_mask(labels: np.ndarray, output_path: Path) -> None:
    """Write a MacsIQView-compatible uint8 binary TIFF."""

    binary = remove_label_touching_borders(labels)
    tifffile.imwrite(output_path, binary, photometric="minisblack", compression="zlib")
