"""Centroid ownership stitching for overlapped tile predictions."""

from __future__ import annotations

import numpy as np

from .tiling import Tile


OwnedObject = tuple[np.ndarray, np.ndarray]


def select_owned_objects(tile_mask: np.ndarray, tile: Tile) -> tuple[int, list[OwnedObject]]:
    """Return objects whose global centroid is inside the tile owned region."""

    labels = np.unique(tile_mask)
    labels = labels[labels != 0]
    owned: list[OwnedObject] = []
    if labels.size == 0:
        return 0, owned
    for label in labels:
        yy, xx = np.nonzero(tile_mask == label)
        if yy.size == 0:
            continue
        centroid_y = float(yy.mean()) + tile.read_y0
        centroid_x = float(xx.mean()) + tile.read_x0
        if not (tile.own_y0 <= centroid_y < tile.own_y1 and tile.own_x0 <= centroid_x < tile.own_x1):
            continue
        global_y = yy + tile.read_y0
        global_x = xx + tile.read_x0
        owned.append((global_y, global_x))
    return int(labels.size), owned


def write_owned_objects(global_mask: np.ndarray, owned_objects: list[OwnedObject], next_label: int) -> int:
    """Write retained complete objects into the global label mask."""

    for global_y, global_x in owned_objects:
        inside = (global_y >= 0) & (global_y < global_mask.shape[0]) & (global_x >= 0) & (global_x < global_mask.shape[1])
        global_mask[global_y[inside], global_x[inside]] = next_label
        next_label += 1
    return next_label


def sequential_relabel(mask: np.ndarray) -> np.ndarray:
    """Relabel a mask sequentially from 1 while preserving background as 0."""

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return mask.astype(np.uint32, copy=False)
    relabeled = np.zeros(mask.shape, dtype=np.uint32)
    for new_label, old_label in enumerate(labels, start=1):
        relabeled[mask == old_label] = new_label
    return relabeled
