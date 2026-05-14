"""Tile geometry for overlap inference and centroid ownership."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Tile:
    """A tile with an owned region and an expanded inference region."""

    row: int
    col: int
    own_y0: int
    own_y1: int
    own_x0: int
    own_x1: int
    read_y0: int
    read_y1: int
    read_x0: int
    read_x1: int


def own_region_size(tile_size: int, overlap_px: int) -> int:
    """Return the non-overlap owned region size."""

    size = tile_size - 2 * overlap_px
    if size <= 0:
        raise ValueError("tile_size must be greater than 2 * overlap")
    return size


def compute_grid(height: int, width: int, tile_size: int, overlap_px: int) -> tuple[int, int]:
    """Compute grid dimensions from image size and owned region size."""

    size = own_region_size(tile_size, overlap_px)
    return ceil(height / size), ceil(width / size)


def iter_tiles(height: int, width: int, tile_size: int, overlap_px: int) -> list[Tile]:
    """Create tiles using non-overlap owned regions expanded by a halo."""

    size = own_region_size(tile_size, overlap_px)
    n_rows, n_cols = compute_grid(height, width, tile_size, overlap_px)
    tiles: list[Tile] = []
    for row in range(n_rows):
        own_y0 = row * size
        own_y1 = min(height, own_y0 + size)
        read_y0 = max(0, own_y0 - overlap_px)
        read_y1 = min(height, own_y1 + overlap_px)
        for col in range(n_cols):
            own_x0 = col * size
            own_x1 = min(width, own_x0 + size)
            read_x0 = max(0, own_x0 - overlap_px)
            read_x1 = min(width, own_x1 + overlap_px)
            tiles.append(Tile(row, col, own_y0, own_y1, own_x0, own_x1, read_y0, read_y1, read_x0, read_x1))
    return tiles
