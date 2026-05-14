"""CLI-only adaptation of the Fusion_analysis v7 segmentation core."""

from __future__ import annotations

import gc
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile

from .gpu_utils import empty_cuda_cache
from .ome_reader import OmeTileReader, choose_nuclear_channel


@dataclass(frozen=True)
class V7Tile:
    """V7-style tile with owned and read regions."""

    row: int
    col: int
    own: tuple[int, int, int, int]
    read: tuple[int, int, int, int]


@dataclass(frozen=True)
class V7Result:
    """Summary of one segmentation run."""

    total_tiles: int
    total_labels: int
    tile_times: list[float]
    label_output: Path
    macsiqview_output: Path


class FastOmeSource:
    """Sample-scoped lazy OME-TIFF zarr source for fast tile reads."""

    def __init__(self, path: Path, logger: logging.Logger) -> None:
        self.path = path
        self.logger = logger
        self.tif: tifffile.TiffFile | None = None
        self.store: Any | None = None
        self.z0: Any | None = None
        self.height = 0
        self.width = 0
        self.ndim = 0
        self.shape: tuple[int, ...] = ()

    def __enter__(self) -> "FastOmeSource":
        import zarr

        self.tif = tifffile.TiffFile(self.path)
        page0 = self.tif.pages[0]
        self.height = int(page0.imagelength)
        self.width = int(page0.imagewidth)
        self.store = self.tif.aszarr()
        z = zarr.open(self.store, mode="r")
        self.z0 = OmeTileReader._select_z0(z)
        self.shape = tuple(int(v) for v in getattr(self.z0, "shape", ()))
        self.ndim = int(getattr(self.z0, "ndim", 0))
        self.logger.info("Fast OME zarr object type: %s", type(z).__name__)
        self.logger.info("Fast OME selected z0 shape: %s", self.shape)
        self.logger.info("Fast OME selected z0 ndim: %d", self.ndim)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.store is not None and hasattr(self.store, "close"):
            try:
                self.store.close()
            except Exception:
                pass
        if self.tif is not None:
            self.tif.close()

    @property
    def channel_count(self) -> int:
        if self.ndim == 3 and self.shape:
            return int(self.shape[0])
        if self.ndim == 4 and len(self.shape) >= 2:
            return int(self.shape[1])
        if self.ndim == 2:
            return 1
        raise RuntimeError(f"Unsupported OME-TIFF zarr dimensions: ndim={self.ndim}")

    def read_channel(self, channel_index: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Read one selected channel tile from the open zarr source."""

        if self.z0 is None:
            raise RuntimeError("OME source is not open.")
        if self.ndim == 3:
            return np.asarray(self.z0[channel_index, y0:y1, x0:x1])
        if self.ndim == 4:
            return np.asarray(self.z0[0, channel_index, y0:y1, x0:x1])
        if self.ndim == 2:
            return np.asarray(self.z0[y0:y1, x0:x1])
        raise RuntimeError(f"Unsupported OME-TIFF zarr dimensions for ROI read: ndim={self.ndim}")


def build_v7_tiles(height: int, width: int, n_rows: int, n_cols: int, overlap_px: int) -> list[V7Tile]:
    """Build the Fusion_analysis v7 n_rows by n_cols tile grid."""

    tile_h = -(-height // n_rows)
    tile_w = -(-width // n_cols)
    tiles: list[V7Tile] = []
    for row in range(n_rows):
        for col in range(n_cols):
            oy0 = row * tile_h
            oy1 = min(oy0 + tile_h, height)
            ox0 = col * tile_w
            ox1 = min(ox0 + tile_w, width)
            ry0 = max(0, oy0 - overlap_px)
            ry1 = min(height, oy1 + overlap_px)
            rx0 = max(0, ox0 - overlap_px)
            rx1 = min(width, ox1 + overlap_px)
            tiles.append(V7Tile(row=row, col=col, own=(oy0, oy1, ox0, ox1), read=(ry0, ry1, rx0, rx1)))
    return tiles


def build_size_tiles(height: int, width: int, tile_size: int, overlap_px: int) -> list[V7Tile]:
    """Build owned-region tiles from an approximate expanded tile size."""

    own_size = tile_size - 2 * overlap_px
    if own_size <= 0:
        raise ValueError("tile_size must be greater than 2 * overlap.")
    n_rows = -(-height // own_size)
    n_cols = -(-width // own_size)
    return build_v7_tiles(height, width, n_rows, n_cols, overlap_px)


def centroids_vectorised(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid arrays for labels 1..max_label using bincount."""

    n_labels = int(mask.max())
    if n_labels == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    height, width = mask.shape
    flat = mask.ravel()
    ys = np.repeat(np.arange(height, dtype=np.float32), width)
    xs = np.tile(np.arange(width, dtype=np.float32), height)
    counts = np.bincount(flat, minlength=n_labels + 2)
    sum_y = np.bincount(flat, weights=ys, minlength=n_labels + 2)
    sum_x = np.bincount(flat, weights=xs, minlength=n_labels + 2)
    valid = counts[1 : n_labels + 1] > 0
    cy = np.where(valid, sum_y[1 : n_labels + 1] / np.maximum(counts[1 : n_labels + 1], 1), -1)
    cx = np.where(valid, sum_x[1 : n_labels + 1] / np.maximum(counts[1 : n_labels + 1], 1), -1)
    return cy, cx


class V7NucleiSegmenter:
    """CellposeModel wrapper matching the Fusion_analysis v7 initialization style."""

    def __init__(self, device: Any, logger: logging.Logger) -> None:
        from cellpose import models as cp_models

        self.logger = logger
        self.model = cp_models.CellposeModel(device=device)
        self.logger.info("Cellpose backend initialized with CellposeModel(device=%s).", device)

    def segment_dapi_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """Segment one DAPI tile using v7-style uint16 to float32 normalization."""

        tile_f32 = tile_data.astype(np.float32) / 65535.0
        dapi = np.ascontiguousarray(tile_f32)
        result = self.model.eval(dapi, diameter=None, do_3D=False)
        masks = result[0] if isinstance(result, tuple) else result
        return np.asarray(masks, dtype=np.uint32)


def write_tiled_tiff(path: Path, arr: np.ndarray, dtype: np.dtype | type, compression: str = "lzw") -> None:
    """Write a 2-D tiled BigTIFF."""

    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        tif.write(
            np.asarray(arr, dtype=dtype),
            tile=(512, 512),
            compression=compression,
            photometric="minisblack",
            metadata=None,
        )


def write_macsiqview_binary_from_labels(label_mmap: np.ndarray, output_path: Path) -> None:
    """Create the MacsIQView binary mask using vectorized 8-neighbor border removal."""

    labels = np.asarray(label_mmap)
    binary = labels != 0
    border = np.zeros(labels.shape, dtype=bool)
    center = labels[1:-1, 1:-1]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = labels[1 + dy : labels.shape[0] - 1 + dy, 1 + dx : labels.shape[1] - 1 + dx]
            border[1:-1, 1:-1] |= (center != 0) & (neighbor != 0) & (neighbor != center)
    binary[border] = False
    write_tiled_tiff(output_path, binary.astype(np.uint8), np.uint8, compression="lzw")


def run_v7_like_segmentation(
    sample_id: str,
    input_tiff: Path,
    markers_csv: Path | None,
    label_output: Path,
    macsiqview_output: Path,
    segmenter: V7NucleiSegmenter,
    logger: logging.Logger,
    progress_callback: Callable[[int, int], None],
    n_rows: int | None,
    n_cols: int | None,
    tile_size: int | None,
    overlap_px: int,
    nuclear_channel: str | None,
) -> V7Result:
    """Run the v7-like fast core for one MACSima OME-TIFF sample."""

    label_output.parent.mkdir(parents=True, exist_ok=True)
    tile_times: list[float] = []
    with FastOmeSource(input_tiff, logger) as source:
        channel_index, channel_names, fallback = choose_nuclear_channel(
            input_tiff, markers_csv, nuclear_channel, source.channel_count
        )
        if fallback:
            logger.warning("No nuclear channel was detected for sample %s. Falling back to channel 0.", sample_id)
        logger.info("Selected nuclear channel index: %d", channel_index)
        if channel_index < len(channel_names):
            logger.info("Selected nuclear channel name: %s", channel_names[channel_index])
        if tile_size:
            tiles = build_size_tiles(source.height, source.width, tile_size, overlap_px)
            grid_desc = "tile_size=%d overlap=%d" % (tile_size, overlap_px)
        else:
            rows = n_rows or 2
            cols = n_cols or 3
            tiles = build_v7_tiles(source.height, source.width, rows, cols, overlap_px)
            grid_desc = "n_rows=%d n_cols=%d overlap=%d" % (rows, cols, overlap_px)
        logger.info("Fast v7-like grid: %s total_tiles=%d", grid_desc, len(tiles))
        progress_callback(0, len(tiles))
        mmap_path = Path(tempfile.gettempdir()) / f"{sample_id}_global_mask_{int(time.time())}.dat"
        global_mask = np.memmap(mmap_path, dtype="uint32", mode="w+", shape=(source.height, source.width))
        global_mask[:] = 0
        global_id_offset = 0
        try:
            for idx, tile in enumerate(tiles, start=1):
                tile_started = time.time()
                oy0, oy1, ox0, ox1 = tile.own
                ry0, ry1, rx0, rx1 = tile.read
                local_oy0 = oy0 - ry0
                local_oy1 = oy1 - ry0
                local_ox0 = ox0 - rx0
                local_ox1 = ox1 - rx0
                logger.info(
                    "Tile start | sample_id=%s tile=%d/%d row=%d col=%d own=(%d:%d,%d:%d) read=(%d:%d,%d:%d)",
                    sample_id,
                    idx,
                    len(tiles),
                    tile.row,
                    tile.col,
                    oy0,
                    oy1,
                    ox0,
                    ox1,
                    ry0,
                    ry1,
                    rx0,
                    rx1,
                )
                read_started = time.time()
                tile_data = source.read_channel(channel_index, ry0, ry1, rx0, rx1)
                logger.info(
                    "Tile read end | sample_id=%s tile=%d/%d elapsed_seconds=%.3f tile_shape=%s",
                    sample_id,
                    idx,
                    len(tiles),
                    time.time() - read_started,
                    tuple(int(v) for v in tile_data.shape),
                )
                eval_started = time.time()
                local_mask = segmenter.segment_dapi_tile(tile_data)
                logger.info(
                    "Cellpose eval end | sample_id=%s tile=%d/%d elapsed_seconds=%.3f tile_shape=%s local_labels=%d",
                    sample_id,
                    idx,
                    len(tiles),
                    time.time() - eval_started,
                    tuple(int(v) for v in local_mask.shape),
                    int(local_mask.max()),
                )
                del tile_data
                n_raw = int(local_mask.max())
                if n_raw > 0:
                    filter_started = time.time()
                    cy, cx = centroids_vectorised(local_mask)
                    keep_labels = [
                        label_idx + 1
                        for label_idx, (lcy, lcx) in enumerate(zip(cy, cx))
                        if local_oy0 <= lcy < local_oy1 and local_ox0 <= lcx < local_ox1
                    ]
                    logger.info(
                        "Centroid ownership filtering end | sample_id=%s tile=%d/%d elapsed_seconds=%.3f local_labels=%d kept_labels=%d",
                        sample_id,
                        idx,
                        len(tiles),
                        time.time() - filter_started,
                        n_raw,
                        len(keep_labels),
                    )
                    if keep_labels:
                        write_started = time.time()
                        lut = np.zeros(n_raw + 1, dtype=np.uint32)
                        for new_id, label in enumerate(keep_labels, start=1):
                            lut[label] = new_id + global_id_offset
                        remapped = lut[local_mask]
                        dst = global_mask[ry0:ry1, rx0:rx1]
                        np.copyto(dst, remapped, where=(remapped > 0))
                        global_id_offset += len(keep_labels)
                        logger.info(
                            "Global mask write end | sample_id=%s tile=%d/%d elapsed_seconds=%.3f kept_labels=%d total_labels=%d",
                            sample_id,
                            idx,
                            len(tiles),
                            time.time() - write_started,
                            len(keep_labels),
                            global_id_offset,
                        )
                        del lut, remapped
                    del cy, cx
                else:
                    logger.info(
                        "Centroid ownership filtering end | sample_id=%s tile=%d/%d elapsed_seconds=0.000 local_labels=0 kept_labels=0",
                        sample_id,
                        idx,
                        len(tiles),
                    )
                del local_mask
                empty_cuda_cache()
                gc.collect()
                elapsed = time.time() - tile_started
                tile_times.append(elapsed)
                logger.info("Tile done | sample_id=%s tile=%d/%d elapsed_seconds=%.3f", sample_id, idx, len(tiles), elapsed)
                progress_callback(idx, len(tiles))
            logger.info("All tiles completed | sample_id=%s total_tiles=%d total_labels=%d", sample_id, len(tiles), global_id_offset)
            flush_started = time.time()
            global_mask.flush()
            logger.info("Global memmap flush end | sample_id=%s elapsed_seconds=%.3f", sample_id, time.time() - flush_started)
            write_started = time.time()
            write_tiled_tiff(label_output, global_mask, np.uint32, compression="lzw")
            logger.info("Label TIFF write end | sample_id=%s elapsed_seconds=%.3f output_path=%s", sample_id, time.time() - write_started, label_output)
            macs_started = time.time()
            write_macsiqview_binary_from_labels(global_mask, macsiqview_output)
            logger.info(
                "MacsIQView binary write end | sample_id=%s elapsed_seconds=%.3f output_path=%s",
                sample_id,
                time.time() - macs_started,
                macsiqview_output,
            )
            return V7Result(
                total_tiles=len(tiles),
                total_labels=global_id_offset,
                tile_times=tile_times,
                label_output=label_output,
                macsiqview_output=macsiqview_output,
            )
        finally:
            del global_mask
            try:
                mmap_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temporary memmap file: %s", mmap_path)
