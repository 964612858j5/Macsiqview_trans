#!/usr/bin/env python3
"""Run GPU-only batch Cellpose nuclei segmentation for MACSima datasets."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from macsima_cellpose.discovery import Sample, discover_samples
from macsima_cellpose.gpu_utils import empty_cuda_cache, is_cuda_oom, require_gpu
from macsima_cellpose.logging_utils import setup_logging
from macsima_cellpose.macsiqview import write_macsiqview_mask
from macsima_cellpose.ome_reader import OmeTileReader, choose_nuclear_channel
from macsima_cellpose.progress import ProgressUI, SampleStatus, format_seconds
from macsima_cellpose.segmentation import NucleiSegmenter
from macsima_cellpose.stitching import paste_owned_objects, sequential_relabel
from macsima_cellpose.tiling import compute_grid, iter_tiles


DEFAULT_ROOT = Path("/mnt/MACSimaDumpling/NTrautwein_Sarcoma_staged")
DEFAULT_OUTPUT = DEFAULT_ROOT / "segmentation"
OOM_FALLBACKS = ((3072, 256), (2048, 256), (1536, 192))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch GPU Cellpose nuclei segmentation for MACSima OME-TIFF datasets.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT, help="Root directory containing 2026* sample folders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for segmentation outputs.")
    parser.add_argument("--only-sample", default=None, help="Process only one sample ID, for example R1_B1_ROI1.")
    parser.add_argument("--dry-run", action="store_true", help="Discover samples and print planned processing without running Cellpose.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output masks.")
    parser.add_argument("--gpu", action="store_true", help="Require CUDA GPU execution. This tool never falls back to CPU.")
    parser.add_argument("--tile-size", type=int, default=3072, help="Expanded tile size used for the first attempt.")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap halo in pixels used for the first attempt.")
    parser.add_argument("--nuclear-channel", default=None, help="Nuclear channel override as zero-based index or channel name.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.")
    return parser


def save_label_mask(path: Path, mask: np.ndarray) -> None:
    """Write full label mask as tiled TIFF."""

    tifffile.imwrite(path, mask.astype(np.uint32, copy=False), bigtiff=True, photometric="minisblack", compression="zlib")


def fallback_sequence(tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """Build OOM retry sequence starting with CLI settings."""

    sequence = [(tile_size, overlap)]
    for fallback in OOM_FALLBACKS:
        if fallback[0] >= tile_size:
            continue
        if fallback not in sequence:
            sequence.append(fallback)
    return sequence


def process_sample(
    sample: Sample,
    segmenter: NucleiSegmenter,
    tile_size: int,
    overlap: int,
    overwrite: bool,
    nuclear_channel: str | None,
    logger: logging.Logger,
) -> str:
    """Process one sample with OOM fallback."""

    if sample.label_output.exists() and sample.macsiqview_output.exists() and not overwrite:
        logger.info("Skipping sample %s because outputs already exist.", sample.sample_id)
        return "skipped"
    sample.label_output.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None
    for attempt_tile_size, attempt_overlap in fallback_sequence(tile_size, overlap):
        try:
            logger.info("Starting sample %s", sample.sample_id)
            logger.info("Input TIFF: %s", sample.input_tiff)
            logger.info("Output label mask: %s", sample.label_output)
            logger.info("Output MacsIQView mask: %s", sample.macsiqview_output)
            logger.info("Tile settings: tile_size=%d overlap=%d batch_size=1", attempt_tile_size, attempt_overlap)
            with OmeTileReader(sample.input_tiff, logger=logger) as reader:
                channel_index, channel_names, fallback = choose_nuclear_channel(
                    sample.input_tiff, sample.markers_csv, nuclear_channel, reader.channel_count
                )
                if fallback:
                    logger.warning("No nuclear channel was detected for sample %s. Falling back to channel 0.", sample.sample_id)
                logger.info("Selected nuclear channel index: %d", channel_index)
                if channel_index < len(channel_names):
                    logger.info("Selected nuclear channel name: %s", channel_names[channel_index])
                height, width = reader.height, reader.width
                n_rows, n_cols = compute_grid(height, width, attempt_tile_size, attempt_overlap)
                logger.info("Image size: height=%d width=%d", height, width)
                logger.info("Computed grid: n_rows=%d n_cols=%d", n_rows, n_cols)
                global_mask = np.zeros((height, width), dtype=np.uint32)
                next_label = 1
                tiles = iter_tiles(height, width, attempt_tile_size, attempt_overlap)
                for idx, tile in enumerate(tiles, start=1):
                    logger.info(
                        "Tile %d/%d row=%d col=%d own=(%d:%d,%d:%d) read=(%d:%d,%d:%d)",
                        idx,
                        len(tiles),
                        tile.row,
                        tile.col,
                        tile.own_y0,
                        tile.own_y1,
                        tile.own_x0,
                        tile.own_x1,
                        tile.read_y0,
                        tile.read_y1,
                        tile.read_x0,
                        tile.read_x1,
                    )
                    tile_image = reader.read_channel_tile(channel_index, tile.read_y0, tile.read_y1, tile.read_x0, tile.read_x1)
                    tile_mask = segmenter.segment_tile(tile_image)
                    next_label = paste_owned_objects(global_mask, tile_mask, tile, next_label)
                    del tile_image, tile_mask
                    empty_cuda_cache()
                    gc.collect()
                final_mask = sequential_relabel(global_mask)
                save_label_mask(sample.label_output, final_mask)
                write_macsiqview_mask(final_mask, sample.macsiqview_output)
                logger.info("Completed sample %s with %d labels.", sample.sample_id, int(final_mask.max()))
                del global_mask, final_mask
                gc.collect()
                empty_cuda_cache()
                return "done"
        except BaseException as exc:
            last_error = exc
            logger.error("Sample %s failed with tile_size=%d overlap=%d: %s", sample.sample_id, attempt_tile_size, attempt_overlap, exc)
            logger.debug("Traceback for sample %s:\n%s", sample.sample_id, traceback.format_exc())
            empty_cuda_cache()
            gc.collect()
            if not is_cuda_oom(exc):
                break
            logger.warning("CUDA OOM detected for sample %s. Trying smaller tile settings if available.", sample.sample_id)
    if last_error:
        raise RuntimeError(f"Sample {sample.sample_id} failed after retries: {last_error}") from last_error
    raise RuntimeError(f"Sample {sample.sample_id} failed for an unknown reason.")


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(__file__).resolve().parent
    logger, log_path = setup_logging(project_dir / "logs", args.log_level)
    logger.info("Log file: %s", log_path)
    logger.info("root_dir: %s", args.root_dir)
    logger.info("output_dir: %s", args.output_dir)
    logger.info("Cellpose model: nuclei")
    logger.info("Initial tile settings: tile_size=%d overlap=%d batch_size=1", args.tile_size, args.overlap)
    samples = discover_samples(args.root_dir, args.output_dir, args.only_sample)
    statuses = [SampleStatus(sample.sample_id, "pending", sample.label_output) for sample in samples]
    ui = ProgressUI()
    if args.dry_run:
        dry_rows = []
        for sample in samples:
            logger.info("Dry run sample %s input=%s label_output=%s macsiqview_output=%s", sample.sample_id, sample.input_tiff, sample.label_output, sample.macsiqview_output)
            dry_rows.append((sample.sample_id, sample.input_tiff, sample.label_output, sample.macsiqview_output))
        ui.print_dry_run(dry_rows)
        return 0
    if not args.gpu:
        logger.warning("The --gpu flag was not provided. GPU execution is still required and CPU fallback is disabled.")
    started = time.time()
    try:
        gpu_info = require_gpu(logger)
    except Exception as exc:
        logger.error("GPU validation failed: %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        for status in statuses:
            status.status = "error"
            status.started_at = datetime.now()
            status.finished_at = status.started_at
            status.error = str(exc)
        ui.print_status(statuses, time.time() - started)
        logger.info("Batch completion summary")
        logger.info("total samples: %d", len(samples))
        logger.info("successful: 0")
        logger.info("failed: %d", len(samples))
        logger.info("skipped: 0")
        logger.info("total runtime: %s", format_seconds(time.time() - started))
        logger.info("output directory: %s", args.output_dir)
        print("Batch completion summary")
        print(f"total samples: {len(samples)}")
        print("successful: 0")
        print(f"failed: {len(samples)}")
        print("skipped: 0")
        print(f"total runtime: {format_seconds(time.time() - started)}")
        print(f"output directory: {args.output_dir}")
        return 1
    logger.info("GPU requirement satisfied: %s", gpu_info)
    segmenter = NucleiSegmenter(logger)
    successful = 0
    failed = 0
    skipped = 0
    for sample, status in zip(samples, statuses):
        status.status = "running"
        status.started_at = datetime.now()
        ui.print_status(statuses, time.time() - started)
        try:
            with ui.running_spinner(sample.sample_id):
                result = process_sample(sample, segmenter, args.tile_size, args.overlap, args.overwrite, args.nuclear_channel, logger)
            status.status = result
            if result == "done":
                successful += 1
            elif result == "skipped":
                skipped += 1
        except Exception as exc:
            failed += 1
            status.status = "error"
            status.error = str(exc)
            logger.error("Sample %s marked as error: %s", sample.sample_id, exc)
            logger.debug("Traceback:\n%s", traceback.format_exc())
        finally:
            status.finished_at = datetime.now()
            ui.print_status(statuses, time.time() - started)
    total_runtime = time.time() - started
    logger.info("Batch completion summary")
    logger.info("total samples: %d", len(samples))
    logger.info("successful: %d", successful)
    logger.info("failed: %d", failed)
    logger.info("skipped: %d", skipped)
    logger.info("total runtime: %s", format_seconds(total_runtime))
    logger.info("output directory: %s", args.output_dir)
    print("Batch completion summary")
    print(f"total samples: {len(samples)}")
    print(f"successful: {successful}")
    print(f"failed: {failed}")
    print(f"skipped: {skipped}")
    print(f"total runtime: {format_seconds(total_runtime)}")
    print(f"output directory: {args.output_dir}")
    return 1 if failed else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
