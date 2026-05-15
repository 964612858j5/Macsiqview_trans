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

from macsima_cellpose.discovery import Sample, discover_samples
from macsima_cellpose.gpu_utils import empty_cuda_cache, is_cuda_oom, require_gpu
from macsima_cellpose.logging_utils import setup_logging
from macsima_cellpose.progress import ProgressContext, ProgressUI, SampleStatus, format_seconds
from macsima_cellpose.v7_core import V7NucleiSegmenter, run_v7_like_segmentation


DEFAULT_ROOT = Path("/mnt/MACSimaDumpling/NTrautwein_Sarcoma_staged")
DEFAULT_OUTPUT = DEFAULT_ROOT / "segmentation"
OOM_FALLBACKS = ((4096, 256), (3072, 256), (2048, 256), (1536, 192))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch GPU Cellpose nuclei segmentation for MACSima OME-TIFF datasets.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT, help="Root directory containing MACSima sample folders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for segmentation outputs.")
    parser.add_argument("--only-sample", default=None, help="Process only one sample ID, for example R1_B1_ROI1.")
    parser.add_argument("--dry-run", action="store_true", help="Discover samples and print planned processing without running Cellpose.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output masks.")
    parser.add_argument("--gpu", action="store_true", help="Require CUDA GPU execution. This tool never falls back to CPU.")
    parser.add_argument("--tile-size", type=int, default=None, help="Optional expanded tile size. Default uses the v7-compatible row and column grid.")
    parser.add_argument("--n-rows", type=int, default=2, help="V7-compatible tile grid rows. Default is 2.")
    parser.add_argument("--n-cols", type=int, default=3, help="V7-compatible tile grid columns. Default is 3.")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap halo in pixels.")
    parser.add_argument("--nuclear-channel", default=None, help="Nuclear channel override as zero-based index or channel name.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level for --verbose-terminal.")
    parser.add_argument("--verbose-terminal", action="store_true", help="Print detailed log messages to the terminal.")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the live dashboard and print simple status lines.")
    return parser


def fallback_sequence(tile_size: int | None, overlap: int) -> list[tuple[int | None, int]]:
    """Build OOM retry sequence while preserving v7 grid mode by default."""

    if tile_size is None:
        return [(None, overlap)]
    sequence: list[tuple[int | None, int]] = [(tile_size, overlap)]
    for fallback in OOM_FALLBACKS:
        if fallback[0] >= tile_size:
            continue
        if fallback not in sequence:
            sequence.append(fallback)
    return sequence


def process_sample(
    sample: Sample,
    segmenter: V7NucleiSegmenter,
    tile_size: int | None,
    overlap: int,
    n_rows: int,
    n_cols: int,
    overwrite: bool,
    nuclear_channel: str | None,
    logger: logging.Logger,
    status: SampleStatus,
    ui: ProgressUI,
) -> str:
    """Process one sample with the v7-like fast core."""

    if sample.label_output.exists() and sample.macsiqview_output.exists() and not overwrite:
        logger.info("Skipping sample %s because outputs already exist.", sample.sample_id)
        status.tile_current = 0
        status.tile_total = 0
        ui.refresh()
        return "skipped"
    sample.label_output.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None
    for attempt_tile_size, attempt_overlap in fallback_sequence(tile_size, overlap):
        try:
            logger.info("Starting sample %s", sample.sample_id)
            logger.info("Input TIFF: %s", sample.input_tiff)
            logger.info("Output label mask: %s", sample.label_output)
            logger.info("Output MacsIQView mask: %s", sample.macsiqview_output)
            if attempt_tile_size:
                logger.info("Tile settings: tile_size=%d overlap=%d batch_size=1", attempt_tile_size, attempt_overlap)
            else:
                logger.info("Tile settings: n_rows=%d n_cols=%d overlap=%d batch_size=1", n_rows, n_cols, attempt_overlap)

            def update_progress(done_tiles: int, total_tiles: int) -> None:
                status.tile_current = done_tiles
                status.tile_total = total_tiles
                ui.refresh()

            sample_started = time.time()
            result = run_v7_like_segmentation(
                sample_id=sample.sample_id,
                input_tiff=sample.input_tiff,
                markers_csv=sample.markers_csv,
                label_output=sample.label_output,
                macsiqview_output=sample.macsiqview_output,
                segmenter=segmenter,
                logger=logger,
                progress_callback=update_progress,
                n_rows=n_rows,
                n_cols=n_cols,
                tile_size=attempt_tile_size,
                overlap_px=attempt_overlap,
                nuclear_channel=nuclear_channel,
            )
            logger.info("Completed sample %s with %d labels.", sample.sample_id, result.total_labels)
            logger.info("Sample total tiles: %d", result.total_tiles)
            for tile_idx, tile_seconds in enumerate(result.tile_times, start=1):
                logger.info(
                    "Sample tile runtime | sample_id=%s tile=%d/%d elapsed_seconds=%.3f",
                    sample.sample_id,
                    tile_idx,
                    result.total_tiles,
                    tile_seconds,
                )
            logger.info("Sample total runtime | sample_id=%s elapsed_seconds=%.3f", sample.sample_id, time.time() - sample_started)
            return "done"
        except BaseException as exc:
            last_error = exc
            logger.error("Sample %s failed with tile_size=%s overlap=%d: %s", sample.sample_id, attempt_tile_size, attempt_overlap, exc)
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
    logger, log_path = setup_logging(project_dir / "logs", args.log_level, verbose_terminal=args.verbose_terminal)
    logger.info("Log file: %s", log_path)
    logger.info("root_dir: %s", args.root_dir)
    logger.info("output_dir: %s", args.output_dir)
    logger.info("Cellpose model: nuclei")
    if args.tile_size:
        logger.info("Initial tile settings: tile_size=%d overlap=%d batch_size=1", args.tile_size, args.overlap)
    else:
        logger.info("Initial tile settings: n_rows=%d n_cols=%d overlap=%d batch_size=1", args.n_rows, args.n_cols, args.overlap)
    samples = discover_samples(args.root_dir, args.output_dir, args.only_sample)
    statuses = [SampleStatus(sample.sample_id, "pending", sample.label_output) for sample in samples]
    use_dashboard = not args.no_dashboard and not args.verbose_terminal
    ui = ProgressUI(use_dashboard=use_dashboard)
    if args.dry_run:
        dry_rows = []
        for sample in samples:
            logger.info(
                "Dry run sample %s input=%s label_output=%s macsiqview_output=%s",
                sample.sample_id,
                sample.input_tiff,
                sample.label_output,
                sample.macsiqview_output,
            )
            dry_rows.append((sample.sample_id, sample.dataset_dir, sample.input_tiff, sample.label_output, sample.macsiqview_output))
        ui.print_dry_run(dry_rows)
        return 0
    if not args.gpu:
        logger.warning("The --gpu flag was not provided. GPU execution is still required and CPU fallback is disabled.")
    started = time.time()
    try:
        require_gpu(logger)
        import torch

        device = torch.device("cuda")
    except Exception as exc:
        logger.error("GPU validation failed: %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        for status in statuses:
            status.status = "error"
            status.started_at = datetime.now()
            status.finished_at = status.started_at
            status.error = str(exc)
        with ProgressContext(ui, statuses):
            ui.refresh()
        if args.no_dashboard or args.verbose_terminal or not ui.rich:
            for status in statuses:
                ui.print_line(ui.sample_line(status))
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

    segmenter = V7NucleiSegmenter(device, logger)
    successful = 0
    failed = 0
    skipped = 0
    with ProgressContext(ui, statuses):
        for sample, status in zip(samples, statuses):
            status.status = "running"
            status.started_at = datetime.now()
            status.finished_at = None
            status.tile_current = 0
            status.tile_total = 0
            ui.refresh()
            if args.no_dashboard or args.verbose_terminal or not ui.rich:
                ui.print_line(ui.sample_line(status))
            try:
                result = process_sample(
                    sample,
                    segmenter,
                    args.tile_size,
                    args.overlap,
                    args.n_rows,
                    args.n_cols,
                    args.overwrite,
                    args.nuclear_channel,
                    logger,
                    status,
                    ui,
                )
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
                ui.refresh()
                if args.no_dashboard or args.verbose_terminal or not ui.rich:
                    ui.print_line(ui.sample_line(status))

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
        if getattr(args, "verbose_terminal", False):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
