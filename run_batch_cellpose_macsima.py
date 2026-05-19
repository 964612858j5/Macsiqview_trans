#!/usr/bin/env python3
"""Batch Cellpose nuclei segmentation for MACSima/MacsiqView backsub OME-TIFFs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from Separate_masks import convert_label_mask_to_macsiqview
from macsima_cellpose.gpu_utils import empty_cuda_cache, is_cuda_oom, require_gpu
from macsima_cellpose.logging_utils import setup_logging
from macsima_cellpose.progress import format_seconds
from macsima_cellpose.v7_core import V7NucleiSegmenter, run_v7_like_segmentation


DEFAULT_ROOT = Path("/mnt/MACSimaDumpling/CRC_V2")
DEFAULT_CENTRAL_OUTPUT_DIR = Path("/mnt/MACSimaDumpling/CRC_V2_ALL_MASKS")
DEFAULT_RESULT_OUTPUT_FOLDER_NAME = "Fusion"
SUMMARY_CSV = "CRC_cellpose_nuclei_summary.csv"
SUMMARY_JSON = "CRC_cellpose_nuclei_summary.json"
OOM_FALLBACKS = ((4096, 256), (3072, 256), (2048, 256), (1536, 192))
SUMMARY_FIELDS = (
    "result_folder",
    "result_folder_path",
    "background_path",
    "input_backsub_ome_tif",
    "output_mask",
    "output_mode",
    "output_root",
    "final_output_dir",
    "nuclei_mask_path",
    "macsiqview_mask_path",
    "status",
    "n_background_found",
    "n_backsub_found",
    "skipped_reason",
    "error_message",
    "start_time",
    "end_time",
    "elapsed_seconds",
)


@dataclass(frozen=True)
class Task:
    """One backsub OME-TIFF segmentation task."""

    result_folder: str
    result_folder_path: Path
    background_path: Path
    input_backsub_ome_tif: Path
    final_output_dir: Path
    nuclei_mask_path: Path
    macsiqview_mask_path: Path
    output_mode: str
    output_root: Path
    n_background_found: int
    n_backsub_found: int


@dataclass
class SummaryRow:
    """CSV/JSON row for one folder status or image task."""

    result_folder: str
    result_folder_path: str
    background_path: str
    input_backsub_ome_tif: str
    output_mask: str
    output_mode: str
    output_root: str
    final_output_dir: str
    nuclei_mask_path: str
    macsiqview_mask_path: str
    status: str
    n_background_found: int
    n_backsub_found: int
    skipped_reason: str = ""
    error_message: str = ""
    start_time: str = ""
    end_time: str = ""
    elapsed_seconds: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Cellpose nuclei segmentation for MACSima staged result folders."
    )
    parser.add_argument("--root", "--root-dir", dest="root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-mode",
        choices=["per-result-folder", "centralized-output"],
        default="centralized-output",
        help="Output organization mode. Default: centralized-output.",
    )
    parser.add_argument(
        "--central-output-dir",
        type=Path,
        default=DEFAULT_CENTRAL_OUTPUT_DIR,
        help="Central output directory used by --output-mode centralized-output.",
    )
    parser.add_argument(
        "--result-output-folder-name",
        default=DEFAULT_RESULT_OUTPUT_FOLDER_NAME,
        help="Per-result-folder output directory name. Default: Fusion.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan inputs and write summary without running Cellpose.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output masks.")
    parser.add_argument("--model-type", default="nuclei", help="Cellpose model type. Default: nuclei.")
    parser.add_argument("--diameter", type=float, default=30.0, help="Cellpose diameter. Use 0 for auto.")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--gpu", action="store_true", help="Run Cellpose on CUDA GPU.")
    parser.add_argument("--workers", type=int, default=2, help="CPU-mode worker count. Default: 2.")
    parser.add_argument("--gpu-workers", type=int, default=1, help="Maximum concurrent GPU Cellpose tasks. Default: 1.")
    parser.add_argument("--tile-size", type=int, default=None, help="Optional expanded tile size.")
    parser.add_argument("--n-rows", type=int, default=2, help="Tile grid rows when --tile-size is not set.")
    parser.add_argument("--n-cols", type=int, default=3, help="Tile grid columns when --tile-size is not set.")
    parser.add_argument("--overlap", type=int, default=256, help="Tile overlap halo in pixels.")
    parser.add_argument("--nuclear-channel", default=None, help="Nuclear channel override as zero-based index or channel name.")
    parser.add_argument("--allow-channel-zero-fallback", action="store_true", help="Use channel 0 if DAPI/nuclei cannot be detected.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--verbose-terminal", action="store_true", help="Also print detailed logs to terminal.")
    return parser


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def input_base_name(input_path: Path) -> str:
    """Return the acquisition base name without the *_backsub.ome.tif suffix."""

    suffix = "_backsub.ome.tif"
    if not input_path.name.endswith(suffix):
        raise ValueError(f"Input file does not end with {suffix}: {input_path}")
    return input_path.name[: -len(suffix)]


def resolve_output_dir(
    result_folder: Path,
    output_mode: str,
    central_output_dir: Path | None,
    result_output_folder_name: str,
) -> Path:
    """Resolve the single output directory for one staged result folder."""

    if output_mode == "per-result-folder":
        return result_folder / result_output_folder_name
    if output_mode == "centralized-output":
        if central_output_dir is None:
            raise ValueError("central_output_dir is required for centralized-output mode.")
        return central_output_dir / result_folder.name
    raise ValueError(f"Unsupported output mode: {output_mode}")


def resolve_output_paths(
    result_folder: Path,
    input_path: Path,
    output_mode: str,
    central_output_dir: Path | None,
    result_output_folder_name: str,
) -> tuple[Path, Path, Path, Path]:
    """Resolve final output dir, nuclei mask, MacsIQView mask, and output root."""

    final_output_dir = resolve_output_dir(result_folder, output_mode, central_output_dir, result_output_folder_name)
    base = input_base_name(input_path)
    nuclei_mask_path = final_output_dir / f"{base}_cellpose_nuclei_mask.tif"
    macsiqview_mask_path = final_output_dir / f"{base}_cellpose_nuclei_mask_MacsIQView.tif"
    output_root = final_output_dir if output_mode == "per-result-folder" else central_output_dir
    if output_root is None:
        raise ValueError("Could not resolve output root.")
    return final_output_dir, nuclei_mask_path, macsiqview_mask_path, output_root


def discover_macsima_backsub_images(
    root: Path,
    output_mode: str = "centralized-output",
    central_output_dir: Path | None = DEFAULT_CENTRAL_OUTPUT_DIR,
    result_output_folder_name: str = DEFAULT_RESULT_OUTPUT_FOLDER_NAME,
) -> list[Task]:
    """Discover valid background backsub OME-TIFF tasks under staged result folders."""

    tasks, _missing_rows, _total_result_folders = discover_macsima_backsub_images_with_status(
        root,
        output_mode,
        central_output_dir,
        result_output_folder_name,
    )
    return tasks


def discover_macsima_backsub_images_with_status(
    root: Path,
    output_mode: str,
    central_output_dir: Path | None,
    result_output_folder_name: str,
) -> tuple[list[Task], list[SummaryRow], int]:
    """Discover tasks plus per-folder missing-background and missing-input rows."""

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    tasks: list[Task] = []
    missing_rows: list[SummaryRow] = []
    result_folders = sorted(path for path in root.iterdir() if path.is_dir())

    for result_folder_path in result_folders:
        final_output_dir = resolve_output_dir(
            result_folder_path,
            output_mode,
            central_output_dir,
            result_output_folder_name,
        )
        output_root = final_output_dir if output_mode == "per-result-folder" else central_output_dir
        if output_root is None:
            raise ValueError("Could not resolve output root.")
        background_paths = sorted(
            path for path in result_folder_path.rglob("background") if path.is_dir() and path.name == "background"
        )
        n_background_found = len(background_paths)
        if n_background_found == 0:
            missing_rows.append(
                SummaryRow(
                    result_folder=result_folder_path.name,
                    result_folder_path=str(result_folder_path),
                    background_path="",
                    input_backsub_ome_tif="",
                    output_mask="",
                    output_mode=output_mode,
                    output_root=str(output_root),
                    final_output_dir=str(final_output_dir),
                    nuclei_mask_path="",
                    macsiqview_mask_path="",
                    status="missing_background",
                    n_background_found=0,
                    n_backsub_found=0,
                )
            )
            continue

        backsub_paths: list[Path] = []
        for background_path in background_paths:
            backsub_paths.extend(sorted(path for path in background_path.glob("*_backsub.ome.tif") if path.is_file()))
        n_backsub_found = len(backsub_paths)
        if n_backsub_found == 0:
            for background_path in background_paths:
                missing_rows.append(
                    SummaryRow(
                        result_folder=result_folder_path.name,
                        result_folder_path=str(result_folder_path),
                        background_path=str(background_path),
                        input_backsub_ome_tif="",
                        output_mask="",
                        output_mode=output_mode,
                        output_root=str(output_root),
                        final_output_dir=str(final_output_dir),
                        nuclei_mask_path="",
                        macsiqview_mask_path="",
                        status="missing_backsub_ome_tif",
                        n_background_found=n_background_found,
                        n_backsub_found=0,
                    )
                )
            continue

        for input_path in backsub_paths:
            task_final_output_dir, nuclei_mask_path, macsiqview_mask_path, task_output_root = resolve_output_paths(
                result_folder_path,
                input_path,
                output_mode,
                central_output_dir,
                result_output_folder_name,
            )
            tasks.append(
                Task(
                    result_folder=result_folder_path.name,
                    result_folder_path=result_folder_path,
                    background_path=input_path.parent,
                    input_backsub_ome_tif=input_path,
                    final_output_dir=task_final_output_dir,
                    nuclei_mask_path=nuclei_mask_path,
                    macsiqview_mask_path=macsiqview_mask_path,
                    output_mode=output_mode,
                    output_root=task_output_root,
                    n_background_found=n_background_found,
                    n_backsub_found=n_backsub_found,
                )
            )
    return tasks, missing_rows, len(result_folders)


def markers_csv_for_task(task: Task) -> Path | None:
    markers = task.background_path / "markers_bs.csv"
    return markers if markers.exists() else None


def fallback_sequence(tile_size: int | None, overlap: int) -> list[tuple[int | None, int]]:
    if tile_size is None:
        return [(None, overlap)]
    sequence: list[tuple[int | None, int]] = [(tile_size, overlap)]
    for fallback in OOM_FALLBACKS:
        if fallback[0] < tile_size and fallback not in sequence:
            sequence.append(fallback)
    return sequence


def task_to_row(task: Task, status: str, skipped_reason: str = "", error_message: str = "") -> SummaryRow:
    return SummaryRow(
        result_folder=task.result_folder,
        result_folder_path=str(task.result_folder_path),
        background_path=str(task.background_path),
        input_backsub_ome_tif=str(task.input_backsub_ome_tif),
        output_mask=str(task.macsiqview_mask_path),
        output_mode=task.output_mode,
        output_root=str(task.output_root),
        final_output_dir=str(task.final_output_dir),
        nuclei_mask_path=str(task.nuclei_mask_path),
        macsiqview_mask_path=str(task.macsiqview_mask_path),
        status=status,
        n_background_found=task.n_background_found,
        n_backsub_found=task.n_backsub_found,
        skipped_reason=skipped_reason,
        error_message=error_message,
    )


def run_one_task(task: Task, args: argparse.Namespace, logger: logging.Logger, device: Any) -> SummaryRow:
    row = task_to_row(task, "failed")
    row.start_time = iso_now()
    started = time.time()
    logger.info("Starting result folder: %s", task.result_folder)
    logger.info("Background folders found: %d", task.n_background_found)
    logger.info("Backsub OME-TIFFs found: %d", task.n_backsub_found)
    logger.info("Input backsub OME-TIFF: %s", task.input_backsub_ome_tif)
    logger.info("Final output directory: %s", task.final_output_dir)
    logger.info("Nuclei mask output: %s", task.nuclei_mask_path)
    logger.info("MacsIQView mask output: %s", task.macsiqview_mask_path)

    try:
        if task.macsiqview_mask_path.exists() and not args.overwrite:
            row.status = "already_done"
            row.skipped_reason = "macsiqview_mask_exists"
            logger.info("Skipping existing MacsIQView output mask: %s", task.macsiqview_mask_path)
            return row

        task.final_output_dir.mkdir(parents=True, exist_ok=True)
        segmenter = V7NucleiSegmenter(
            device=device,
            logger=logger,
            model_type=args.model_type,
            diameter=None if args.diameter == 0 else args.diameter,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            gpu=args.gpu,
        )
        last_error: BaseException | None = None
        for attempt_tile_size, attempt_overlap in fallback_sequence(args.tile_size, args.overlap):
            try:
                run_v7_like_segmentation(
                    sample_id=task.input_backsub_ome_tif.stem,
                    input_tiff=task.input_backsub_ome_tif,
                    markers_csv=markers_csv_for_task(task),
                    label_output=task.nuclei_mask_path,
                    macsiqview_output=None,
                    segmenter=segmenter,
                    logger=logger,
                    progress_callback=lambda _done, _total: None,
                    n_rows=args.n_rows,
                    n_cols=args.n_cols,
                    tile_size=attempt_tile_size,
                    overlap_px=attempt_overlap,
                    nuclear_channel=args.nuclear_channel,
                    require_detected_nuclear_channel=not args.allow_channel_zero_fallback,
                )
                convert_label_mask_to_macsiqview(task.nuclei_mask_path, task.macsiqview_mask_path)
                row.status = "success"
                logger.info("Task succeeded: nuclei=%s macsiqview=%s", task.nuclei_mask_path, task.macsiqview_mask_path)
                return row
            except BaseException as exc:
                last_error = exc
                logger.error(
                    "Task failed with tile_size=%s overlap=%d input=%s: %s",
                    attempt_tile_size,
                    attempt_overlap,
                    task.input_backsub_ome_tif,
                    exc,
                )
                logger.debug("Traceback:\n%s", traceback.format_exc())
                empty_cuda_cache()
                gc.collect()
                if not is_cuda_oom(exc):
                    break
                logger.warning("CUDA OOM detected. Trying smaller tile settings if available.")
        if last_error:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError("Task failed for an unknown reason.")
    except Exception as exc:
        row.status = "failed"
        row.error_message = str(exc)
        logger.error("Task marked failed: input=%s error=%s", task.input_backsub_ome_tif, exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return row
    finally:
        row.end_time = iso_now()
        row.elapsed_seconds = round(time.time() - started, 3)
        empty_cuda_cache()
        gc.collect()


def write_summary(root: Path, rows: list[SummaryRow]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / SUMMARY_CSV
    json_path = root / SUMMARY_JSON
    serialised = [asdict(row) for row in rows]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(serialised)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(serialised, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return csv_path, json_path


def write_per_result_summaries(rows: list[SummaryRow]) -> None:
    grouped: dict[str, list[SummaryRow]] = {}
    for row in rows:
        if not row.final_output_dir:
            continue
        grouped.setdefault(row.final_output_dir, []).append(row)
    for output_dir, output_rows in grouped.items():
        write_summary(Path(output_dir), output_rows)


def summary_root_for_run(args: argparse.Namespace) -> Path:
    if args.output_mode == "centralized-output":
        return args.central_output_dir
    return args.root


def print_completion(total_result_folders: int, total_tasks: int, rows: list[SummaryRow], elapsed: float) -> None:
    counts = Counter(row.status for row in rows)
    print("Batch completion summary")
    print(f"total result folders: {total_result_folders}")
    print(f"total tasks: {total_tasks}")
    print(f"success: {counts.get('success', 0)}")
    print(f"already_done: {counts.get('already_done', 0)}")
    print(f"missing_background: {counts.get('missing_background', 0)}")
    print(f"missing_backsub_ome_tif: {counts.get('missing_backsub_ome_tif', 0)}")
    print(f"failed: {counts.get('failed', 0)}")
    print(f"dry_run: {counts.get('dry_run', 0)}")
    print(f"total runtime: {format_seconds(elapsed)}")


def run(args: argparse.Namespace) -> int:
    project_dir = Path(__file__).resolve().parent
    logger, log_path = setup_logging(project_dir / "logs", args.log_level, verbose_terminal=args.verbose_terminal)
    started = time.time()
    logger.info("Log file: %s", log_path)
    logger.info("Root: %s", args.root)
    logger.info("Output mode: %s", args.output_mode)
    logger.info("Central output dir: %s", args.central_output_dir)
    logger.info("Result output folder name: %s", args.result_output_folder_name)
    logger.info("Model type: %s", args.model_type)
    logger.info("Cellpose params: diameter=%s cellprob_threshold=%.3f flow_threshold=%.3f", args.diameter, args.cellprob_threshold, args.flow_threshold)
    logger.info("Execution: gpu=%s workers=%d gpu_workers=%d dry_run=%s overwrite=%s", args.gpu, args.workers, args.gpu_workers, args.dry_run, args.overwrite)

    tasks, missing_rows, total_result_folders = discover_macsima_backsub_images_with_status(
        args.root,
        args.output_mode,
        args.central_output_dir,
        args.result_output_folder_name,
    )
    logger.info("Discovered result folders: %d", total_result_folders)
    logger.info("Discovered Cellpose tasks: %d", len(tasks))
    for row in missing_rows:
        logger.warning(
            "Discovery status=%s result_folder=%s backgrounds=%d backsubs=%d",
            row.status,
            row.result_folder_path,
            row.n_background_found,
            row.n_backsub_found,
        )

    rows: list[SummaryRow] = list(missing_rows)
    if args.dry_run:
        for task in tasks:
            row = task_to_row(task, "dry_run")
            row.start_time = iso_now()
            row.end_time = row.start_time
            rows.append(row)
            logger.info(
                "Dry run task input=%s nuclei=%s macsiqview=%s",
                task.input_backsub_ome_tif,
                task.nuclei_mask_path,
                task.macsiqview_mask_path,
            )
        csv_path, json_path = write_summary(summary_root_for_run(args), rows)
        logger.info("Summary CSV: %s", csv_path)
        logger.info("Summary JSON: %s", json_path)
        print_completion(total_result_folders, len(tasks), rows, time.time() - started)
        print(f"summary csv: {csv_path}")
        print(f"summary json: {json_path}")
        return 0

    device: Any = None
    if args.gpu:
        try:
            require_gpu(logger)
            import torch

            device = torch.device("cuda")
        except Exception as exc:
            logger.error("GPU validation failed: %s", exc)
            logger.debug("Traceback:\n%s", traceback.format_exc())
            for task in tasks:
                row = task_to_row(task, "failed", error_message=str(exc))
                row.start_time = iso_now()
                row.end_time = row.start_time
                rows.append(row)
            csv_path, json_path = write_summary(summary_root_for_run(args), rows)
            print_completion(total_result_folders, len(tasks), rows, time.time() - started)
            print(f"summary csv: {csv_path}")
            print(f"summary json: {json_path}")
            return 1

    worker_count = max(1, int(args.gpu_workers if args.gpu else args.workers))
    if args.gpu:
        worker_count = min(worker_count, max(1, int(args.workers)))
    logger.info("Effective segmentation workers: %d", worker_count)

    if worker_count == 1:
        for task in tasks:
            rows.append(run_one_task(task, args, logger, device))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {executor.submit(run_one_task, task, args, logger, device): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    logger.error("Unhandled task failure input=%s error=%s", task.input_backsub_ome_tif, exc)
                    logger.debug("Traceback:\n%s", traceback.format_exc())
                    row = task_to_row(task, "failed", error_message=str(exc))
                    row.start_time = iso_now()
                    row.end_time = row.start_time
                    rows.append(row)

    csv_path, json_path = write_summary(summary_root_for_run(args), rows)
    write_per_result_summaries(rows)
    elapsed = time.time() - started
    logger.info("Summary CSV: %s", csv_path)
    logger.info("Summary JSON: %s", json_path)
    print_completion(total_result_folders, len(tasks), rows, elapsed)
    print(f"summary csv: {csv_path}")
    print(f"summary json: {json_path}")
    return 1 if any(row.status == "failed" for row in rows) else 0


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
