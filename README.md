# MACSima Batch Cellpose Nuclei Segmentation

This project runs a MACSima/MacsiqView staged-data batch pipeline:

1. Discover `*/background/*_backsub.ome.tif`.
2. Generate a Cellpose nuclei label mask.
3. Convert the label mask to a MacsIQView-compatible binary mask.

## Input Discovery

The batch script treats each first-level folder under `--root` as one staged result folder. Folder names may contain spaces and do not need to start with `2026`, `EXP`, or any other fixed prefix.

Valid inputs are discovered recursively with both conditions:

- The direct parent directory is named `background`.
- The filename ends with `_backsub.ome.tif`.

The script does not use `pseudochannel.tif`, existing old masks, or `IO_output_cp_masks.png`.

## Output Modes

Default mode is centralized output:

```text
--output-mode centralized-output
--central-output-dir /mnt/MACSimaDumpling/CRC_V2_ALL_MASKS
```

Each staged result folder receives its own subdirectory under the central output directory:

```text
<central_output_dir>/<result_folder_name>/
  rack-01-well-A01-roi-002-exp-2_cellpose_nuclei_mask.tif
  rack-01-well-A01-roi-002-exp-2_cellpose_nuclei_mask_MacsIQView.tif
```

Per-result-folder mode writes into a named folder inside each staged result folder:

```text
<result_folder>/Fusion/
  rack-01-well-A01-roi-002-exp-2_cellpose_nuclei_mask.tif
  rack-01-well-A01-roi-002-exp-2_cellpose_nuclei_mask_MacsIQView.tif
```

The folder name is controlled by `--result-output-folder-name`; default is `Fusion`.

## Summary

Summary files are written as:

```text
CRC_cellpose_nuclei_summary.csv
CRC_cellpose_nuclei_summary.json
```

In centralized mode, the global summary is written to `--central-output-dir`. In per-result-folder mode, the global summary is written to `--root`. Completed runs also write per-result summary files in each final output directory.

Summary statuses include `success`, `already_done`, `missing_background`, `missing_backsub_ome_tif`, `failed`, and `dry_run`.

## Terminal And Logs

By default, the terminal uses a clean single-line live status when running in an interactive terminal:

```text
⠋ running | EXP_..._staged | step=cellpose | tile=12/48 | start=14:32:08 | elapsed=02:15 | eta=06:40
```

Each task then prints one final line such as `✓ done`, `✗ failed`, or `- skipped`. Third-party stdout/stderr and Python warnings are redirected to the batch log by default.

Log files are written under:

```text
<central_output_dir>/logs/
```

for centralized output, or:

```text
<root>/logs/
```

for per-result-folder output.

Useful display flags:

```bash
--no-live-status
--quiet-third-party
--no-quiet-third-party
```

## Default CRC Run

```bash
cd /sda1/Nadya/20260514_cellpose

python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --gpu
```

## Centralized Output

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --output-mode centralized-output \
  --central-output-dir /mnt/MACSimaDumpling/CRC_V2_ALL_MASKS \
  --gpu
```

## Per-Result-Folder Output

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --output-mode per-result-folder \
  --result-output-folder-name Fusion \
  --gpu
```

## Dry Run

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --dry-run
```

## Skip List

Use `--skip-list` to skip selected first-level staged result folders by folder name. The file may be `.txt` or `.csv`, has one column with no header, ignores blank lines and lines beginning with `#`, and strips surrounding whitespace.

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --skip-list /mnt/MACSimaDumpling/CRC_V2/skip_samples.txt \
  --gpu
```

Skipped folders are not scanned internally and are recorded in the summary as `skipped_by_user` with `skipped_reason=in_skip_list`.

## Overwrite Existing Masks

The pipeline supports resume with four output states:

- No nuclei mask and no MacsIQView mask: run full Cellpose segmentation and conversion.
- Nuclei mask exists but MacsIQView mask is missing: run conversion-only mode and do not rerun Cellpose.
- Both masks exist: mark `already_done` and skip unless `--overwrite` is provided.
- MacsIQView mask exists but nuclei mask is missing: mark `inconsistent_state`, log a warning, and skip by default.

The summary records `pipeline_state`, `conversion_only`, `nuclei_mask_exists`, and `macsiqview_mask_exists`.

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --model-type nuclei \
  --gpu \
  --overwrite
```

## Channel Selection

The pipeline searches OME metadata and `markers_bs.csv` for nuclear channel names using keywords such as `DAPI`, `Hoechst`, `DNA`, `nucleus`, and `nuclei`.

By default, if no nuclear channel is detected, that task is marked `failed` and the batch continues. Override the channel when needed:

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --gpu \
  --nuclear-channel DAPI
```

For legacy behavior, allow channel 0 fallback:

```bash
python run_batch_cellpose_macsima.py \
  --root /mnt/MACSimaDumpling/CRC_V2 \
  --gpu \
  --allow-channel-zero-fallback
```

## Parallelism

- CPU mode uses `--workers`.
- GPU mode limits concurrent Cellpose tasks with `--gpu-workers`; the default is `1` to avoid CUDA OOM.
- Per-task tile processing remains sequential and uses overlap stitching from the existing v7-like core.
