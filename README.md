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

## Overwrite Existing Masks

The skip check uses the final MacsIQView mask. If `*_cellpose_nuclei_mask_MacsIQView.tif` exists, the task is marked `already_done` unless `--overwrite` is provided.

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
