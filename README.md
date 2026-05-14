# MACSima Batch Cellpose Nuclei Segmentation

This project provides a production-oriented GPU batch segmentation pipeline for MACSima spatial proteomics OME-TIFF datasets. It runs Cellpose with the nuclei model only, processes samples serially, uses overlap tile inference, and stitches objects by centroid ownership.

## Behavior

- Discovers dataset folders under a root directory when their names start with `2026`.
- Finds input files matching `1/rack-*/background/*_backsub.ome.tif`.
- Parses sample IDs such as `R1_B1_ROI1` from dataset folder names.
- Writes outputs to the configured segmentation directory.
- Requires CUDA GPU execution and never falls back to CPU.
- Uses Cellpose nuclei with `diameter=None` and default Cellpose parameters.
- Processes overlapped tiles sequentially to reduce CUDA memory pressure.
- Keeps complete tile objects only when their global centroid falls inside the non-overlap owned region.
- Writes a full uint32 label TIFF and a MacsIQView-compatible uint8 binary TIFF.
- Continues batch processing after per-sample failures.

## Default Command

```bash
python run_batch_cellpose_macsima.py \
  --root-dir /mnt/MACSimaDumpling/NTrautwein_Sarcoma_staged \
  --output-dir /mnt/MACSimaDumpling/NTrautwein_Sarcoma_staged/segmentation \
  --overlap 256 \
  --gpu \
  --overwrite
```

By default, the terminal shows a clean one-line-per-sample status dashboard. Detailed tile, channel, timing, and traceback information is still written to the timestamped log file.
The default segmentation grid is the v7-compatible `--n-rows 2 --n-cols 3` layout, giving 6 tiles for each sample unless `--tile-size` is provided.

## Single Sample

```bash
python run_batch_cellpose_macsima.py \
  --root-dir /mnt/MACSimaDumpling/NTrautwein_Sarcoma_staged \
  --only-sample R1_B1_ROI1 \
  --overlap 256 \
  --gpu \
  --overwrite
```

## Dry Run

```bash
python run_batch_cellpose_macsima.py \
  --root-dir /mnt/MACSimaDumpling/NTrautwein_Sarcoma_staged \
  --dry-run
```

Dry run mode discovers samples and prints input and output paths without running Cellpose.

## Outputs

For each sample, the output directory receives:

- `{sample_id}.tiff`: full nuclei label mask.
- `{sample_id}_MacsIQView.tif`: MacsIQView-compatible binary mask.

## Tiling and Stitching

The default v7-compatible tile settings are:

- `n_rows = 2`
- `n_cols = 3`
- `overlap_px = 256`
- `batch_size = 1`

In default mode, the owned region grid follows Fusion_analysis v7:

```text
tile_h = ceil(height / n_rows)
tile_w = ceil(width / n_cols)
```

Each owned region is expanded by the overlap halo for inference context. After Cellpose predicts labels on the expanded tile, each object's centroid is checked in local tile coordinates. Complete objects are retained only if their centroid falls inside the tile owned region. Retained objects are remapped with a global label offset and streamed into a `uint32` memmap-backed global mask.

If `--tile-size` is provided, the CLI computes a grid from `tile_size - 2 * overlap_px`.

## CUDA OOM Fallback

If CUDA out-of-memory occurs for a sample, the same sample is retried with:

1. `tile_size=2048`, `overlap=256`
2. `tile_size=1536`, `overlap=192`

If all attempts fail, the sample is marked as error and the batch continues.

## Channel Selection

The pipeline automatically searches OME metadata and `markers_bs.csv` for nuclear channels using these keywords:

- `DAPI`
- `Hoechst`
- `DNA`
- `nucleus`
- `nuclei`

If no nuclear channel is detected, channel 0 is used and a warning is logged. A channel can be forced with:

```bash
--nuclear-channel 0
--nuclear-channel DAPI
```

## Logs

Every run writes a timestamped log file under `logs/`:

```text
logs/batch_cellpose_YYYYMMDD_HHMMSS.log
```

Logs include paths, GPU information, Cellpose model settings, tile settings, retries, CUDA OOM events, timings, output paths, and traceback details.

## Terminal Output

Default terminal output uses a dynamic dashboard when `rich` is available. Each sample occupies one line with status, sample ID, completed tiles, total tiles, elapsed time, and estimated remaining time.

Use detailed terminal logs only when needed:

```bash
python run_batch_cellpose_macsima.py --verbose-terminal
```

Disable the live dashboard and print simple line-by-line status:

```bash
python run_batch_cellpose_macsima.py --no-dashboard
```
