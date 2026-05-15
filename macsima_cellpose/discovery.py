"""Dataset discovery and sample identity parsing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path


SAMPLE_ID_RE = re.compile(r"(R\d+_[A-Za-z0-9]+_ROI\d+)")
RACK_ID_RE = re.compile(r"\bwell-([A-Za-z0-9]+).*?\broi-?(\d+)", re.IGNORECASE)
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")
LOGGER = logging.getLogger("macsima_cellpose")


@dataclass(frozen=True)
class Sample:
    """Input and output paths for one MACSima sample."""

    sample_id: str
    dataset_dir: Path
    input_tiff: Path
    markers_csv: Path | None
    label_output: Path
    macsiqview_output: Path


def parse_sample_id(dataset_dir: Path) -> str:
    """Parse sample IDs such as R1_B1_ROI1 from a dataset folder name."""

    match = SAMPLE_ID_RE.search(dataset_dir.name)
    if not match:
        raise ValueError(f"Could not parse sample ID from dataset folder: {dataset_dir.name}")
    return match.group(1)


def parse_rack_sample_id(rack_dir: Path) -> str | None:
    """Parse fallback sample IDs such as B01_ROI001 from rack directory names."""

    match = RACK_ID_RE.search(rack_dir.name)
    if not match:
        return None
    well, roi = match.groups()
    return f"{well.upper()}_ROI{roi}"


def sanitize_sample_id(value: str) -> str:
    """Return a filesystem-safe sample ID."""

    sample_id = SAFE_ID_RE.sub("_", value.replace(" ", "_"))
    return sample_id or "sample"


def sample_id_from_path(sample_dir: Path, rack_dir: Path) -> str:
    """Generate a sample ID without depending on a sample folder naming scheme."""

    match = SAMPLE_ID_RE.search(sample_dir.name)
    if match:
        return match.group(1)
    rack_sample_id = parse_rack_sample_id(rack_dir)
    if rack_sample_id:
        return sanitize_sample_id(rack_sample_id)
    return sanitize_sample_id(sample_dir.name)


def unique_sample_id(sample_id: str, seen_sample_ids: dict[str, int], input_tiff: Path) -> str:
    """Ensure a sample ID is unique within one discovery run."""

    count = seen_sample_ids.get(sample_id, 0) + 1
    seen_sample_ids[sample_id] = count
    if count == 1:
        return sample_id
    unique_id = f"{sample_id}_{count}"
    LOGGER.warning(
        "Duplicate sample_id detected during discovery. Using %s instead of %s for input_tif=%s",
        unique_id,
        sample_id,
        input_tiff,
    )
    return unique_id


def discover_samples(root_dir: Path, output_dir: Path, only_sample: str | None = None) -> list[Sample]:
    """Discover MACSima background TIFF files under root_dir."""

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    samples: list[Sample] = []
    seen_sample_ids: dict[str, int] = {}
    all_tiffs = sorted(root_dir.rglob("*_backsub.ome.tif"))
    valid_tiffs = [tif_path for tif_path in all_tiffs if tif_path.parent.name == "background"]
    skipped_non_background = len(all_tiffs) - len(valid_tiffs)
    LOGGER.info("Discovery root_dir: %s", root_dir)
    LOGGER.info("Discovery *_backsub.ome.tif files found: %d", len(all_tiffs))
    LOGGER.info("Discovery valid background TIFFs accepted: %d", len(valid_tiffs))
    LOGGER.info("Discovery TIFFs skipped because parent directory is not background: %d", skipped_non_background)
    for input_tiff in valid_tiffs:
        background_dir = input_tiff.parent
        rack_dir = background_dir.parent
        level_1_dir = rack_dir.parent
        dataset_dir = level_1_dir.parent
        sample_id = unique_sample_id(sample_id_from_path(dataset_dir, rack_dir), seen_sample_ids, input_tiff)
        if only_sample and sample_id != only_sample:
            continue
        markers_csv = input_tiff.parent / "markers_bs.csv"
        LOGGER.info("Discovery accepted sample_id=%s input_tif=%s sample_dir=%s", sample_id, input_tiff, dataset_dir)
        samples.append(
            Sample(
                sample_id=sample_id,
                dataset_dir=dataset_dir,
                input_tiff=input_tiff,
                markers_csv=markers_csv if markers_csv.exists() else None,
                label_output=output_dir / f"{sample_id}.tiff",
                macsiqview_output=output_dir / f"{sample_id}_MacsIQView.tif",
            )
        )
    if only_sample and not samples:
        raise FileNotFoundError(f"Requested sample was not found: {only_sample}")
    return samples
