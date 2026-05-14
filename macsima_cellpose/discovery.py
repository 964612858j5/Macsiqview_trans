"""Dataset discovery and sample identity parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SAMPLE_ID_RE = re.compile(r"_(R\d+_[A-Z]\d+_ROI\d+)_")


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


def discover_samples(root_dir: Path, output_dir: Path, only_sample: str | None = None) -> list[Sample]:
    """Discover all 2026* MACSima sample folders under root_dir."""

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    samples: list[Sample] = []
    for dataset_dir in sorted(p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("2026")):
        sample_id = parse_sample_id(dataset_dir)
        if only_sample and sample_id != only_sample:
            continue
        candidates = sorted(dataset_dir.glob("1/rack-*/background/*_backsub.ome.tif"))
        if not candidates:
            raise FileNotFoundError(f"No backsub OME-TIFF found for sample {sample_id}: {dataset_dir}")
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple backsub OME-TIFF files found for sample {sample_id}: {dataset_dir}")
        input_tiff = candidates[0]
        markers_csv = input_tiff.parent / "markers_bs.csv"
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
