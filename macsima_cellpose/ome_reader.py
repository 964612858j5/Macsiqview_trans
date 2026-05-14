"""OME-TIFF channel selection and tile reading."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


NUCLEAR_KEYWORDS = ("dapi", "hoechst", "dna", "nucleus", "nuclei")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def ome_channel_names(path: Path) -> list[str]:
    """Extract channel names from OME XML when present."""

    try:
        with tifffile.TiffFile(path) as tif:
            xml = tif.ome_metadata
        if not xml:
            return []
        root = ET.fromstring(xml)
        names: list[str] = []
        for elem in root.iter():
            if _local_name(elem.tag) == "Channel":
                name = elem.attrib.get("Name") or elem.attrib.get("ID") or ""
                names.append(name)
        return names
    except Exception:
        return []


def marker_names(markers_csv: Path | None) -> list[str]:
    """Read channel or marker names from markers_bs.csv."""

    if not markers_csv or not markers_csv.exists():
        return []
    try:
        with markers_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            return []
        preferred = ("marker", "marker_name", "channel", "channel_name", "name", "target")
        fieldnames = [name for name in (reader.fieldnames or []) if name]
        column = next((name for name in fieldnames if name.lower() in preferred), fieldnames[0])
        return [str(row.get(column, "")).strip() for row in rows]
    except Exception:
        return []


def find_nuclear_channel(names: list[str]) -> int | None:
    """Return the first channel index matching nuclear channel keywords."""

    for idx, name in enumerate(names):
        lower = name.lower()
        if any(keyword in lower for keyword in NUCLEAR_KEYWORDS):
            return idx
    return None


def parse_channel_override(override: str | None, names: list[str], channel_count: int) -> int | None:
    """Parse a channel override as either zero-based index or channel name."""

    if not override:
        return None
    text = override.strip()
    if re.fullmatch(r"\d+", text):
        index = int(text)
        if index < 0 or index >= channel_count:
            raise ValueError(f"Nuclear channel index is out of range: {index}")
        return index
    for idx, name in enumerate(names):
        if name.lower() == text.lower():
            return idx
    raise ValueError(f"Nuclear channel name was not found: {text}")


class OmeTileReader:
    """Read a selected channel from OME-TIFF tiles without loading all channels when possible."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tif = tifffile.TiffFile(path)
        self.series = self._tif.series[0]
        self.axes = self.series.axes
        self.shape = self.series.shape
        self._store: Any | None = None
        self._array: Any | None = None
        try:
            import zarr

            self._store = self.series.aszarr()
            self._array = zarr.open(self._store, mode="r")
        except Exception:
            self._array = None
        self.height, self.width = self._infer_hw()
        self.channel_count = self._infer_channel_count()

    def close(self) -> None:
        """Close open TIFF resources."""

        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
        self._tif.close()

    def __enter__(self) -> "OmeTileReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _infer_hw(self) -> tuple[int, int]:
        if "Y" not in self.axes or "X" not in self.axes:
            raise ValueError(f"OME-TIFF series does not expose Y and X axes: axes={self.axes}")
        return int(self.shape[self.axes.index("Y")]), int(self.shape[self.axes.index("X")])

    def _infer_channel_count(self) -> int:
        return int(self.shape[self.axes.index("C")]) if "C" in self.axes else 1

    def read_channel_tile(self, channel_index: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Read one 2-D tile from the selected channel."""

        if channel_index < 0 or channel_index >= self.channel_count:
            raise ValueError(f"Channel index is out of range: {channel_index}")
        index: list[Any] = []
        for axis in self.axes:
            if axis == "Y":
                index.append(slice(y0, y1))
            elif axis == "X":
                index.append(slice(x0, x1))
            elif axis == "C":
                index.append(channel_index)
            else:
                index.append(0)
        if self._array is not None:
            tile = np.asarray(self._array[tuple(index)])
        else:
            data = self.series.asarray()
            tile = np.asarray(data[tuple(index)])
        return np.squeeze(tile)


def choose_nuclear_channel(path: Path, markers_csv: Path | None, override: str | None, channel_count: int) -> tuple[int, list[str], bool]:
    """Choose a nuclear channel from override, OME metadata, markers CSV, or fallback."""

    names = ome_channel_names(path)
    marker_list = marker_names(markers_csv)
    if marker_list and (not names or len(marker_list) >= len(names)):
        names = marker_list
    if len(names) < channel_count:
        names = names + [f"channel_{idx}" for idx in range(len(names), channel_count)]
    override_index = parse_channel_override(override, names, channel_count)
    if override_index is not None:
        return override_index, names, False
    detected = find_nuclear_channel(names)
    if detected is not None and detected < channel_count:
        return detected, names, False
    return 0, names, True
