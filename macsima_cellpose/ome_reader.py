"""OME-TIFF channel selection and tile reading."""

from __future__ import annotations

import csv
import logging
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
    """Read selected-channel OME-TIFF tiles using tifffile's lazy zarr interface."""

    def __init__(self, path: Path, logger: logging.Logger | None = None) -> None:
        self.path = path
        self.logger = logger or logging.getLogger(__name__)
        self.channel_names = ome_channel_names(path)
        self.height, self.width = self._parse_image_size()
        self.z0_shape, self.z0_ndim = self._inspect_zarr()
        self.channel_count = self._infer_channel_count()

    def close(self) -> None:
        """Compatibility no-op. TIFF and zarr stores are opened per read call."""

        return None

    def __enter__(self) -> "OmeTileReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _parse_image_size(self) -> tuple[int, int]:
        """Use page 0 dimensions as the full image size, matching the proven MACSima loader."""

        with tifffile.TiffFile(self.path) as tif:
            page0 = tif.pages[0]
            return int(page0.imagelength), int(page0.imagewidth)

    @staticmethod
    def _select_z0(z: Any) -> Any:
        """Select the first array from a tifffile zarr object or group."""

        if hasattr(z, "ndim"):
            return z
        if not hasattr(z, "keys"):
            return z
        keys = list(z.keys())
        if "0" in keys:
            try:
                return z[0]
            except Exception:
                return z["0"]
        try:
            return next(iter(z.values()))
        except Exception as exc:
            raise RuntimeError("Could not select an array from the OME-TIFF zarr group.") from exc

    def _open_z0(self) -> tuple[Any, Any, Any]:
        """Open a fresh TiffFile and zarr array for one ROI read."""

        import zarr

        tif = tifffile.TiffFile(self.path)
        try:
            store = tif.aszarr()
            z = zarr.open(store, mode="r")
            z0 = self._select_z0(z)
            self.logger.debug("OME zarr object type: %s", type(z).__name__)
            self.logger.debug("OME selected z0 shape: %s", getattr(z0, "shape", None))
            self.logger.debug("OME selected z0 ndim: %s", getattr(z0, "ndim", None))
            return tif, store, z0
        except Exception:
            tif.close()
            raise

    def _inspect_zarr(self) -> tuple[tuple[int, ...], int]:
        """Inspect zarr shape without caching the zarr object."""

        tif = None
        store = None
        try:
            tif, store, z0 = self._open_z0()
            shape = tuple(int(v) for v in getattr(z0, "shape", ()))
            ndim = int(getattr(z0, "ndim", 0))
            return shape, ndim
        except Exception as exc:
            raise RuntimeError(f"OME-TIFF zarr inspection failed: {exc}") from exc
        finally:
            if store is not None and hasattr(store, "close"):
                try:
                    store.close()
                except Exception:
                    pass
            if tif is not None:
                tif.close()

    def _infer_channel_count(self) -> int:
        if self.z0_ndim == 3 and self.z0_shape:
            return int(self.z0_shape[0])
        if self.z0_ndim == 4 and len(self.z0_shape) >= 2:
            return int(self.z0_shape[1])
        if self.z0_ndim == 2:
            return 1
        if self.channel_names:
            return len(self.channel_names)
        raise RuntimeError(f"Unsupported OME-TIFF zarr dimensions during channel inference: ndim={self.z0_ndim}")

    def read_channel_tile(self, channel_index: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Read one 2-D tile from the selected channel."""

        if channel_index < 0 or channel_index >= self.channel_count:
            raise ValueError(f"Channel index is out of range: {channel_index}")
        self.logger.debug(
            "Reading OME tile: channel_index=%d y0=%d y1=%d x0=%d x1=%d",
            channel_index,
            y0,
            y1,
            x0,
            x1,
        )
        tif = None
        store = None
        try:
            tif, store, z0 = self._open_z0()
            ndim = int(getattr(z0, "ndim", 0))
            if ndim == 3:
                tile = np.asarray(z0[channel_index, y0:y1, x0:x1])
            elif ndim == 4:
                tile = np.asarray(z0[0, channel_index, y0:y1, x0:x1])
            elif ndim == 2:
                tile = np.asarray(z0[y0:y1, x0:x1])
            else:
                raise RuntimeError(f"Unsupported OME-TIFF zarr dimensions for ROI read: ndim={ndim}")
            return np.squeeze(tile).copy()
        except Exception as exc:
            raise RuntimeError(
                "OME-TIFF zarr ROI read failed "
                f"for channel_index={channel_index}, y={y0}:{y1}, x={x0}:{x1}: {exc}"
            ) from exc
        finally:
            if store is not None and hasattr(store, "close"):
                try:
                    store.close()
                except Exception:
                    pass
            if tif is not None:
                tif.close()


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
