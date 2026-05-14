"""Terminal progress rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path


@dataclass
class SampleStatus:
    """Progress state for one sample."""

    sample_id: str
    status: str
    output_path: Path
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str = ""


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds for terminal output."""

    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def estimate_remaining(completed: int, total: int, elapsed_seconds: float) -> str:
    """Estimate remaining batch runtime from completed samples."""

    if completed <= 0:
        return "unknown"
    remaining = total - completed
    return format_seconds((elapsed_seconds / completed) * remaining)


class ProgressUI:
    """Small wrapper around rich with a plain-text fallback."""

    def __init__(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            self.console = Console()
            self.table_cls = Table
            self.rich = True
        except Exception:
            self.console = None
            self.table_cls = None
            self.rich = False

    def print_plan(self, statuses: list[SampleStatus]) -> None:
        """Print the planned work."""

        if self.rich:
            table = self.table_cls(title="Planned samples")
            table.add_column("sample_id")
            table.add_column("status")
            table.add_column("output path")
            for status in statuses:
                table.add_row(status.sample_id, status.status, str(status.output_path))
            self.console.print(table)
            return
        print("Planned samples")
        for status in statuses:
            print(f"{status.sample_id}\t{status.status}\t{status.output_path}")

    def print_dry_run(self, rows: list[tuple[str, Path, Path, Path]]) -> None:
        """Print dry-run input and output paths."""

        if self.rich:
            table = self.table_cls(title="Dry run plan")
            table.add_column("sample_id")
            table.add_column("input TIFF")
            table.add_column("label output")
            table.add_column("MacsIQView output")
            for sample_id, input_tiff, label_output, macsiqview_output in rows:
                table.add_row(sample_id, str(input_tiff), str(label_output), str(macsiqview_output))
            self.console.print(table)
            return
        print("Dry run plan")
        for sample_id, input_tiff, label_output, macsiqview_output in rows:
            print(f"{sample_id}\t{input_tiff}\t{label_output}\t{macsiqview_output}")

    def print_status(self, statuses: list[SampleStatus], elapsed_seconds: float) -> None:
        """Print current sample statuses."""

        completed = sum(1 for item in statuses if item.status in {"done", "skipped", "error"})
        eta = estimate_remaining(completed, len(statuses), elapsed_seconds)
        if self.rich:
            table = self.table_cls(title=f"Batch status | elapsed {format_seconds(elapsed_seconds)} | remaining {eta}")
            table.add_column("sample_id")
            table.add_column("status")
            table.add_column("elapsed")
            table.add_column("output path")
            for item in statuses:
                elapsed = ""
                if item.started_at:
                    stop = item.finished_at or datetime.now()
                    elapsed = format_seconds((stop - item.started_at).total_seconds())
                marker = "running" if item.status == "running" else item.status
                table.add_row(item.sample_id, marker, elapsed, str(item.output_path))
            self.console.print(table)
            return
        print(f"Batch status | elapsed {format_seconds(elapsed_seconds)} | remaining {eta}")
        for item in statuses:
            print(f"{item.sample_id}\t{item.status}\t{item.output_path}")

    def running_spinner(self, sample_id: str):
        """Return a rich spinner context for the currently running sample."""

        if self.rich:
            return self.console.status(f"Running sample {sample_id}", spinner="dots")
        return nullcontext()
