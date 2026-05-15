"""Clean terminal progress rendering for batch segmentation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SampleStatus:
    """Progress state for one sample."""

    sample_id: str
    status: str
    output_path: Path
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tile_current: int = 0
    tile_total: int = 0
    error: str = ""


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds for logs and summaries."""

    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def format_clock(seconds: float | None) -> str:
    """Format a compact terminal timer as MM:SS or HH:MM:SS."""

    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class ProgressUI:
    """One-line-per-sample dashboard with a simple fallback."""

    def __init__(self, use_dashboard: bool = True) -> None:
        self.use_dashboard = use_dashboard
        self.rich = False
        self.live: Any | None = None
        self.spinner: Any | None = None
        self.statuses: list[SampleStatus] = []
        if use_dashboard:
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.spinner import Spinner
                from rich.table import Table
                from rich.text import Text

                self.console = Console()
                self.live_cls = Live
                self.spinner = Spinner("dots", style="bright_cyan")
                self.table_cls = Table
                self.text_cls = Text
                self.rich = True
            except Exception:
                self.console = None
                self.live_cls = None
                self.table_cls = None
                self.text_cls = None
        else:
            self.console = None
            self.live_cls = None
            self.table_cls = None
            self.text_cls = None

    def start(self, statuses: list[SampleStatus]) -> None:
        """Start the live dashboard when available."""

        self.statuses = statuses
        if self.rich and self.live_cls is not None:
            self.live = self.live_cls(
                self._render_table(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
                screen=False,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="visible",
            )
            self.live.start(refresh=True)
        elif not self.use_dashboard:
            self.print_line("Batch started.")

    def stop(self) -> None:
        """Stop the live dashboard."""

        if self.live is not None:
            self.live.stop()
            self.live = None

    def refresh(self) -> None:
        """Refresh the dashboard after internal status updates."""

        if self.live is not None:
            self.live.update(self._render_table(), refresh=True)

    def print_line(self, message: str) -> None:
        """Print a simple line when the dashboard is disabled or unavailable."""

        print(message)

    def print_dry_run(self, rows: list[tuple[str, Path, Path, Path, Path]]) -> None:
        """Print dry-run input and output paths."""

        print("Dry run plan")
        print("sample_id\tsample_dir\tinput_tif\toutput_label_path\toutput_macsiqview_path")
        for sample_id, sample_dir, input_tiff, label_output, macsiqview_output in rows:
            print(f"{sample_id}\t{sample_dir}\t{input_tiff}\t{label_output}\t{macsiqview_output}")

    def sample_line(self, status: SampleStatus) -> str:
        """Return one simple status line for non-dashboard mode."""

        return (
            f"{status.status:<8} {status.sample_id:<16} "
            f"{status.tile_current}/{status.tile_total:<6} {self._elapsed_eta_text(status)}"
        )

    def _status_renderable(self, status: SampleStatus) -> Any:
        if not self.rich or self.text_cls is None:
            return status.status
        if status.status == "pending":
            return self.text_cls("pending", style="yellow")
        if status.status == "running":
            return self.spinner
        if status.status == "done":
            return self.text_cls("done", style="green")
        if status.status == "error":
            return self.text_cls("error", style="red")
        if status.status == "skipped":
            return self.text_cls("skipped", style="dim white")
        return self.text_cls(status.status)

    def _elapsed_seconds(self, status: SampleStatus) -> float:
        if not status.started_at:
            return 0.0
        stop = status.finished_at or datetime.now()
        return max(0.0, (stop - status.started_at).total_seconds())

    def _estimated_remaining_seconds(self, status: SampleStatus) -> float | None:
        if status.status == "done":
            return 0.0
        if status.status != "running" or status.tile_current <= 0 or status.tile_total <= 0:
            return None
        remaining_tiles = max(0, status.tile_total - status.tile_current)
        average_tile_time = self._elapsed_seconds(status) / status.tile_current
        return average_tile_time * remaining_tiles

    def _elapsed_eta_text(self, status: SampleStatus) -> str:
        elapsed = self._elapsed_seconds(status)
        eta = self._estimated_remaining_seconds(status)
        return f"{format_clock(elapsed)} / {format_clock(eta)}"

    def _render_table(self) -> Any:
        table = self.table_cls(show_header=True, header_style="bold", show_lines=False, box=None, expand=False)
        table.add_column("status", no_wrap=True)
        table.add_column("sample_id", no_wrap=True)
        table.add_column("tile", justify="right", no_wrap=True)
        table.add_column("elapsed / remaining", justify="right", no_wrap=True)
        for status in self.statuses:
            table.add_row(
                self._status_renderable(status),
                status.sample_id,
                f"{status.tile_current}/{status.tile_total}",
                self._elapsed_eta_text(status),
            )
        return table


class ProgressContext:
    """Context manager wrapper for dashboard lifecycle."""

    def __init__(self, ui: ProgressUI, statuses: list[SampleStatus]) -> None:
        self.ui = ui
        self.statuses = statuses

    def __enter__(self) -> ProgressUI:
        self.ui.start(self.statuses)
        return self.ui

    def __exit__(self, *_exc: object) -> None:
        time.sleep(0.05)
        self.ui.refresh()
        self.ui.stop()
