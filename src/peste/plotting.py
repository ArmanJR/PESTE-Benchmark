"""Deterministic SVG rendering for leaderboard comparisons."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from xml.sax.saxutils import escape, quoteattr

LOGGER = logging.getLogger(__name__)


class LeaderboardPlotRow(Protocol):
    """Metrics required by the leaderboard plot."""

    @property
    def model_id(self) -> str: ...

    @property
    def wer(self) -> float: ...

    @property
    def cer(self) -> float: ...

    @property
    def memory_efficiency(self) -> float: ...

    @property
    def peak_cuda_reserved_gib(self) -> float: ...


@dataclass(frozen=True, slots=True)
class MetricSpec:
    title: str
    unit: str
    css_class: str
    value: Callable[[LeaderboardPlotRow], float]
    formatter: Callable[[float], str]


ACCURACY_METRICS = (
    MetricSpec("WER ↓", "percent", "wer", lambda row: row.wer * 100, lambda value: f"{value:.1f}%"),
    MetricSpec("CER ↓", "percent", "cer", lambda row: row.cer * 100, lambda value: f"{value:.1f}%"),
)
MEMORY_METRICS = (
    MetricSpec(
        "Peak CUDA reserved ↓",
        "GiB",
        "memory",
        lambda row: row.peak_cuda_reserved_gib,
        lambda value: f"{value:.2f}",
    ),
    MetricSpec(
        "Word accuracy / GiB ↑",
        "score",
        "efficiency",
        lambda row: row.memory_efficiency,
        lambda value: f"{value:.1f}",
    ),
)


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1.0, 1.25, 1.5, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    raise AssertionError("The final ceiling candidate must accept every positive value")


def _tick(value: float, unit: str) -> str:
    if unit == "percent":
        return f"{value:g}%"
    return f"{value:g}"


def _short_label(value: str, limit: int = 34) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _render_leaderboard_svg(
    suite_id: str,
    rows: Sequence[LeaderboardPlotRow],
    metrics: Sequence[MetricSpec],
    *,
    board: str,
    accessible_title: str,
    accessible_description: str,
    repositories: dict[str, str] | None = None,
) -> str:
    """Render one scalable leaderboard plot from already ranked rows."""
    ordered = list(rows)
    model_repositories = repositories or {}
    width = 900
    margin = 24
    label_width = 235
    panel_gap = 24
    panel_width = (width - 2 * margin - label_width - panel_gap * (len(metrics) - 1)) / len(metrics)
    value_width = 58
    bar_width = panel_width - value_width
    row_height = 36
    rows_top = 66
    footer_height = 18
    height = rows_top + len(ordered) * row_height + footer_height if ordered else 92

    scales = [
        _nice_ceiling(max((metric.value(row) for row in ordered), default=0.0))
        for metric in metrics
    ]
    scaled_metrics = list(zip(metrics, scales, strict=True))

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="leaderboard-plot-title leaderboard-plot-description">'
        ),
        f'<title id="leaderboard-plot-title">{escape(accessible_title)}</title>',
        (
            '<desc id="leaderboard-plot-description">'
            f"{len(ordered)} ranked models from {escape(suite_id)}. "
            f"{escape(accessible_description)}"
            "</desc>"
        ),
        "<style>",
        "  :root { color-scheme: light dark; --plot-text: #111827; --plot-muted: #6b7280;",
        "    --plot-grid: #d1d5db; --plot-track: #e5e7eb; --wer: #d97706;",
        "    --cer: #2563eb; --memory: #7c3aed; --efficiency: #059669; }",
        "  @media (prefers-color-scheme: dark) { :root {",
        "    --plot-text: #f3f4f6; --plot-muted: #9ca3af; --plot-grid: #374151;",
        "    --plot-track: #1f2937; --wer: #fbbf24; --cer: #60a5fa;",
        "    --memory: #c084fc; --efficiency: #34d399; } }",
        (
            "  text { fill: #111827; fill: var(--plot-text); font-family: ui-sans-serif, "
            'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }'
        ),
        "  .plot-subtitle, .tick { fill: #6b7280; fill: var(--plot-muted); }",
        "  .plot-subtitle { font-size: 12px; }",
        "  .metric-title { font-size: 14px; font-weight: 500; }",
        "  .metric-unit, .tick { font-size: 11px; }",
        "  .model-label { font-size: 13px; font-weight: 500; }",
        "  .rank, .value { font-size: 12px; }",
        "  .rank { fill: #6b7280; fill: var(--plot-muted); }",
        "  .value { font-weight: 500; }",
        ("  .grid, .row-line { stroke: #d1d5db; stroke: var(--plot-grid); stroke-width: 1; }"),
        "  .grid { opacity: 0.55; }",
        "  .row-line { opacity: 0.35; }",
        "  .track { fill: #e5e7eb; fill: var(--plot-track); }",
        "  .bar.wer { fill: #d97706; fill: var(--wer); }",
        "  .bar.cer { fill: #2563eb; fill: var(--cer); }",
        "  .bar.memory { fill: #7c3aed; fill: var(--memory); }",
        "  .bar.efficiency { fill: #059669; fill: var(--efficiency); }",
        "  a:hover .model-label { text-decoration: underline; }",
        "</style>",
    ]

    if not ordered:
        lines.append(
            f'<text class="plot-subtitle" x="{margin}" y="54">'
            "No complete official results yet</text>"
        )
    else:
        panel_start = margin + label_width
        plot_bottom = rows_top + len(ordered) * row_height
        lines.append(f'<text class="metric-title" x="{margin}" y="24">Model</text>')
        for metric_index, (metric, scale) in enumerate(scaled_metrics):
            panel_x = panel_start + metric_index * (panel_width + panel_gap)
            lines.extend(
                [
                    (
                        f'<text class="metric-title" x="{panel_x:.1f}" '
                        f'y="24">{escape(metric.title)}</text>'
                    ),
                    (
                        f'<text class="metric-unit" x="{panel_x:.1f}" '
                        f'y="40">{escape(metric.unit)}</text>'
                    ),
                ]
            )
            for fraction in (0.0, 0.5, 1.0):
                tick_x = panel_x + bar_width * fraction
                tick_value = scale * fraction
                anchor = "start" if fraction == 0 else "middle" if fraction == 0.5 else "end"
                lines.extend(
                    [
                        (
                            f'<line class="grid" x1="{tick_x:.1f}" y1="45" '
                            f'x2="{tick_x:.1f}" y2="{plot_bottom}" />'
                        ),
                        (
                            f'<text class="tick" x="{tick_x:.1f}" y="59" '
                            f'text-anchor="{anchor}">'
                            f"{escape(_tick(tick_value, metric.unit))}</text>"
                        ),
                    ]
                )

        for rank, row in enumerate(ordered, start=1):
            row_top = rows_top + (rank - 1) * row_height
            center_y = row_top + row_height / 2
            bar_y = center_y - 8
            label = _short_label(row.model_id, limit=30)
            lines.append(
                f'<line class="row-line" x1="{margin}" y1="{row_top}" '
                f'x2="{width - margin}" y2="{row_top}" />'
            )
            lines.append(f'<text class="rank" x="{margin}" y="{center_y + 4:.1f}">{rank}</text>')
            model_text = (
                f'<text class="model-label" x="{margin + 24}" y="{center_y + 4:.1f}">'
                f"<title>{escape(row.model_id)}</title>{escape(label)}</text>"
            )
            repository = model_repositories.get(row.model_id)
            if repository is None:
                lines.append(model_text)
            else:
                url = f"https://huggingface.co/{repository}"
                lines.append(f"<a href={quoteattr(url)}>{model_text}</a>")

            for metric_index, (metric, scale) in enumerate(scaled_metrics):
                panel_x = panel_start + metric_index * (panel_width + panel_gap)
                value = metric.value(row)
                rendered_width = bar_width * value / scale
                lines.extend(
                    [
                        (
                            f'<rect class="track" x="{panel_x:.1f}" y="{bar_y:.1f}" '
                            f'width="{bar_width:.1f}" height="16" rx="3" />'
                        ),
                        (
                            f'<rect class="bar {metric.css_class}" x="{panel_x:.1f}" '
                            f'y="{bar_y:.1f}" width="{rendered_width:.1f}" '
                            'height="16" rx="3" />'
                        ),
                        (
                            f'<text class="value" x="{panel_x + bar_width + 8:.1f}" '
                            f'y="{center_y + 4:.1f}">{escape(metric.formatter(value))}</text>'
                        ),
                    ]
                )
        lines.append(
            f'<line class="row-line" x1="{margin}" y1="{plot_bottom}" '
            f'x2="{width - margin}" y2="{plot_bottom}" />'
        )

    lines.extend(["</svg>", ""])
    LOGGER.debug(
        "Rendered leaderboard SVG",
        extra={
            "suite": suite_id,
            "board": board,
            "ranked_models": len(ordered),
            "width": width,
            "height": height,
        },
    )
    return "\n".join(lines)


def render_accuracy_svg(
    suite_id: str,
    rows: Sequence[LeaderboardPlotRow],
    repositories: dict[str, str] | None = None,
) -> str:
    """Render the normalized-accuracy plot in leaderboard order."""
    ordered = sorted(rows, key=lambda row: (row.wer, row.cer, row.model_id))
    return _render_leaderboard_svg(
        suite_id,
        ordered,
        ACCURACY_METRICS,
        board="accuracy",
        accessible_title="PESTE normalized accuracy leaderboard",
        accessible_description="Compared by word error rate and character error rate.",
        repositories=repositories,
    )


def render_memory_svg(
    suite_id: str,
    rows: Sequence[LeaderboardPlotRow],
    repositories: dict[str, str] | None = None,
) -> str:
    """Render the memory-efficiency plot in leaderboard order."""
    ordered = sorted(rows, key=lambda row: (-row.memory_efficiency, row.wer, row.model_id))
    return _render_leaderboard_svg(
        suite_id,
        ordered,
        MEMORY_METRICS,
        board="memory_efficiency",
        accessible_title="PESTE accuracy per peak CUDA memory leaderboard",
        accessible_description=(
            "Compared by peak CUDA reserved memory and word accuracy per reserved GiB."
        ),
        repositories=repositories,
    )
