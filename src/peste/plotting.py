"""Deterministic SVG rendering for leaderboard comparisons."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection, Sequence
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
    def audio_throughput_x(self) -> float: ...

    @property
    def rtf(self) -> float: ...

    @property
    def speed_valid(self) -> bool: ...


class ParetoPlotRow(LeaderboardPlotRow, Protocol):
    """Metrics and uncertainty required by the Pareto plot."""

    @property
    def cer_ci_lower(self) -> float: ...

    @property
    def cer_ci_upper(self) -> float: ...


@dataclass(frozen=True, slots=True)
class MetricSpec:
    title: str
    unit: str
    css_class: str
    value: Callable[[LeaderboardPlotRow], float]
    formatter: Callable[[float], str]


ACCURACY_METRICS = (
    MetricSpec("CER ↓", "percent", "cer", lambda row: row.cer * 100, lambda value: f"{value:.1f}%"),
    MetricSpec("WER ↓", "percent", "wer", lambda row: row.wer * 100, lambda value: f"{value:.1f}%"),
)
SPEED_METRICS = (
    MetricSpec(
        "Audio throughput ↑",
        "× real time",
        "throughput",
        lambda row: row.audio_throughput_x,
        lambda value: f"{value:.2f}×",
    ),
    MetricSpec(
        "RTF ↓",
        "processing / audio",
        "rtf",
        lambda row: row.rtf,
        lambda value: f"{value:.4f}",
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
    longest_label = max((len(row.model_id) for row in ordered), default=0)
    label_width = max(235, min(680, 42 + longest_label * 7))
    width = 900 + label_width - 235
    margin = 24
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
        "    --cer: #2563eb; --throughput: #059669; --rtf: #7c3aed; }",
        "  @media (prefers-color-scheme: dark) { :root {",
        "    --plot-text: #f3f4f6; --plot-muted: #9ca3af; --plot-grid: #374151;",
        "    --plot-track: #1f2937; --wer: #fbbf24; --cer: #60a5fa;",
        "    --throughput: #34d399; --rtf: #c084fc; } }",
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
        "  .bar.throughput { fill: #059669; fill: var(--throughput); }",
        "  .bar.rtf { fill: #7c3aed; fill: var(--rtf); }",
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
            label = row.model_id
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
    ordered = sorted(rows, key=lambda row: (row.cer, row.wer, row.model_id))
    return _render_leaderboard_svg(
        suite_id,
        ordered,
        ACCURACY_METRICS,
        board="accuracy",
        accessible_title="PESTE normalized accuracy leaderboard",
        accessible_description=(
            "Ranked by whitespace-insensitive character error rate; word error rate is also shown."
        ),
        repositories=repositories,
    )


def render_speed_svg(
    suite_id: str,
    rows: Sequence[LeaderboardPlotRow],
    repositories: dict[str, str] | None = None,
) -> str:
    """Render valid steady-state speed results in leaderboard order."""
    ordered = sorted(
        (row for row in rows if row.speed_valid),
        key=lambda row: (-row.audio_throughput_x, row.cer, row.wer, row.model_id),
    )
    return _render_leaderboard_svg(
        suite_id,
        ordered,
        SPEED_METRICS,
        board="speed",
        accessible_title="PESTE steady-state speed leaderboard",
        accessible_description="Ranked by audio throughput; real-time factor is also shown.",
        repositories=repositories,
    )


def _log_ticks(lower: float, upper: float) -> list[float]:
    ticks: list[float] = []
    minimum_exponent = math.floor(math.log10(lower))
    maximum_exponent = math.ceil(math.log10(upper))
    for exponent in range(minimum_exponent, maximum_exponent + 1):
        magnitude = 10.0**exponent
        for multiplier in (1.0, 2.0, 5.0):
            candidate = multiplier * magnitude
            if lower <= candidate <= upper:
                ticks.append(candidate)
    return ticks


def _log_bounds(values: Sequence[float]) -> tuple[float, float]:
    logarithms = [math.log10(value) for value in values]
    lower = min(logarithms)
    upper = max(logarithms)
    span = upper - lower
    if span == 0:
        return lower - 0.5, upper + 0.5
    padding = span * 0.08
    return lower - padding, upper + padding


def _pareto_tick(value: float, *, throughput: bool) -> str:
    if throughput:
        return f"{value:g}×"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def render_pareto_svg(
    suite_id: str,
    rows: Sequence[ParetoPlotRow],
    frontier_model_ids: Collection[str],
    statistically_dominated_model_ids: Collection[str],
    repositories: dict[str, str] | None = None,
) -> str:
    """Render the CER/throughput Pareto frontier with CER confidence intervals."""
    ordered = sorted(
        (row for row in rows if row.speed_valid),
        key=lambda row: (-row.audio_throughput_x, row.cer, row.wer, row.model_id),
    )
    model_repositories = repositories or {}
    frontier = set(frontier_model_ids)
    statistically_dominated = set(statistically_dominated_model_ids)
    width = 1440
    height = max(560, 104 + len(ordered) * 19)
    plot_left = 78.0
    plot_right = 720.0
    plot_top = 88.0
    plot_bottom = float(height - 78)
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="pareto-plot-title pareto-plot-description">'
        ),
        '<title id="pareto-plot-title">PESTE accuracy-speed Pareto efficiency</title>',
        (
            '<desc id="pareto-plot-description">'
            f"CER versus steady-state audio throughput for {len(ordered)} speed-valid models "
            f"from {escape(suite_id)}. Lower CER and higher throughput are better."
            "</desc>"
        ),
        "<style>",
        "  :root { color-scheme: light dark; --plot-text: #111827; --plot-muted: #6b7280;",
        "    --plot-grid: #d1d5db; --plot-frontier: #2563eb; --plot-dominated: #9ca3af;",
        "    --plot-supported: #dc2626; --plot-error: #4b5563; --plot-surface: #ffffff; }",
        "  @media (prefers-color-scheme: dark) { :root { --plot-text: #f3f4f6;",
        "    --plot-muted: #9ca3af; --plot-grid: #374151; --plot-frontier: #60a5fa;",
        "    --plot-dominated: #6b7280; --plot-supported: #f87171;",
        "    --plot-error: #d1d5db; --plot-surface: #111827; } }",
        (
            "  text { fill: #111827; fill: var(--plot-text); font-family: ui-sans-serif, "
            'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }'
        ),
        "  .subtitle, .tick, .legend { fill: #6b7280; fill: var(--plot-muted); }",
        "  .subtitle { font-size: 12px; }",
        "  .tick, .legend { font-size: 11px; }",
        "  .axis-title { font-size: 13px; font-weight: 500; }",
        "  .model-label { font-size: 11px; font-weight: 500; }",
        "  .grid { stroke: #d1d5db; stroke: var(--plot-grid); stroke-width: 1; opacity: 0.55; }",
        "  .axis { stroke: #6b7280; stroke: var(--plot-muted); stroke-width: 1.2; }",
        "  .frontier-line { fill: none; stroke: #2563eb; stroke: var(--plot-frontier);",
        "    stroke-width: 2.5; stroke-linejoin: round; opacity: 0.75; }",
        "  .error-bar { stroke: #4b5563; stroke: var(--plot-error); stroke-width: 1.2; }",
        "  .point { stroke: #ffffff; stroke: var(--plot-surface); stroke-width: 2; }",
        "  .point.frontier { fill: #2563eb; fill: var(--plot-frontier); }",
        "  .point.dominated { fill: #9ca3af; fill: var(--plot-dominated); }",
        "  .point.supported-dominated { fill: #dc2626; fill: var(--plot-supported); }",
        "  a:hover .model-label { text-decoration: underline; }",
        "</style>",
    ]
    if not ordered:
        lines.extend(
            [
                '<text class="subtitle" x="24" y="54">No complete speed-valid results yet</text>',
                "</svg>",
                "",
            ]
        )
        return "\n".join(lines)
    if any(row.audio_throughput_x <= 0 for row in ordered):
        raise ValueError("Pareto throughput values must be positive for logarithmic plotting")
    positive_cer_values = [
        value
        for row in ordered
        for value in (row.cer, row.cer_ci_lower, row.cer_ci_upper)
        if value > 0
    ]
    if not positive_cer_values:
        positive_cer_values = [1e-6]
    cer_floor = min(positive_cer_values) / 2
    throughput_log_min, throughput_log_max = _log_bounds(
        [row.audio_throughput_x for row in ordered]
    )
    cer_log_min, cer_log_max = _log_bounds([max(value, cer_floor) for value in positive_cer_values])

    def x_position(value: float) -> float:
        fraction = (math.log10(value) - throughput_log_min) / (
            throughput_log_max - throughput_log_min
        )
        return plot_left + fraction * plot_width

    def y_position(value: float) -> float:
        fraction = (math.log10(max(value, cer_floor)) - cer_log_min) / (cer_log_max - cer_log_min)
        return plot_top + fraction * plot_height

    throughput_lower = 10**throughput_log_min
    throughput_upper = 10**throughput_log_max
    cer_lower = 10**cer_log_min
    cer_upper = 10**cer_log_max
    for tick in _log_ticks(throughput_lower, throughput_upper):
        x = x_position(tick)
        lines.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{plot_top:.1f}" '
                f'x2="{x:.1f}" y2="{plot_bottom:.1f}" />',
                f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 18:.1f}" '
                f'text-anchor="middle">{escape(_pareto_tick(tick, throughput=True))}</text>',
            ]
        )
    for tick in _log_ticks(cer_lower, cer_upper):
        y = y_position(tick)
        lines.extend(
            [
                f'<line class="grid" x1="{plot_left:.1f}" y1="{y:.1f}" '
                f'x2="{plot_right:.1f}" y2="{y:.1f}" />',
                f'<text class="tick" x="{plot_left - 10:.1f}" y="{y + 4:.1f}" '
                f'text-anchor="end">{escape(_pareto_tick(tick, throughput=False))}</text>',
            ]
        )
    lines.extend(
        [
            f'<line class="axis" x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" '
            f'x2="{plot_right:.1f}" y2="{plot_bottom:.1f}" />',
            f'<line class="axis" x1="{plot_left:.1f}" y1="{plot_top:.1f}" '
            f'x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" />',
            f'<text class="axis-title" x="{(plot_left + plot_right) / 2:.1f}" '
            f'y="{height - 26}" '
            'text-anchor="middle">Audio throughput (× real time, log scale) →</text>',
            f'<text class="axis-title" x="18" y="{height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 18 {height / 2:.1f})">'
            "Accuracy (CER ↓, inverted log scale)</text>",
            '<text class="axis-title" x="18" y="78" text-anchor="middle">↑</text>',
            '<text class="subtitle" x="512" y="70">Ideal direction: upper-right ↗</text>',
            '<circle class="point frontier" cx="92" cy="28" r="6" />',
            '<text class="legend" x="104" y="32">Pareto frontier</text>',
            '<circle class="point dominated" cx="218" cy="28" r="6" />',
            '<text class="legend" x="230" y="32">Point-dominated</text>',
            '<circle class="point supported-dominated" cx="350" cy="28" r="6" />',
            '<text class="legend" x="362" y="32">Supported dominance</text>',
            '<line class="error-bar" x1="516" y1="21" x2="516" y2="35" />',
            '<line class="error-bar" x1="512" y1="21" x2="520" y2="21" />',
            '<line class="error-bar" x1="512" y1="35" x2="520" y2="35" />',
            '<text class="legend" x="526" y="32">CER 95% CI</text>',
        ]
    )
    frontier_points = sorted(
        (row for row in ordered if row.model_id in frontier),
        key=lambda row: row.audio_throughput_x,
    )
    if len(frontier_points) > 1:
        points = " ".join(
            f"{x_position(row.audio_throughput_x):.1f},{y_position(row.cer):.1f}"
            for row in frontier_points
        )
        lines.append(f'<polyline class="frontier-line" points="{points}" />')
    roster_x = 760.0
    roster_top = 92.0
    lines.extend(
        [
            f'<text class="axis-title" x="{roster_x:.1f}" y="70">All models</text>',
            f'<text class="subtitle" x="{roster_x:.1f}" y="84">Numbered in speed order</text>',
        ]
    )
    for index, row in enumerate(ordered):
        x = x_position(row.audio_throughput_x)
        y = y_position(row.cer)
        lower_y = y_position(row.cer_ci_lower)
        upper_y = y_position(row.cer_ci_upper)
        lines.extend(
            [
                f'<line class="error-bar" x1="{x:.1f}" y1="{upper_y:.1f}" '
                f'x2="{x:.1f}" y2="{lower_y:.1f}" />',
                f'<line class="error-bar" x1="{x - 4:.1f}" y1="{upper_y:.1f}" '
                f'x2="{x + 4:.1f}" y2="{upper_y:.1f}" />',
                f'<line class="error-bar" x1="{x - 4:.1f}" y1="{lower_y:.1f}" '
                f'x2="{x + 4:.1f}" y2="{lower_y:.1f}" />',
            ]
        )
        if row.model_id in frontier:
            point_class = "frontier"
        elif row.model_id in statistically_dominated:
            point_class = "supported-dominated"
        else:
            point_class = "dominated"
        title = (
            f"{row.model_id}: CER {row.cer:.4f} "
            f"({row.cer_ci_lower:.4f}–{row.cer_ci_upper:.4f}), "
            f"throughput {row.audio_throughput_x:.3f}×"
        )
        point = (
            f'<circle class="point {point_class}" cx="{x:.1f}" cy="{y:.1f}" r="7">'
            f"<title>{escape(title)}</title></circle>"
        )
        point_number = (
            f'<text class="legend" x="{x:.1f}" y="{y + 3.5:.1f}" '
            f'text-anchor="middle">{index + 1}</text>'
        )
        roster_y = roster_top + index * 19
        model_text = (
            f'<text class="model-label" x="{roster_x:.1f}" y="{roster_y:.1f}">'
            f"<title>{escape(title)}</title>{index + 1}. {escape(row.model_id)}</text>"
        )
        repository = model_repositories.get(row.model_id)
        if repository is None:
            lines.extend([point, point_number, model_text])
        else:
            url = f"https://huggingface.co/{repository}"
            lines.append(f"<a href={quoteattr(url)}>{point}{point_number}{model_text}</a>")
    lines.extend(["</svg>", ""])
    LOGGER.debug(
        "Rendered Pareto SVG",
        extra={
            "suite": suite_id,
            "ranked_models": len(ordered),
            "frontier_models": len(frontier),
            "statistically_dominated_models": len(statistically_dominated),
            "width": width,
            "height": height,
        },
    )
    return "\n".join(lines)
