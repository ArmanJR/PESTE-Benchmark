"""Deterministic, data-driven accuracy, speed, and Pareto plot tests."""

from dataclasses import dataclass
from xml.etree import ElementTree

from peste.plotting import render_accuracy_svg, render_pareto_svg, render_speed_svg


@dataclass(frozen=True, slots=True)
class PlotRow:
    model_id: str
    wer: float
    cer: float
    audio_throughput_x: float
    rtf: float
    speed_valid: bool = True
    cer_ci_lower: float = 0.05
    cer_ci_upper: float = 0.15


def test_accuracy_plot_adapts_to_models_and_metric_ranges() -> None:
    rows = [
        PlotRow("model-beta", 0.95, 1.43, 3.8, 1 / 3.8),
        PlotRow("model-alpha", 0.20, 0.06, 45.4, 1 / 45.4),
    ]
    svg = render_accuracy_svg("future-suite", rows, {"model-alpha": "organization/model-alpha"})
    assert 'width="900" height="156"' in svg
    assert "PESTE normalized accuracy leaderboard" in svg
    assert svg.index(">model-alpha</text>") < svg.index(">model-beta</text>")
    assert "150%" in svg
    assert "https://huggingface.co/organization/model-alpha" in svg


def test_accuracy_plot_uses_cer_ranking_and_presents_cer_first() -> None:
    rows = [
        PlotRow("wer-leader", 0.10, 0.20, 10.0, 0.1),
        PlotRow("cer-leader", 0.20, 0.10, 10.0, 0.1),
    ]
    svg = render_accuracy_svg("future-suite", rows)
    assert svg.index(">cer-leader</text>") < svg.index(">wer-leader</text>")
    assert svg.index("CER ↓") < svg.index("WER ↓")


def test_speed_plot_ranks_throughput_and_excludes_invalid_runs() -> None:
    rows = [
        PlotRow("accuracy-leader", 0.10, 0.05, 2.0, 0.5),
        PlotRow("speed-leader", 0.20, 0.10, 20.0, 0.05),
        PlotRow("resumed", 0.01, 0.01, 100.0, 0.01, speed_valid=False),
    ]
    svg = render_speed_svg("future-suite", rows)
    assert "PESTE steady-state speed leaderboard" in svg
    assert svg.index(">speed-leader</text>") < svg.index(">accuracy-leader</text>")
    assert ">resumed</text>" not in svg
    assert "Audio throughput ↑" in svg
    assert "RTF ↓" in svg


def test_empty_plot_has_an_accessible_state() -> None:
    svg = render_accuracy_svg("empty-suite", [])
    assert "0 ranked models from empty-suite" in svg
    assert "No complete official results yet" in svg


def test_pareto_plot_renders_frontier_dominance_and_uncertainty() -> None:
    rows = [
        PlotRow("fast-frontier", 0.20, 0.20, 100.0, 0.01, cer_ci_lower=0.18, cer_ci_upper=0.22),
        PlotRow("accurate-frontier", 0.10, 0.05, 10.0, 0.1, cer_ci_lower=0.04, cer_ci_upper=0.06),
        PlotRow("dominated", 0.30, 0.30, 5.0, 0.2, cer_ci_lower=0.25, cer_ci_upper=0.35),
        PlotRow("resumed", 0.01, 0.01, 500.0, 0.002, speed_valid=False),
    ]
    svg = render_pareto_svg(
        "future-suite",
        rows,
        {"fast-frontier", "accurate-frontier"},
        {"dominated"},
    )
    assert "PESTE accuracy-speed Pareto efficiency" in svg
    assert "Audio throughput (× real time, log scale)" in svg
    assert "Accuracy (CER ↓, inverted log scale)" in svg
    assert 'text-anchor="middle">↑</text>' in svg
    assert "Ideal direction: upper-right ↗" in svg
    assert 'class="frontier-line"' in svg
    assert 'class="point frontier"' in svg
    assert 'class="point supported-dominated"' in svg
    assert 'class="error-bar"' in svg
    assert ">resumed</text>" not in svg
    root = ElementTree.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    point_y = {
        title.text.split(":", maxsplit=1)[0]: float(circle.attrib["cy"])
        for circle in root.findall(".//svg:circle", namespace)
        if (title := circle.find("svg:title", namespace)) is not None and title.text is not None
    }
    assert point_y["accurate-frontier"] < point_y["fast-frontier"]
