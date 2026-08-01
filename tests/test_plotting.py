"""Deterministic, data-driven leaderboard plot tests."""

from dataclasses import dataclass

from peste.plotting import render_accuracy_svg, render_memory_svg


@dataclass(frozen=True, slots=True)
class PlotRow:
    model_id: str
    wer: float
    cer: float
    memory_efficiency: float
    peak_cuda_reserved_gib: float


def test_accuracy_plot_adapts_to_models_and_metric_ranges() -> None:
    rows = [
        PlotRow("model-beta", 0.95, 1.43, 3.8, 19.05),
        PlotRow("model-alpha", 0.20, 0.06, 45.4, 1.75),
    ]

    svg = render_accuracy_svg(
        "future-suite",
        rows,
        {"model-alpha": "organization/model-alpha"},
    )

    assert 'width="900" height="156"' in svg
    assert "PESTE normalized accuracy leaderboard" in svg
    assert svg.index(">model-alpha</text>") < svg.index(">model-beta</text>")
    assert "150%" in svg
    assert ">95.0%</text>" in svg
    assert "https://huggingface.co/organization/model-alpha" in svg


def test_memory_plot_uses_efficiency_ranking() -> None:
    rows = [
        PlotRow("accuracy-leader", 0.10, 0.05, 2.0, 2.0),
        PlotRow("efficiency-leader", 0.20, 0.10, 20.0, 1.0),
    ]

    svg = render_memory_svg("future-suite", rows)

    assert "PESTE accuracy per peak CUDA memory leaderboard" in svg
    assert svg.index(">efficiency-leader</text>") < svg.index(">accuracy-leader</text>")
    assert "Peak CUDA reserved ↓" in svg
    assert "Word accuracy / GiB ↑" in svg


def test_empty_plot_has_an_accessible_state() -> None:
    svg = render_accuracy_svg("empty-suite", [])

    assert "0 ranked models from empty-suite" in svg
    assert "No complete official results yet" in svg
