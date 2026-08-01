"""Corpus WER/CER and efficiency contract tests."""

import pytest

from psst.metrics import (
    aggregate_scores,
    edit_counts,
    memory_efficiency,
    score_sample,
    word_accuracy_pct,
)


def test_edit_operation_counts() -> None:
    counts = edit_counts(["a", "b", "c"], ["a", "x", "c", "d"])
    assert counts.substitutions == 1
    assert counts.deletions == 0
    assert counts.insertions == 1
    assert counts.reference_units == 3
    assert counts.rate == pytest.approx(2 / 3)


def test_corpus_rates_sum_counts_instead_of_averaging_samples() -> None:
    corpus = aggregate_scores([score_sample("یک دو سه", "یک"), score_sample("چهار", "پنج")])
    assert corpus.words.errors == 3
    assert corpus.words.reference_units == 4
    assert corpus.wer == pytest.approx(0.75)
    assert corpus.characters.reference_units == len("یکدوسهچهار")


def test_cer_removes_whitespace() -> None:
    score = score_sample("ا ب", "اب")
    assert score.characters.errors == 0


def test_empty_reference_rejected_and_empty_prediction_permitted() -> None:
    with pytest.raises(ValueError, match="Normalized reference"):
        score_sample("...", "متن")
    score = score_sample("یک دو", "")
    assert score.words.deletions == 2


def test_memory_efficiency_formula_and_floor() -> None:
    assert word_accuracy_pct(0.25) == 75.0
    assert memory_efficiency(0.25, 5.0) == 15.0
    assert memory_efficiency(1.5, 2.0) == 0.0
    with pytest.raises(ValueError, match="greater than zero"):
        memory_efficiency(0.1, 0)
