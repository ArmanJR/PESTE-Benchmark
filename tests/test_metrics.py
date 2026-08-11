"""Corpus WER/CER contract tests."""

import pytest

from peste.metrics import aggregate_scores, edit_counts, score_sample, word_accuracy_pct


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


def test_fa_v2_scores_digit_and_word_number_styles_equally() -> None:
    score = score_sample(
        "جنگ شش\u200cروزه ۱۹۶۷ است",
        "جنگ شش روزه هزار و نهصد و شصت و هفت است",
        version="fa-v2",
    )
    assert score.normalized_reference == score.normalized_prediction
    assert score.words.errors == 0
    assert score.characters.errors == 0


def test_empty_reference_rejected_and_empty_prediction_permitted() -> None:
    with pytest.raises(ValueError, match="Normalized reference"):
        score_sample("...", "متن")
    score = score_sample("یک دو", "")
    assert score.words.deletions == 2


def test_word_accuracy_floor() -> None:
    assert word_accuracy_pct(0.25) == 75.0
    assert word_accuracy_pct(1.5) == 0.0
