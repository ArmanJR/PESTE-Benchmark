"""Corpus-level ASR error metrics."""

from collections.abc import Sequence
from dataclasses import dataclass

from peste.normalization import normalize


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            raise ValueError("Cannot calculate an error rate with an empty reference")
        return self.errors / self.reference_units

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_units=self.reference_units + other.reference_units,
        )


@dataclass(frozen=True, slots=True)
class SampleScore:
    normalized_reference: str
    normalized_prediction: str
    words: EditCounts
    characters: EditCounts


@dataclass(frozen=True, slots=True)
class CorpusScore:
    words: EditCounts
    characters: EditCounts

    @property
    def wer(self) -> float:
        return self.words.rate

    @property
    def cer(self) -> float:
        return self.characters.rate


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return Levenshtein operation counts with deterministic tie resolution."""
    rows: list[list[tuple[int, int, int, int]]] = [
        [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    ]
    for ref_index in range(1, len(reference) + 1):
        row: list[tuple[int, int, int, int]] = [(ref_index, 0, ref_index, 0)]
        for hyp_index in range(1, len(hypothesis) + 1):
            if reference[ref_index - 1] == hypothesis[hyp_index - 1]:
                row.append(rows[ref_index - 1][hyp_index - 1])
                continue
            diagonal = rows[ref_index - 1][hyp_index - 1]
            above = rows[ref_index - 1][hyp_index]
            left = row[hyp_index - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2] + 1, above[3]),
                (left[0] + 1, left[1], left[2], left[3] + 1),
            )
            row.append(min(candidates, key=lambda value: (value[0], value[1], value[2], value[3])))
        rows.append(row)
    _, substitutions, deletions, insertions = rows[-1][-1]
    return EditCounts(substitutions, deletions, insertions, len(reference))


def score_sample(reference: str, prediction: str, version: str = "fa-v1") -> SampleScore:
    normalized_reference = normalize(reference, version)
    normalized_prediction = normalize(prediction, version)
    if not normalized_reference:
        raise ValueError("Normalized reference must not be empty")
    reference_characters = list(normalized_reference.replace(" ", ""))
    prediction_characters = list(normalized_prediction.replace(" ", ""))
    return SampleScore(
        normalized_reference=normalized_reference,
        normalized_prediction=normalized_prediction,
        words=edit_counts(normalized_reference.split(), normalized_prediction.split()),
        characters=edit_counts(reference_characters, prediction_characters),
    )


def aggregate_scores(samples: Sequence[SampleScore]) -> CorpusScore:
    if not samples:
        raise ValueError("At least one scored sample is required")
    empty = EditCounts(0, 0, 0, 0)
    words = empty
    characters = empty
    for sample in samples:
        words += sample.words
        characters += sample.characters
    return CorpusScore(words=words, characters=characters)


def word_accuracy_pct(wer: float) -> float:
    return 100.0 * max(0.0, 1.0 - wer)
