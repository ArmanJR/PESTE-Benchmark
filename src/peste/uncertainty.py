"""Deterministic utterance-bootstrap uncertainty for corpus error rates."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    point: float
    lower: float
    upper: float


def _validated_counts(
    errors: Sequence[int], reference_units: Sequence[int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    error_array = np.asarray(errors, dtype=np.int64)
    reference_array = np.asarray(reference_units, dtype=np.int64)
    if (
        error_array.ndim != 1
        or reference_array.ndim != 1
        or len(error_array) != len(reference_array)
    ):
        raise ValueError("Errors and reference units must have the same non-zero length")
    if len(error_array) == 0:
        raise ValueError("Errors and reference units must have the same non-zero length")
    if np.any(error_array < 0):
        raise ValueError("Error counts must be non-negative")
    if np.any(reference_array <= 0):
        raise ValueError("Reference units must be positive")
    return error_array, reference_array


def _interval(values: NDArray[np.float64], confidence_level: float) -> tuple[float, float]:
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


@lru_cache(maxsize=16)
def _bootstrap_indices(sample_count: int, replicates: int, seed: int) -> NDArray[np.int64]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    generator = np.random.default_rng(seed)
    return generator.integers(0, sample_count, size=(replicates, sample_count), dtype=np.int64)


@lru_cache(maxsize=256)
def _sampled_rates(
    errors: tuple[int, ...],
    reference_units: tuple[int, ...],
    replicates: int,
    seed: int,
) -> NDArray[np.float64]:
    error_array, reference_array = _validated_counts(errors, reference_units)
    indices = _bootstrap_indices(len(error_array), replicates, seed)
    return error_array[indices].sum(axis=1) / reference_array[indices].sum(axis=1)


def bootstrap_rate(
    errors: Sequence[int],
    reference_units: Sequence[int],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> BootstrapEstimate:
    """Estimate a percentile interval for one corpus error rate."""
    error_array, reference_array = _validated_counts(errors, reference_units)
    sampled_rates = _sampled_rates(
        tuple(int(value) for value in error_array),
        tuple(int(value) for value in reference_array),
        replicates,
        seed,
    )
    lower, upper = _interval(sampled_rates, confidence_level)
    estimate = BootstrapEstimate(
        point=float(error_array.sum() / reference_array.sum()),
        lower=lower,
        upper=upper,
    )
    LOGGER.debug(
        "Calculated bootstrap corpus error-rate interval",
        extra={
            "samples": len(error_array),
            "replicates": replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "point": estimate.point,
            "lower": estimate.lower,
            "upper": estimate.upper,
        },
    )
    return estimate


def bootstrap_paired_rate_difference(
    first_errors: Sequence[int],
    second_errors: Sequence[int],
    reference_units: Sequence[int],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> BootstrapEstimate:
    """Estimate a percentile interval for a paired corpus-rate difference."""
    first_array, reference_array = _validated_counts(first_errors, reference_units)
    second_array, second_reference = _validated_counts(second_errors, reference_units)
    if not np.array_equal(reference_array, second_reference):
        raise ValueError("Paired systems must use identical reference units")
    first_rates = _sampled_rates(
        tuple(int(value) for value in first_array),
        tuple(int(value) for value in reference_array),
        replicates,
        seed,
    )
    second_rates = _sampled_rates(
        tuple(int(value) for value in second_array),
        tuple(int(value) for value in reference_array),
        replicates,
        seed,
    )
    sampled_differences = first_rates - second_rates
    lower, upper = _interval(sampled_differences, confidence_level)
    estimate = BootstrapEstimate(
        point=float((first_array.sum() - second_array.sum()) / reference_array.sum()),
        lower=lower,
        upper=upper,
    )
    LOGGER.debug(
        "Calculated paired bootstrap error-rate difference",
        extra={
            "samples": len(first_array),
            "replicates": replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "point": estimate.point,
            "lower": estimate.lower,
            "upper": estimate.upper,
        },
    )
    return estimate
