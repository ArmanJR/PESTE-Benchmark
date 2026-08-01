"""Deterministic bootstrap uncertainty tests."""

import pytest

from peste.uncertainty import bootstrap_paired_rate_difference, bootstrap_rate


def test_bootstrap_rate_is_deterministic() -> None:
    first = bootstrap_rate([0, 2, 1], [5, 10, 5], seed=7, replicates=2_000)
    second = bootstrap_rate([0, 2, 1], [5, 10, 5], seed=7, replicates=2_000)

    assert first == second
    assert first.point == pytest.approx(3 / 20)
    assert first.lower <= first.point <= first.upper


def test_paired_bootstrap_distinguishes_resolved_and_unresolved_differences() -> None:
    resolved = bootstrap_paired_rate_difference([0, 0], [1, 2], [10, 20], seed=11, replicates=2_000)
    unresolved = bootstrap_paired_rate_difference([0, 1], [1, 0], [1, 1], seed=11, replicates=2_000)

    assert resolved.point == pytest.approx(-0.1)
    assert resolved.upper < 0
    assert unresolved.lower < 0 < unresolved.upper


def test_bootstrap_rejects_invalid_sample_counts() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        bootstrap_rate([1], [1, 2], seed=1, replicates=100)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_rate([1], [0], seed=1, replicates=100)
