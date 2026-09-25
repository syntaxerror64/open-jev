"""Unit tests for the benchmark statistics collector (stage 1, agent A)."""

import math

import pytest

from benchmarks.stats import Stats, summarize


def test_summarize_known_values() -> None:
    s = summarize([5.0, 1.0, 3.0])
    assert (s.n, s.min, s.median, s.max) == (3, 1.0, 3.0, 5.0)
    assert s.mean == pytest.approx(3.0)
    assert s.std == pytest.approx(math.sqrt(8 / 3))  # population std


def test_summarize_is_order_independent() -> None:
    assert summarize([1.0, 2.0, 3.0]) == summarize([3.0, 1.0, 2.0])


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize([])
