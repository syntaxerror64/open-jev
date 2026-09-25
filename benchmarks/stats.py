"""Statistics collector and timer helpers for the stage-1 benchmarks.

Everything the benchmark layer needs to turn raw `perf_counter` samples into a
reportable row lives here:

    Stats       -- frozen summary of a set of samples (ms)
    summarize   -- samples -> Stats, population std via statistics.pstdev
    repeat      -- warmup + n timed runs of a callable, in milliseconds
    median_of   -- the noise-robust comparison statistic (risk R1)
    time_ms     -- a single timed call, in milliseconds

Why medians: CPU timings on a shared machine are noisy (risk R1). Comparisons
in the tests and in the CLI use `median_of` / `Stats.median` so that one OS
preemption cannot flip an assertion.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Stats:
    """Summary of `n` timing samples, all in milliseconds.

    `std` is the *population* standard deviation (`statistics.pstdev`), not the
    sample one: a benchmark run reports every repeat it made, so the samples
    are the whole population of interest.
    """

    n: int
    min: float
    median: float
    mean: float
    std: float
    max: float


def summarize(samples: Sequence[float]) -> Stats:
    """Reduce timing samples to a `Stats`.

    Raises:
        ValueError: if `samples` is empty -- a summary of nothing is a bug in
            the caller (missing repeats/warmup), not a zero-valued result.
    """
    if not samples:
        raise ValueError("summarize() needs at least one sample")
    values = [float(x) for x in samples]
    return Stats(
        n=len(values),
        min=min(values),
        median=float(statistics.median(values)),
        mean=float(statistics.fmean(values)),
        std=float(statistics.pstdev(values)),  # population std
        max=max(values),
    )


def repeat(
    fn: Callable[[], object],
    *,
    n: int = 7,
    warmup: int = 2,
) -> list[float]:
    """Run `fn` `warmup` times untimed, then `n` times timed.

    Returns the `n` durations in milliseconds. Warmup exists so that lazy
    allocations, one-time op registration and thread pools settle before the
    first timed sample (risk R1).

    Raises:
        ValueError: if `n < 1` or `warmup < 0`.
    """
    if n < 1:
        raise ValueError(f"repeat() needs n >= 1, got {n}")
    if warmup < 0:
        raise ValueError(f"repeat() needs warmup >= 0, got {warmup}")
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def median_of(samples: Sequence[float]) -> float:
    """Median of timing samples (ms); the statistic all comparisons use."""
    if not samples:
        raise ValueError("median_of() needs at least one sample")
    return float(statistics.median(samples))


def time_ms(fn: Callable[[], object]) -> float:
    """Time one call of `fn`, in milliseconds."""
    return repeat(fn, n=1, warmup=0)[0]
