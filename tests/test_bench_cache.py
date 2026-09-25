"""Cache reuse benchmarks: cold encode vs cache hit (stage 1, agent A).

Spec: stage 1, step 2. Two theses under test:

1. `encode_state` (open_jev/main.py, O(T^2), paid once) is more expensive than
   `_readout` (O(M*T*N) per question batch) -- but only for *large* T and
   *small* N (risk R3), hence T~400 and N=4 here.
2. One encode + N readouts beats N full forward passes (README todo #3).

Measurement hygiene (risk R1): `model.eval()`, `torch.no_grad()`, warmup runs
before timed runs, >= 5 repeats, comparisons on the MEDIAN, and torch pinned to
one thread for the duration of each test (restored afterwards).

Note on T: the shared TINY model from conftest.py caps states at
`max_state_len=128`; at that length the O(T^2) advantage disappears (R3), so
test 1 builds a local model that keeps every TINY hyperparameter but raises the
state cap to fit T ~ 400. Test 2 needs no such override: its inequality holds
for any positive timings.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from benchmarks.stats import median_of, repeat, summarize
from open_jev.main import (
    Choice,
    HashTokenizer,
    Jev,
    Noul,
    Score,
    flatten_state,
)

REPEATS = 7  # >= 5 per risk R1
WARMUP = 2  # >= 2 per risk R1
TARGET_T = 400  # "large T" per step 2 / risk R3
LONG_STATE_CAP = 512  # must fit TARGET_T; TINY's own cap (128) cannot
LONG_T_MIN = 350  # sanity floor: the state really is "long"


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    """Pin torch to one core per test (risk R1), restoring the previous value."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def long_model(cfg) -> Jev:
    """TINY model (conftest) with only the state cap raised to fit T ~ 400."""
    torch.manual_seed(0)
    return Jev(replace(cfg, max_state_len=LONG_STATE_CAP)).eval()


@pytest.fixture(scope="module")
def long_state(cfg) -> dict:
    """Deterministic state flattening to >= TARGET_T tokens (no RNG involved)."""
    tokenizer = HashTokenizer(cfg.vocab_size)
    records: list[dict] = []
    while True:
        i = len(records)
        records.append(
            {
                "id": f"r{i}",
                "amount": round(i * 1.5, 2),
                "note": "duplicate charge dispute pending review",
            }
        )
        state = {"records": records}
        ids, _ = flatten_state(state, tokenizer, LONG_STATE_CAP)
        if len(ids) >= TARGET_T:
            return state


@pytest.fixture(scope="module")
def long_questions() -> list:
    """N = 4 questions: the small-N regime (risk R3)."""
    return [
        Noul("The customer is requesting a refund.", key="wants_refund"),
        Choice(
            "Which team should handle this?",
            options=["billing", "technical", "account"],
            key="route",
        ),
        Score(
            "How frustrated is the customer?",
            labels=["calm", "annoyed", "frustrated", "very frustrated"],
            key="frustration",
        ),
        Noul("This charge was already refunded.", key="already_refunded"),
    ]


def test_cache_hit_is_cheaper_than_cold_encode(
    long_model, long_state, long_questions
) -> None:
    # encode_state (main.py:634) is O(T^2); _readout (main.py:664) is
    # O(M*T*N). At large T and small N the cold encode must cost more.
    long_model.eval()
    with torch.no_grad():
        t_enc = median_of(
            repeat(
                lambda: long_model.encode_state([long_state]),
                n=REPEATS,
                warmup=WARMUP,
            )
        )
        cache = long_model.encode_state([long_state])
        t_hit = median_of(
            repeat(
                lambda: long_model._readout(cache, long_questions),
                n=REPEATS,
                warmup=WARMUP,
            )
        )
        # Both timers must describe the same input: a fresh cold encode has to
        # produce a cache that reads out bit-identically to the timed one.
        fresh = long_model.encode_state([long_state])
        assert torch.equal(
            long_model._readout(fresh, long_questions),
            long_model._readout(cache, long_questions),
        )

    T = cache.hidden.shape[1]
    assert LONG_T_MIN <= T <= long_model.cfg.max_state_len, T
    assert t_hit < t_enc, (
        f"cache hit {t_hit:.3f} ms not cheaper than cold encode {t_enc:.3f} ms "
        f"at T={T}, N={len(long_questions)}"
    )


def test_caching_n_queries_beats_n_full_forwards(model, state, questions) -> None:
    # cached = 1 encode + N readouts ; uncached = N full forward passes.
    # The inequality is algebraic in the shared medians, but the second assert
    # measures real forwards, so it guards the API against a regression that
    # would silently re-encode inside the readout path.
    model.eval()
    n = len(questions)
    with torch.no_grad():
        enc = summarize(
            repeat(lambda: model.encode_state([state]), n=REPEATS, warmup=WARMUP)
        )
        cache = model.encode_state([state])
        hit = summarize(
            repeat(lambda: model._readout(cache, questions), n=REPEATS, warmup=WARMUP)
        )
        full = summarize(
            repeat(lambda: model([state], questions), n=REPEATS, warmup=WARMUP)
        )

    cached_total_ms = enc.median + n * hit.median
    assert cached_total_ms < n * (enc.median + hit.median), (
        f"cached {cached_total_ms:.3f} ms vs N*(encode+readout) "
        f"{n * (enc.median + hit.median):.3f} ms"
    )
    assert cached_total_ms < n * full.median, (
        f"cached {cached_total_ms:.3f} ms vs {n} full forwards "
        f"{n * full.median:.3f} ms"
    )
