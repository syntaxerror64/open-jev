"""Deterministic synthetic inputs for the stage-1 benchmarks (agent B).

Spec: stage 1, contract section.

    make_state(total_tokens, seed=0)  -> dict   # flattens to exactly T tokens
    make_questions(n)                 -> list    # Noul / Choice / Score mix

Both generators are pure functions: the same arguments always produce the
same objects, so two runs of `python -m benchmarks.cli` compare like with
like (risk R1, fixed seed).

State length is verified with the REAL `flatten_state` from `open_jev.main` --
the very function `Jev.encode_state` runs -- so "T tokens" in the report is
the token count the model actually sees, not an estimate from our own rules.
"""

from __future__ import annotations

import random

from open_jev.main import (
    Choice,
    HashTokenizer,
    Noul,
    Question,
    Score,
    flatten_state,
)

# Any vocab size works for *counting* tokens: HashTokenizer.encode() emits one
# id per whitespace word no matter the vocab size, so the flattened length is
# independent of it. 32_000 is JevConfig's default vocabulary.
_BUILD_VOCAB = 32_000

_WORDS = (
    "duplicate charge dispute pending review refund transaction customer "
    "invoice payment retry escalated resolved ledger settlement chargeback "
    "receipt policy entitlement escalation"
).split()

_STATUSES = ("captured", "pending", "refunded", "disputed")


def make_state(total_tokens: int, seed: int = 0) -> dict:
    """A deterministic (by `seed`) state that flattens to `total_tokens` tokens.

    The state is a growing list of synthetic ledger records. After every
    record we re-flatten with the real `flatten_state`, capped at exactly
    `total_tokens`; because the cap equals the target, reaching
    `len(ids) >= total_tokens` means the state really is `total_tokens`
    tokens long after `Jev.encode_state`'s own truncation. Every record
    contributes at least one token, so the loop always terminates.

    Raises:
        ValueError: if `total_tokens < 1`.
    """
    if total_tokens < 1:
        raise ValueError(f"make_state() needs total_tokens >= 1, got {total_tokens}")

    rng = random.Random(seed)
    tokenizer = HashTokenizer(_BUILD_VOCAB)
    records: list[dict] = []
    while True:
        i = len(records)
        records.append(
            {
                "id": f"r{seed}-{i}",
                "amount": round(rng.uniform(1.0, 999.99), 2),
                "status": rng.choice(_STATUSES),
                "note": " ".join(
                    rng.choice(_WORDS) for _ in range(rng.randint(2, 5))
                ),
            }
        )
        state = {"seed": seed, "records": records}
        ids, _ = flatten_state(state, tokenizer, total_tokens)
        if len(ids) >= total_tokens:
            return state


def make_questions(n: int) -> list[Question]:
    """A deterministic mix of the three primitives: Noul / Choice / Score.

    Cycling over the types keeps every read-out head exercised regardless of
    `n` (for `n >= 3`), and keys `q0, q1, ...` stay unique so answers are
    attributable.

    Raises:
        ValueError: if `n < 1`.
    """
    if n < 1:
        raise ValueError(f"make_questions() needs n >= 1, got {n}")

    questions: list[Question] = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            questions.append(
                Noul(f"Statement {i} about the record is true.", key=f"q{i}")
            )
        elif kind == 1:
            questions.append(
                Choice(
                    f"Which action fits record {i}?",
                    options=["refund", "retry", "escalate"],
                    key=f"q{i}",
                )
            )
        else:
            questions.append(
                Score(
                    f"How urgent is record {i}?",
                    labels=["low", "normal", "high", "critical"],
                    key=f"q{i}",
                )
            )
    return questions
