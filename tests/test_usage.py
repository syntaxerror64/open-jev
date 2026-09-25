"""Token accounting for stage 12: `Jev.usage(states, questions) -> list[Usage]`.

Stage 12 (Шаг 1) pins their envelope: every request
carries `usage: {input_tokens, output_tokens}`. Ours is one `Usage` per state,
symmetrically with `forward`, counting the input it really eats -- the flattened
state plus every part of every question (text, Choice options, Score labels) --
and reporting `output_tokens == 0`, because System One generates no text at all.

Written RED first (TDD): neither `Usage` nor `Jev.usage` exists yet, so every
test dies at its first reference to either.
"""

from __future__ import annotations

import torch

from open_jev import Choice, Score, flatten_state


def _question_tokens(model, questions) -> int:
    """Tokens the questions cost, built only from the public tokenizer surface.

    `forward` encodes question text in `_readout` and Choice options in
    `_encode_options`, both via `tokenizer.encode_batch(..., max_question_len)`.
    `encode_batch` only right-pads its rows, so one string's own cost is
    `len(tokenizer.encode(s, max_question_len))` -- the Tokenizer protocol's
    single-string entry point, not a re-implementation of it. Score labels go
    through the same `encode` call (risk R2: the count must include options AND
    labels of Choice/Score, or it understates what the questions contribute).
    """
    tok = model.tokenizer
    max_len = model.cfg.max_question_len
    total = 0
    for q in questions:
        total += len(tok.encode(q.text, max_len))
        if isinstance(q, Choice):
            total += sum(len(tok.encode(o, max_len)) for o in q.options)
        elif isinstance(q, Score):
            total += sum(len(tok.encode(lbl, max_len)) for lbl in q.labels)
    return total


def test_usage_returns_one_entry_per_state(model, state, questions) -> None:
    """N states in, N `Usage`s out -- row-wise symmetry with `forward`."""
    # Pinned import path: Шаг 2 defines `Usage` in `open_jev/main.py` AND
    # re-exports it in `open_jev/__init__.py` next to the Answer types, which
    # is where stage-11's contract test imports its wire types from too.
    from open_jev import Usage

    other = {"ticket": {"status": "open"}}
    usages = model.usage([state, other], questions)

    assert isinstance(usages, list)
    assert len(usages) == 2
    assert all(isinstance(u, Usage) for u in usages)


def test_usage_fields_and_zero_output(model, state, questions) -> None:
    """System One generates nothing: input counted, output always exactly zero."""
    (usage,) = model.usage([state], questions)

    assert type(usage.input_tokens) is int
    assert type(usage.output_tokens) is int
    assert usage.input_tokens > 0
    assert usage.output_tokens == 0


def test_usage_input_equals_manual_token_count(model, state, questions) -> None:
    """Pin the semantics: state + every question part, from the public surface.

    The expectation is assembled from the very calls the real encoding path
    makes -- `flatten_state(state, tokenizer, cfg.max_state_len)`, exactly what
    `Jev.encode_state` runs per state, for the state side; `tokenizer.encode`
    for question text, Choice options and Score labels. Nothing about the
    internals is copied twice, so the pin stays honest: same input, one number,
    and that number is NOT just the state.
    """
    other = {"ticket": {"status": "open"}}
    states = [state, other]

    # Real tokens per state; PAD is a batching artifact, not input text, so it
    # is never counted.
    state_tokens = [
        len(flatten_state(s, model.tokenizer, model.cfg.max_state_len)[0])
        for s in states
    ]
    question_tokens = _question_tokens(model, questions)
    assert question_tokens > 0  # the questions contribute: it is NOT just the state

    usages = model.usage(states, questions)

    assert [u.input_tokens for u in usages] == [
        n + question_tokens for n in state_tokens
    ]


def test_usage_is_pure_and_deterministic(model, state, questions) -> None:
    """Usage only counts: no mutation, no growth, no effect in either direction."""
    states = [state, {"ticket": {"status": "open"}}]

    with torch.no_grad():
        before = model.forward(states, questions)

    first = model.usage(states, questions)
    second = model.usage(states, questions)
    assert first == second

    with torch.no_grad():
        after = model.forward(states, questions)
    assert after == before  # usage left the model and its answers untouched

    assert model.usage(states, questions) == first  # forward left usage untouched
