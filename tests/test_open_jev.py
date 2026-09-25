"""Structural guarantees of open_jev, run as a push gate.

These tests assert what the architecture claims to guarantee *by construction*:
typed answers, distributions that sum to one, question isolation, cache reuse,
and deterministic tokenization. The weights are random, so nothing here asserts
semantic correctness -- only structure.
"""

from __future__ import annotations

import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

from open_jev.main import (
    Choice,
    ChoiceAnswer,
    HashTokenizer,
    JevConfig,
    Noul,
    NoulAnswer,
    RLCDLoss,
    Score,
    ScoreAnswer,
    flatten_state,
    stable_hash,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-5


# --- input validation ------------------------------------------------------


def test_config_requires_divisible_heads() -> None:
    with pytest.raises(ValueError):
        JevConfig(d_model=65, n_heads=4)


def test_choice_needs_at_least_two_options() -> None:
    with pytest.raises(ValueError):
        Choice("Pick", options=["only"], key="k")


def test_choice_options_must_be_unique() -> None:
    with pytest.raises(ValueError):
        Choice("Pick", options=["a", "a"], key="k")


def test_score_needs_at_least_two_levels() -> None:
    with pytest.raises(ValueError):
        Score("Rate", labels=["low"], key="k")


# --- tokenizer / state flattening ------------------------------------------


def test_stable_hash_is_crc32_and_process_stable() -> None:
    assert stable_hash("refund") == zlib.crc32(b"refund")
    assert stable_hash("refund") == stable_hash("refund")


def test_tokenizer_ids_stay_inside_vocab() -> None:
    tok = HashTokenizer(512)
    ids = tok.encode("the customer wants a refund now", max_len=6)
    assert 1 <= len(ids) <= 6
    assert all(1 <= i < 512 for i in ids)
    # An empty string still yields one token rather than an empty sequence.
    assert tok.encode("", max_len=8) == [1]


def test_flatten_paths_stay_aligned_and_respect_max_len(state) -> None:
    tok = HashTokenizer(512)
    ids, paths = flatten_state(state, tok, max_len=17)
    assert len(ids) == len(paths)
    assert len(ids) <= 17
    assert all(len(p) == 3 for p in paths)


def test_key_order_preserves_token_and_path_identity(state) -> None:
    """Shuffling dict keys keeps every token's (depth, path hash) identity.

    Sibling indices move with the reorder by design; the content-addressed
    path hash is the invariant that makes a key recognisable regardless of
    where it sits in the object.
    """
    tok = HashTokenizer(512)
    shuffled = {k: state[k] for k in reversed(list(state.keys()))}

    ids_a, paths_a = flatten_state(state, tok, 128)
    ids_b, paths_b = flatten_state(shuffled, tok, 128)

    assert sorted(ids_a) == sorted(ids_b)
    assert sorted((i, d, h) for i, (d, _, h) in zip(ids_a, paths_a)) == sorted(
        (i, d, h) for i, (d, _, h) in zip(ids_b, paths_b)
    )


# --- typed answers ---------------------------------------------------------


def test_forward_returns_one_typed_answer_per_question_per_state(
    model, state, questions
) -> None:
    other = {"ticket": {"status": "open"}}
    with torch.no_grad():
        results = model([state, other], questions)

    assert len(results) == 2
    for row in results:
        assert len(row) == len(questions)
        assert [type(a) for a in row] == [NoulAnswer, ChoiceAnswer, ScoreAnswer]
        assert [a.key for a in row] == ["wants_refund", "route", "frustration"]


def test_choice_answer_is_always_a_declared_option(model, state) -> None:
    q = Choice("Route it", options=["billing", "technical", "account"], key="r")
    with torch.no_grad():
        answer = model([state], [q])[0][0]

    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice in q.options
    assert set(answer.probabilities) == set(q.options)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=TOL)
    assert all(0.0 <= p <= 1.0 for p in answer.probabilities.values())


def test_noul_probability_lies_in_the_unit_interval(model, state, questions) -> None:
    with torch.no_grad():
        answers = model([state], questions)[0]
    for answer in answers:
        if isinstance(answer, NoulAnswer):
            assert 0.0 <= answer.noul <= 1.0


def test_score_distribution_sums_to_one_and_score_is_its_expectation(
    model, state
) -> None:
    labels = ["calm", "annoyed", "frustrated", "very frustrated"]
    q = Score("How frustrated?", labels=labels, key="f")
    with torch.no_grad():
        answer = model([state], [q])[0][0]

    assert isinstance(answer, ScoreAnswer)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=TOL)
    expectation = sum(i * p for i, p in enumerate(answer.probabilities.values()))
    assert answer.score == pytest.approx(expectation, abs=TOL)
    assert 0.0 <= answer.score <= len(labels) - 1


def test_confidence_is_epistemic_and_bounded(model, state, questions) -> None:
    with torch.no_grad():
        answers = model([state], questions)[0]
    confidences = [a.confidence for a in answers if not isinstance(a, NoulAnswer)]
    assert confidences
    assert all(0.0 <= c <= 1.0 for c in confidences)


# --- isolation and caching -------------------------------------------------


def test_questions_do_not_influence_each_other(model, state, questions) -> None:
    """Asking a question alone or alongside others must give the same answer."""
    route = questions[1]
    with torch.no_grad():
        alone = model([state], [route])[0][0]
        together = model([state], questions)[0][1]

    assert alone.choice == together.choice
    for key in alone.probabilities:
        assert alone.probabilities[key] == pytest.approx(
            together.probabilities[key], abs=TOL
        )
    assert alone.confidence == pytest.approx(together.confidence, abs=TOL)


def test_encoded_state_is_cacheable_and_reusable(model, state, questions) -> None:
    with torch.no_grad():
        cache = model.encode_state([state])
        first = model._readout(cache, questions)
        second = model._readout(cache, questions)

    assert cache.batch_size == 1
    assert torch.equal(first, second)


def test_empty_state_is_accepted_without_nan(model) -> None:
    questions = [
        Noul("Anything here?", key="n"),
        Choice("Route", options=["a", "b"], key="c"),
        Score("Rate", labels=["low", "high"], key="s"),
    ]
    with torch.no_grad():
        answers = model([{}], questions)[0]

    assert len(answers) == 3
    for answer in answers:
        values = (
            [answer.noul]
            if isinstance(answer, NoulAnswer)
            else list(answer.probabilities.values())
        )
        assert all(math_isfinite(v) for v in values)


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


# --- training path ---------------------------------------------------------


def test_noul_logits_are_binary_and_normalized(model, state, questions) -> None:
    with torch.no_grad():
        outputs = model.logits([state], questions)
    for (probs, conf), question in zip(outputs, questions):
        expected_k = 2 if isinstance(question, Noul) else (
            len(question.options) if isinstance(question, Choice)
            else len(question.labels)
        )
        assert probs.shape == (1, expected_k)
        assert probs.sum().item() == pytest.approx(1.0, abs=TOL)
        assert conf.shape == (1,)
        assert 0.0 <= conf.item() <= 1.0


def test_rlcd_loss_is_finite_and_differentiable(model, state, questions) -> None:
    shuffled = {k: state[k] for k in reversed(list(state.keys()))}
    targets = [
        torch.tensor([[0.08, 0.92]]),
        torch.tensor([[0.80, 0.05, 0.15]]),
        torch.tensor([[0.02, 0.10, 0.48, 0.40]]),
    ]

    model.train()
    try:
        loss = RLCDLoss()(model, [state], questions, targets, augmented_states=[shuffled])
        loss.backward()
    finally:
        model.eval()

    assert loss.item() == loss.item() and abs(loss.item()) != float("inf")
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


# --- shipped demos ---------------------------------------------------------


@pytest.mark.parametrize("script", ["forward.py", "example.py"])
def test_shipped_demo_scripts_run(script: str) -> None:
    proc = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
