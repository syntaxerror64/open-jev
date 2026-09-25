"""Answer contract: `type` discriminator, `legend`, and wire-form `to_dict()`.

Stage 11 (Шаг 1) pins the Typesafe wire
contract: every answer carries a string `type` discriminator, ScoreAnswer
carries an int-keyed `legend` (level number -> label, like their Python SDK),
and `to_dict()` emits their JSON byte-for-byte -- without the `key` field,
which lives only as the key of the answers map in their transport.

These tests are written RED first (TDD): the dataclasses currently expose
only `key`/values, so `type`, `legend` and `to_dict` are expected to be
missing until Шаг 2 (Implementer) lands in open_jev/main.py.
"""

from __future__ import annotations

import json

import pytest
import torch

from open_jev import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

TOL = 1e-5


# --- discriminator ---------------------------------------------------------


def test_answers_carry_type_discriminator(model, state) -> None:
    """Each answer exposes its kind as a string literal, not the class."""
    questions = [
        Noul("The customer is requesting a refund.", key="wants_refund"),
        Choice(
            "Which team should handle this?",
            options=["billing", "technical", "account"],
            key="route",
        ),
        Score(
            "How frustrated is the customer?",
            labels=["bad", "ok", "great"],
            key="frustration",
        ),
    ]
    with torch.no_grad():
        answers = model([state], questions)[0]

    assert [a.type for a in answers] == ["noul", "choice", "score"]
    assert all(isinstance(a.type, str) for a in answers)


# --- legend ----------------------------------------------------------------


def test_score_answer_legend_maps_level_numbers_to_labels(model, state) -> None:
    """`legend` maps int level numbers to labels, like their Python SDK."""
    q = Score("Rate the outcome", labels=["bad", "ok", "great"], key="s")
    with torch.no_grad():
        answer = model([state], [q])[0][0]

    assert isinstance(answer, ScoreAnswer)
    assert answer.legend == {0: "bad", 1: "ok", 2: "great"}
    assert all(isinstance(level, int) for level in answer.legend)


# --- to_dict: their documented wire examples -------------------------------


def test_to_dict_matches_typesafe_choice_example() -> None:
    """DICT-EXACT match of the Typesafe doc example; `key` must be absent."""
    answer = ChoiceAnswer(
        key="whatever",
        choice="returns",
        probabilities={"shipping": 0.0, "returns": 1.0, "billing": 0.0},
        confidence=1.0,
    )
    assert answer.to_dict() == {
        "type": "choice",
        "choice": "returns",
        "confidence": 1.0,
        "probabilities": {"shipping": 0.0, "returns": 1.0, "billing": 0.0},
    }


def test_to_dict_noul_example() -> None:
    """`{"type":"noul","noul":0.72}` -- no `key` in the wire form."""
    assert NoulAnswer(key="k", noul=0.72).to_dict() == {
        "type": "noul",
        "noul": 0.72,
    }


def test_to_dict_score_uses_level_number_keys_and_json_dumps(
    model, state
) -> None:
    """Wire form: probabilities and legend keyed by strings "0".."L-1"."""
    labels = ["bad", "ok", "great"]
    q = Score("Rate the outcome", labels=labels, key="s")
    with torch.no_grad():
        answer = model([state], [q])[0][0]

    assert isinstance(answer, ScoreAnswer)
    d = answer.to_dict()

    level_keys = {str(i) for i in range(len(labels))}
    assert set(d["probabilities"]) == level_keys
    assert set(d["legend"]) == level_keys
    # The wire legend keeps level-number -> label; only keys stringify.
    assert d["legend"] == {"0": "bad", "1": "ok", "2": "great"}
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=TOL)

    text = json.dumps(d)
    assert json.loads(text) == d


# --- characterization: existing union order is unchanged -------------------


def test_answer_types_field_does_not_break_existing_union_order(
    model, state, questions
) -> None:
    """Adding `type` must not reorder or alter the Answer union members."""
    with torch.no_grad():
        row = model([state], questions)[0]

    assert [type(a) for a in row] == [NoulAnswer, ChoiceAnswer, ScoreAnswer]
