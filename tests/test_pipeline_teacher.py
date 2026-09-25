"""Stage-4, step 2 (red -> green): the teacher layer -- sources of soft targets.

The first three tests are the code from
stage 4, Шаг 2, copied verbatim; where the design
writes ``teach(...)`` the arguments are filled in (same states, same questions,
same ``repeats`` for both calls), so the determinism check stays exactly the
one-liner.

The remaining tests pin down what the spec leaves open:

* ``StubTeacher`` never returns a vertex/one-hot target -- the rule of
  open_jev/main.py:807 that pipeline.data.validate_soft_targets enforces, so
  the stub must satisfy it too (it is what the loop trains on, risk R1);
* ``ApiTeacher`` is verified against a FAKE client only -- open question
  O2: a real API teacher needs a key/authorization/payment this repo does not
  hold, so no network call is made or tested here;
* ``HFLocalTeacher`` gets a factory/construction smoke test plus the
  lazy-``ImportError`` path -- open question O3: ``transformers`` is not
  installed and CPU inference (one forward per state x question x repeats)
  is far too slow for the suite, so its generation loop stays untested.
"""

from __future__ import annotations

import importlib.util

import pytest, torch
from open_jev.main import Choice, Noul, Score
from pipeline.teacher import (
    ApiTeacher,
    HFLocalTeacher,
    StubTeacher,
    frequencies,
    make_teacher,
)

# Exactly the three primitives of Шаг 2, one of each kind.
QUESTIONS = [
    Noul("x", key="a"),
    Choice("y", options=["p", "q"], key="b"),
    Score("z", labels=["lo", "mid", "hi"], key="c"),
]
STATES = [{"s": 1}]


def test_stub_teacher_returns_valid_soft_frequencies() -> None:
    t = StubTeacher(seed=0)
    qs = [Noul("x", key="a"), Choice("y", options=["p", "q"], key="b"),
          Score("z", labels=["lo", "mid", "hi"], key="c")]
    out = t.teach([{"s": 1}], qs, repeats=10)
    assert len(out) == len(qs)
    for probs, q in zip(out, qs):
        assert probs.shape[-1] == (2 if isinstance(q, Noul)
                                   else len(q.options) if isinstance(q, Choice)
                                   else len(q.labels))
        assert probs.sum(-1).item() == pytest.approx(1.0, abs=1e-6)


def test_stub_teacher_is_deterministic_for_a_seed() -> None:
    assert StubTeacher(seed=3).teach(STATES, QUESTIONS, repeats=10) == StubTeacher(seed=3).teach(STATES, QUESTIONS, repeats=10)


def test_frequencies_convert_counts_to_distribution() -> None:
    assert frequencies([7, 3]) == pytest.approx([0.7, 0.3])   # main.py:810-811


def test_stub_teacher_never_returns_one_hot_targets() -> None:
    # main.py:807-817: one-hot manufactures overconfidence, so it is forbidden
    # for EVERY backend -- train.py feeds stub output straight into
    # validate_soft_targets (pipeline/data.py), which rejects max >= 1 - 1e-6.
    for seed in range(20):
        out = StubTeacher(seed=seed).teach(STATES, QUESTIONS, repeats=10)
        assert len(out) == len(QUESTIONS)
        for probs in out:
            assert probs.dtype == torch.float32     # matches the model dtype
            assert probs.min().item() > 0.0         # strictly inside ...
            assert probs.max().item() < 1.0 - 1e-6  # ... the simplex


class _FakeClient:
    """Fake ``complete(prompt) -> Sequence[float]`` client: NO network (O2)."""

    def __init__(self, reply: list[float]) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, prompt: str) -> list[float]:
        self.calls += 1
        assert "State:" in prompt and "Question:" in prompt
        return self.reply


def test_api_teacher_averages_client_samples_into_soft_targets() -> None:
    fake = _FakeClient([0.7, 0.3])
    out = ApiTeacher(client=fake).teach(
        STATES, [Choice("y", options=["p", "q"], key="b")], repeats=10
    )
    # 1 state x 1 question x repeats -- the ensemble of main.py:810-812:
    # ten samples are averaged, so a 7/3 split becomes the target 0.7.
    assert fake.calls == 10
    (probs,) = out
    assert probs.shape[-1] == 2
    assert probs.sum(-1).item() == pytest.approx(1.0, abs=1e-6)
    assert probs[0, 0].item() == pytest.approx(0.7, abs=1e-6)


def test_api_teacher_requires_a_client_and_rejects_hard_replies() -> None:
    # O2: no key/authorization in this repo -> a clear error, never a call.
    with pytest.raises(RuntimeError, match="client"):
        ApiTeacher().teach(STATES, QUESTIONS, repeats=1)
    # main.py:807: a one-hot reply from the API is a contract violation.
    with pytest.raises(ValueError, match="soft"):
        ApiTeacher(client=_FakeClient([1.0, 0.0])).teach(
            STATES, [Choice("y", options=["p", "q"], key="b")], repeats=2
        )


def test_make_teacher_builds_each_backend() -> None:
    # Factory signature fixed for pipeline.train (agent В):
    # make_teacher(kind: str, **kwargs) -> Teacher.
    assert isinstance(make_teacher("stub", seed=0), StubTeacher)
    assert isinstance(make_teacher("api"), ApiTeacher)
    assert isinstance(make_teacher("hf_local", model_id="gpt2"), HFLocalTeacher)
    with pytest.raises(ValueError, match="unknown teacher kind"):
        make_teacher("gpt6")


def test_hf_local_teacher_fails_lazily_without_transformers() -> None:
    # O3: transformers is NOT installed; teach() imports it lazily and the
    # error must name the package instead of crashing the suite at import time.
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed: lazy-import error path unreachable")
    t = make_teacher("hf_local", model_id="gpt2")
    with pytest.raises(ImportError, match="transformers"):
        t.teach(STATES, [Noul("x", key="a")], repeats=1)
