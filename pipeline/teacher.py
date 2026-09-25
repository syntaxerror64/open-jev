"""Teacher layer: the source of SOFT training targets (Шаг 2, step 2).

Why this module exists boils down to one rule, written out in
open_jev/main.py:807-821: NEVER train on hard labels. Cross-entropy against a
one-hot target manufactures overconfidence -- the exact failure Jev is built to
avoid -- so every backend here returns *frequencies*: one ``[B, K]`` float
tensor per question whose rows sum to 1 and never sit on a simplex vertex.
``pipeline.data.validate_soft_targets`` re-checks the same invariants on disk.

Why frequencies and not "the answer": a frontier ensemble sampled across
paraphrases, option orderings and temperatures *is* the calibration signal
(main.py:812-817). Ten samples splitting 7/3 yield the target 0.7 -- that
arithmetic is :func:`frequencies`, and ``repeats`` on every backend is the
ensemble size (default 10, ``TrainConfig.teacher_repeats`` in
pipeline/config.py).

Backends and their honest limits:

* ``StubTeacher`` -- deterministic (``random.Random(seed)`` recreated per
  call), CPU-only, no network. It validates pipeline *mechanics*, never model
  quality: with no real data or teacher yet (known gaps O1/R1) its votes are random,
  so stub-trained metrics mean nothing.
* ``ApiTeacher`` -- open question **O2**: a real API teacher needs a key/
  authorization/payment this repo does not hold. So the contract is
  ``client.complete(prompt) -> Sequence[float]`` plus a mock mode; tests cover
  it with a fake client only. **No network call, auth, retry or response
  parsing is implemented or tested here.**
* ``HFLocalTeacher`` -- open question **O3**: a local HF model runs
  without a network but on CPU costs one forward pass per state x question x
  ``repeats``, far too slow for the suite. ``transformers`` is imported lazily
  inside ``teach`` (the package is not installed; a missing one raises an
  actionable ``ImportError``), and the suite smoke-tests construction/factory
  only -- **the generation loop is untested**, toy volumes only.

Factory (signature fixed for pipeline.train, agent В):
``make_teacher(kind: str, **kwargs) -> Teacher`` with ``kind`` in
``{"stub", "api", "hf_local"}``.
"""

from __future__ import annotations

import json
import math
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

from open_jev.main import Choice, Noul, Question, Score, StateValue

__all__ = [
    "ApiTeacher",
    "HFLocalTeacher",
    "StubTeacher",
    "TargetList",
    "Teacher",
    "frequencies",
    "make_teacher",
]


class TargetList(list[Tensor]):
    """A ``list`` of ``[B, K]`` target tensors whose ``==`` is a plain ``bool``.

    ``list.__eq__`` compares tensors element-wise and yields one *tensor* per
    pair; the truth value of a multi-element tensor is ambiguous, so the
    determinism assertion (``teach(...) == teach(...)``) would raise instead of
    passing. This subclass compares with :func:`torch.equal` and returns
    ``bool``. Everything else -- ``len``, ``zip``, indexing, feeding it to
    ``RLCDLoss`` -- is ordinary list behaviour.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, list):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(torch.equal(a, b) for a, b in zip(self, other))

    def __ne__(self, other: object) -> bool:
        # list defines __ne__ at the C level, so Python's usual "derive __ne__
        # from __eq__" does NOT kick in here -- without this override, ``a != b``
        # would fall back to list's element-wise tensor comparison.
        if not isinstance(other, list):
            return NotImplemented
        return not self.__eq__(other)


class Teacher(ABC):
    """Anything that turns (states, questions) into soft targets."""

    @abstractmethod
    def teach(
        self,
        states: Sequence[StateValue],
        questions: Sequence[Question],
        repeats: int = 10,
    ) -> list[Tensor]:
        """Return one ``[B, K]`` float32 tensor per question, ``B = len(states)``.

        Contract every backend obeys (main.py:807-821):

        * each row is a distribution: non-negative and summing to 1;
        * never one-hot -- no mass on a simplex vertex, hard labels are
          forbidden by construction;
        * ``repeats`` is the ensemble size the target is built from, so a 7/3
          split over ten samples becomes 0.7 (main.py:810-812);
        * backends that hold a seed are deterministic: identical calls return
          identical tensors (risk R3).
        """
        raise NotImplementedError


def frequencies(counts: Sequence[float]) -> list[float]:
    """Ensemble vote counts -> distribution: ``frequencies([7, 3]) == [0.7, 0.3]``.

    The arithmetic of main.py:810-811 ("if ten samples split 7/3, the target
    is 0.7"). Negative counts or a zero total raise ``ValueError``: normalising
    those would invent probability mass instead of measuring disagreement.
    """
    values = [float(c) for c in counts]
    if any(c < 0 for c in values):
        raise ValueError(f"frequencies: negative counts are not votes: {list(counts)!r}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"frequencies: counts must sum to > 0, got {list(counts)!r}")
    return [c / total for c in values]


@dataclass
class StubTeacher(Teacher):
    """Deterministic CPU stand-in: seeded random votes, no model, no network.

    ``teach`` rebuilds ``random.Random(self.seed)`` on every call, so the same
    (seed, states, questions, repeats) yields bit-identical tensors no matter
    how often or in which order it is called -- what the determinism
    test relies on.

    Honest caveat (risk R1): the votes are *random*, not learned. This
    backend proves pipeline mechanics (shapes, sums, determinism), never model
    quality; metrics from a stub-trained model stay meaningless until real
    data and a real teacher exist (O1/O2).
    """

    seed: int = 0

    def teach(
        self,
        states: Sequence[StateValue],
        questions: Sequence[Question],
        repeats: int = 10,
    ) -> TargetList:
        _check_repeats(repeats)
        rng = random.Random(self.seed)
        out = TargetList()
        for q in questions:
            k = _n_options(q)
            rows = [_draw_frequencies(rng, k, repeats) for _ in states]
            out.append(torch.tensor(rows, dtype=torch.float32))
        return out


@dataclass
class ApiTeacher(Teacher):
    """Frontier-model teacher behind an injected client (open question O2).

    Contract: ``client.complete(prompt: str) -> Sequence[float]`` returns one
    soft frequency per option of the question (length K, summing to 1,
    strictly positive). ``teach`` calls it ``repeats`` times per (state,
    question) and averages the replies -- the ensemble rule of main.py:810-812
    -- then renormalises. A wrong-arity, off-sum or hard (zero-mass) reply
    raises ``ValueError``; a missing/incomplete client raises ``RuntimeError``
    *before* any call.

    HONEST LIMIT (O2): this repo holds no API key, so nothing HTTP-shaped is
    implemented or tested -- no auth, retries, text parsing, or rate limits,
    and the suite drives it only through a fake client. To go live, wrap your
    endpoint in an object exposing ``complete`` and pass it as ``client=``.
    """

    client: Any = None

    def teach(
        self,
        states: Sequence[StateValue],
        questions: Sequence[Question],
        repeats: int = 10,
    ) -> TargetList:
        _check_repeats(repeats)
        complete = getattr(self.client, "complete", None)
        if not callable(complete):
            raise RuntimeError(
                "ApiTeacher needs a client exposing complete(prompt) -> "
                "Sequence[float] of soft frequencies. A real API teacher "
                "requires a key/authorization this repo does not hold ("
                "O2): inject either a wrapper around your endpoint or a test "
                "fake -- this module makes no network calls itself."
            )
        instruction = "\nReply with one soft frequency per option; they sum to 1."
        out = TargetList()
        for q in questions:
            k = _n_options(q)
            rows: list[list[float]] = []
            for s in states:
                prompt = _prompt(s, q) + instruction
                totals = [0.0] * k
                for _ in range(repeats):
                    reply = list(complete(prompt))
                    _check_frequencies(reply, k, f"ApiTeacher reply for {q.key or q.text!r}")
                    totals = [t + v for t, v in zip(totals, reply)]
                rows.append(frequencies(totals))
            out.append(torch.tensor(rows, dtype=torch.float32))
        return out


@dataclass
class HFLocalTeacher(Teacher):
    """Local Hugging Face causal-LM teacher on CPU (open question O3).

    ``transformers`` is imported lazily inside ``teach``: importing this module
    -- or building the backend through the factory -- must work without the
    package (it is not installed), and a missing one raises an ``ImportError``
    that names the fix instead of crashing test collection. The model is
    sampled ``repeats`` times per (state, question); the earliest option
    mentioned in each completion is one vote, and the votes become
    frequencies (unanimity lifted off the vertex, as in ``StubTeacher``).

    HONEST LIMIT (O3): on CPU this costs one forward pass per state x question
    x ``repeats`` -- orders of magnitude slower than the stub, reasonable for
    toy volumes only (README should carry this limitation). The suite
    therefore smoke-tests construction/factory alone and **never runs the
    generation loop below, which is consequently untested**.
    """

    model_id: str = "gpt2"
    device: str = "cpu"
    temperature: float = 0.8
    seed: int = 0
    max_new_tokens: int = 8

    _tokenizer: Any = field(init=False, default=None, repr=False, compare=False)
    _model: Any = field(init=False, default=None, repr=False, compare=False)

    def teach(
        self,
        states: Sequence[StateValue],
        questions: Sequence[Question],
        repeats: int = 10,
    ) -> TargetList:
        _check_repeats(repeats)
        tokenizer, model = self._load()
        out = TargetList()
        sample_index = 0
        for q in questions:
            labels = _option_labels(q)
            rows: list[list[float]] = []
            for s in states:
                prompt = _prompt(s, q) + "\nReply with exactly one option."
                counts = [0] * len(labels)
                for _ in range(repeats):
                    text = self._sample(tokenizer, model, prompt, sample_index)
                    sample_index += 1
                    hit = _best_option(text, labels)
                    if hit is not None:
                        counts[hit] += 1
                if sum(counts) == 0:
                    raise RuntimeError(
                        f"HFLocalTeacher: {self.model_id!r} produced no "
                        f"recognizable option for question {q.key or q.text!r}"
                    )
                rows.append(_votes_to_frequencies(counts))
            out.append(torch.tensor(rows, dtype=torch.float32))
        return out

    def _load(self) -> tuple[Any, Any]:
        """Import transformers and cache the model -- lazily, on first use."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HFLocalTeacher requires the optional 'transformers' package "
                "(pip install transformers). It is deliberately not a pipeline "
                "dependency: CPU inference (one forward per state x question x "
                "repeats) is far too slow for the test suite -- open question O3. Use "
                "StubTeacher in tests."
            ) from exc
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )
        return self._tokenizer, self._model

    def _sample(self, tokenizer: Any, model: Any, prompt: str, sample_index: int) -> str:
        """One seeded, sampled completion; untested code path (see class docstring)."""
        torch.manual_seed(self.seed + sample_index)  # reproducibility, risk R3
        encoded = {k: v.to(self.device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][encoded["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_teacher(kind: str, **kwargs: Any) -> Teacher:
    """Build a backend by name: ``make_teacher(kind: str, **kwargs) -> Teacher``.

    ``kind`` is one of ``"stub"`` (deterministic, no network),
    ``"api"`` (pass ``client=...``, open question O2) or ``"hf_local"`` (needs
    ``transformers``, open question O3); ``kwargs`` go straight to the backend
    constructor (``seed=0``, ``model_id="gpt2"``, ...). Unknown kinds raise
    ``ValueError`` listing the options. This exact signature is what the
    training loop should call (agent В).
    """
    try:
        backend = _TEACHERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown teacher kind {kind!r}; expected one of {sorted(_TEACHERS)}"
        ) from None
    return backend(**kwargs)


_TEACHERS: dict[str, type[Teacher]] = {
    "stub": StubTeacher,
    "api": ApiTeacher,
    "hf_local": HFLocalTeacher,
}


def _n_options(question: Question) -> int:
    """K for a question: 2 for a Noul (main.py:790-792), else the declared arity."""
    if isinstance(question, Noul):
        return 2
    if isinstance(question, Choice):
        return len(question.options)
    if isinstance(question, Score):
        return len(question.labels)
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _option_labels(question: Question) -> tuple[str, ...]:
    """Option names in the exact column order ``Jev.logits`` produces.

    A Noul is ``[P(false), P(true)]`` (main.py:791-792), so its labels are
    ``("false", "true")``; Choice/Score are their own declared order.
    """
    if isinstance(question, Noul):
        return ("false", "true")
    if isinstance(question, Choice):
        return tuple(question.options)
    if isinstance(question, Score):
        return tuple(question.labels)
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _prompt(state: StateValue, question: Question) -> str:
    """The one prompt both network-capable backends share (deterministic order)."""
    return (
        f"State: {json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)}\n"
        f"Question: {question.text}\n"
        f"Options: {' | '.join(_option_labels(question))}"
    )


def _check_repeats(repeats: int) -> None:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1 (ensemble size), got {repeats!r}")


def _check_frequencies(values: Sequence[float], expected: int, source: str) -> None:
    """Validate a client/backend reply against the soft-target contract."""
    vals = [float(v) for v in values]
    if len(vals) != expected:
        raise ValueError(f"{source}: expected {expected} frequencies, got {len(vals)}")
    if any(not math.isfinite(v) for v in vals):
        raise ValueError(f"{source}: non-finite frequency in {vals!r}")
    if any(v < 0 for v in vals):
        raise ValueError(f"{source}: negative frequency in {vals!r}")
    total = sum(vals)
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"{source}: frequencies must sum to 1, got {total!r}")
    if min(vals) <= 0.0:
        raise ValueError(
            f"{source}: hard/zero probability {vals!r} -- soft targets must stay "
            f"strictly inside the simplex, one-hot is forbidden (main.py:807)"
        )


def _votes_to_frequencies(counts: Sequence[int]) -> list[float]:
    """Vote counts -> soft frequencies, lifting unanimity off the vertex.

    A unanimous ensemble (10/0) would be one-hot, which main.py:807 forbids, so
    every silent option gets one pseudo-vote; any other split stays the exact
    ensemble frequency (7/3 -> 0.7). At least two options always carry mass:
    K >= 2 everywhere (Choice/Score enforce it, a Noul is binary).
    """
    return frequencies([max(int(c), 1) for c in counts])


def _draw_frequencies(rng: random.Random, k: int, repeats: int) -> list[float]:
    """``repeats`` categorical draws from ``rng`` -> one soft row of length k."""
    counts = [0] * k
    for _ in range(repeats):
        counts[rng.randrange(k)] += 1
    return _votes_to_frequencies(counts)


def _best_option(text: str, labels: Sequence[str]) -> int | None:
    """Index of the option mentioned first in ``text``, or ``None`` if none.

    Earliest match wins (ties -> lowest index); options match as whole tokens,
    case-insensitively, so "q" inside "quiet" is not a vote for "q".
    """
    best: tuple[int, int] | None = None
    for i, label in enumerate(labels):
        m = re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text, flags=re.IGNORECASE)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), i)
    return None if best is None else best[1]
