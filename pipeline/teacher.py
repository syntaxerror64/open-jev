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
  without a network but on CPU costs seconds per forward, far too slow for
  the suite. ``transformers`` is imported lazily inside ``_load`` (a missing
  one raises an actionable ``ImportError``), and tests drive both of its
  modes -- ``mode="logits"`` (one forward, softmax over option first tokens)
  and ``mode="sample"`` (seeded generation votes) -- through a fake
  model/tokenizer injected over ``_load``: **no test loads real weights or
  touches the network**, toy volumes only. Its disk cache
  (``cache_dir/<sha256>.json``, stage-07 Шаг 5) makes a resume or a second
  epoch replay stored rows instead of paying inference, and it is consulted
  only AFTER ``_load`` so the lazy ``ImportError`` survives a warm cache
  (Шаг 6).

Factory (signature fixed for pipeline.train, agent В):
``make_teacher(kind: str, **kwargs) -> Teacher`` with ``kind`` in
``{"stub", "api", "hf_local"}``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
    "CACHE_VERSION",
    "HFLocalTeacher",
    "StubTeacher",
    "TargetList",
    "Teacher",
    "frequencies",
    "make_teacher",
]

#: Version stamp baked into every disk-cache key (§Шаг 5). Bump it when
#: the prompt format, the logits/votes arithmetic or the tokenisation changes:
#: old entries then read as stale and are recomputed instead of served.
CACHE_VERSION: int = 1


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

    ``transformers`` is imported lazily inside ``_load``: importing this
    module -- or building the backend through the factory -- must work
    without the package, and a missing one raises an ``ImportError`` that
    names the fix instead of crashing test collection.

    Two modes (stage-07 Шаги 3-4), selected by ``mode``:

    * ``mode="logits"`` (default, the primary mode): ONE forward per
      (state, question) under ``no_grad``; ``softmax`` over the last
      position's logits at the first-token id of every option label
      (``_option_first_token_ids``: candidates ``" "+label`` then ``label``)
      -> one ``[K]`` row summing to 1. If a label has no encodable candidate
      or two labels collide on one id, THAT question falls back to the
      sample path -- the fixed rule (never sequence-scoring).
    * ``mode="sample"``: the model generates ``max_new_tokens`` tokens
      ``repeats`` times per (state, question); the earliest option mentioned
      in each completion is one vote, and the votes become frequencies
      (unanimity lifted off the vertex, as in ``StubTeacher``).

    HONEST LIMIT (O3): on CPU one forward costs seconds and sampling costs
    more (batching buys nothing -- compute-bound), so real weights stay out
    of the fast suite: tests inject a fake model/tokenizer over ``_load``,
    and a real model is reasonable for toy volumes only (README should carry
    this limitation).

    DISK CACHE (stage-07 Шаг 5): every (state, question) row is stored as
    ``cache_dir/<sha256>.json`` whose key covers ``CACHE_VERSION``,
    ``model_id``, ``mode``, the exact prompt, ``repeats``, ``temperature``,
    ``seed`` and ``max_new_tokens`` (sorted JSON) -- resume and the second
    epoch replay bytes instead of paying inference, while changing any input
    that can move the answer invalidates the entry. Writes are atomic (tmp
    file in the SAME directory + ``os.replace``); an unreadable/unwritable
    ``cache_dir`` degrades to a miss, never to a failed ``teach``. ``None``/
    ``""`` turns caching off. ``cache_hits``/``cache_misses`` count lookups
    per instance. Crucially, the cache is read and written only AFTER
    ``_load()``: a pre-warm cache can never launder away the lazy
    ``transformers`` ImportError (Шаг 6, pinned by test).
    """

    mode: str = "logits"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    temperature: float = 0.8
    seed: int = 0
    max_new_tokens: int = 4  # one option word suffices; 8 cost ~2x on CPU (Шаг 4)
    cache_dir: str = "cache/hf"

    _tokenizer: Any = field(init=False, default=None, repr=False, compare=False)
    _model: Any = field(init=False, default=None, repr=False, compare=False)
    # Observable cache bookkeeping (§Шаг 5): NOT constructor arguments and
    # NOT part of equality -- two teachers with the same config stay equal
    # however differently their histories went.
    cache_hits: int = field(init=False, default=0, repr=False, compare=False)
    cache_misses: int = field(init=False, default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in ("logits", "sample"):
            raise ValueError(
                f"HFLocalTeacher: unknown mode {self.mode!r}; expected "
                "'logits' (one forward, softmax over option first tokens) or "
                "'sample' (seeded generation votes)"
            )

    def teach(
        self,
        states: Sequence[StateValue],
        questions: Sequence[Question],
        repeats: int = 10,
    ) -> TargetList:
        _check_repeats(repeats)
        # _load() runs FIRST and unconditionally: the disk cache is consulted
        # only below, so a pre-warm cache can never bypass the lazy ImportError
        # (Шаг 6 pins exactly this ordering).
        tokenizer, model = self._load()
        out = TargetList()
        sample_index = 0
        for q in questions:
            labels = _option_labels(q)
            # Logits mode: unique first-token ids exist -> exact distribution;
            # otherwise (or in sample mode) this question uses the votes path.
            first_ids = (
                _option_first_token_ids(tokenizer, labels)
                if self.mode == "logits"
                else None
            )
            rows: list[list[float]] = []
            for s in states:
                prompt = _prompt(s, q) + "\nReply with exactly one option."
                key = self._cache_key(prompt, repeats)
                cached = self._cache_get(key, len(labels))
                if cached is not None:
                    rows.append(cached)  # warm: replay, no forward/generate
                    continue
                if first_ids is not None:
                    row = self._logits_row(tokenizer, model, prompt, first_ids)
                else:
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
                    row = _votes_to_frequencies(counts)
                rows.append(row)
                self._cache_put(key, row)  # cold: pay once, replay forever
            out.append(torch.tensor(rows, dtype=torch.float32))
        return out

    def _cache_key(self, prompt: str, repeats: int) -> str:
        """sha256 of the sorted key payload -> the entry's file stem (Шаг 5).

        Every input that can move the answer is in there: the code generation
        (``CACHE_VERSION``), the weights (``model_id``), the arithmetic
        (``mode``, ``repeats``, ``temperature``, ``max_new_tokens``), the RNG
        (``seed``) and the exact prompt (which embeds state + question text +
        options). Sorting the JSON keeps the key independent of dict order.
        """
        payload = {
            "version": CACHE_VERSION,
            "model_id": self.model_id,
            "mode": self.mode,
            "prompt": prompt,
            "repeats": repeats,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_get(self, key: str, n_options: int) -> list[float] | None:
        """Stored row for ``key``, or ``None`` -- counted either way.

        Anything unreadable (missing file, broken JSON, wrong version, wrong
        arity, non-finite/negative mass) is a MISS, i.e. the entry is
        recomputed and rewritten: the cache may only save time, never change
        an answer. ``cache_dir`` falsy -> caching is off: no lookup, no count.
        """
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, key + ".json")
        row: list[float] | None = None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            probs = data.get("probs") if isinstance(data, dict) else None
            if (
                isinstance(data, dict)
                and data.get("version") == CACHE_VERSION
                and isinstance(probs, list)
                and len(probs) == n_options
            ):
                values = [float(v) for v in probs]
                if (
                    all(math.isfinite(v) and v >= 0.0 for v in values)
                    and sum(values) > 0.0
                ):
                    row = values
        except (OSError, ValueError, TypeError):
            row = None
        if row is None:
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return row

    def _cache_put(self, key: str, probs: Sequence[float]) -> None:
        """Write ``{version, probs}`` atomically: tmp file in the SAME dir + replace.

        ``os.replace`` is atomic within one filesystem, so a reader never sees
        a half-written entry and a crash mid-write leaves the old file intact.
        An unwritable ``cache_dir`` silently degrades to "no cache" -- the
        cache is an optimisation, a failed teach() would be a regression.
        """
        if not self.cache_dir:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = os.path.join(self.cache_dir, f".{key}.{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"version": CACHE_VERSION, "probs": [float(v) for v in probs]},
                    fh,
                )
            os.replace(tmp, os.path.join(self.cache_dir, key + ".json"))
        except OSError:
            pass  # never fail a teach() because of a cache write

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
        """One seeded, sampled completion (the ``mode="sample"`` votes path)."""
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

    def _logits_row(
        self, tokenizer: Any, model: Any, prompt: str, candidate_ids: Sequence[int]
    ) -> list[float]:
        """One forward -> softmax over the options' first-token ids -> one row.

        The prompt is byte-identical to the sample path's (the disk cache keys
        on exactly this string), and ``repeats`` does not apply: the softmax IS
        the distribution, one forward per (state, question). Nothing is
        clipped -- a near-one-hot row is a model fact the downstream
        validation deals with (main.py:807 mitigation is an open
        question), never a reason to edit probabilities here.
        """
        encoded = {
            k: v.to(self.device)
            for k, v in tokenizer(prompt, return_tensors="pt").items()
        }
        with torch.no_grad():
            last_logits = model(**encoded).logits[0, -1]
        probs = torch.softmax(last_logits[list(candidate_ids)].float(), dim=0)
        return probs.tolist()


def make_teacher(kind: str, **kwargs: Any) -> Teacher:
    """Build a backend by name: ``make_teacher(kind: str, **kwargs) -> Teacher``.

    ``kind`` is one of ``"stub"`` (deterministic, no network),
    ``"api"`` (pass ``client=...``, open question O2) or ``"hf_local"`` (needs
    ``transformers``, open question O3); ``kwargs`` go straight to the backend
    constructor (``seed=0``, ``mode="logits"``, ...). Unknown kinds raise
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


def _option_first_token_ids(tokenizer: Any, labels: Sequence[str]) -> list[int] | None:
    """First-token id per option label, or ``None`` when the mapping is not unique.

    Candidates per label, in order: ``" " + label`` then ``label`` -- byte-level
    BPE (GPT-2/Qwen style) keeps the leading space inside the token, and the
    probe scored ``[' yes', ' no']``. Only the FIRST id of a candidate
    is used: a multi-token label scores on its opening token.

    ``None`` -- which makes ``HFLocalTeacher.teach`` fall back to
    ``mode="sample"`` for this question, the fixed rule (never
    sequence-scoring) -- when a label has no encodable candidate at all
    (ambiguous tokenization, no unique id) or when two labels collide on one
    id (the softmax would compare an option with itself). Any tokenizer
    failure while encoding counts as "no id": the votes path is the
    always-safe fallback and already distinguishes full labels.
    """
    ids: list[int] = []
    for label in labels:
        first: int | None = None
        for candidate in (" " + label, label):
            try:
                encoded = tokenizer.encode(candidate, add_special_tokens=False)
                first = int(encoded[0]) if encoded else None
            except Exception:
                first = None
            if first is not None:
                break
        if first is None:
            return None
        ids.append(first)
    if len(set(ids)) != len(ids):
        return None
    return ids


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
