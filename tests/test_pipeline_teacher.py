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
  lazy-``ImportError`` path -- open question O3: CPU inference is far too
  slow for the suite, so stage-07 Шаги 3-6 drive BOTH of its modes through
  fakes (``_FakeLM``/``_FakeTokenizer`` injected over ``HFLocalTeacher._load``):
  no FAST test loads a real HF model or touches the network. The single
  exception is ``test_real_qwen_logits_teach`` at the end of this file (
  stage 7 Шаг 7, ``@pytest.mark.slow``): it loads the real Qwen 0.5B from the
  LOCAL HF snapshot with ``HF_HUB_OFFLINE=1`` -- still no network. Шаг 5 pins the disk
  cache (hits across calls *and* instances, key coverage, one
  ``<sha256>.json`` per entry); Шаг 6 pins that the cache is read only AFTER
  ``_load()``, so the lazy ``ImportError`` survives a pre-warm cache under an
  import block.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

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


@pytest.fixture(autouse=True)
def _hermetic_cache_dir(monkeypatch, tmp_path) -> None:
    """Every test runs with its cwd in a fresh temp dir.

    The Шаг 3-4 tests pin ``_FakeLM`` forward/generate counters while using
    HFLocalTeacher's DEFAULT ``cache_dir="cache/hf"`` -- a RELATIVE path. With
    the disk cache wired (Шаг 5) that path resolves into the repo and survives
    between runs, so a second ``pytest`` would hand those tests a warm entry
    and fail their ``forwards == 1`` / ``generate_calls`` assertions (it did:
    4 failed, 12 passed on run #2). A per-test cwd keeps them green on EVERY
    run, leaves the repo's ``cache/`` untouched and makes the file independent
    of where pytest was started from. The Шаг 5-6 tests pass absolute
    ``tmp_path`` cache dirs, so they are unaffected.
    """
    monkeypatch.chdir(tmp_path)


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


# NOTE (stage-07 Шаг 6): the old test_hf_local_teacher_fails_lazily_without_
# transformers SKIPPED itself whenever transformers was installed -- which it
# is here -- so the lazy-ImportError path went untested. It is replaced by
# test_lazy_importerror_survives_cache_and_import_block at the bottom of this
# file: it blocks the import via sys.modules instead of hoping for a missing
# package, and it runs ALWAYS.


# --- Stage-07 Шаги 3-4: fakes over HFLocalTeacher._load -- no real HF model,
# no network, no transformers import. Shared fixture for Шаги 3-6 (§Шаг 3).


class _FakeLM(torch.nn.Module):
    """Minimal causal-LM stand-in with counted forwards.

    ``forward`` yields fixed last-position logits (``{token_id: value}``,
    everything else 0) wrapped the way an HF causal-LM output is (``.logits``);
    ``generate`` records its kwargs and appends ``max_new_tokens`` copies of
    one seeded-random "answer" token (drawn from the torch RNG that ``_sample``
    seeds per call), so the votes path runs end-to-end deterministically.
    """

    def __init__(
        self,
        logits_by_id: dict[int, float] | None = None,
        answer_ids: tuple[int, ...] = (10, 11),
    ) -> None:
        super().__init__()
        self.logits_by_id = dict(logits_by_id or {})
        self.answer_ids = tuple(answer_ids)
        self.forwards = 0
        self.generate_calls: list[dict] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.forwards += 1
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 64)
        for token_id, value in self.logits_by_id.items():
            logits[:, -1, token_id] = value
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids=None, max_new_tokens=8, do_sample=False, **kwargs):
        self.generate_calls.append(
            {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        )
        pick = int(torch.randint(0, len(self.answer_ids), (1,)).item())
        new_tokens = torch.full(
            (input_ids.shape[0], int(max_new_tokens)),
            self.answer_ids[pick],
            dtype=input_ids.dtype,
        )
        return torch.cat([input_ids, new_tokens], dim=1)


class _FakeTokenizer:
    """Exact-match tokenizer: ONLY the listed candidate strings encode at all.

    ``encode`` returns ``[]`` for anything else -- that is the "ambiguous
    tokenization / no unique id" case the logits path must survive by falling
    back to ``mode="sample"``. The prompt ``__call__`` hands back a fixed
    short sequence (its content never reaches the assertions), ``decode`` maps
    ids back through the vocab (unknown id -> ""), and ``eos_token_id``
    satisfies ``_sample``.
    """

    def __init__(self, vocab: dict[str, list[int]]) -> None:
        self.vocab = {text: list(ids) for text, ids in vocab.items()}
        self.eos_token_id = 0
        self.id_to_token = {
            ids[0]: text for text, ids in self.vocab.items() if ids
        }

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(self.vocab.get(text, []))

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        ids = [1, 2, 3]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self.id_to_token.get(int(i), "") for i in ids)


def _install_fakes(monkeypatch, tokenizer, model) -> None:
    """Inject the fakes exactly where ``teach`` gets the real backend."""
    monkeypatch.setattr(HFLocalTeacher, "_load", lambda self: (tokenizer, model))


def test_logits_softmaxes_option_first_tokens(monkeypatch) -> None:
    # Шаг 3 red 1: mode="logits" does ONE forward per (state, question) and
    # softmaxes the last-position logits of the options' first-token ids.
    # Noul labels come from the same source _best_option uses
    # (pipeline.teacher._option_labels -> ("false", "true"), main.py:791-792).
    tok = _FakeTokenizer(
        {" false": [10], " true": [11], "false": [10], "true": [11]}
    )
    lm = _FakeLM(logits_by_id={10: 2.0, 11: 0.0})
    _install_fakes(monkeypatch, tok, lm)
    t = HFLocalTeacher(mode="logits")

    (probs,) = t.teach([{"s": 1}], [Noul("x", key="a")], repeats=4)

    assert probs.shape == (1, 2)                       # [B, K], K = 2 for a Noul
    assert probs.dtype == torch.float32
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert probs[0, 0].item() == pytest.approx(0.8808, abs=1e-3)  # softmax(2.0, 0.0)
    assert probs[0, 1].item() == pytest.approx(0.1192, abs=1e-3)
    assert lm.forwards == 1        # repeats never multiply the logits path
    assert lm.generate_calls == []  # ... and sampling never happens


def test_logits_handles_leading_space_and_multitoken_options(monkeypatch) -> None:
    # Шаг 3 red 2: candidate ids are tried as " "+label then label (byte-level
    # BPE keeps the leading space in the token); a multi-token option scores
    # on its FIRST id only; no id may be None; a label with no encodable
    # candidate ("zz") means "ambiguous, no unique id" -> sample fallback.
    from pipeline.teacher import _option_first_token_ids

    tok = _FakeTokenizer({" yes": [7], "no": [9], " tw": [21, 22]})

    ids = _option_first_token_ids(tok, ("yes", "no", "tw"))
    assert ids == [7, 9, 21]  # " yes" wins for yes; " tw" contributes only 21
    assert ids is not None and all(isinstance(i, int) for i in ids)

    lm = _FakeLM(answer_ids=(7,))  # decode -> " yes" -> recognized by _best_option
    _install_fakes(monkeypatch, tok, lm)
    t = HFLocalTeacher(mode="logits")
    (probs,) = t.teach(
        [{"s": 1}], [Choice("y", options=["yes", "zz"], key="b")], repeats=3
    )
    assert lm.generate_calls, "no id for 'zz' -> fallback to mode='sample'"
    assert lm.forwards == 0  # no wasted forward when ids are not unique
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert probs[0].max().item() < 1.0  # votes path lifts unanimity off the vertex


def test_logits_token_collision_falls_back_to_sampling(monkeypatch) -> None:
    # Шаг 3 red 3: two options colliding on ONE first-token id -> fallback to
    # mode="sample" FOR THAT QUESTION (fixed rule -- never sequence-
    # scoring): generate runs >= 1 time, the row still sums to 1 and is not
    # one-hot (_votes_to_frequencies lifts unanimity).
    tok = _FakeTokenizer({" aa": [10], " bb": [10]})  # both options -> id 10
    lm = _FakeLM(answer_ids=(10,))  # decode -> "aa"/"bb" -> recognized
    _install_fakes(monkeypatch, tok, lm)
    t = HFLocalTeacher(mode="logits")

    (probs,) = t.teach(
        [{"s": 1}], [Choice("y", options=["aa", "bb"], key="b")], repeats=5
    )

    assert len(lm.generate_calls) >= 1  # the sample fallback actually ran
    assert lm.forwards == 0             # collision detected before any forward
    assert probs.shape == (1, 2)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert probs[0].max().item() < 1.0 - 1e-6  # not one-hot


def test_logits_mode_is_deterministic(tmp_path, monkeypatch) -> None:
    # Шаг 3 red 4: two instances with different EMPTY cache_dir -> equal
    # TargetList. The logits path draws no random numbers, so the RNG (and the
    # cache, which does not exist yet in Шаг 3) cannot influence the output.
    tok = _FakeTokenizer(
        {
            " false": [10], " true": [11],
            " p": [12], " q": [13],
            " lo": [14], " mid": [15], " hi": [16],
        }
    )
    lm = _FakeLM(
        logits_by_id={10: 1.5, 11: 0.0, 12: 0.7, 13: 0.1, 14: 0.3, 15: 0.9, 16: 0.2}
    )
    _install_fakes(monkeypatch, tok, lm)
    cache_a, cache_b = tmp_path / "one", tmp_path / "two"
    cache_a.mkdir(), cache_b.mkdir()  # different, both empty

    first = HFLocalTeacher(mode="logits", cache_dir=str(cache_a))
    second = HFLocalTeacher(mode="logits", cache_dir=str(cache_b))
    out_a = first.teach(STATES, QUESTIONS, repeats=2)
    out_b = second.teach(STATES, QUESTIONS, repeats=2)

    assert out_a == out_b  # TargetList.__eq__ -> torch.equal -> bool
    assert lm.forwards == 6        # 3 questions x 1 state, once per instance pair
    assert lm.generate_calls == []


def test_sample_mode_keeps_contract_with_four_tokens(monkeypatch) -> None:
    # Шаг 4 red: mode="sample" keeps the votes contract but its DEFAULT
    # max_new_tokens must be 4 (today it is 8 — a word of an option is enough,
    # 8 tokens cost ~2x for nothing on CPU).
    tok = _FakeTokenizer({" false": [10], " true": [11]})
    lm = _FakeLM(answer_ids=(10,))
    _install_fakes(monkeypatch, tok, lm)
    t = HFLocalTeacher(mode="sample", seed=0)

    first = t.teach(STATES, [Noul("x", key="a")], repeats=6)
    second = t.teach(STATES, [Noul("x", key="a")], repeats=6)

    assert first == second  # same seed -> identical outputs (risk R3)
    assert lm.generate_calls, "sample mode must go through model.generate"
    for call in lm.generate_calls:
        assert call["max_new_tokens"] == 4  # <- the Шаг 4 red (default is 8 today)
        assert call["do_sample"] is True

    # Pinned defaults: logits is the primary mode, 4 new tokens, Qwen instead
    # of the dead gpt2 default.
    fresh = HFLocalTeacher()
    assert fresh.mode == "logits"
    assert fresh.max_new_tokens == 4
    assert fresh.model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    # An empty decode must still raise the EXISTING verbatim error: nothing in
    # the votes path's messages changes with the new default.
    silent_tok, silent_lm = _FakeTokenizer({}), _FakeLM(answer_ids=(10,))
    _install_fakes(monkeypatch, silent_tok, silent_lm)
    with pytest.raises(RuntimeError, match="no recognizable option"):
        HFLocalTeacher(mode="sample").teach(STATES, [Noul("x", key="a")], repeats=2)


def test_hf_local_teacher_rejects_unknown_mode() -> None:
    # __post_init__ guards the mode: only "logits" and "sample" exist, and the
    # error names the field (config.py validates mode too, but this backend
    # must not rely on its caller — make_teacher passes kwargs straight in).
    with pytest.raises(ValueError, match="mode"):
        HFLocalTeacher(mode="greedy")


# --- Stage-07 Шаг 5: disk cache (resume / second epoch must not pay inference) ---


def test_cache_hits_on_second_teach_and_across_instances(tmp_path, monkeypatch) -> None:
    # Шаг 5 red 1: teach() twice on ONE instance plus once on a NEW instance with
    # the same cache_dir. The model must run only on the FIRST call: B*Q misses
    # up front, then 2*B*Q hits accumulated on instance #1 and B*Q hits on the
    # fresh instance #2; outputs byte-equal across calls/instances; one
    # <sha256>.json file per (state, question) entry inside cache_dir.
    tok = _FakeTokenizer(
        {
            " false": [10], " true": [11],
            " p": [12], " q": [13],
            " lo": [14], " mid": [15], " hi": [16],
        }
    )
    lm = _FakeLM(
        logits_by_id={10: 1.5, 11: 0.0, 12: 0.7, 13: 0.1, 14: 0.3, 15: 0.9, 16: 0.2}
    )
    _install_fakes(monkeypatch, tok, lm)
    cache = tmp_path / "hf"
    states = [{"s": 1}, {"s": 2}]   # B = 2
    entries = 2 * len(QUESTIONS)    # B * Q = 6 prompt entries (3 questions)

    first = HFLocalTeacher(mode="logits", cache_dir=str(cache))
    out_cold = first.teach(states, QUESTIONS, repeats=3)
    assert (first.cache_hits, first.cache_misses) == (0, entries)
    assert lm.forwards == entries  # cold: one forward per (state, question)

    out_warm = first.teach(states, QUESTIONS, repeats=3)
    # shorthand "B*Q miss, 2*B*Q hit" = scenario totals: misses happen once
    # (teach #1), the two warm replays (here + instance #2 below) add 2*B*Q hits.
    assert (first.cache_hits, first.cache_misses) == (entries, entries)
    assert lm.forwards == entries  # warm: NOT a single new forward

    second = HFLocalTeacher(mode="logits", cache_dir=str(cache))
    out_fresh = second.teach(states, QUESTIONS, repeats=3)
    assert (second.cache_hits, second.cache_misses) == (entries, 0)
    assert lm.forwards == entries  # a NEW instance pays nothing either
    # scenario totals: exactly B*Q misses were ever paid, 2*B*Q hits collected
    assert first.cache_hits + second.cache_hits == 2 * entries
    assert first.cache_misses + second.cache_misses == entries

    # byte-equal rows across calls and instances (same JSON round-trip value)
    assert out_cold == out_warm == out_fresh  # TargetList.__eq__ -> torch.equal
    for cold, fresh in zip(out_cold, out_fresh):
        assert cold.numpy().tobytes() == fresh.numpy().tobytes()

    files = [p.name for p in cache.glob("*.json")]
    assert len(files) == entries, f"one <sha256>.json per entry, got {sorted(files)}"
    assert all(re.fullmatch(r"[0-9a-f]{64}\.json", name) for name in files)


def test_cache_key_covers_model_mode_prompt_seed(tmp_path, monkeypatch) -> None:
    # Шаг 5 red 2: the key must cover model_id, mode, seed and the question text
    # (plus prompt/repeats/temperature/max_new_tokens/version): an identical
    # repeat is a hit, changing ANY of those components is a miss -- the model
    # runs again instead of being served the stale entry.
    tok = _FakeTokenizer({" false": [10], " true": [11]})
    lm = _FakeLM(logits_by_id={10: 2.0, 11: 0.0}, answer_ids=(10,))
    _install_fakes(monkeypatch, tok, lm)
    cache = str(tmp_path / "hf")
    question = [Noul("x", key="a")]

    def run(**kwargs):
        t = HFLocalTeacher(cache_dir=cache, **kwargs)
        return t, t.teach(STATES, question, repeats=4)

    base, out0 = run(mode="logits")
    assert (base.cache_hits, base.cache_misses) == (0, 1)
    assert lm.forwards == 1

    same, again = run(mode="logits")
    assert (same.cache_hits, same.cache_misses) == (1, 0)
    assert lm.forwards == 1  # identical key -> no forward
    assert again == out0

    run(mode="logits", model_id="other/model")
    assert lm.forwards == 2  # model_id is part of the key

    run(mode="logits", seed=7)
    assert lm.forwards == 3  # seed is part of the key

    t_text = HFLocalTeacher(mode="logits", cache_dir=cache)
    t_text.teach(STATES, [Noul("y", key="a")], repeats=4)
    assert (t_text.cache_hits, t_text.cache_misses) == (0, 1)
    assert lm.forwards == 4  # different question text -> different prompt -> miss

    # mode is part of the key: same prompt, model_id, seed and repeats=4 -- the
    # sample path must NOT be served the logits entry (generate must run).
    before = len(lm.generate_calls)
    t_mode, _ = run(mode="sample")
    assert (t_mode.cache_hits, t_mode.cache_misses) == (0, 1)
    assert len(lm.generate_calls) > before


def test_lazy_importerror_survives_cache_and_import_block(monkeypatch, tmp_path) -> None:
    # Шаг 6 (REPLACES the old self-skipping test): transformers IS installed in
    # this environment, so the missing package is simulated with
    # sys.modules["transformers"] = None (the interpreter then refuses the
    # import outright). The cache_dir is PRE-FILLED with a valid entry for the
    # exact prompt, and teach() must STILL raise ImportError naming
    # "transformers": the cache is read only AFTER _load(), so a warm cache can
    # never launder away the lazy import (§Шаг 5: чтение/запись кэша
    # после _load()). Runs ALWAYS -- no skip.
    tok = _FakeTokenizer({" false": [10], " true": [11]})
    lm = _FakeLM(logits_by_id={10: 2.0, 11: 0.0})
    cache = tmp_path / "hf"

    # 1) pre-fill the cache through the fakes (real key, real JSON entry) ...
    with pytest.MonkeyPatch.context() as mp:
        _install_fakes(mp, tok, lm)
        warm = HFLocalTeacher(mode="logits", cache_dir=str(cache))
        warm.teach(STATES, [Noul("x", key="a")], repeats=4)
    assert lm.forwards == 1
    assert [p.name for p in cache.glob("*.json")], "cache entry was not pre-filled"

    # 2) ... then block transformers: the cache must NOT answer for _load().
    monkeypatch.setitem(sys.modules, "transformers", None)
    cold = HFLocalTeacher(mode="logits", cache_dir=str(cache))
    with pytest.raises(ImportError, match="transformers"):
        cold.teach(STATES, [Noul("x", key="a")], repeats=4)
    assert (cold.cache_hits, cold.cache_misses) == (0, 0)  # died before any lookup


# --- Stage-07 Шаг 7: the ONE slow test -- REAL Qwen weights, offline from the
# local HF snapshot (the fast suite above never loads a real model).


@pytest.mark.slow  # load 0.5B weights + 2 forwards on CPU: ~5-15s (Шаг 7)
def test_real_qwen_logits_teach(tmp_path, monkeypatch) -> None:
    """``mode="logits"`` through the REAL Qwen2.5-0.5B-Instruct (Шаг 7).

    Skips if ``transformers`` is missing or the snapshot
    ``models--Qwen--Qwen2.5-0.5B-Instruct`` is absent from the local HF cache;
    ``HF_HUB_OFFLINE=1`` is set BEFORE anything imports transformers (the env
    is read into ``huggingface_hub.constants`` at IMPORT time; the only import
    after that point in this test is ``pytest.importorskip``), so the test can
    never touch the network or download anything.
    """
    import os

    # Set BEFORE the only thing here that may import transformers
    # (the importorskip below): the env is read into huggingface_hub at import.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("transformers")

    # Resolve the hub cache exactly as huggingface_hub does: HF_HUB_CACHE env
    # -> HUGGINGFACE_HUB_CACHE env -> (HF_HOME env or
    # $(XDG_CACHE_HOME|~/.cache)/huggingface)/hub.
    hub = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME")
            or os.path.join(
                os.environ.get("XDG_CACHE_HOME")
                or os.path.join(os.path.expanduser("~"), ".cache"),
                "huggingface",
            ),
            "hub",
        )
    )
    snapshot = os.path.join(hub, "models--Qwen--Qwen2.5-0.5B-Instruct")
    if not os.path.isdir(snapshot):
        pytest.skip(f"Qwen snapshot absent from HF cache: {snapshot}")

    # 1 state x 1 Noul; a CLEAN cache_dir per instance (the autouse fixture
    # already chdirs into tmp_path, but the design asks for explicit dirs).
    q = [QUESTIONS[0]]  # Noul -> labels ("false", "true") -> K = 2

    first = HFLocalTeacher(mode="logits", cache_dir=str(tmp_path / "one"))
    cold = first.teach(STATES, q, repeats=5)
    (probs,) = cold
    assert probs.shape == (1, 2)  # [B, K], B = 1, K = 2
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
    # Risk R3 IS this assertion: a real model that ever produced a
    # vertex row fails here. Never weaken it -- report it as a finding.
    assert probs.max().item() < 1.0 - 1e-6  # contract: not one-hot

    # Repeat teach() on the SAME instance: served from the disk cache
    # (hits are counted per (state, question) lookup) with equal values.
    warm = first.teach(STATES, q, repeats=5)
    assert first.cache_hits > 0
    assert warm == cold  # TargetList.__eq__ -> torch.equal -> bool

    # Determinism WITHOUT cache: a second instance with a DIFFERENT, EMPTY
    # cache_dir pays its own forward and must agree byte-for-byte. Instances
    # are taught strictly sequentially and the first model is released first:
    # two live HF loads would double the ~1.4 GB RSS (never two at once).
    del first
    second = HFLocalTeacher(mode="logits", cache_dir=str(tmp_path / "two"))
    fresh = second.teach(STATES, q, repeats=5)
    assert fresh == cold
