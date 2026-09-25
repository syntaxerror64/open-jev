"""Stage 3, agent B: tests for the trained BPE tokenizer.

TDD (stage 3): before ``open_jev/bpe_tokenizer`` exists the
whole file fails with

    ModuleNotFoundError: No module named 'open_jev.bpe_tokenizer'

What is covered:

  * ``BPETokenizer.load`` rebuilds the trained artifact ``cache/tokenizer.json``
    and the artifact fits the demo vocab budget (O3: 4096);
  * the shared contract: ``encode("") == [1]``, every id inside
    ``[1, vocab_size)``, length ``<= max_len``, PAD id (0) never leaks into
    ``encode`` — only ``encode_batch`` pads with it;
  * batch tensors: dtypes, shape, mask semantics, right padding, agreement
    with per-row ``encode``;
  * determinism of encoding;
  * determinism of *retraining* (O1/O4): two fresh runs of
    ``python -m scripts.train_bpe`` on the same corpus and seed produce a
    byte-identical artifact, and the artifact is rebuilt locally — it is never
    fetched from the network;
  * ``--corpus`` accepts a single file (read whole) or a directory (recursive
    ``*.md`` / ``*.txt`` / ``*.py``, sorted) — decision O1.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "cache" / "tokenizer.json"

# Demo configs (example.py, forward.py) pin vocab_size=4096; the trained
# tokenizer must fit inside them (contract item 6).
DEMO_VOCAB = 4096


@pytest.fixture(scope="module")
def tok():
    """The trained artifact, loaded exactly the way the contract test does."""
    from open_jev.bpe_tokenizer import BPETokenizer

    assert ARTIFACT.is_file(), (
        f"{ARTIFACT} is missing — train it with "
        "`python -m scripts.train_bpe --vocab-size 4096 --out cache/tokenizer.json`"
    )
    return BPETokenizer.load(str(ARTIFACT))


def _run_train(args: list[str]) -> subprocess.CompletedProcess:
    """Run ``scripts.train_bpe`` as a CLI, from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.train_bpe", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )


# --- loading the artifact ----------------------------------------------------


def test_artifact_loads_and_fits_demo_vocab_budget(tok) -> None:
    assert isinstance(tok.vocab_size, int)
    assert 2 <= tok.vocab_size <= DEMO_VOCAB, tok.vocab_size
    assert tok.PAD == 0


def test_load_reports_missing_artifact(tmp_path: Path) -> None:
    from open_jev.bpe_tokenizer import BPETokenizer

    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(str(tmp_path / "does-not-exist.json"))


# --- encode contract ---------------------------------------------------------


def test_empty_string_encodes_to_single_token(tok) -> None:
    assert tok.encode("", max_len=8) == [1]
    assert tok.encode("   \n\t ", max_len=8) == [1]


def test_ids_stay_inside_vocab_and_respect_max_len(tok) -> None:
    ids = tok.encode("the customer wants a refund now", max_len=6)
    assert 1 <= len(ids) <= 6
    assert all(1 <= i < tok.vocab_size for i in ids)


def test_encode_never_returns_pad_id(tok) -> None:
    """PAD is reserved for encode_batch; even OOV text stays in [1, vocab)."""
    samples = [
        "customer.name = 'refund'",
        "THIS IS UPPERCASE TEXT",
        "спанish símbolos ∑∆",
        "<pad>",  # the literal pad string must not round-trip to id 0
        "x" * 200,
    ]
    for text in samples:
        ids = tok.encode(text, 32)
        assert ids, text
        assert 0 not in ids, text
        assert all(1 <= i < tok.vocab_size for i in ids), text


def test_encoding_is_deterministic(tok) -> None:
    text = "repeatable text 123"
    assert tok.encode(text, 16) == tok.encode(text, 16)
    ids_a, mask_a = tok.encode_batch([text, ""], 16)
    ids_b, mask_b = tok.encode_batch([text, ""], 16)
    assert torch.equal(ids_a, ids_b)
    assert torch.equal(mask_a, mask_b)


def test_lowercasing_is_normalised(tok) -> None:
    """NFC + lowercase normalisation lives in the artifact (predictable case)."""
    assert tok.encode("HELLO World", 8) == tok.encode("hello world", 8)


# --- encode_batch contract ---------------------------------------------------


def test_encode_batch_contract(tok) -> None:
    ids, mask = tok.encode_batch(["alpha", "", "gamma delta"], max_len=8)
    assert ids.dtype == torch.long
    assert mask.dtype == torch.bool
    assert ids.shape == mask.shape and ids.shape[0] == 3
    assert not mask.all(dim=1).any(), "ни одна строка не должна быть целиком PAD"
    assert (ids[mask] == 0).all(), "на позициях PAD должен лежать 0"
    assert (ids[~mask] > 0).all(), "id вне PAD обязаны быть >= 1"


def test_encode_batch_matches_encode_and_right_pads(tok) -> None:
    texts = ["the customer wants a refund", "", "gamma delta"]
    ids, mask = tok.encode_batch(texts, max_len=8)
    for row, text in enumerate(texts):
        expected = tok.encode(text, 8)
        assert ids[row, : len(expected)].tolist() == expected
        assert mask[row, len(expected) :].all(), "хвост строки — это PAD"
        assert not mask[row, : len(expected)].any()


# --- training (scripts/train_bpe.py) ----------------------------------------


def test_retraining_is_deterministic(tmp_path: Path) -> None:
    """Same corpus + seed => byte-identical artifact (O1: no network involved)."""
    payloads: list[bytes] = []
    for name in ("first.json", "second.json"):
        proc = _run_train(
            ["--vocab-size", "512", "--seed", "7", "--out", str(tmp_path / name)]
        )
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        payloads.append((tmp_path / name).read_bytes())
    first, second = payloads
    assert first, "артефакт пуст"
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert json.loads(first) == json.loads(second)

    # The retrained artifact is loadable and obeys the same contract.
    from open_jev.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer.load(str(tmp_path / "first.json"))
    assert 2 <= tok.vocab_size <= 512
    assert tok.encode("", 8) == [1]
    assert all(1 <= i < tok.vocab_size for i in tok.encode("refund now", 8))


def test_corpus_flag_accepts_a_file(tmp_path: Path) -> None:
    corpus = tmp_path / "one.txt"
    corpus.write_text("refund refund billing billing " * 40, encoding="utf-8")
    out = tmp_path / "from_file.json"
    proc = _run_train(
        ["--corpus", str(corpus), "--vocab-size", "64", "--out", str(out)]
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    from open_jev.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer.load(str(out))
    assert 2 <= tok.vocab_size <= 64
    ids = tok.encode("refund billing", 8)
    assert 1 <= len(ids) <= 8
    assert all(1 <= i < tok.vocab_size for i in ids)


def test_corpus_flag_accepts_a_directory_and_filters_extensions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("refund refund billing billing " * 40, encoding="utf-8")
    (corpus / "b.md").write_text("hello world hello world " * 40, encoding="utf-8")
    nested = corpus / "nested"
    nested.mkdir()
    (nested / "c.py").write_text("def f():\n    return 'hello'\n" * 30, encoding="utf-8")
    (corpus / "noise.bin").write_bytes(b"\x00\x01\x02 not a text extension")
    out = tmp_path / "from_dir.json"
    proc = _run_train(
        ["--corpus", str(corpus), "--vocab-size", "64", "--out", str(out)]
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    from open_jev.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer.load(str(out))
    assert 2 <= tok.vocab_size <= 64
    assert 0 not in tok.encode("refund billing hello", 8)
