"""Red-first coverage for ``eval/compare.py`` (stage 9, step 5).

The tracked trace: one markdown file (``eval/COMPARISON.md`` in real runs)
assembled from TWO in-process ``eval.cli.main`` runs -- a random baseline and
the trained checkpoint -- plus ``hashlib.sha256`` over the raw checkpoint
bytes.

Fixtures are built locally in ``tmp_path`` (never in conftest): a 2-row
dataset from ``eval.dataset`` and a tiny stage-4 pipeline checkpoint saved
with ``pipeline.checkpoint.save_checkpoint`` (plain-dict cfg, same 9
JevConfig model fields as tests/test_checkpoint.py:9-11). ``--reports-dir``
points BOTH cli runs under ``tmp_path`` so the test stays hermetic -- the
default ``eval/out`` is reserved for real runs (gitignored).

Deliberately NO "trained beats random" assertion: 2 fixture rows at
n_bins=10 cannot guarantee it (flaky); the honest comparison lives in
``eval/COMPARISON.md`` on the 100-row holdout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import eval.compare
from eval.dataset import generate, write_jsonl
from open_jev.main import Jev, JevConfig
from pipeline.checkpoint import save_checkpoint

# The same 9 model fields as tests/test_checkpoint.py:9-11.
MODEL_CFG = dict(vocab_size=512, d_model=32, n_heads=4, d_ff=64,
                 n_state_layers=1, n_question_layers=1,
                 n_readout_layers=1, n_slots=4, max_state_len=64)


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.jsonl"
    write_jsonl(generate(2, seed=0), path)
    return path


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = Jev(JevConfig(**MODEL_CFG))
    return save_checkpoint(tmp_path / "ck.pt", model, dict(MODEL_CFG),
                           step=1, metrics={"ece": 0.1})


def _argv(tmp_path: Path, ds: Path, ck: Path, out: Path) -> list[str]:
    return ["--dataset", str(ds), "--checkpoint", str(ck), "--out", str(out),
            "--reports-dir", str(tmp_path / "reports")]


def test_compare_writes_tracked_markdown(tmp_path) -> None:
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    out = tmp_path / "COMPARISON.md"
    assert eval.compare.main(_argv(tmp_path, ds, ck, out)) == 0

    text = out.read_text(encoding="utf-8")
    assert "| metric | random | checkpoint |" in text   # both column headers
    assert "ece" in text
    # 64 hex chars = sha256 over the checkpoint BYTES (raw .pt, no digest inside)
    assert re.search(r"\b[0-9a-f]{64}\b", text)
    assert "checkpoint:" in text                        # trained model id
    assert "step" in text                               # training step row
    # Both raw reports landed under --reports-dir (gitignored in real runs).
    assert (tmp_path / "reports" / "random" / "report.json").is_file()
    assert (tmp_path / "reports" / "trained" / "report.json").is_file()


def test_missing_checkpoint_is_clean_error(tmp_path) -> None:
    ds = _dataset(tmp_path)
    missing = tmp_path / "nope.pt"
    out = tmp_path / "COMPARISON.md"
    # Clean SystemExit/ValueError naming the path, no traceback, no output file.
    with pytest.raises((SystemExit, ValueError)) as exc:
        eval.compare.main(_argv(tmp_path, ds, missing, out))
    assert str(missing) in str(exc.value)
    assert not out.exists()


def test_missing_dataset_is_clean_error(tmp_path) -> None:
    ck = _checkpoint(tmp_path)
    missing = tmp_path / "nope.jsonl"
    out = tmp_path / "COMPARISON.md"
    with pytest.raises((SystemExit, ValueError)) as exc:
        eval.compare.main(_argv(tmp_path, missing, ck, out))
    assert str(missing) in str(exc.value)
    assert not out.exists()
