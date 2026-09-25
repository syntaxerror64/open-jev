"""Stage-08 Шаг 3 (red -> green): the ``python -m pipeline.distill`` CLI.

Spec: stage 8, Шаг 3 — ``main()`` layered on agent А's
core (``distill_file`` / ``distill_rows`` / ``row_key``, which stay untouched)
and providing:

* flags ``--data --out --teacher-json --seed`` PLUS individual overrides of
  the teacher keys ``kind/model_id/mode/temperature/max_new_tokens/repeats``
  (flag wins over the JSON block; ``--seed`` is the fallback injected the way
  ``pipeline/train.py:216-218`` does it),
* progress with ``ETA`` per taught row on stderr and a final
  ``rows/written/skipped/elapsed`` summary on stderr — a second identical run
  resumes and mentions ``skipped``,
* scripts/fetch_data.py-style errors: ``ValueError``/``OSError`` → message on
  stderr, exit 1, ``--out`` never created.

Every CLI check goes through ``_run_cli`` (a subprocess with ``cwd=REPO``, the
``tests/test_fetch_data.py:25-28`` pattern): the module form *is* the contract
under test, so a missing ``main`` shows up exactly the way a user would see it.
The last test guards the Шаг-4 fixture this agent ships
(``tests/fixtures/precomputed.jsonl``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from pipeline.data import build_batch, load_jsonl
from scripts.fetch_data import template_questions, uniform_targets

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "precomputed.jsonl"


def _run_cli(args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run ``python <args>`` from the repo root (pattern: test_fetch_data.py:25)."""
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=timeout,
    )


def _write_src(path: Path, n: int = 4) -> Path:
    """Small stage-6 JSONL: ``{state, questions, targets}``, uniform placeholders.

    The question list is the fetch_data.py template and is IDENTICAL on every
    row (pipeline/data.py:153 refuses mixed lists); placeholders are soft, so
    ``build_batch`` inside ``distill_file`` validates the source before the
    teacher ever runs.
    """
    rows = [
        {
            "state": {"instruction": f"Sum the first {i + 1} prime numbers",
                      "input": ""},
            "questions": template_questions(),
            "targets": uniform_targets(),
        }
        for i in range(n)
    ]
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _args(src: Path, out: Path, *extra: str) -> list[str]:
    """Spec's argv (Шаг 3, lines 76-78) with room for extra flags."""
    return [
        "-m", "pipeline.distill",
        "--data", str(src),
        "--out", str(out),
        "--teacher-json", '{"kind":"stub","seed":0}',
        "--seed", "0",
        *extra,
    ]


def test_cli_stub_teacher_exit0_and_resume(tmp_path) -> None:
    """Exit 0, ``--out`` written, ETA on stderr; rerun = resume with skipped."""
    src = _write_src(tmp_path / "src.jsonl")
    out = tmp_path / "out.jsonl"
    args = _args(src, out)

    p = _run_cli(args)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert out.is_file(), f"--out не создан:\n{p.stderr}"
    assert "ETA" in p.stderr, f"прогресс с ETA обязан идти в stderr:\n{p.stderr}"
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4

    p2 = _run_cli(args)  # повторный запуск = resume на практике
    assert p2.returncode == 0, f"stdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"
    assert "skipped" in p2.stderr, f"итог resume обязан упоминать skipped:\n{p2.stderr}"
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


def test_cli_flags_override_teacher_json(tmp_path) -> None:
    """Precedence: an individual flag beats the JSON value; ``--seed`` is a fallback.

    ``--repeats 3`` must win over ``"repeats": 5`` from ``--teacher-json``,
    while a ``seed`` INSIDE the JSON pins the run (``kwargs.setdefault`` —
    train.py:216-218), i.e. ``--seed 0`` does not override it.
    """
    src = _write_src(tmp_path / "src.jsonl")
    out = tmp_path / "out.jsonl"
    p = _run_cli(
        [
            "-m", "pipeline.distill",
            "--data", str(src),
            "--out", str(out),
            "--teacher-json", '{"kind":"stub","seed":7,"repeats":5}',
            "--kind", "stub",
            "--repeats", "3",
            "--seed", "0",
        ]
    )
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert '"repeats": 3' in p.stderr, f"--repeats обязан выигрывать у JSON:\n{p.stderr}"
    assert '"seed": 7' in p.stderr, f"seed из JSON обязан пиниться:\n{p.stderr}"
    assert out.is_file()


def test_cli_unknown_teacher_key_exits_1_without_out(tmp_path) -> None:
    """A typo'd ``--teacher-json`` key fails loudly BEFORE make_teacher/write."""
    src = _write_src(tmp_path / "src.jsonl")
    out = tmp_path / "out.jsonl"
    p = _run_cli(
        ["-m", "pipeline.distill", "--data", str(src), "--out", str(out),
         "--teacher-json", '{"kind":"stub","bogus":1}', "--seed", "0"]
    )
    assert p.returncode == 1, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "unknown teacher config key(s)" in p.stderr, p.stderr
    assert not out.exists(), "ошибка учителя не должна создавать --out"


def test_cli_precomputed_is_not_a_live_teacher(tmp_path) -> None:
    """``kind=precomputed`` is a TRAIN mode, never a live distill teacher."""
    src = _write_src(tmp_path / "src.jsonl")
    out = tmp_path / "out.jsonl"
    p = _run_cli(_args(src, out, "--kind", "precomputed"))
    assert p.returncode == 1, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert "unknown teacher kind" in p.stderr, p.stderr
    assert not out.exists()


def test_precomputed_fixture_is_stage6_and_soft() -> None:
    """Шаг-4 fixture: 8-16 stage-6 rows with REAL soft (7/3) targets.

    ``build_batch`` is the contract the consumer test relies on
    (``data_path="tests/fixtures/precomputed.jsonl"``, ``batch_size=2``): one
    shared question list, one 1-D target per question, every row summing to 1,
    every class holding at least one vote — soft, never one-hot — and
    byte-stable ``sort_keys`` lines like every other JSONL this repo writes.
    """
    rows = load_jsonl(FIXTURE)
    assert 8 <= len(rows) <= 16, f"ожидали 8-16 строк, получили {len(rows)}"
    assert set(rows[0]) == {"state", "questions", "targets"}
    assert set(rows[0]["state"]) == {"instruction", "input"}
    assert rows[0]["questions"] == template_questions()

    batch = build_batch(rows)  # identical questions + validate_soft_targets
    for t in batch.targets:    # [B, K] float32
        assert t.dtype == torch.float32
        assert t.shape[0] == len(rows)
        assert torch.allclose(t.sum(-1), torch.ones(len(rows)), atol=1e-4, rtol=0.0)
        assert t.min().item() >= 0.0
        assert (t > 0.0).all(), "каждый класс обязан получить >= 1 голос"
        assert (t < 1.0 - 1e-6).all(), "цели мягкие (7/3), не one-hot"

    lines = FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == len(rows)
    for line, row in zip(lines, rows):
        assert line == json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
