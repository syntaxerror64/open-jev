"""Stage-08 Шаги 1-2 (red -> green): ядро дистилляции — distill_rows/distill_file.

Четыре теста — это красные из stage 8, Шаги 1-2,
сделанные самодостаточными: ``fixture_rows`` строит строки схемы этапа 6 в
манере ``rows_from_fixture`` из tests/test_pipeline_data.py (мягкие
плейсхолдеры, один список вопросов во всех строках), а ``FakeTeacher`` /
``OneHotTeacher`` держат контракт ``pipeline.teacher.Teacher``
(``teach(states, questions, repeats) -> list[Tensor[B, K]]``) без
transformers и сети.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pipeline.data import load_jsonl, validate_soft_targets
from pipeline.distill import distill_file, distill_rows, row_key

QUESTIONS = [
    {"type": "noul", "text": "a > 0.5", "key": "a_gt_half"},
    {"type": "choice", "text": "which feature dominates",
     "options": ["a", "b"], "key": "dominant"},
    {"type": "score", "text": "how high is c",
     "labels": ["low", "mid", "high"], "key": "c_level"},
]


def fixture_rows(n: int = 4) -> list[dict]:
    """Строки схемы этапа 6 с мягкими плейсхолдерами (не one-hot).

    Одинаковый список вопросов в каждой строке — build_batch (data.py:153)
    отказывается смешивать списки; плейсхолдеры суммируются в 1 и проходят
    validate_soft_targets — дистилляция их подменяет.
    """
    rows = []
    for i in range(n):
        rows.append({
            "state": {"a": round((i + 1) / (n + 1), 4), "b": i % 4,
                      "flag": bool(i % 2)},
            "questions": [dict(q) for q in QUESTIONS],
            "targets": [[0.5, 0.5], [0.6, 0.4], [1 / 3, 1 / 3, 1 / 3]],
        })
    return rows


class FakeTeacher:
    """Фейковый учитель: мягкие 7/3-строки, без transformers."""

    def teach(self, states, questions, repeats=10):
        return [torch.tensor([[0.7, 0.3]] * len(states)) for _ in questions]


def write_src(path: Path, rows: list[dict]) -> Path:
    """Источник в байт-стабильном формате строк репозитория."""
    path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    return path


def test_distill_writes_valid_targets(tmp_path) -> None:
    rows_in = fixture_rows(4)
    rows_out = distill_rows(rows_in, FakeTeacher(), repeats=10)
    assert len(rows_out) == len(rows_in)
    for r_in, r in zip(rows_in, rows_out):
        # Каждая строка, которую distill запишет, проходит
        # validate_soft_targets ДО записи. Цель строки — плоский [K], а
        # validate_soft_targets требует [B, K], поэтому [K] валидируется как
        # батч B=1 (unsqueeze): первоисточник писал `torch.tensor(t)` без unsqueeze —
        # для 1-D [K] это "must be a [B, K] tensor"; формат on-disk остаётся
        # плоским [K], как читает build_batch (data.py:170-177).
        validate_soft_targets([torch.tensor(t).unsqueeze(0) for t in r["targets"]])
        assert len(r["targets"]) == len(r["questions"])
        assert r["state"] == r_in["state"]
        assert r["questions"] == rows_in[0]["questions"]
    # distill_rows не мутирует вход: плейсхолдеры источника на месте.
    assert rows_in[0]["targets"] == [[0.5, 0.5], [0.6, 0.4], [1 / 3, 1 / 3, 1 / 3]]


def test_rerun_skips_done_rows(tmp_path) -> None:
    rows = fixture_rows(4)
    n = len(rows)
    src = write_src(tmp_path / "src.jsonl", rows)
    out = tmp_path / "out.jsonl"
    s1 = distill_file(src, out, FakeTeacher(), repeats=10)
    assert s1 == {"rows": n, "written": n, "skipped": 0}
    # Resume-матчинг игнорирует цели: дистиллированная строка хэшируется
    # как её источник.
    assert [row_key(r) for r in load_jsonl(out)] == [row_key(r) for r in rows]
    s2 = distill_file(src, out, FakeTeacher(), repeats=10)
    assert s2["skipped"] == n and s2["written"] == 0
    assert len(load_jsonl(out)) == n


def test_distill_is_byte_stable(tmp_path) -> None:
    src = write_src(tmp_path / "src.jsonl", fixture_rows(4))
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    distill_file(src, a, FakeTeacher())
    distill_file(src, b, FakeTeacher())
    assert a.read_bytes() == b.read_bytes()
    assert len(a.read_text(encoding="utf-8").splitlines()) == 4


def test_one_hot_teacher_fails_loudly(tmp_path) -> None:
    class OneHotTeacher:
        def teach(self, states, questions, repeats=10):
            return [torch.tensor([[1.0, 0.0]] * len(states)) for _ in questions]

    src = write_src(tmp_path / "src.jsonl", fixture_rows(4))
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="one-hot|soft"):
        distill_file(src, out, OneHotTeacher(), repeats=10)
    assert not out.exists()
    # Ни tmp, ни промежуточных файлов: валидация идёт ДО записи (Шаг 2).
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src.jsonl"]
