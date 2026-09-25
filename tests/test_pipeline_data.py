"""Stage-4, step 1 (red -> green): TrainConfig + the data layer's contract.

The four tests below are the code from stage 4,
Шаг 1, copied verbatim; `rows_from_fixture` is the local helper the
second test refers to (it builds eval/dataset.py-style rows whose targets are
soft ensemble vote splits -- integer counts turned into a distribution, never
one-hot).
"""

from __future__ import annotations

import pytest, torch
from pipeline.config import TrainConfig
from pipeline.data import build_batch, load_jsonl, validate_soft_targets


def rows_from_fixture(n: int = 4) -> list[dict]:
    """Build ``n`` rows in the eval/dataset.py shape with SOFT targets.

    Each target is an ensemble vote split over 10 samples -- 7/3 becomes
    [0.7, 0.3] exactly as in open_jev/main.py:810-812 -- and every class gets
    at least one vote, so no row is one-hot (the rule
    ``validate_soft_targets`` enforces). Question dicts mirror the QUESTIONS
    template of eval/dataset.py; states are small JSON-friendly scalars.
    """
    questions = [
        {"type": "noul", "text": "a > 0.5", "key": "a_gt_half"},
        {"type": "choice", "text": "which feature dominates",
         "options": ["a", "b"], "key": "dominant"},
        {"type": "score", "text": "how high is c",
         "labels": ["low", "mid", "high"], "key": "c_level"},
    ]
    # Integer votes per question, rotated across rows; counts >= 1 and
    # summing to 10, so every distribution is soft (never a vertex).
    votes = [
        [[7, 3], [6, 4], [8, 2], [5, 5], [9, 1]],
        [[4, 6], [5, 5], [7, 3], [3, 7], [6, 4]],
        [[5, 3, 2], [4, 4, 2], [6, 3, 1], [2, 4, 4], [3, 5, 2]],
    ]
    rows: list[dict] = []
    for i in range(n):
        targets = []
        for question_votes in votes:
            counts = question_votes[i % len(question_votes)]
            targets.append([c / sum(counts) for c in counts])
        rows.append({
            "state": {"a": round((i + 1) / (n + 1), 4), "b": round(1 - (i + 1) / (n + 1), 4),
                      "c": i % 4, "flag": bool(i % 2)},
            "questions": [dict(q) for q in questions],
            "targets": targets,
        })
    return rows


def test_loader_returns_states_questions_targets(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text('{"state": {"a": 1}, "questions": ["q?"], '                 '"targets": [[0.7, 0.3]]}\n', encoding="utf-8")
    rows = load_jsonl(p)
    assert len(rows) == 1 and set(rows[0]) >= {"state", "questions", "targets"}

def test_build_batch_targets_sum_to_one() -> None:
    # Первоисточник (Шаг 1) писал `t.sum() == 1.0`, но batch.targets[q] — тензор [B, K]
    # (контракт RLCDLoss), и сумма всех элементов равна B. Смысл проверки:
    # каждая СТРОКА батча — мягкое распределение, суммирующееся в 1.
    batch = build_batch(rows_from_fixture(n=4))
    for t in batch.targets:
        assert t.sum(dim=-1).tolist() == pytest.approx([1.0] * t.shape[0], abs=1e-6)
        assert t.min().item() >= 0.0

def test_one_hot_targets_are_rejected() -> None:
    # main.py:807-817: one-hot производит переуверенность — запрещён
    with pytest.raises(ValueError, match="soft"):
        validate_soft_targets([torch.tensor([[1.0, 0.0]])])

def test_config_roundtrip(tmp_path) -> None:
    cfg = TrainConfig(steps=5, lr=1e-4, seed=0)
    cfg.save(tmp_path / "c.json"); assert TrainConfig.load(tmp_path / "c.json") == cfg
