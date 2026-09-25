"""JSONL rows -> validated soft-target batches for the stage-4 train loop.

Two rules from open_jev/main.py:807-821 drive this module:

1. NEVER train on hard labels. A one-hot target manufactures overconfidence --
   the exact failure this model exists to avoid -- so ``validate_soft_targets``
   rejects one-hots outright: targets must be soft ensemble frequencies (ten
   samples splitting 7/3 -> 0.7). On-disk rows therefore carry one
   distribution per question, each summing to 1.
2. Keep the I/O dependency-free. The spec forbids new dependencies and the
   repo has no YAML/TOML anywhere, so rows live in plain JSONL -- the format
   eval/dataset.py already established -- parsed with stdlib ``json``. The
   only third-party import here is torch, needed to build the [B, K] tensors
   that RLCDLoss consumes.

Row schema (one JSON object per line)::

    {"state": {...}, "questions": [{"type": "noul", ...}, ...],
     "targets": [[p0, p1, ...], ...]}

``targets[i]`` is the soft distribution for ``questions[i]`` (main.py:893-898,
RLCDLoss docstring). Questions may also arrive already built as
Noul/Choice/Score objects; :func:`build_batch` accepts both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from open_jev.main import Choice, Noul, Question, Score, StateValue

__all__ = ["Batch", "build_batch", "load_jsonl", "validate_soft_targets"]


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, one per non-blank line.

    Blank lines are skipped; anything that is not a JSON object fails at load
    time with the offending line number, rather than surfacing later as a
    confusing ``KeyError`` in the middle of a training run.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def validate_soft_targets(targets: Sequence[Tensor]) -> list[Tensor]:
    """Check every [B, K] target tensor; return them as float32.

    Raises ``ValueError`` when a tensor is not a batch of soft distributions:
    wrong rank, non-finite or negative mass, rows that do not sum to 1, or --
    the rule this function exists for -- a one-hot row. open_jev/main.py:807-817:
    hard labels manufacture overconfidence, so only soft teacher frequencies
    may pass; the message for that case names the fix ("soft").
    """
    checked: list[Tensor] = []
    for q, target in enumerate(targets):
        t = torch.as_tensor(target, dtype=torch.float32)
        if t.ndim != 2 or t.shape[-1] < 2:
            raise ValueError(
                f"question {q}: targets must be a [B, K] tensor with K >= 2, "
                f"got shape {tuple(t.shape)}"
            )
        if not torch.isfinite(t).all():
            raise ValueError(f"question {q}: targets contain NaN/inf")
        if bool((t < 0).any()):
            raise ValueError(
                f"question {q}: negative target mass -- soft targets must be "
                f"probabilities in [0, 1]"
            )
        sums = t.sum(-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=0.0):
            raise ValueError(
                f"question {q}: target rows must sum to 1 (soft distributions), "
                f"got {sums.tolist()}"
            )
        if bool((t.max(dim=-1).values >= 1.0 - 1e-6).any()):
            raise ValueError(
                f"question {q}: one-hot target rows are forbidden -- train on "
                f"soft teacher frequencies, never hard labels "
                f"(open_jev/main.py:807-817)"
            )
        checked.append(t)
    return checked


@dataclass
class Batch:
    """A micro-batch in exactly the order ``RLCDLoss.__call__`` takes them.

    ``loss_fn(model, batch.states, batch.questions, batch.targets)`` is the
    intended call. ``targets[q]`` is the float32 [B, K_q] soft distribution
    for ``questions[q]``; every row sums to 1 (guaranteed by
    :func:`validate_soft_targets`).
    """

    states: list[StateValue]
    questions: list[Question]
    targets: list[Tensor]


def build_batch(rows: Sequence[Mapping[str, Any]]) -> Batch:
    """Turn JSONL rows into (states, questions, targets) ready for RLCDLoss.

    * ``questions`` are normalized once from the on-disk dict form (eval/
      dataset.py's ``{"type": ...}``) -- or accepted as Noul/Choice/Score
      already -- so ``model.logits`` can consume them directly;
    * ``targets`` are stacked per question into float32 [B, K] and run
      through :func:`validate_soft_targets`, so a one-hot row dies here with
      a clear message instead of inside the loss;
    * every row must agree on its question list: one batch, one question set,
      otherwise the stacked tensors would silently mix different tasks.
    """
    if not rows:
        raise ValueError("build_batch: need at least one row")
    required = ("state", "questions", "targets")
    questions: list[Question] | None = None
    columns: list[list[Tensor]] = []
    states: list[StateValue] = []

    for i, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"row {i}: missing key(s) {sorted(missing)}")

        row_questions = _questions_of(row, i)
        if questions is None:
            if not row_questions:
                raise ValueError(f"row {i}: 'questions' must not be empty")
            questions = row_questions
            columns = [[] for _ in questions]
        elif row_questions != questions:
            raise ValueError(
                f"row {i}: questions differ from row 0 -- a batch shares one "
                f"question list"
            )

        row_targets = row["targets"]
        if not _is_listy(row_targets):
            raise ValueError(
                f"row {i}: 'targets' must be a list, "
                f"got {type(row_targets).__name__}"
            )
        if len(row_targets) != len(questions):
            raise ValueError(
                f"row {i}: {len(row_targets)} target rows for "
                f"{len(questions)} questions"
            )
        for q, target in enumerate(row_targets):
            t = torch.as_tensor(target, dtype=torch.float32)
            if t.ndim != 1:
                raise ValueError(
                    f"row {i}, question {q}: expected a 1-D soft target, "
                    f"got shape {tuple(t.shape)}"
                )
            columns[q].append(t)

        states.append(row["state"])

    assert questions is not None  # rows is non-empty, so it was initialised
    targets: list[Tensor] = []
    for q, column in enumerate(columns):
        widths = {t.shape[0] for t in column}
        if len(widths) != 1:
            raise ValueError(
                f"question {q}: ragged target widths across rows: "
                f"{sorted(widths)}"
            )
        targets.append(torch.stack(column))
    return Batch(states, questions, validate_soft_targets(targets))


def _is_listy(value: Any) -> bool:
    """True for list/tuple but not for str/bytes (which are also Sequences)."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _questions_of(row: Mapping[str, Any], i: int) -> list[Question]:
    """Coerce one row's ``questions`` into Noul/Choice/Score objects."""
    raw = row["questions"]
    if not _is_listy(raw):
        raise ValueError(
            f"row {i}: 'questions' must be a list, got {type(raw).__name__}"
        )
    return [_coerce_question(q) for q in raw]


def _coerce_question(q: Any) -> Question:
    """On-disk dict (eval/dataset.py style) or built Question -> Question.

    A bare string is read as a noul statement: with no options or labels
    declared, K=2 is the only primitive a lone piece of text can address.
    """
    if isinstance(q, (Noul, Choice, Score)):
        return q
    if isinstance(q, str):
        return Noul(q)
    if isinstance(q, Mapping):
        kind = q.get("type")
        text = q.get("text")
        if not isinstance(text, str):
            raise ValueError(f"question {q!r}: missing string 'text'")
        key = q.get("key", "")
        if kind == "noul":
            return Noul(text, key=key)
        if kind == "choice":
            # Choice validates >= 2 unique options itself (main.py:149-154).
            return Choice(text, options=list(q.get("options") or []), key=key)
        if kind == "score":
            # Score validates >= 2 levels itself (main.py:169-171).
            return Score(text, labels=list(q.get("labels") or []), key=key)
        raise ValueError(
            f"question {q!r}: unknown type {kind!r} "
            f"(expected 'noul', 'choice' or 'score')"
        )
    raise ValueError(
        f"question {q!r}: expected a question dict or Question, "
        f"got {type(q).__name__}"
    )
