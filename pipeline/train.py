"""Stage-4 training loop and CLI: batch -> RLCDLoss -> AdamW -> schedule ->
clip -> periodic eval -> checkpoint.

One run in fixed order (stage 4, Декомпозиция,
agent B): sample a micro-batch in a *fixed* order, get SOFT targets from the
teacher layer (never the dataset's hard-ish rows -- open_jev/main.py:807-821),
forward through ``Jev.logits`` inside ``RLCDLoss``, ``backward``, warmup-scaled
LR, ``clip_grad_norm_``, AdamW step, and every ``eval_every`` steps a report
from the stage-2 harness (``eval.metrics``, risk R4). At the end (and on every
CLI run) ``checkpoint.pt`` + ``metrics.json`` land in ``out_dir``.

Data and teacher, given open questions O1/O2 (no corpus, no API key exists):

* rows come from ``cfg.data_path`` if that file exists, otherwise a synthetic
  set from ``eval/dataset.generate`` seeded by ``cfg.seed`` is materialised
  there (risk R3; ``data/train/`` is gitignored) -- except under
  ``kind="precomputed"``, where a missing file is a loud ``ValueError``
  before any generation;
* the teacher is built from ``cfg.teacher`` (stage 7, Шаг 2):
  ``make_teacher(kind, **block kwargs)`` with ``seed=cfg.seed`` injected when
  the block does not pin one, and ``repeats=cfg.teacher.get("repeats",
  cfg.teacher_repeats)`` feeding ``teach(...)``. The default block
  ``{"kind": "stub"}`` reproduces the old hardcoded
  ``make_teacher("stub", seed=cfg.seed)`` call byte for byte, so configs
  written before the field existed train exactly as before;
* with ``kind="precomputed"`` no teacher is built and no uniform substitution
  is applied: the rows' own soft targets reach the loss verbatim, guarded by
  ``build_batch``'s ``validate_soft_targets``. For every other kind the
  dataset's formula targets are *not* used for training -- some are
  deliberately sharpened to near one-hot, which ``validate_soft_targets``
  (rightly) rejects as hard labels -- so they only serve as the reference for
  the periodic eval, exactly like stage 2's ``eval.cli``.

Reproducibility (risk R3): one ``cfg.seed`` for ``torch`` + ``random``, sample
order drawn from a *local* ``random.Random(cfg.seed)`` (immune to global RNG
drift), and the seed + torch version recorded in every checkpoint (done by
``pipeline.checkpoint.save_checkpoint``).

Resume: ``train(cfg, resume=path)`` restores weights, AdamW state and
``step_start`` from the checkpoint; the model *architecture* always comes from
the checkpoint's config (that is what the weights describe), while the loop
knobs (``steps``, ``lr``, ...) come from the passed ``cfg``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from eval.dataset import generate as generate_rows, write_jsonl
from eval.metrics import brier, ece, nll
from open_jev.main import (
    Choice,
    Jev,
    JevConfig,
    Noul,
    Question,
    RLCDLoss,
    Score,
    StateValue,
)
from pipeline.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from pipeline.config import TrainConfig
from pipeline.data import build_batch, load_jsonl, validate_soft_targets
from pipeline.teacher import make_teacher

__all__ = ["History", "StepRecord", "main", "train"]

#: Synthetic pool bounds when ``cfg.data_path`` does not exist (risk O1).
_MIN_ROWS = 64
_MAX_ROWS = 10_000


@dataclass
class StepRecord:
    """One optimised step: what the ``history[i].loss`` reads."""

    step: int
    loss: float
    lr: float
    parts: dict[str, float] = field(default_factory=dict)


class History(list):
    """``list[StepRecord]`` plus ``step_start`` -- the resume cursor.

    A plain list keeps the assertions (``history[-1].loss <
    history[0].loss``) working; ``step_start`` is the global step the run
    *began* at (0 for a fresh run, the checkpoint's step after ``resume=`` --
    the resume test asserts ``run2.step_start == 5``).
    """

    def __init__(
        self, records: Sequence[StepRecord] = (), *, step_start: int = 0
    ) -> None:
        super().__init__(records)
        self.step_start = step_start


def train(
    cfg: TrainConfig, *, out: str | Path | None = None, resume: str | Path | None = None
) -> History:
    """Run ``cfg.steps`` optimisation steps; write checkpoint + metrics.

    Args:
        cfg: the run configuration (loop, model shape via ``model_kwargs``).
        out: artifact directory; defaults to ``cfg.out_dir``. Receives
            ``checkpoint.pt`` and ``metrics.json``.
        resume: a ``checkpoint.pt`` to continue from -- restores weights,
            optimizer state and the step counter (``history.step_start``).

    Returns:
        The run's :class:`History`: one record per step performed, and
        ``step_start`` set to where the run picked up.
    """
    # Risk R2 (CPU speed): the TINY loop is two orders of magnitude faster
    # single-threaded -- torch's default intra-op threads burn more time in
    # barrier spin-waits than in math on a loaded box (measured: forward pass
    # 1777ms with the default 4 threads vs 9ms with 1). Same choice as
    # benchmarks/cli.py:118; the host process's setting is restored on the
    # way out, and a fixed thread count also removes a source of numeric
    # jitter from the seeded run (risk R3).
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _train(cfg, out=out, resume=resume)
    finally:
        torch.set_num_threads(previous_threads)


def _train(
    cfg: TrainConfig, *, out: str | Path | None, resume: str | Path | None
) -> History:
    """The actual loop; called by :func:`train` with threads pinned to 1."""
    # Risk R3: one seed for torch + random, drawn before the model is built so
    # initialisation and dropout follow a fixed stream.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Teacher selection (stage 7, Шаг 2): the block names the
    # backend and carries its kwargs; ``kind`` and ``repeats`` are loop
    # directives, so they never reach the backend constructor. Work on a copy:
    # ``cfg.teacher`` must survive untouched into the checkpoint.
    teacher_cfg = dict(cfg.teacher)
    kind = teacher_cfg.pop("kind", "stub")
    repeats = teacher_cfg.pop("repeats", cfg.teacher_repeats)

    if kind == "precomputed":
        # Verbatim row targets, no backend, NO synthetic fallback: generating
        # rows here would train on uniform placeholders instead of the
        # distilled targets and mask the missing dataset (Шаг 2, test 5).
        precomputed_path = Path(cfg.data_path)
        if not precomputed_path.exists():
            raise ValueError(
                f"{precomputed_path}: dataset file not found -- "
                f"teacher kind 'precomputed' trains on the row targets in "
                f"cfg.data_path and never falls back to synthetic generation"
            )

    out_dir = Path(out) if out is not None else Path(cfg.out_dir)
    rows = _load_rows(cfg)
    if kind == "precomputed":
        # Rows keep their own targets; build_batch validates them (soft,
        # non-one-hot) and they flow to RLCDLoss as-is.
        train_rows = rows
    else:
        # Placeholder rows keep build_batch validating/coercing states+questions;
        # the real training targets are produced by the teacher right after.
        train_rows = [_with_uniform_targets(row) for row in rows]
    order = _sample_order(len(rows), cfg.seed)
    eval_states, eval_questions, eval_targets = _eval_batch(rows, train_rows, cfg)

    step_start = 0
    if resume is not None:
        ckpt = read_checkpoint(resume)
        # Architecture follows the checkpoint (that is what its weights are);
        # loop knobs (steps/lr/...) follow the caller's cfg.
        resume_cfg = TrainConfig(**ckpt["config"])
        model = load_checkpoint(
            resume, model_factory=lambda: Jev(JevConfig(**resume_cfg.model_kwargs()))
        )
        if "optimizer" not in ckpt:
            raise ValueError(
                f"{resume}: no optimizer state in checkpoint -- cannot resume "
                "training (re-save it through pipeline.train or "
                "save_checkpoint(..., optimizer=...))"
            )
        step_start = int(ckpt["step"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        # load_state_dict restores the *saved* hyperparameters; the new run's
        # cfg wins for the knobs it owns (LR is re-asserted every step anyway).
        for group in optimizer.param_groups:
            group["weight_decay"] = cfg.weight_decay
    else:
        model = Jev(JevConfig(**cfg.model_kwargs()))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    model.train()

    if kind == "precomputed":
        # No backend at all: the targets come from ``built.targets`` below.
        teacher = None
    else:
        kwargs = dict(teacher_cfg)
        kwargs.setdefault("seed", cfg.seed)  # injected unless the block pins it
        teacher = make_teacher(kind, **kwargs)
    loss_fn = RLCDLoss()
    history = History(step_start=step_start)
    end_step = step_start + cfg.steps
    last_eval: dict[str, float] = {}

    for step in range(step_start + 1, end_step + 1):
        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = _batch_of(train_rows, order, step, cfg.batch_size)
        built = build_batch(batch)
        if teacher is None:
            # precomputed: build_batch already ran validate_soft_targets.
            targets = built.targets
        else:
            targets = validate_soft_targets(
                teacher.teach(built.states, built.questions, repeats=repeats)
            )
        loss = loss_fn(model, built.states, built.questions, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        parts = dict(loss_fn.parts)  # main.py:957-964: nll/brier/evidential/ece/...
        history.append(
            StepRecord(step=step, loss=float(loss.item()), lr=lr, parts=parts)
        )
        print(
            f"step {step}/{end_step} lr={lr:g} "
            + " ".join(f"{k}={v:.4f}" for k, v in parts.items()),
            flush=True,
        )

        if step % cfg.eval_every == 0 or step == end_step:
            last_eval = _evaluate(model, eval_states, eval_questions, eval_targets)
            print(
                f"  eval@{step} " + " ".join(f"{k}={v:.4f}" for k, v in last_eval.items()),
                flush=True,
            )

    final = history[-1]
    metrics = {
        "step": final.step,
        "steps": len(history),
        "step_start": history.step_start,
        "seed": cfg.seed,
        "torch_version": str(torch.__version__),
        "final": final.parts,
        "eval": last_eval,
        "history": [
            {"step": r.step, "loss": r.loss, "lr": r.lr} for r in history
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / "checkpoint.pt",
        model,
        cfg,
        final.step,
        {"train": final.parts, "eval": last_eval},
        optimizer=optimizer,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"done: step {history.step_start} -> {final.step} | loss "
        f"{history[0].loss:.4f} -> {final.loss:.4f} | "
        f"checkpoint: {out_dir / 'checkpoint.pt'}",
        flush=True,
    )
    return history


def _load_rows(cfg: TrainConfig) -> list[dict]:
    """Dataset rows: ``cfg.data_path`` if it exists, else fresh synthetic data.

    Risk O1 says no real corpus exists, so a missing file is filled from
    ``eval/dataset.generate(n, seed=cfg.seed)`` and written to ``data_path``
    (reproducible artifact; ``data/train/`` is gitignored). An existing file
    is trusted as-is -- delete it to regenerate under a different seed.
    The fallback is for teacher-driven runs only: ``_train`` refuses to call
    this with ``kind="precomputed"`` when the file is missing, because
    substituting synthetic rows there would hide the absence of the very
    targets the run is supposed to train on.
    """
    path = Path(cfg.data_path)
    if path.exists():
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"{path}: dataset is empty")
        return rows
    n_rows = min(max(_MIN_ROWS, cfg.steps * cfg.batch_size), _MAX_ROWS)
    rows = generate_rows(n_rows, seed=cfg.seed)
    write_jsonl(rows, path)
    return rows


def _sample_order(n: int, seed: int) -> list[int]:
    """Fixed sample order from a *local* RNG (risk R3: reproducibility)."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def _batch_of(
    rows: Sequence[dict], order: Sequence[int], step: int, batch_size: int
) -> list[dict]:
    """The ``batch_size`` rows for ``step``, cycling through the fixed order.

    Indexing by the *global* step (not a per-run counter) is what makes a
    resumed run continue with the samples the unresumed run would have seen.
    """
    n = len(rows)
    base = (step - 1) * batch_size
    return [rows[order[(base + j) % n]] for j in range(batch_size)]


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup over ``cfg.warmup_steps``, then flat ``cfg.lr``."""
    if cfg.warmup_steps > 0 and step <= cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    return cfg.lr


def _with_uniform_targets(row: dict) -> dict:
    """Copy of ``row`` with uniform targets -- a valid stand-in for schema.

    ``build_batch`` insists on targets, but the dataset's own formula targets
    must NOT reach training: the odd rows of eval/dataset.generate are
    sharpened to near one-hot, which ``validate_soft_targets`` rejects as hard
    labels (open_jev/main.py:807-817). Uniform rows pass validation, so
    build_batch keeps doing its real job -- coercing questions, checking the
    rows agree -- while the teacher supplies the actual soft targets.
    """
    out = dict(row)
    out["targets"] = [[1.0 / k] * k for k in _widths(row["questions"])]
    return out


def _widths(questions: Sequence[Any]) -> list[int]:
    """K per question: 2 for a noul, else the declared arity (main.py:790-792)."""
    widths: list[int] = []
    for q in questions:
        if isinstance(q, Noul) or isinstance(q, str):
            widths.append(2)
        elif isinstance(q, Choice):
            widths.append(len(q.options))
        elif isinstance(q, Score):
            widths.append(len(q.labels))
        elif isinstance(q, dict):
            kind = q.get("type")
            if kind in (None, "noul"):
                widths.append(2)
            elif kind == "choice":
                widths.append(len(q.get("options") or []))
            elif kind == "score":
                widths.append(len(q.get("labels") or []))
            else:
                raise ValueError(f"unknown question type {kind!r} in dataset")
        else:
            raise ValueError(f"unknown question {q!r} in dataset")
    return widths


def _eval_batch(
    rows: Sequence[dict], train_rows: Sequence[dict], cfg: TrainConfig
) -> tuple[list[StateValue], list[Question], list[Tensor]]:
    """A fixed slice of rows -> (states, questions, formula/disk targets).

    States/questions are built through ``build_batch`` (placeholder targets,
    so validation runs); the reference targets come from the *original* rows,
    mirroring stage-2 ``eval.cli``, which scores against the dataset labels.
    """
    n = min(len(rows), max(cfg.batch_size, 8))
    built = build_batch(list(train_rows[:n]))
    columns = list(zip(*(row["targets"] for row in rows[:n])))
    targets = [torch.tensor(col, dtype=torch.float32) for col in columns]
    return built.states, built.questions, targets


def _evaluate(
    model: Jev,
    states: Sequence[StateValue],
    questions: Sequence[Question],
    targets: Sequence[Tensor],
) -> dict[str, float]:
    """Periodic evaluation through the stage-2 harness (risk R4).

    ece/brier/nll from ``eval.metrics`` -- the same functions the stage-2
    report uses, so numbers stay comparable across stages. The model is put
    back into whatever mode it was in (dropout must not leak into eval).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            preds = model.logits(states, questions)
            if len(preds) != len(targets):
                raise ValueError(
                    f"eval: {len(preds)} predictions vs {len(targets)} target columns"
                )
            out: dict[str, float] = {}
            for name, fn in (("ece", ece), ("brier", brier), ("nll", nll)):
                values = []
                for i, ((probs, _), target) in enumerate(zip(preds, targets)):
                    if probs.shape != target.shape:
                        raise ValueError(
                            f"eval question {i}: model predicts {tuple(probs.shape)} "
                            f"but targets are {tuple(target.shape)}"
                        )
                    values.append(float(fn(probs, target)))
                out[name] = sum(values) / len(values)
            return out
    finally:
        if was_training:
            model.train()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m pipeline.train --config ... [--steps N] [--resume P]``."""
    p = argparse.ArgumentParser(
        prog="python -m pipeline.train",
        description="Pretrain/distil Jev on soft teacher targets (stage 4).",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="TrainConfig JSON (default: built-in TINY defaults)",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="override the config's step count (e.g. 30 for the gate)",
    )
    p.add_argument(
        "--resume", type=Path, default=None,
        help="checkpoint.pt to continue from (restores step + optimizer)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="artifact directory (default: the config's out_dir)",
    )
    args = p.parse_args(argv)

    if args.steps is not None and args.steps < 1:
        p.error("--steps must be >= 1")
    if args.config is not None:
        cfg = TrainConfig.load(args.config)
    elif args.resume is not None:
        # No --config: the run being resumed defines the run.
        cfg = TrainConfig(**read_checkpoint(args.resume)["config"])
    else:
        cfg = TrainConfig()
    if args.steps is not None:
        cfg.steps = args.steps

    train(cfg, out=args.out, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
