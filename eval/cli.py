"""Eval CLI: score a labeled JSONL dataset and emit report.json + report.md.

Usage::

    python -m eval.cli --dataset data/eval/synth.jsonl --out eval/out/

Pipeline: load rows -> group rows that share one question spec (the model
applies one question list to every state, Jev.logits) -> one batched forward
per group, plus a second forward over ``shuffle_keys`` restatements of the
same states for the consistency metric -> aggregate ece/brier/nll/consistency
over questions -> build_report() -> write both renderings into --out.

Weights are random until stage 4, so the report carries ``model: "random"``
and the standing warning (risk R1); ``n_bins`` is pinned to 10 so numbers
stay comparable across runs (risk R4).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from eval.dataset import load_jsonl
from eval.metrics import brier, ece, nll
from eval.report import build_report, consistency
from eval.shift import shuffle_keys
from open_jev.main import Choice, Jev, JevConfig, Noul, Score

__all__ = ["evaluate", "main"]


def _build_questions(specs: Sequence[dict]) -> list:
    """Turn dataset question dicts into typed primitives of open_jev."""
    questions = []
    for spec in specs:
        kind = spec.get("type")
        if kind == "noul":
            questions.append(Noul(spec["text"], key=spec.get("key", "")))
        elif kind == "choice":
            questions.append(Choice(spec["text"], spec["options"],
                                    key=spec.get("key", "")))
        elif kind == "score":
            questions.append(Score(spec["text"], spec["labels"],
                                   key=spec.get("key", "")))
        else:
            raise ValueError(f"unknown question type {kind!r} in dataset")
    return questions


def _group_rows(rows: Sequence[dict]) -> list[tuple[list[dict], list[int]]]:
    """Split rows into groups sharing one question spec (batch per group)."""
    groups: dict[str, tuple[list[dict], list[int]]] = {}
    for i, row in enumerate(rows):
        if not all(k in row for k in ("state", "questions", "targets")):
            raise ValueError(f"row {i}: needs state, questions and targets")
        key = repr(row["questions"])  # spec equality => one shared forward
        groups.setdefault(key, ([], []))[0].append(row)
        groups[key][1].append(i)
    return list(groups.values())


def evaluate(rows: Sequence[dict], *, seed: int = 0,
             n_bins: int = 10) -> tuple[dict, dict]:
    """Score rows with a freshly seeded model; returns (metrics, meta).

    Metrics are means over questions of the per-question values, so a report
    row is always a scalar regardless of dataset shape.
    """
    if not rows:
        raise ValueError("dataset is empty - nothing to evaluate")
    torch.manual_seed(seed)
    model = Jev(JevConfig(vocab_size=1024, d_model=32, n_heads=4, d_ff=64,
                          n_state_layers=1, n_question_layers=1,
                          n_readout_layers=1, n_slots=4,
                          max_state_len=64)).eval()

    totals: dict[str, list[float]] = {"ece": [], "brier": [], "nll": [],
                                      "consistency": []}
    n_questions = 0
    with torch.no_grad():
        for rows_same, idxs in _group_rows(rows):
            questions = _build_questions(rows_same[0]["questions"])
            states = [r["state"] for r in rows_same]
            # Consistency compares predictions against semantic restatements
            # of the SAME states: key order carries no meaning (main.py:255).
            augmented = [shuffle_keys(r["state"], seed + i)
                         for r, i in zip(rows_same, idxs)]
            base = model.logits(states, questions)
            aug = model.logits(augmented, questions)

            for j, ((probs, _), (aug_probs, _)) in enumerate(zip(base, aug)):
                target = torch.tensor(
                    [r["targets"][j] for r in rows_same], dtype=torch.float32
                )
                if target.shape != probs.shape:
                    raise ValueError(
                        f"row targets[{j}] gives {tuple(target.shape)}, "
                        f"model predicts {tuple(probs.shape)}"
                    )
                totals["ece"].append(float(ece(probs, target, n_bins)))
                totals["brier"].append(float(brier(probs, target)))
                totals["nll"].append(float(nll(probs, target)))
                totals["consistency"].append(
                    consistency(probs, aug_probs)
                )
                n_questions += 1

    metrics = {name: sum(values) / len(values)
               for name, values in totals.items()}
    return metrics, {"rows": len(rows), "questions": n_questions,
                     "seed": seed}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: load dataset, evaluate, write report.json and report.md."""
    p = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="Evaluate the model on a labeled JSONL dataset.",
    )
    p.add_argument("--dataset", type=Path, default=Path("data/eval/synth.jsonl"))
    p.add_argument("--out", type=Path, default=Path("eval/out"),
                   help="output directory for report.json / report.md")
    p.add_argument("--seed", type=int, default=0,
                   help="model init + augmentation seed (default: 0)")
    p.add_argument("--n-bins", type=int, default=10,
                   help="ECE bins, pinned in report metadata (risk R4)")
    args = p.parse_args(argv)

    rows = load_jsonl(args.dataset)
    metrics, meta = evaluate(rows, seed=args.seed, n_bins=args.n_bins)
    report = build_report(metrics, model="random",
                          dataset=str(args.dataset), n_bins=args.n_bins,
                          **meta)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(report["json"], encoding="utf-8")
    (args.out / "report.md").write_text(report["markdown"], encoding="utf-8")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"wrote {args.out / 'report.json'} and {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
