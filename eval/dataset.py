"""Synthetic labeled dataset for the eval harness: JSONL I/O + generation.

Row schema (one JSON object per line)::

    {"state": {...}, "questions": [...], "targets": [[p0, p1, ...], ...]}

``targets[i]`` is a soft distribution (sums to 1) for ``questions[i]`` --
never a hard label: hard targets manufacture overconfidence, which is the
failure this model exists to avoid (see the RLCDLoss comment block,
open_jev/main.py:807-821).

Formula labeling (risk R2): there is no teacher yet, so ground truth comes
from an explicit formula over the row's own state. Even rows get *calibrated*
targets (the raw formula probability); odd rows get *miscalibrated* ones (the
same distribution sharpened by ``GAMMA``, i.e. overconfident). The regime is
a function of the row index, so a consumer can recover it without extra
fields. Everything is drawn from ``random.Random(seed)`` only, so the same
seed yields byte-identical output on any machine (risk R5).
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["load_jsonl", "write_jsonl", "generate", "main"]

#: Overconfidence exponent for miscalibrated rows (risk R2).
GAMMA = 4.0

#: Question template shared by every row; per-row answers live in `targets`.
QUESTIONS: list[dict] = [
    {"type": "noul", "text": "a > 0.5", "key": "a_gt_half"},
    {"type": "choice", "text": "which feature dominates",
     "options": ["a", "b"], "key": "dominant"},
    {"type": "score", "text": "how high is c",
     "labels": ["low", "mid", "high"], "key": "c_level"},
]


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, one per non-blank line."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    """Write dicts as one JSON object per line (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm(weights: Iterable[float]) -> list[float]:
    """Normalize to a distribution; uniform if the weights sum to <= 0."""
    w = [max(float(x), 0.0) for x in weights]
    total = sum(w)
    if total <= 0.0:
        return [1.0 / len(w)] * len(w)
    return [x / total for x in w]


def _sharpen(dist: Sequence[float], gamma: float = GAMMA) -> list[float]:
    """Push a distribution toward its argmax: overconfident, still sums to 1."""
    return _norm(x ** gamma for x in dist)


def generate(n: int, seed: int = 0) -> list[dict]:
    """Build ``n`` synthetic rows deterministically from ``seed``.

    Per-question formula targets:
      * noul  ``[1 - a, a]`` -- soft P("a > 0.5") taken as ``a`` itself;
      * choice ``_norm([a, b])`` -- which feature dominates;
      * score ``_rotate([0.5, 0.3, 0.2], c)`` -- ordinal mass rotated by ``c``.

    Even rows keep the raw formula (calibrated); odd rows sharpen it by
    ``GAMMA`` (miscalibrated). Only ``random.Random(seed)`` is consulted, in a
    fixed call order, so output depends on nothing else.
    """
    rng = random.Random(seed)
    base_score = [0.5, 0.3, 0.2]
    rows: list[dict] = []
    for i in range(n):
        a = round(rng.random(), 4)
        b = round(rng.random(), 4)
        c = rng.randint(0, 3)
        state = {
            "a": a,
            "b": b,
            "c": c,
            "flag": bool(rng.getrandbits(1)),
            "label": rng.choice(["x", "y", "z"]),
        }
        calibrated = [
            [1.0 - a, a],                       # noul
            _norm([a, b]),                      # choice over ["a", "b"]
            _norm(base_score[c:] + base_score[:c]),  # score, rotated by c
        ]
        if i % 2 == 0:
            targets = calibrated
        else:  # miscalibrated: same belief, sharpened into overconfidence
            targets = [_sharpen(t) for t in calibrated]
        rows.append({
            "state": state,
            "questions": list(QUESTIONS),
            "targets": targets,
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m eval.dataset --generate --n 200 --out data/eval/...``."""
    p = argparse.ArgumentParser(
        prog="python -m eval.dataset",
        description="Generate the synthetic JSONL eval dataset.",
    )
    p.add_argument("--generate", action="store_true",
                   help="required: actually write the dataset")
    p.add_argument("--n", type=int, default=200,
                   help="number of rows (default: 200)")
    p.add_argument("--seed", type=int, default=0,
                   help="generation seed (default: 0)")
    p.add_argument("--out", type=Path, default=Path("data/eval/synth.jsonl"),
                   help="output path (default: data/eval/synth.jsonl)")
    args = p.parse_args(argv)
    if not args.generate:
        p.error("--generate is required")
    if args.n < 0:
        p.error("--n must be >= 0")
    rows = generate(args.n, seed=args.seed)
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
