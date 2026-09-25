#!/usr/bin/env python
"""Fetch Stanford Alpaca and turn it into ``pipeline.data`` train/holdout rows.

Usage::

    python -m scripts.fetch_data [--out data/real] [--seed 0]
                                 [--sample-size 600] [--train-size 500]
                                 [--url URL] [--from-local PATH]

Stage-06 pipeline (Шаг 4 «зелёный»):

* **source** — a JSON *array* of ``{instruction, input, output}`` objects.
  ``--from-local PATH`` reads that file and never touches the network (this
  is what the gate tests use); otherwise ``fetch()`` downloads the alpaca
  URL with stdlib ``urllib.request`` and caches the bytes at
  ``cache/alpaca_data.json`` (already gitignored), reusing the cache on
  later runs.
* **sample + split** — ``random.Random(seed).sample(pool, sample_size)``;
  the first ``--train-size`` sampled items become ``train.jsonl``, the
  remainder ``holdout.jsonl``. Sampling is without replacement, so the two
  splits never share a state.
* **transform** — each item becomes one row of the ``pipeline/data.py``
  schema: ``state`` (instruction/input clipped to 600 chars), the fixed
  three-question template (identical for every row — ``build_batch``
  refuses mixed question lists), and uniform soft targets that pass
  ``validate_soft_targets`` (sums to 1, never one-hot).
* **byte-stable output** — each row is
  ``json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n"`` in UTF-8,
  so two runs with the same seed write byte-identical files.

Errors: any bad input (broken JSON, non-array source, sample larger than
the pool) raises ``ValueError``; ``main`` prints ``str(exc)`` to stderr and
returns exit code 1. Standard library only — no new dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/"
    "alpaca_data.json"
)
DEFAULT_CACHE = REPO_ROOT / "cache" / "alpaca_data.json"

# Characters kept per state field; longer text is clipped (risk O2).
MAX_TEXT = 600


def template_questions() -> list[dict]:
    """The fixed question template, a fresh copy per call.

    Contract per spec «Контракт шаблонных вопросов» — identical for every
    row (``pipeline/data.py:153`` rejects mixed question lists): self-
    containment (choice, K=2), need for external context (noul, K=2) and
    output-format strictness (score, K=3).
    """
    return [
        {
            "type": "choice",
            "text": "Is the instruction executable with only the provided input?",
            "key": "self_contained",
            "options": ["self-contained", "needs-external-context"],
        },
        {
            "type": "noul",
            "text": "The requested answer depends on facts not given in the instruction.",
            "key": "requires_external_context",
        },
        {
            "type": "score",
            "text": "How strictly does the instruction pin down the output format?",
            "key": "output_format_specificity",
            "labels": ["unspecified", "loose", "strict"],
        },
    ]


def uniform_targets() -> list[list[float]]:
    """Uniform placeholder distribution per question (K = 2, 2, 3).

    Replaced by real teacher frequencies after distillation (stage-08);
    until then it satisfies ``validate_soft_targets``: floats summing to 1
    and never one-hot (max mass 0.5 < 1).
    """
    return [[0.5, 0.5], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]]


def _clip(value: Any) -> str:
    """Coerce a state field to ``str[:600]`` (``None``/missing → ``""``)."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value[:MAX_TEXT]


def transform(items: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Convert ``{instruction, input, output}`` items into schema rows.

    Only ``instruction``/``input`` survive (as the clipped ``state``);
    ``output`` is dropped — this stage prepares *judged* states, not
    answers. Each row carries a fresh copy of the template questions so no
    two rows alias the same dict.
    """
    rows: list[dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"item {i}: expected a JSON object with instruction/input/"
                f"output, got {type(item).__name__}"
            )
        rows.append(
            {
                "state": {
                    "instruction": _clip(item.get("instruction")),
                    "input": _clip(item.get("input")),
                },
                "questions": template_questions(),
                "targets": uniform_targets(),
            }
        )
    return rows


def load_source(path: str | Path) -> list[dict]:
    """Read the source file: a JSON array of alpaca-style objects.

    The source is an array (alpaca_data.json, the fixtures), *not* JSONL,
    so it goes through ``json.load``; anything unparsable or not a list
    fails with the ``"...: invalid JSON: ..."`` wording the CLI contract
    promises (and ``main`` prints to stderr, exit 1).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: invalid JSON: expected a JSON array, "
            f"got {type(data).__name__}"
        )
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{path}: invalid JSON: expected an object at index {i}, "
                f"got {type(item).__name__}"
            )
    return data


def fetch(url: str = DEFAULT_URL, cache: Path = DEFAULT_CACHE) -> Path:
    """Return the cached alpaca file, downloading it once if absent.

    The download lands in ``cache/alpaca_data.json`` (gitignored) via a
    ``.part`` file that is validated as a JSON array *before* the rename,
    so a truncated download can never poison the cache. stdlib only:
    ``urllib.request``.
    """
    if cache.is_file():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            part.write_bytes(response.read())
        load_source(part)  # validate before the cache becomes permanent
    except Exception:
        part.unlink(missing_ok=True)
        raise
    part.replace(cache)
    return cache


def sample_and_split(
    pool: Sequence[Mapping[str, Any]], sample_size: int, train_size: int, seed: int
) -> tuple[list, list]:
    """Deterministically sample ``sample_size`` items and split them.

    First ``train_size`` sampled items → train, the rest → holdout.
    Sampling is without replacement, so the splits cannot overlap.
    """
    if sample_size > len(pool):
        raise ValueError(
            f"sample-size {sample_size} exceeds the {len(pool)} items in the pool"
        )
    if train_size < 0:
        raise ValueError(f"train-size {train_size} must not be negative")
    if train_size > sample_size:
        raise ValueError(
            f"train-size {train_size} exceeds sample-size {sample_size} — "
            f"the holdout split would always be empty"
        )
    sampled = random.Random(seed).sample(list(pool), sample_size)
    return sampled[:train_size], sampled[train_size:]


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as sorted-key JSONL — byte-stable across runs."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="data/real",
        help="output directory for train.jsonl and holdout.jsonl "
        "(default: data/real)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the sampling RNG; same seed → byte-identical "
        "outputs (default: 0)",
    )
    parser.add_argument(
        "--from-local",
        metavar="PATH",
        default=None,
        help="read the source JSON array from PATH and skip the network "
        "(used by the offline gate)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500,
        help="rows in train.jsonl; the sampled remainder goes to "
        "holdout.jsonl (default: 500)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=600,
        help="items sampled from the pool before splitting; must be <= "
        "the pool size (default: 600)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="alpaca URL to download when --from-local is not given "
        f"(default: {DEFAULT_URL})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.from_local is not None:
            source = Path(args.from_local)
        else:
            source = fetch(args.url)
        pool = load_source(source)
        train_items, holdout_items = sample_and_split(
            pool, args.sample_size, args.train_size, args.seed
        )
        train_rows = transform(train_items)
        holdout_rows = transform(holdout_items)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_rows(out / "train.jsonl", train_rows)
        write_rows(out / "holdout.jsonl", holdout_rows)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"source: {source} ({len(pool)} items)")
    print(f"seed: {args.seed} sample-size: {args.sample_size}")
    print(f"train: {len(train_rows)} -> {out / 'train.jsonl'}")
    print(f"holdout: {len(holdout_rows)} -> {out / 'holdout.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
