#!/usr/bin/env python
"""Train the BPE tokenizer artifact used by ``open_jev.bpe_tokenizer``.

Usage::

    python -m scripts.train_bpe [--corpus PATH] [--vocab-size 4096]
                                [--out cache/tokenizer.json] [--seed 0]

Corpus (decision O1 — no network access, nothing to download):

* **default** (no ``--corpus``): a fixed, deterministically ordered list of
  text files from this repository itself — ``README.md``, ``example.py``,
  ``forward.py``, ``open_jev/*.py``, ``eval/*.py``, ``benchmarks/README.md``.
  Each glob is expanded in sorted order, so two runs on the same tree read
  exactly the same bytes in exactly the same order.
* ``--corpus PATH``: ``PATH`` may be a **file** (read whole) or a **directory**
  (collected recursively over ``*.md`` / ``*.txt`` / ``*.py``, sorted by path).

Output: a JSON artifact for ``BPETokenizer.load``. By default it lands in
``cache/tokenizer.json``; ``cache/`` is deliberately *not* committed to git
(decision O4 — rebuild locally from the fixed corpus and seed instead).

Determinism: the file order is fixed and the ``tokenizers`` BPE trainer has no
randomness, so retraining yields a byte-identical artifact; ``--seed`` pins
``random`` bookkeeping so whole runs stay reproducible end to end.

Vocab size (decision O3): the default is 4096, matching the demo configs
(``example.py``, ``forward.py``); the trained size never exceeds the requested
one and always keeps ``0`` = ``<pad>`` and ``1`` = ``<unk>``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# Decision O1: the default corpus, in a fixed order (globs sorted internally).
DEFAULT_CORPUS: tuple[str, ...] = (
    "README.md",
    "example.py",
    "forward.py",
    "open_jev/*.py",
    "eval/*.py",
    "benchmarks/README.md",
)

# Extensions collected when --corpus points at a directory.
TEXT_SUFFIXES = frozenset({".md", ".txt", ".py"})


def default_corpus_files(root: Path = REPO_ROOT) -> list[Path]:
    """Deterministic default corpus: DEFAULT_CORPUS patterns, globs sorted."""
    files: list[Path] = []
    for pattern in DEFAULT_CORPUS:
        files.extend(sorted(p for p in root.glob(pattern) if p.is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def collect_corpus(corpus: str | None) -> list[Path]:
    """Resolve ``--corpus`` (file or directory) or the default repo corpus."""
    if corpus is None:
        files = default_corpus_files()
        if not files:
            raise SystemExit(
                "default corpus is empty (no README.md/example.py/... found) "
                "-- pass --corpus <PATH> explicitly"
            )
        return files

    path = Path(corpus)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        )
        if not files:
            raise SystemExit(f"--corpus {path}: no *.md/*.txt/*.py files found")
        return files
    raise SystemExit(f"--corpus path does not exist: {path}")


def build_tokenizer(vocab_size: int):
    """Fresh untrained BPE tokenizer + trainer honouring the id contract."""
    try:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "the 'tokenizers' package is required: "
            "pip install -r requirements-dev.txt"
        ) from exc

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Predictable normalisation: NFC first, then lowercase, so "HELLO" and
    # "hello" encode identically regardless of the input casing.
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC(), normalizers.Lowercase()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        # PAD must land on id 0 and UNK on id 1 (contract: ids in [1, vocab),
        # 0 reserved for padding in encode_batch).
        special_tokens=["<pad>", "<unk>"],
        show_progress=False,
    )
    return tokenizer, trainer


def train(corpus: str | None, vocab_size: int, seed: int):
    """Train on the resolved corpus; returns ``(tokenizer, corpus_files)``."""
    if vocab_size < 2:
        raise SystemExit("--vocab-size must be >= 2 (0=PAD, 1=UNK are reserved)")

    # The trainer itself is deterministic; the seed pins interpreter
    # bookkeeping so a whole run is reproducible end to end.
    random.seed(seed)

    files = collect_corpus(corpus)
    tokenizer, trainer = build_tokenizer(vocab_size)
    tokenizer.train([str(f) for f in files], trainer)

    actual = tokenizer.get_vocab_size()
    if actual > vocab_size:
        raise SystemExit(
            f"trainer produced vocab_size={actual} > requested {vocab_size}"
        )
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id != 0:
        raise SystemExit(f"<pad> got id {pad_id}; the contract requires 0")
    return tokenizer, files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        metavar="PATH",
        default=None,
        help="text file (read whole) or directory (recursive *.md/*.txt/*.py, "
        "sorted). Default: fixed list of repo text files (README.md, "
        "example.py, forward.py, open_jev/*.py, eval/*.py, "
        "benchmarks/README.md) — no network used.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="target vocabulary size incl. <pad>/<unk> (default: 4096, the "
        "demo configs' vocab_size; the config default 32000 also fits)",
    )
    parser.add_argument(
        "--out",
        default="cache/tokenizer.json",
        help="artifact path for BPETokenizer.load "
        "(default: cache/tokenizer.json, not committed to git)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for reproducible runs; corpus order is fixed, so the "
        "trained vocab does not depend on it (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer, files = train(args.corpus, args.vocab_size, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))

    print(f"corpus: {len(files)} file(s)")
    for path in files:
        print(f"  {path}")
    print(f"vocab_size: {tokenizer.get_vocab_size()} (requested {args.vocab_size})")
    print(f"seed: {args.seed}")
    print(f"out: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
