"""Assemble the tracked random-vs-checkpoint comparison (stage 9, step 5).

Usage::

    python -m eval.compare --dataset data/real/holdout.targets.jsonl \
        --checkpoint runs/real/checkpoint.pt --out eval/COMPARISON.md

Runs ``eval.cli.main`` twice **in-process** -- no subprocess, so the whole
comparison is deterministic, offline and runs under the pre-push gate --
first a random baseline, then the trained checkpoint, into
``<reports-dir>/random`` and ``<reports-dir>/trained`` (default
``eval/out/{random,trained}``, both gitignored). It then reads the two
``report.json`` files, hashes the raw checkpoint BYTES with
``hashlib.sha256`` (the pipeline ``.pt`` carries no embedded digest) and
assembles one markdown file: the reproduce commands, a
``| metric | random | checkpoint |`` table over ece/brier/nll/consistency
plus rows/questions/seed/n_bins/step/sha256, and the framing below.

Why a script and not a docs section: ``docs/`` is in .gitignore, so nothing
written there would ever be tracked -- ``eval/COMPARISON.md`` is the only
trace git sees (``git check-ignore`` confirmed it is not ignored).

Framing (risk R3): ECE/Brier/NLL against distilled teacher targets are
***teacher-consistency*** (student agreement with HFLocalTeacher), **not
absolute truth** -- honester than uniform stubs, but still not "model
quality". The standing RANDOM_WEIGHTS_WARNING stays inside BOTH raw reports
(report.py is deliberately untouched this stage); the compensation for it
lives here, in the tracked file, and the raw reports stay gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import eval.cli

__all__ = ["main"]

#: Table rows, in report order (means over questions, scalars by construction).
METRICS = ("ece", "brier", "nll", "consistency")

#: Risk R3: what the numbers are -- and what they are NOT. Rendered verbatim
#: into COMPARISON.md so a reader of the tracked trace cannot mistake
#: teacher-consistency for model quality.
FRAMING = """\
ECE / Brier / NLL считаются против дистиллированных таргетов учителя, а не
против ground truth: это ***teacher-consistency*** (согласованность студента
с HFLocalTeacher), **не абсолютная правда** -- честнее uniform-заглушек, но и
это не «качество модели». Consistency -- симметричный KL между предсказаниями
на исходных и `shuffle_keys`-переформулировках одних и тех же состояний.

Число строк и вопросов-групп зафиксировано в таблице (`rows` -- строк
датасета, `questions` -- вопрос-групп; оба прогона считали один и тот же
датасет при одном seed, поэтому колонки совпадают).

Сырые отчёты лежат в `{random_out}/report.json` + `report.md` и
`{trained_out}/report.json` + `report.md` -- оба каталога gitignored
(при реальном запуске это `eval/out/{{random,trained}}`), этот файл --
единственный трекаемый след. Warning «Model weights are random» из
`eval/report.py` остаётся в ОБОИХ отчётах (report.py на этапе не правился):
для checkpoint-прогона случайных весов уже нет, но и сама метрика остаётся
teacher-consistency -- рамка задана здесь, в COMPARISON.md."""


def _require_file(path: Path, what: str) -> None:
    """Fail with a clean, path-bearing message instead of a traceback."""
    if not path.is_file():
        raise SystemExit(f"error: {what} not found: {path}")


def _read_report(directory: Path) -> dict:
    """Parse one ``report.json`` written by ``eval.cli`` (clean error if absent)."""
    path = directory / "report.json"
    if not path.is_file():
        raise SystemExit(f"error: expected report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cli(argv: list[str], what: str) -> None:
    """One in-process ``eval.cli.main`` run; a nonzero exit fails loudly."""
    rc = eval.cli.main(argv)
    if rc != 0:
        raise SystemExit(f"error: {what} failed with exit code {rc}: "
                         f"{' '.join(argv)}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: evaluate random + checkpoint in-process, write one markdown file."""
    p = argparse.ArgumentParser(
        prog="python -m eval.compare",
        description="Evaluate a random baseline and a checkpoint on the same "
                    "dataset (two in-process eval.cli runs) and write one "
                    "tracked markdown comparison.",
    )
    p.add_argument("--dataset", type=Path, required=True,
                   help="labeled JSONL, e.g. the distilled holdout targets")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="stage-4 pipeline checkpoint (.pt) to compare against "
                        "the random baseline")
    p.add_argument("--out", type=Path, required=True,
                   help="markdown file to write (e.g. eval/COMPARISON.md -- "
                        "the only tracked trace; docs/ is gitignored)")
    p.add_argument("--reports-dir", type=Path, default=Path("eval/out"),
                   help="directory for the two raw reports: "
                        "<dir>/random and <dir>/trained (default: eval/out -- "
                        "gitignored; tests point this at tmp_path)")
    args = p.parse_args(argv)

    _require_file(args.dataset, "dataset")
    _require_file(args.checkpoint, "checkpoint")

    random_out = args.reports_dir / "random"
    trained_out = args.reports_dir / "trained"
    # Both runs share dataset + seed (+ pinned n_bins=10) so the columns are
    # comparable; torch.manual_seed inside evaluate equalizes the augmentations.
    _run_cli(["--dataset", str(args.dataset), "--out", str(random_out),
              "--seed", "0"], "random baseline run")
    _run_cli(["--dataset", str(args.dataset), "--out", str(trained_out),
              "--checkpoint", str(args.checkpoint), "--seed", "0"],
             "checkpoint run")

    random_report = _read_report(random_out)
    trained_report = _read_report(trained_out)
    # Raw bytes of the pipeline .pt -- no digest is embedded in it (and the v1
    # open_jev.checkpoint double-write digest is a different thing entirely).
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    commands = [
        f".venv/bin/python -m eval.cli --dataset {args.dataset} "
        f"--out {random_out} --seed 0",
        f".venv/bin/python -m eval.cli --dataset {args.dataset} "
        f"--checkpoint {args.checkpoint} --out {trained_out} --seed 0",
        f".venv/bin/python -m eval.compare --dataset {args.dataset} "
        f"--checkpoint {args.checkpoint} --out {args.out} "
        f"--reports-dir {args.reports_dir}",
    ]
    lines = [
        "# Random vs trained checkpoint (holdout)",
        "",
        "## Reproduce",
        "",
        "```bash",
        *commands,
        "```",
        "",
        f"model: `{random_report['model']}` vs `{trained_report['model']}`",
        "",
        "## Что измеряют числа (рамка, R3)",
        "",
        *FRAMING.format(random_out=random_out, trained_out=trained_out).splitlines(),
        "",
        "## Сравнение",
        "",
        "| metric | random | checkpoint |",
        "| --- | --- | --- |",
    ]
    for name in METRICS:
        lines.append(f"| {name} | {random_report['metrics'][name]} | "
                     f"{trained_report['metrics'][name]} |")
    lines += [
        # Random report has no `step` -> "-" (consistent for both step/sha256).
        f"| rows | {random_report['rows']} | {trained_report['rows']} |",
        f"| questions | {random_report['questions']} | "
        f"{trained_report['questions']} |",
        f"| seed | {random_report['seed']} | {trained_report['seed']} |",
        f"| n_bins | {random_report['n_bins']} | {trained_report['n_bins']} |",
        f"| step | - | {trained_report.get('step', '-')} |",
        f"| sha256 | - | {digest} |",
        "",
    ]
    text = "\n".join(lines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
