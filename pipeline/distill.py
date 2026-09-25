"""Stage-8 distillation: real soft targets for stage-6 rows, JSONL -> JSONL.

Design (stage 8, Шаги 1-2): the CORE of the distill
pass -- the functions, not the CLI. ``python -m pipeline.distill``'s
``main()`` (agent B, stage-08) is layered on top of this file later and
drives these three entry points; nothing here parses argv.

What the core guarantees:

1. **Real targets, any teacher.** ``distill_rows`` runs every row through
   ``Teacher.teach`` (pipeline/teacher.py contract) and REPLACES the row's
   placeholder ``targets`` with the teacher's soft distribution -- one flat
   ``[K]`` list per question, exactly the shape ``pipeline.data.build_batch``
   reads back (data.py:170-177), so the output file trains as-is with
   ``kind="precomputed"``.
2. **Validation before bytes (risk R2).** Every teacher answer goes through
   ``pipeline.data.validate_soft_targets`` before anything is written: a
   teacher returning one-hot / NaN / negative / off-sum rows raises
   ``ValueError`` loudly (naming the row and question) and ``out`` is never
   created -- hard labels manufacture overconfidence (open_jev/main.py:807-817),
   so a bad teacher must die, not write.
3. **Resumable and byte-stable.** ``distill_file`` skips rows already present
   in ``out`` (matched by ``row_key`` -- sha256 of the canonical
   ``{state, questions}`` JSON, since targets are the very thing that
   changes), writes ``json.dumps(row, ensure_ascii=False, sort_keys=True) +
   "\\n"`` lines (the byte-stable format scripts/fetch_data.py established),
   and lands them through a tmp file in the target directory + ``os.replace``
   so a reader only ever sees the old file or the complete new one.

Row schema is pipeline/data.py's ``{"state", "questions", "targets"}`` with
the SAME question list on every row; source rows go through ``build_batch``
once, which enforces that schema and hands back the coerced ``Question``
objects (Noul/Choice/Score) a real backend needs -- dicts on disk, question
objects in ``teach``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pipeline.config import _TEACHER_KEYS
from pipeline.data import build_batch, load_jsonl, validate_soft_targets
from pipeline.teacher import make_teacher

__all__ = ["distill_file", "distill_rows", "row_key"]


def row_key(row: Mapping[str, Any]) -> str:
    """sha256 of the canonical ``{state, questions}`` JSON of ``row``.

    ``targets`` is excluded on purpose: distillation is exactly what changes
    it, so a source row and its distilled copy in ``out`` must hash equal for
    resume to match them (resume по хэшу ``(state, questions)``).
    ``sort_keys=True`` makes the hash depend on the content, not on dict
    insertion order, i.e. canonical across runs and files.
    """
    try:
        payload = {"state": row["state"], "questions": row["questions"]}
    except KeyError as exc:
        raise ValueError(f"row_key: row is missing {exc}") from exc
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def distill_rows(
    rows: Sequence[Mapping[str, Any]], teacher: Any, *, repeats: int = 10
) -> list[dict]:
    """Replace every row's placeholder ``targets`` with the teacher's soft ones.

    One ``teacher.teach([state], questions, repeats=repeats)`` call per row
    (the rows share one coerced question list -- ``build_batch`` insists on
    it, data.py:153); each answer must survive ``validate_soft_targets`` as a
    ``[1, K]`` batch before the row is emitted, so one-hot / NaN / negative /
    off-sum output raises ``ValueError`` prefixed with the row index and the
    question -- loudly, BEFORE ``distill_file`` writes anything.

    Returns fresh dicts; the inputs keep their placeholder targets.
    """
    if not rows:
        return []
    # Schema (state/questions/targets, one shared question list) + coerced
    # Question objects for the backend.
    built = build_batch(list(rows))
    distilled: list[dict] = []
    for i, row in enumerate(rows):
        try:
            produced = teacher.teach(
                [row["state"]], built.questions, repeats=repeats
            )
            targets = _checked_targets(produced, len(built.questions))
        except ValueError as exc:
            raise ValueError(f"row {i}: {exc}") from exc
        distilled.append(dict(row, targets=targets))
    return distilled


def distill_file(
    src: str | Path, out: str | Path, teacher: Any, *, repeats: int = 10
) -> dict:
    """Distill ``src`` JSONL into ``out``; resumable, byte-stable, atomic.

    Rows already present in ``out`` (matched by :func:`row_key`) are counted
    as ``skipped`` and copied through untouched -- only pending rows reach
    ``teacher``, so a slow real-data pass can be interrupted and restarted.

    Nothing touches ``out`` until every pending row has been taught AND
    validated: a failing teacher raises with ``out`` absent (fresh run) or
    unchanged (rerun), and the bytes land via tmp file + ``os.replace`` in
    the target directory.

    Returns ``{"rows", "written", "skipped"}`` where ``rows`` is the number
    of source rows and ``written + skipped == rows``.
    """
    src_path, out_path = Path(src), Path(out)
    rows = load_jsonl(src_path)
    done: dict[str, dict] = {}
    if out_path.exists():
        done = {row_key(r): r for r in load_jsonl(out_path)}

    keys = [row_key(r) for r in rows]
    pending = [row for row, key in zip(rows, keys) if key not in done]
    # Validates EVERY pending row before a single byte is written (Шаг 2).
    new_rows = distill_rows(pending, teacher, repeats=repeats)

    final: list[dict] = []
    fresh = iter(new_rows)
    for key in keys:
        final.append(done[key] if key in done else next(fresh))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Same-dir tmp + rename, as pipeline/teacher.py's disk cache does: a
    # reader sees the old file or the complete new one, never a torn write.
    tmp = out_path.parent / f".{out_path.name}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in final:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"rows": len(rows), "written": len(new_rows), "skipped": len(rows) - len(new_rows)}


def _checked_targets(produced: Any, n_questions: int) -> list[list[float]]:
    """``teach`` output -> per-question flat ``[K]`` lists, validated first.

    A single-state call must answer with one ``[1, K]`` tensor per question;
    each is run through ``validate_soft_targets`` (finite, non-negative,
    sums to 1, never one-hot) and only then flattened to the on-disk ``[K]``
    row -- validation of the values that will actually be written, not of a
    different representation of them.
    """
    produced = list(produced)
    if len(produced) != n_questions:
        raise ValueError(
            f"expected {n_questions} target row(s) (one per question), "
            f"teacher returned {len(produced)}"
        )
    tensors: list[torch.Tensor] = []
    for q, target in enumerate(produced):
        t = torch.as_tensor(target)
        if t.ndim != 2 or t.shape[0] != 1:
            raise ValueError(
                f"question {q}: a single-state teach() call must return a "
                f"[1, K] tensor, got shape {tuple(t.shape)}"
            )
        tensors.append(t)
    checked = validate_soft_targets(tensors)
    return [t[0].tolist() for t in checked]


# ---------------------------------------------------------------------------
# CLI (stage 8, Шаг 3): ``python -m pipeline.distill``.
# Agent Б's section, appended on top of agent А's core above -- the four core
# functions keep their exact behaviour; everything below only calls them.
# ---------------------------------------------------------------------------

#: ``--teacher-json`` default, verbatim: the LIVE HF teacher.
#: The CLI exists to run a real teacher, so silently falling back to stub
#: semantics is not an option; tests pass ``kind=stub`` explicitly and
#: ``hf_local`` never runs in the fast gate (manual / ``slow`` only).
DEFAULT_TEACHER_JSON = (
    '{"kind":"hf_local","model_id":"Qwen/Qwen2.5-0.5B-Instruct","mode":"logits"}'
)

#: Teacher-config keys with a dedicated ``--flag`` -- the argparse mirror of
#: ``_TEACHER_KEYS`` (pipeline/config.py:49). ``seed`` has its own ``--seed``
#: (a fallback, not an override) and ``device``/``cache_dir`` stay JSON-only;
#: ``kind`` and ``repeats`` are loop directives popped before ``make_teacher``
#: (train.py:152-154), exactly as specified.
_TEACHER_FLAGS: tuple[str, ...] = (
    "kind",
    "model_id",
    "mode",
    "temperature",
    "max_new_tokens",
    "repeats",
)


class _ProgressTeacher:
    """Teacher proxy printing ``row i/n ETA ...`` to stderr once per taught row.

    ``distill_file`` is atomic and silent by design (agent А's core), so
    progress wraps the one thing it calls exactly once per pending row -- the
    teacher -- instead of touching the core: the ETA is computed from measured
    per-row time (Шаг 3: прогресс+ETA в stderr), never fabricated, and a
    failing teacher simply stops the lines with ``--out`` untouched.
    """

    def __init__(self, teacher: Any, total: int, start: float) -> None:
        self._teacher = teacher
        self._total = total
        self._start = start
        self._done = 0

    def teach(self, states: Any, questions: Any, repeats: int = 10) -> Any:
        produced = self._teacher.teach(states, questions, repeats=repeats)
        self._done += 1
        elapsed = time.perf_counter() - self._start
        eta = elapsed / self._done * (self._total - self._done)
        print(
            f"row {self._done}/{self._total} ETA {eta:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        return produced


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m pipeline.distill --data S --out O [teacher flags]``.

    Teacher precedence (Шаг 3): ``--teacher-json`` provides the block,
    each individual flag that was passed overrides its key -- flag wins --
    and ``--seed`` is only the FALLBACK injected via
    ``kwargs.setdefault("seed", args.seed)``, i.e. a ``seed`` inside the JSON
    pins the run exactly like ``pipeline/train.py:216-218`` does for
    ``TrainConfig.teacher``. Unknown JSON keys fail with the same wording as
    ``TrainConfig._validate_teacher`` BEFORE ``make_teacher`` runs.

    Errors follow scripts/fetch_data.py:263-289: ``ValueError``/``OSError``
    prints ``str(exc)`` to stderr and returns 1, so a bad/one-hot teacher
    exits non-zero with ``--out`` never created (the core validates before it
    writes a byte). Progress + ETA and the final
    ``rows/written/skipped/elapsed`` summary both go to stderr; ``elapsed`` is
    measured here because ``distill_file`` reports no time of its own.
    """
    p = argparse.ArgumentParser(
        prog="python -m pipeline.distill",
        description="Distill stage-6 JSONL rows through any pipeline.teacher "
        "backend into real soft targets (resumable; progress and the final "
        "summary go to stderr).",
    )
    p.add_argument(
        "--data", required=True, metavar="PATH",
        help="source JSONL: stage-6 rows {state, questions, targets}",
    )
    p.add_argument(
        "--out", required=True, metavar="PATH",
        help="output JSONL with teacher targets; existing rows are skipped "
        "(resume by sha256 of {state, questions})",
    )
    p.add_argument(
        "--teacher-json", default=DEFAULT_TEACHER_JSON, metavar="JSON",
        help=f"teacher block mirroring TrainConfig.teacher "
        f"(default: {DEFAULT_TEACHER_JSON})",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="fallback teacher seed when --teacher-json doesn't pin one "
        "(default: 0)",
    )
    for _name in _TEACHER_FLAGS:
        _type = {"temperature": float, "max_new_tokens": int, "repeats": int}.get(
            _name, str
        )
        p.add_argument(
            "--" + _name.replace("_", "-"),
            dest=_name,
            type=_type,
            default=None,
            help=f"override {_name!r} from --teacher-json (flag wins)",
        )
    args = p.parse_args(argv)

    started = time.perf_counter()
    try:
        teacher_cfg = json.loads(args.teacher_json)
        if not isinstance(teacher_cfg, dict):
            raise ValueError(
                f"--teacher-json: expected a JSON object, got "
                f"{type(teacher_cfg).__name__}"
            )
        # Individual flags win over the merged JSON values.
        for name in _TEACHER_FLAGS:
            value = getattr(args, name)
            if value is not None:
                teacher_cfg[name] = value
        unknown = sorted(set(teacher_cfg) - set(_TEACHER_KEYS))
        if unknown:
            # Same wording as TrainConfig._validate_teacher (config.py:147-149).
            raise ValueError(f"unknown teacher config key(s): {unknown}")

        kind = teacher_cfg.pop("kind", None)
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                '--teacher-json/--kind must name a teacher kind, e.g. "stub"'
            )
        repeats = teacher_cfg.pop("repeats", 10)  # loop directive -> distill_file
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats!r}")

        # ETA denominator = rows the teacher will actually see (resume-aware,
        # computed the same way distill_file computes it).
        rows = load_jsonl(args.data)
        done: set[str] = set()
        if Path(args.out).exists():
            done = {row_key(r) for r in load_jsonl(args.out)}
        pending = sum(1 for row in rows if row_key(row) not in done)

        kwargs = teacher_cfg
        kwargs.setdefault("seed", args.seed)  # injected unless the JSON pins it
        print(
            f"distill: {args.data} -> {args.out} "
            f"({len(rows)} rows, {pending} pending)",
            file=sys.stderr,
            flush=True,
        )
        print(
            "teacher: "
            + json.dumps(dict(kwargs, kind=kind, repeats=repeats), sort_keys=True),
            file=sys.stderr,
            flush=True,
        )

        # Risk R5: one torch thread for the whole pass (train.py:131-136).
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            teacher = make_teacher(kind, **kwargs)
            stats = distill_file(
                args.data,
                args.out,
                _ProgressTeacher(teacher, pending, started),
                repeats=repeats,
            )
        finally:
            torch.set_num_threads(previous_threads)
        elapsed = time.perf_counter() - started
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"rows {stats['rows']} written {stats['written']} "
        f"skipped {stats['skipped']} elapsed {elapsed:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
