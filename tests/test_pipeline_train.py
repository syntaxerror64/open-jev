"""Stage-4, step 3 (red -> green): the training loop, checkpoints and the CLI.

The first four tests are stage 4, Шаг 3: loss
falls over 30 seeded steps on synthetic data, a checkpoint roundtrip restores
identical logits, ``resume`` continues at ``step_start == 5``, and the saved
file carries the ``{"model", "config", "step", "metrics"}` skeleton stage 5
will publish. The fifth (``slow``) launches ``python -m pipeline.train`` in a
subprocess and checks exit code + artifacts, mirroring the success
criteria.

Helpers come from tests/conftest.py where they fit: the session ``cfg``
(JevConfig), ``model`` (seeded, eval mode), ``state`` and ``questions``
fixtures replace the illustrative ``tiny_model``/``STATE``/``Q``.

Appended below (stage 07, Шаг 2): the teacher is
chosen by ``TrainConfig.teacher`` instead of the hardcoded stub -- kwargs and
``repeats`` flow out of the block with the seed injected, ``kind="precomputed"``
trains on the rows' own targets (no backend, no uniform substitution, no
synthetic fallback for a missing data file), and the control test pins the
default config to the exact old ``make_teacher("stub", seed=cfg.seed)`` call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from open_jev.main import Jev
from pipeline.checkpoint import load_checkpoint, save_checkpoint
from pipeline.config import TrainConfig
from pipeline.teacher import StubTeacher
from pipeline.train import train

REPO = Path(__file__).resolve().parent.parent


def train_cfg(out: Path, **overrides) -> TrainConfig:
    """TINY TrainConfig (tests/conftest.py:12-22 model shape) for one test.

    ``out`` and ``data_path`` point into the test's tmp dir so no unit test
    writes into the repo's ``runs/`` or ``data/train/``; the model fields are
    TrainConfig's own defaults, which already mirror the conftest TINY shape.
    """
    base = dict(
        steps=30,
        lr=1e-3,
        seed=0,
        batch_size=2,
        eval_every=10,
        out_dir=str(out),
        data_path=str(Path(out) / "synth.jsonl"),
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_training_step_reduces_loss_on_synthetic_data(tmp_path) -> None:
    # Deterministic seed (risk R3): synthetic rows from eval/dataset.generate,
    # soft targets from StubTeacher (open question O1/O2 - no real data/teacher yet).
    history = train(train_cfg(tmp_path, steps=30))
    assert len(history) == 30
    assert history[-1].loss < history[0].loss


def test_checkpoint_roundtrip_restores_identical_logits(
    tmp_path, cfg, model, state, questions
) -> None:
    path = tmp_path / "ck.pt"
    save_checkpoint(
        path, model=model, cfg=train_cfg(tmp_path), step=1, metrics={"ece": 0.1}
    )
    m2 = load_checkpoint(path, model_factory=lambda: Jev(cfg))
    with torch.no_grad():
        a = model.logits([state], questions)
        b = m2.logits([state], questions)
    assert len(a) == len(b)
    for (p1, _), (p2, _) in zip(a, b):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_resume_continues_from_checkpoint(tmp_path) -> None:
    run1 = train(train_cfg(tmp_path, steps=5))
    assert run1.step_start == 0
    assert run1[-1].step == 5

    # optimizer state + step are restored: the next step executes without a
    # KeyError on the (uninitialised) AdamW state of any parameter.
    run2 = train(
        train_cfg(tmp_path, steps=5), resume=tmp_path / "checkpoint.pt"
    )
    assert run2.step_start == 5
    assert len(run2) == 5
    assert run2[0].step == 6
    assert run2[-1].step == 10


def test_checkpoint_contains_metadata_required_by_stage5(tmp_path, cfg, model) -> None:
    path = tmp_path / "ck.pt"
    save_checkpoint(
        path, model=model, cfg=train_cfg(tmp_path), step=1, metrics={"ece": 0.1}
    )
    ck = torch.load(path, map_location="cpu")  # владелец формата — этап 5,
    assert {"model", "config", "step", "metrics"} <= set(ck)  # здесь — минимальный каркас
    assert ck["step"] == 1
    assert ck["metrics"] == {"ece": 0.1}
    # risk R3: seed and torch version travel with the weights
    assert ck["seed"] == 0
    assert ck["torch_version"] == torch.__version__


@pytest.mark.slow  # subprocess + torch import: ~5s, more on a loaded box
def test_tiny_run_end_to_end() -> None:
    """``python -m pipeline.train --config pipeline/examples/tiny.json``.

    Exit 0, artifacts where the config says they go (``runs/tiny/``), and a
    metrics.json whose history shows the loss falling. Marked ``slow`` when
    the run takes more than a few seconds (subprocess + torch import).
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline.train",
         "--config", "pipeline/examples/tiny.json"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    out = REPO / "runs" / "tiny"
    assert (out / "checkpoint.pt").is_file()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["step"] >= 1
    assert metrics["history"], "metrics.json must record the loss history"
    assert metrics["history"][-1]["loss"] < metrics["history"][0]["loss"]


# --- stage 7, Шаг 2: the teacher comes from cfg.teacher ---


def _k_of(question) -> int:
    """K for a built question: declared arity, else binary (a noul)."""
    if hasattr(question, "options"):
        return len(question.options)
    if hasattr(question, "labels"):
        return len(question.labels)
    return 2


def precomputed_rows(targets_row: list[float], n: int = 4) -> list[dict]:
    """``n`` eval/dataset.py-style rows whose single noul carries the target.

    ``targets_row`` lands in every row verbatim -- that is exactly what the
    precomputed branch must train on (or reject, when it is one-hot).
    """
    questions = [{"type": "noul", "text": "a > 0.5", "key": "a_gt_half"}]
    rows: list[dict] = []
    for i in range(n):
        a = round((i + 1) / (n + 1), 4)
        rows.append(
            {
                "state": {"a": a, "b": round(1 - a, 4)},
                "questions": [dict(q) for q in questions],
                "targets": [list(targets_row)],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    """Materialise rows as JSONL (the on-disk shape ``_load_rows`` reads)."""
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_teacher_is_built_from_config_seed_injected(tmp_path, monkeypatch) -> None:
    """Шаг 2.1: one ``make_teacher`` call, kind from the block, seed from cfg.

    The spy stands in for the backend factory, so a run never loads a real
    teacher; the recorded call is the whole observable contract.
    """
    calls: list[tuple] = []

    def spy(kind, **kwargs):
        calls.append((kind, kwargs))
        return StubTeacher(**kwargs)

    monkeypatch.setattr("pipeline.train.make_teacher", spy)
    cfg = train_cfg(tmp_path, steps=1, seed=7, teacher={"kind": "stub"})
    train(cfg)
    assert len(calls) == 1, f"expected exactly one teacher build, got {calls}"
    kind, kwargs = calls[0]
    assert kind == "stub" == cfg.teacher["kind"]
    assert kwargs == {"seed": cfg.seed}  # injected from the config, not hardcoded


def test_teacher_kwargs_and_repeats_come_from_block(tmp_path, monkeypatch) -> None:
    """Шаг 2.2: block kwargs reach the factory; ``kind``/``repeats`` do not.

    ``repeats`` is a loop directive: it must arrive in ``teach(...)`` instead
    of the backend constructor (``HFLocalTeacher`` has no such kwarg).
    """
    calls: list[tuple] = []
    seen_repeats: list[int] = []

    class SpyTeacher:
        """Records ``repeats``; answers with uniform (validation-safe) targets."""

        def teach(self, states, questions, repeats=10):
            seen_repeats.append(repeats)
            return [
                torch.full((len(states), _k_of(q)), 1.0 / _k_of(q))
                for q in questions
            ]

    def spy(kind, **kwargs):
        calls.append((kind, kwargs))
        return SpyTeacher()

    monkeypatch.setattr("pipeline.train.make_teacher", spy)
    cfg = train_cfg(
        tmp_path,
        steps=1,
        teacher={"kind": "hf_local", "model_id": "m", "mode": "logits", "repeats": 4},
    )
    train(cfg)
    assert calls == [("hf_local", {"model_id": "m", "mode": "logits", "seed": 0})]
    assert seen_repeats == [4]


def test_precomputed_trains_on_row_targets_verbatim(tmp_path, monkeypatch) -> None:
    """Шаг 2.3: row targets reach the loss as-is -- no teacher, no uniform swap.

    Both ``make_teacher`` and ``_with_uniform_targets`` become raising spies:
    under ``kind="precomputed"`` neither may run. A fake ``RLCDLoss`` captures
    the targets so the assertion can see the row's own 0.9 rather than the
    uniform 0.5 a substitution would have left behind.
    """
    data = tmp_path / "distilled.jsonl"
    write_rows(data, precomputed_rows([0.9, 0.1]))

    def boom(*args, **kwargs):
        raise AssertionError("precomputed must not build a teacher or swap targets")

    monkeypatch.setattr("pipeline.train.make_teacher", boom)
    monkeypatch.setattr("pipeline.train._with_uniform_targets", boom)

    captured: list[list[torch.Tensor]] = []

    class FakeLoss:
        """RLCDLoss stand-in: record the targets, return a differentiable 0."""

        def __init__(self):
            self.parts = {"nll": 0.0}

        def __call__(self, model, states, questions, targets):
            captured.append(targets)
            return (next(iter(model.parameters())) * 0.0).sum()

    monkeypatch.setattr("pipeline.train.RLCDLoss", FakeLoss)

    cfg = train_cfg(
        tmp_path, steps=1, teacher={"kind": "precomputed"}, data_path=str(data)
    )
    train(cfg)

    assert captured, "RLCDLoss must have been called once for the one step"
    first_row = captured[0][0][0]  # first question, first row of the batch
    assert first_row.tolist() == pytest.approx([0.9, 0.1])
    assert float(first_row[0]) == pytest.approx(0.9)  # not the uniform 0.5


def test_precomputed_still_rejects_one_hot_rows(tmp_path) -> None:
    """Шаг 2.4: one-hot rows die in ``build_batch`` -- nothing substitutes them."""
    data = tmp_path / "one_hot.jsonl"
    write_rows(data, precomputed_rows([1.0, 0.0]))
    cfg = train_cfg(
        tmp_path, steps=1, teacher={"kind": "precomputed"}, data_path=str(data)
    )
    with pytest.raises(ValueError, match="soft"):
        train(cfg)


def test_precomputed_missing_data_path_fails_loud(tmp_path) -> None:
    """Шаг 2.5: a missing file is a ValueError, never synthetic generation.

    ``_load_rows`` silently materialises synthetic rows for a missing
    ``data_path``; under ``kind="precomputed"`` that fallback would mask the
    error and train on uniform placeholders instead of the distilled targets.
    """
    missing = tmp_path / "no_such_dataset.jsonl"
    cfg = train_cfg(
        tmp_path, steps=1, teacher={"kind": "precomputed"}, data_path=str(missing)
    )
    with pytest.raises(ValueError, match="not found|missing"):
        train(cfg)
    assert not missing.exists(), "no synthetic dataset may be generated in its place"


def test_default_config_keeps_stub_call(tmp_path, monkeypatch) -> None:
    """Шаг 2.6 (control): a config without a ``teacher`` kwarg builds the stub.

    Byte-identical back-compat with the old hardcoded
    ``make_teacher("stub", seed=cfg.seed)``; only the run's file locations
    move into ``tmp_path`` so the control writes nothing into the repo.
    """
    calls: list[tuple] = []

    def spy(kind, **kwargs):
        calls.append((kind, kwargs))
        return StubTeacher(**kwargs)

    monkeypatch.setattr("pipeline.train.make_teacher", spy)
    cfg = TrainConfig(
        steps=1,
        out_dir=str(tmp_path / "out"),
        data_path=str(tmp_path / "synth.jsonl"),
    )
    assert cfg.teacher == {"kind": "stub"}  # default: the block was never passed
    train(cfg)
    assert calls == [("stub", {"seed": 0})]


# --- stage 8, Шаги 4-5: real example + precomputed fixture run


def test_train_on_precomputed_targets_falls(tmp_path):
    # spec amendment (found by agent 08-C): at steps=5 the final-vs-first batch
    # comparison is deterministic red (seed 0 noise: 1.038 > 1.024); steps=10
    # keeps the assertion verbatim and green, still << a second.
    cfg = TrainConfig(steps=10, batch_size=2, eval_every=5, seed=0, lr=1e-3,
                      out_dir=str(tmp_path), data_path="tests/fixtures/precomputed.jsonl",
                      teacher={"kind": "precomputed"})      # поле этапа 7 (dict)
    history = train(cfg)
    assert history[-1].loss < history[0].loss                  # < секунд
    assert (tmp_path / "checkpoint.pt").is_file() and (tmp_path / "metrics.json").is_file()


def test_real_example_config_validates():
    from pipeline.config import MODEL_FIELDS  # local: module imports are stage-07 frozen
    cfg = TrainConfig.load("pipeline/examples/real.json")   # незнакомый ключ = ValueError
    assert set(cfg.model_kwargs()) == set(MODEL_FIELDS)     # модель собирается
    assert (cfg.data_path, cfg.out_dir) == ("data/real/train.targets.jsonl", "runs/real")
    assert cfg.teacher["kind"] == "precomputed"
    assert (cfg.steps, cfg.batch_size, cfg.eval_every, cfg.seed) == (200, 4, 50, 0)
