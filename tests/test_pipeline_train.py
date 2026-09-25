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
