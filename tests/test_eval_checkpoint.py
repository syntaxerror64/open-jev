"""Red-first coverage for ``eval.cli --checkpoint`` (stage 9, steps 1-4).

Fixtures are built locally in ``tmp_path`` (never in conftest): a tiny
stage-4 pipeline checkpoint saved with ``pipeline.checkpoint.save_checkpoint``
(layout key ``"model"`` -- NOT the v1 ``state_dict`` layout of
``open_jev.checkpoint.save``) plus a 2-row dataset from ``eval.dataset``.

Deliberately NO "trained beats random ECE" assertion: 2-3 fixture rows at
n_bins=10 cannot guarantee it (flaky); the honest comparison lives in
``eval/COMPARISON.md`` on the 100-row holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import eval.cli
from eval.dataset import generate, write_jsonl
from open_jev.main import Jev, JevConfig
from pipeline.checkpoint import save_checkpoint

METRIC_KEYS = {"ece", "brier", "nll", "consistency"}

# The same 9 model fields as tests/test_checkpoint.py:9-11.
MODEL_CFG = dict(vocab_size=512, d_model=32, n_heads=4, d_ff=64,
                 n_state_layers=1, n_question_layers=1,
                 n_readout_layers=1, n_slots=4, max_state_len=64)


def _tiny(seed: int, **overrides) -> Jev:
    torch.manual_seed(seed)
    return Jev(JevConfig(**{**MODEL_CFG, **overrides}))


def _save(tmp_path: Path, model: Jev, cfg: dict, *, step: int = 3,
          name: str = "ck.pt") -> Path:
    """One checkpoint in the pipeline layout; ``cfg`` is a plain dict on
    purpose -- TrainConfig-style loop fields must be filtered at load time."""
    return save_checkpoint(tmp_path / name, model, cfg, step,
                           metrics={"ece": 0.0})


def _dataset(tmp_path: Path, n: int = 2) -> Path:
    path = tmp_path / "dataset.jsonl"
    write_jsonl(generate(n, seed=0), path)
    return path


def _run(ds: Path, out: Path, ck: Path | None = None) -> int:
    argv = ["--dataset", str(ds), "--out", str(out)]
    if ck is not None:
        argv += ["--checkpoint", str(ck)]
    return eval.cli.main(argv)


def test_checkpoint_flag_loads_model_and_marks_report(tmp_path) -> None:
    # cfg = 9 model fields + loop fields lr/steps/teacher that JevConfig does
    # NOT know: loading must filter them, not choke on them.
    ck = _save(tmp_path, _tiny(0),
               {**MODEL_CFG, "lr": 1e-4, "steps": 3, "teacher": "stub"})
    out = tmp_path / "out"
    assert _run(_dataset(tmp_path), out, ck) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["model"] == f"checkpoint:{ck}"
    assert report["step"] == 3
    assert set(report["metrics"]) == METRIC_KEYS


def test_no_flag_report_is_unchanged(tmp_path) -> None:
    # Characterization (written BEFORE the --checkpoint edits): the default
    # run must stay exactly as it was -- random model, no step, byte-stable.
    ds = _dataset(tmp_path)
    first, second = tmp_path / "o1", tmp_path / "o2"
    assert _run(ds, first) == 0
    assert _run(ds, second) == 0
    raw_first = (first / "report.json").read_bytes()
    assert raw_first == (second / "report.json").read_bytes()
    report = json.loads(raw_first)
    assert report["model"] == "random"
    assert set(report["metrics"]) == METRIC_KEYS
    assert "step" not in report
    assert {"rows", "questions", "seed"} <= set(report)
    assert report["rows"] == 2 and report["questions"] > 0


def test_missing_checkpoint_file_is_clean_error(tmp_path) -> None:
    missing = tmp_path / "nope.pt"
    # SystemExit/ValueError with the path in the text = no traceback escapes;
    # FileNotFoundError/AttributeError from torch.load would be a traceback.
    with pytest.raises((SystemExit, ValueError)) as exc:
        _run(_dataset(tmp_path), tmp_path / "out", missing)
    assert str(missing) in str(exc.value)


def test_state_dict_config_mismatch_is_clean_error(tmp_path) -> None:
    # config says d_model=32 but the weights came from a d_model=64 model;
    # loop fields in the same cfg prove the filter still applies on this path.
    ck = _save(tmp_path, _tiny(0, d_model=64),
               {**MODEL_CFG, "d_model": 32, "lr": 1e-4, "steps": 3,
                "teacher": "stub"})
    with pytest.raises((SystemExit, ValueError)) as exc:
        _run(_dataset(tmp_path), tmp_path / "out", ck)
    assert str(ck) in str(exc.value)


def test_missing_required_keys_is_named_error(tmp_path) -> None:
    # read_checkpoint already raises ValueError naming the keys -- the CLI
    # must not re-wrap it into a generic failure that hides them.
    ck = tmp_path / "ck.pt"
    torch.save({"model": {}, "config": dict(MODEL_CFG), "metrics": {}}, ck)
    with pytest.raises(ValueError) as exc:
        _run(_dataset(tmp_path), tmp_path / "out", ck)
    msg = str(exc.value)
    assert str(ck) in msg
    assert "missing key" in msg
    assert "step" in msg


def test_metrics_depend_on_checkpoint_weights(tmp_path) -> None:
    # Same dataset, same eval seed, different checkpoint weights -> the
    # metrics must move: they are computed from the loaded state_dict.
    rows = generate(3, seed=0)
    ck_a = _save(tmp_path, _tiny(0), dict(MODEL_CFG), name="a.pt")
    ck_b = _save(tmp_path, _tiny(1), dict(MODEL_CFG), name="b.pt")
    model_a, _ = eval.cli._load_checkpoint_model(ck_a)
    model_b, _ = eval.cli._load_checkpoint_model(ck_b)
    metrics_a, _ = eval.cli.evaluate(rows, model=model_a, seed=0)
    metrics_b, _ = eval.cli.evaluate(rows, model=model_b, seed=0)
    assert set(metrics_a) == set(metrics_b) == METRIC_KEYS
    assert any(metrics_a[k] != metrics_b[k] for k in METRIC_KEYS)


def test_checkpoint_evaluation_is_deterministic(tmp_path) -> None:
    rows = generate(2, seed=0)
    ck = _save(tmp_path, _tiny(0), dict(MODEL_CFG))
    model, step = eval.cli._load_checkpoint_model(ck)
    assert step == 3
    first, meta_first = eval.cli.evaluate(rows, model=model, seed=0)
    second, meta_second = eval.cli.evaluate(rows, model=model, seed=0)
    assert first == second
    assert meta_first == meta_second
