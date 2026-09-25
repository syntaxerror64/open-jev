"""Stage-07, step 1 (red -> green): the ``TrainConfig.teacher`` block.

The four tests are the code from stage 7, Шаг 1, verbatim in
intent: a JSON-serializable ``teacher`` dict selects the stage-6 teacher at
config level (kind/mode/repeats/...), old configs without the field keep
loading byte-for-byte as ``{"kind": "stub"}``, and both unknown subkeys and
invalid ``kind``/``mode`` values fail loudly instead of silently no-op'ing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipeline.config import TrainConfig

TINY = Path(__file__).resolve().parent.parent / "pipeline" / "examples" / "tiny.json"


def test_teacher_field_roundtrips_through_json(tmp_path) -> None:
    teacher = {
        "kind": "hf_local",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "mode": "logits",
        "repeats": 5,
    }
    cfg = TrainConfig(teacher=teacher)
    out = tmp_path / "c.json"
    cfg.save(out)
    assert TrainConfig.load(out) == cfg
    # In the file the ``teacher`` key is a JSON object, not a string/number.
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(raw["teacher"], dict)
    assert raw["teacher"]["kind"] == "hf_local"


def test_teacher_defaults_to_stub_and_old_config_still_loads() -> None:
    assert TrainConfig().teacher == {"kind": "stub"}
    # Back-compat is a hard requirement: pipeline/examples/tiny.json has no
    # ``teacher`` key at all and must keep loading as the stub.
    assert TrainConfig.load(TINY).teacher == {"kind": "stub"}


def test_unknown_teacher_subkey_is_rejected(tmp_path) -> None:
    # via load(): the typo'd subkey must fail, not be dropped
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"teacher": {"kind": "stub", "modle_id": "x"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown teacher"):
        TrainConfig.load(bad)
    # ...and the same validation runs on direct construction
    with pytest.raises(ValueError, match="unknown teacher"):
        TrainConfig(teacher={"modle_id": "x"})


def test_teacher_kind_and_mode_values_validated() -> None:
    with pytest.raises(ValueError):
        TrainConfig(teacher={"kind": "gpt6"})
    with pytest.raises(ValueError):
        TrainConfig(teacher={"kind": "stub", "mode": "greedy"})
