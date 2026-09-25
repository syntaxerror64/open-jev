"""Stage 13: the public Score contract is 2..10 levels (docs.typesafe.ai).

The model's *architectural capacity* (`JevConfig.max_score_levels`, 16) is a
separate thing from the *question contract* (`Score`, 2..10) and must not be
"aligned" to it -- see `test_config_capacity_default_is_untouched`.
"""

from __future__ import annotations

import pytest
import torch

from open_jev.main import JevConfig, Score, ScoreAnswer

TOL = 1e-5


def test_score_accepts_exactly_ten_levels() -> None:
    """The upper bound is inclusive: exactly 10 levels is valid."""
    s = Score("Rate", labels=[f"L{i}" for i in range(10)], key="k")

    assert len(s.labels) == 10
    assert list(s.labels) == [f"L{i}" for i in range(10)]


def test_score_rejects_eleven_levels() -> None:
    """11 levels must raise a clean ValueError naming the limit (10) and the
    offending count (11). RED today: no upper bound is validated yet."""
    labels = [f"L{i}" for i in range(11)]

    with pytest.raises(ValueError, match=r"10") as excinfo:
        Score("Rate", labels=labels, key="k")

    assert "11" in str(excinfo.value)


def test_two_level_and_min_validation_still_hold() -> None:
    """Characterization of the existing lower bound (main.py:175-177)."""
    two = Score("Rate", labels=["low", "high"], key="k")
    assert len(two.labels) == 2

    with pytest.raises(ValueError, match=r">= 2"):
        Score("Rate", labels=["only"], key="k")


def test_forward_handles_ten_levels(model, state) -> None:
    """K=10 semantics survive a forward pass: probabilities sum to 1."""
    q = Score("Rate", labels=[f"L{i}" for i in range(10)], key="ten")
    with torch.no_grad():
        answer = model([state], [q])[0][0]

    assert isinstance(answer, ScoreAnswer)
    assert len(answer.probabilities) == 10
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=TOL)
    assert all(0.0 <= p <= 1.0 for p in answer.probabilities.values())


def test_config_capacity_default_is_untouched() -> None:
    """Pin: capacity stays 16. It is the ARCHITECTURAL head size, not the
    question contract's 10.

    Pipeline checkpoints store `asdict(TrainConfig)` (pipeline/config.py),
    which has no `max_score_levels` field -> loaders fall back to this default,
    which sets `ScoreHead.deltas` shape to `max_score_levels - 2` (main.py:632).
    Lowering 16 -> 10 shrinks deltas 14 -> 8, `load_state_dict` then raises
    RuntimeError, breaking `runs/real/checkpoint.pt` and `dist/jev-real.*`
    (release v0.2.0) plus eval/cli.py and export.
    """
    cfg_max = JevConfig()
    assert cfg_max.max_score_levels == 16
