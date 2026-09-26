"""Stage 14 pins: `confidence_kind` = "evidential" (default) | "spread".

Their confidence comes from how flat the distribution is (docs.typesafe.ai):

    clamp((K * peak - 1) / (K - 1), 0, 1)

Ours comes from the evidential head, `1 - K / S` (`ConfidenceHead`). Stage 14
gives the choice instead of replacing anything: the default keeps every bit of
existing semantics, `spread` switches only the answers-assembly path in
`forward`, and `logits` -- the training path RLCDLoss reads -- stays evidential
forever (its confidence must be stop-gradient-independent of the probabilities).

`spread_confidence` is imported inside the tests that pin it so a missing
symbol fails those tests individually instead of erroring the module at
collection time.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from open_jev.main import (
    ChoiceAnswer,
    Jev,
    JevConfig,
    NoulAnswer,
    ScoreAnswer,
)

TOL = 1e-5


def _spread_model(cfg: JevConfig) -> Jev:
    """The `confidence_kind="spread"` variant of `cfg`.

    Seeded exactly like the session `model` fixture, but inside `fork_rng` so
    the global RNG stream other tests see is untouched. Same seed + same
    construction stream => bit-identical weights to the default model, which
    `test_logits_always_returns_evidential_confidence` relies on.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return Jev(replace(cfg, confidence_kind="spread")).eval()


def test_default_confidence_kind_is_evidential(model, state, questions) -> None:
    """The default is "evidential"; the default forward path is untouched."""
    assert JevConfig().confidence_kind == "evidential"

    with torch.no_grad():
        answers = model([state], questions)[0]
    confidences = [a.confidence for a in answers if not isinstance(a, NoulAnswer)]
    assert confidences
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_spread_matches_typesafe_demo_pairs() -> None:
    """Pin their worked examples (and the degenerate K < 2 guard)."""
    from open_jev.main import spread_confidence

    # Their K=3 examples: (3 * peak - 1) / 2, at their rounding.
    assert spread_confidence([0.61, 0.25, 0.14]) == pytest.approx(0.42, abs=0.005)
    assert spread_confidence([0.84, 0.10, 0.06]) == pytest.approx(0.76, abs=TOL)
    # Uniform -> 0, one-hot -> 1, any K >= 2.
    assert spread_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=TOL)
    assert spread_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=TOL)
    # K = 2: (2 * 0.9 - 1) / 1 = 0.8.
    assert spread_confidence([0.9, 0.1]) == pytest.approx(0.8, abs=TOL)
    # K = 4, uniform -> 0.
    assert spread_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0, abs=TOL)
    # Their "peak 0.40 -> 0.20" example is FOUR options (docs example
    # requested_resolution: [0.34, 0.40, 0.02, 0.24] -> 0.2). The same peak
    # over K=3 yields 0.10 -- their own demo preset [40, 33, 27] shows 0.10.
    assert spread_confidence([0.34, 0.40, 0.02, 0.24]) == pytest.approx(0.20, abs=TOL)
    assert spread_confidence([0.40, 0.35, 0.25]) == pytest.approx(0.10, abs=TOL)
    # Flatter than 1/K: impossible for probabilities summing to 1, but the
    # clamp is what keeps such input at 0 instead of going negative.
    assert spread_confidence([0.33, 0.33, 0.33]) == pytest.approx(0.0, abs=TOL)
    # K < 2 must raise, not divide by zero.
    with pytest.raises(ValueError):
        spread_confidence([0.5])
    with pytest.raises(ValueError):
        spread_confidence([])


def test_spread_kind_used_in_forward_answers(cfg, state, questions) -> None:
    """With `spread`, Choice/Score confidences are the pure function of probs."""
    from open_jev.main import spread_confidence

    spread_model = _spread_model(cfg)
    with torch.no_grad():
        answers = spread_model([state], questions)[0]

    checked = 0
    for answer in answers:
        if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
            assert answer.confidence == pytest.approx(
                spread_confidence(answer.probabilities.values()), abs=TOL
            )
            checked += 1
    assert checked == 2  # one Choice + one Score in the shared questions


def test_noul_has_no_confidence_either_kind(model, cfg, state, questions) -> None:
    """Noul answers never carry confidence -- both kinds, same contract."""
    with torch.no_grad():
        default_rows = model([state], questions)
    noul_default = [a for row in default_rows for a in row if isinstance(a, NoulAnswer)]
    assert noul_default
    for answer in noul_default:
        assert not hasattr(answer, "confidence")

    spread_model = _spread_model(cfg)
    with torch.no_grad():
        spread_rows = spread_model([state], questions)
    noul_spread = [a for row in spread_rows for a in row if isinstance(a, NoulAnswer)]
    assert noul_spread
    for answer in noul_spread:
        assert not hasattr(answer, "confidence")


def test_invalid_confidence_kind_is_clean_error() -> None:
    """A bad kind fails fast in `Jev.__init__`, naming offender and allowed set."""
    cfg = JevConfig(confidence_kind="entropy")
    with pytest.raises(ValueError) as excinfo:
        Jev(cfg)
    message = str(excinfo.value)
    assert "entropy" in message
    assert "evidential" in message
    assert "spread" in message


def test_logits_always_returns_evidential_confidence(
    model, cfg, state, questions
) -> None:
    """`logits` keeps evidential confidence even when answers use `spread`."""
    spread_model = _spread_model(cfg)
    with torch.no_grad():
        answers = spread_model([state], questions)[0]
        outputs = spread_model.logits([state], questions)

    # Shape/semantics pinned: one (probs, conf) tuple per question, both
    # tensors, conf scalar in [0, 1] -- exactly what RLCDLoss unpacks.
    assert len(outputs) == len(questions)
    for output in outputs:
        assert isinstance(output, tuple) and len(output) == 2
        probs, conf = output
        assert isinstance(probs, torch.Tensor) and probs.ndim == 2
        assert isinstance(conf, torch.Tensor) and conf.shape == (1,)
        assert 0.0 <= conf.item() <= 1.0

    # EXPECTED divergence: answers carry the spread value while logits still
    # carries the evidential head's. This is the pin that logits was NOT
    # switched -- RLCDLoss.evidential assumes conf is independent of probs.
    compared = 0
    for answer, (_, conf) in zip(answers, outputs):
        if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
            assert abs(conf.item() - answer.confidence) > 1e-6
            compared += 1
    assert compared == 2  # one Choice + one Score in the shared questions

    # Positive pin: logits conf IS the evidential value -- identical weights
    # (same seed) mean it equals the default model's forward conf exactly.
    with torch.no_grad():
        default_answers = model([state], questions)[0]
    for default, (_, conf) in zip(default_answers, outputs):
        if isinstance(default, (ChoiceAnswer, ScoreAnswer)):
            assert conf.item() == pytest.approx(default.confidence, abs=TOL)
