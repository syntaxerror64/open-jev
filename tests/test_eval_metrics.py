import math

import pytest, torch
from eval.metrics import brier, ece, nll

def test_perfectly_calibrated_example_has_zero_ece() -> None:
    # 10 примеров: уверенность 0.9, и целевая частота действительно 0.9
    probs = torch.tensor([[0.1, 0.9]] * 10)
    target = torch.tensor([[0.1, 0.9]] * 10)
    assert ece(probs, target, n_bins=10).item() == pytest.approx(0.0, abs=1e-6)

def test_overconfident_predictions_have_large_ece() -> None:
    # модель всегда уверена в 0.99 и всегда права на 0 — разрыв 0.99
    probs = torch.tensor([[0.01, 0.99]] * 10)
    target = torch.tensor([[1.0, 0.0]] * 10)      # класс 0 всегда верен
    assert ece(probs, target, n_bins=10).item() > 0.9

def test_brier_prefers_correct_confident_over_spread() -> None:
    one_hot = torch.tensor([[0.0, 1.0]])
    spread = torch.tensor([[0.5, 0.5]])
    confident_correct = torch.tensor([[0.1, 0.9]])
    assert brier(confident_correct, one_hot) < brier(confident_correct, spread)

def test_nll_known_value() -> None:
    p = torch.tensor([[0.5, 0.5]])
    assert nll(p, torch.tensor([[0.5, 0.5]])).item() == pytest.approx(-math.log(0.5))

def test_metrics_match_rlcd_loss_statics() -> None:
    # eval и лосс обязаны считать одно и то же, иначе отчёт врёт про обучение
    from open_jev.main import RLCDLoss
    probs, target = torch.rand(8, 4), torch.rand(8, 4)
    target = target / target.sum(-1, keepdim=True)
    assert brier(probs, target).item() == pytest.approx(
        RLCDLoss.brier(probs, target).item(), rel=1e-6)
