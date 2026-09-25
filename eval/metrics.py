"""Pure evaluation metrics: ECE, Brier score, NLL.

Semantics are delegated to the ``RLCDLoss`` statics wherever possible: the
evaluation report and the training loss must compute the *same* numbers,
otherwise the report lies about training (test_metrics_match_rlcd_loss_statics
guards this). Only ECE is implemented locally, because the loss version is an
instance method pinned to ``self.n_ece_bins`` while evaluation needs a plain
function with an explicit ``n_bins`` knob -- the report records ``n_bins`` in
its metadata so stage-2 and stage-4 numbers stay comparable (risk R4).

All functions are pure: probs/target are [B, K] tensors, the return value is a
scalar tensor (``.item()`` works).
"""

from __future__ import annotations

from torch import Tensor

from open_jev.main import RLCDLoss

__all__ = ["brier", "ece", "nll"]


def ece(probs: Tensor, target: Tensor, n_bins: int = 10) -> Tensor:
    """Expected Calibration Error over equal-width confidence bins.

    Confidence = max prob per row, accuracy = target mass under the argmax
    class. Bins are the ``n_bins`` equal segments of [0, 1]; the result is the
    fraction of rows in a bin times |mean confidence - mean accuracy| in that
    bin, summed over bins. Semantics mirror ``RLCDLoss.ece``
    (open_jev/main.py:872-882) as a standalone function of ``n_bins``.
    """
    conf, idx = probs.max(-1)
    acc = target.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    total = probs.new_zeros(())
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        w = ((conf >= lo) & (conf < hi)).to(probs.dtype)
        if w.sum() > 0:
            total = total + w.mean() * ((w * (conf - acc)).sum() / w.sum()).abs()
    return total


def brier(probs: Tensor, target: Tensor) -> Tensor:
    """Brier score, delegated to the loss so eval and training agree."""
    return RLCDLoss.brier(probs, target)


def nll(probs: Tensor, target: Tensor) -> Tensor:
    """Soft cross-entropy NLL, delegated to the loss's ``soft_nll``."""
    return RLCDLoss.soft_nll(probs, target)
