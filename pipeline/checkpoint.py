"""Minimal, resumable checkpoint I/O for the stage-4 training loop.

The published checkpoint *format* belongs to stage 5 (``open_jev/checkpoint.py``,
another agent): this module deliberately implements only the minimal skeleton
the stage-4 spec asks for -- a ``dict`` with at least::

    {"model": state_dict, "config": asdict(TrainConfig), "step": int,
     "metrics": dict}

plus the two extras risk R3 demands travel with the weights (``seed`` and
``torch_version``) and -- because a resume must continue *optimising*, not just
re-infer -- the AdamW ``optimizer`` state, saved whenever the caller passes it.

Two more rules shape the code:

* Everything stored must survive ``torch.load`` under the safe default
  (``weights_only=True``, the default since torch 2.6): plain dicts, lists,
  numbers, strings and tensors only -- hence ``asdict(cfg)`` rather than the
  dataclass instance, and no pickled callables.
* ``load_checkpoint`` returns the *model* (the roundtrip test calls
  ``m2.logits(...)`` straight away), so training code that also needs
  ``step``/``config``/``optimizer`` reads the raw dict via
  :func:`read_checkpoint` instead of fishing it out of the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["load_checkpoint", "read_checkpoint", "save_checkpoint"]

#: Keys the stage-5 format guarantees (Шаг 3, metadata test).
REQUIRED_KEYS: frozenset[str] = frozenset({"model", "config", "step", "metrics"})


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: Any,
    step: int,
    metrics: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Write one checkpoint; returns the path (parents are created).

    Args:
        model: the module whose ``state_dict`` is stored.
        cfg: the run's ``TrainConfig`` (or any dataclass / mapping); stored as
            a plain dict so ``torch.load`` needs no pickle of our classes.
        step: global step the run reached -- what ``train(..., resume=)`'
            restarts from (``step_start``).
        metrics: whatever the run measured at save time (loss parts, eval).
        optimizer: pass it during training so a resume restores AdamW's
            moment buffers; without it a resumed run would restart the
            optimiser cold (the resume test requires no ``KeyError``).

    The checkpoint also records ``seed`` and ``torch_version`` (risk R3):
    stage 5 needs them to verify a run can be reproduced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    ckpt: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config,
        "step": int(step),
        "metrics": dict(metrics),
        "seed": getattr(cfg, "seed", None),
        # str() because torch.__version__ is a TorchVersion instance, which
        # torch.load(weights_only=True) refuses to unpickle as-is.
        "torch_version": str(torch.__version__),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)
    return path


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load the raw checkpoint dict, validating the minimal skeleton.

    ``weights_only=False`` because this is our own file and the stage-5 format
    may later carry richer metadata; the stage-4 skeleton itself is all plain
    data either way. Missing keys fail here with one message naming them,
    instead of a ``KeyError`` deep inside the training loop.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"{path}: expected a checkpoint dict, got {type(ckpt).__name__}"
        )
    missing = REQUIRED_KEYS - set(ckpt)
    if missing:
        raise ValueError(f"{path}: checkpoint missing key(s): {sorted(missing)}")
    return ckpt


def load_checkpoint(path: str | Path, model_factory: Callable[[], nn.Module]) -> nn.Module:
    """Rebuild the model from ``path`` and return it in eval mode.

    ``model_factory`` takes no arguments and supplies the architecture (it may
    read the checkpoint's ``config`` itself, as ``pipeline.train`` does); all
    weights then come from the file, so roundtripped logits match bit-for-bit.
    The returned model is ``eval()``-mode -- inference is the common case;
    ``pipeline.train`` flips it back with ``model.train()``.
    """
    ckpt = read_checkpoint(path)
    model = model_factory()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
