"""Eval report generator: JSON + markdown (risks R1, R4).

Risk R1: weights are random until stage 4, so every report the harness
produces *now* carries ``model: "random"`` and a warning that the numbers
are mechanics-only and must not be interpreted as quality measurements.
The warning ships inside the JSON payload (so it survives machine reads)
and as a blockquote in the markdown (so it survives human reads).

Risk R4: ECE depends on the bin count (``n_ece_bins=10`` by default,
open_jev/main.py:843), so ``n_bins`` is pinned to 10 in the report metadata.
Stage-2 and stage-4 numbers are only comparable if the binning matches.
"""

from __future__ import annotations

import json
from typing import Any

from torch import Tensor

from open_jev.main import RLCDLoss

__all__ = ["build_report", "consistency"]

DEFAULT_N_BINS = 10

RANDOM_WEIGHTS_WARNING = (
    "Model weights are random; these numbers are evaluation mechanics only "
    "and are not interpretable as quality metrics until stage 4 (trained "
    "weights)."
)


def consistency(probs: Tensor, augmented_probs: Tensor) -> float:
    """Symmetric KL between predictions and their augmented restatements.

    Delegates to ``RLCDLoss.symmetric_kl`` (open_jev/main.py:856) so the
    consistency term in the loss and the consistency metric in the report
    are the same number -- otherwise the report lies about training.
    """
    return float(RLCDLoss.symmetric_kl(probs, augmented_probs))


def build_report(metrics: dict, model: str, **meta: Any) -> dict:
    """Render metrics into parallel JSON and markdown representations.

    Args:
        metrics: metric name -> scalar value (e.g. ece, brier, nll,
            consistency). Keys are preserved as given.
        model: model identifier; stage 2 passes ``"random"`` (risk R1).
        **meta: extra metadata (e.g. dataset path, n, seed). ``n_bins`` is
            pinned to 10 unless explicitly overridden (risk R4).

    Returns:
        ``{"json": <str>, "markdown": <str>}`` -- the JSON string parses
        with :mod:`json` and carries ``model``, ``n_bins``, ``warning`` and
        ``metrics``; the markdown carries a ``| metric | value |`` table.
    """
    payload: dict[str, Any] = {
        "model": model,
        "n_bins": meta.pop("n_bins", DEFAULT_N_BINS),
        "metrics": dict(metrics),
        "warning": RANDOM_WEIGHTS_WARNING,
        **meta,
    }
    json_str = json.dumps(payload, indent=2, sort_keys=True)

    lines = [
        "# Eval report",
        "",
        f"model: `{payload['model']}`",
        f"n_bins: {payload['n_bins']}",
        "",
        f"> **Warning:** {RANDOM_WEIGHTS_WARNING}",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value} |" for name, value in payload["metrics"].items()]
    return {"json": json_str, "markdown": "\n".join(lines) + "\n"}
