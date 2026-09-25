"""Export a training run's checkpoint to a publishable artifact + manifest.

Usage::

    python -m scripts.export_checkpoint --run runs/<name> --out dist/<file>.pt

Input — ``<run>/checkpoint.pt`` in either layout the repo produces:

* stage-4 pipeline layout (``save_checkpoint`` contract):
  ``{"model", "config", "step", "metrics"}``;
* open-jev v1 layout (``open_jev.checkpoint.save``): ``{"state_dict",
  "config", "step", "metrics", "format_version", "torch_version", "seed",
  "sha256"}``.

Output — the artifact at ``--out`` written through
:func:`open_jev.checkpoint.save` (v1 format, preserving ``step``/``metrics``
/``seed``, double-write digest per R2) plus a sidecar manifest at
``--out``.with_suffix(".json")::

    {"sha256", "size_bytes", "format_version", "config", "step", "metrics",
     "source", "torch_version"}

The manifest ``sha256`` is the digest of the **written artifact bytes**
(read back from disk after ``save()``), so it equals ``sha256sum <out>`` —
that is what the offline verification in the success criteria checks
(Risk R2: digest after the write, never before).

The input is fully read into memory before the output is opened, so
``--out`` may even point at the input file without corrupting it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from open_jev.checkpoint import FORMAT_VERSION, save
from open_jev.main import JevConfig

__all__ = ["export_checkpoint", "main"]


class _Shim:
    """What ``open_jev.checkpoint.save`` needs: ``cfg`` + ``state_dict()``.

    ``save()`` is written for a ``Jev`` instance but only ever touches those
    two members, so a loaded checkpoint dict can travel through the exact same
    serialization path (and therefore the same double-write digest scheme) as
    a live model — no second implementation of the format in this script.
    """

    def __init__(self, cfg: JevConfig, state_dict: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self._state = dict(state_dict)

    def state_dict(self) -> dict[str, Any]:
        return self._state


def _read(path: Path) -> dict[str, Any]:
    """Load the input payload; every failure mode gets a plain-language error."""
    if not path.is_file():
        raise SystemExit(
            f"error: input checkpoint not found: {path}\n"
            "       expected <run>/checkpoint.pt — train the run first "
            "(e.g. `python -m pipeline.train`), then export."
        )
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # unreadable/corrupt/pickled-foreign-object
        raise SystemExit(f"error: cannot read {path}: {exc}") from exc
    if not isinstance(ck, dict):
        raise SystemExit(
            f"error: {path}: expected a dict payload, "
            f"got {type(ck).__name__}"
        )
    return ck


def _normalize(
    ck: dict[str, Any], source: Path
) -> tuple[JevConfig, dict[str, Any], int, dict[str, float], int]:
    """Accept both layouts; return ``(cfg, state_dict, step, metrics, seed)``.

    The stage-4 pipeline layout stores weights under ``"model"`` and open-jev
    v1 under ``"state_dict"``; everything else (config/step/metrics) is
    common, ``seed`` exists only in v1. Config keys that are not
    ``JevConfig`` fields (a ``TrainConfig`` written wholesale, say) are
    dropped with a note — the published config must rebuild the model exactly,
    so model fields are kept and loop fields are not part of the artifact.
    """
    if "state_dict" in ck:
        state = ck["state_dict"]
    elif "model" in ck:
        state = ck["model"]
    else:
        raise SystemExit(
            f"error: {source}: unrecognized checkpoint layout, "
            f"keys={sorted(ck)} — expected 'model' (stage-4 pipeline) or "
            "'state_dict' (open-jev v1)"
        )
    if not isinstance(state, Mapping) or not state:
        raise SystemExit(
            f"error: {source}: weight dict is empty or not a mapping "
            f"({type(state).__name__})"
        )
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise SystemExit(f"error: {source}: 'model'/'state_dict' must map to tensors")

    config = ck.get("config")
    if not isinstance(config, Mapping):
        raise SystemExit(
            f"error: {source}: 'config' must be a dict of JevConfig fields, "
            f"got {type(config).__name__}"
        )
    known = {f.name for f in dataclasses.fields(JevConfig)}
    dropped = sorted(set(config) - known)
    kwargs = {k: v for k, v in config.items() if k in known}
    if not kwargs:
        raise SystemExit(
            f"error: {source}: config carries no JevConfig fields: {sorted(config)}"
        )
    if dropped:
        print(
            f"note: dropping non-model config key(s) {dropped} from {source}",
            file=sys.stderr,
        )
    try:
        cfg = JevConfig(**kwargs)
    except Exception as exc:
        raise SystemExit(f"error: {source}: config does not build a JevConfig: {exc}")

    step = ck.get("step", 0)
    if isinstance(step, bool) or not isinstance(step, int):
        raise SystemExit(f"error: {source}: 'step' must be an int, got {step!r}")

    raw_metrics = ck.get("metrics", {})
    if not isinstance(raw_metrics, Mapping):
        raise SystemExit(
            f"error: {source}: 'metrics' must be a mapping, "
            f"got {type(raw_metrics).__name__}"
        )
    # Nested sections (pipeline/train.py logs {"train": {...}, "eval": {...}})
    # flatten to dotted scalar keys; anything non-numeric is an error.
    metrics: dict[str, float] = {}

    def _flatten(prefix: str, mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                _flatten(name, value)
            else:
                metrics[name] = float(value)

    try:
        _flatten("", raw_metrics)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"error: {source}: metrics must be scalar: {exc}")

    seed = 0
    for src in (ck, config):
        candidate = src.get("seed")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            seed = candidate
            break

    return cfg, dict(state), step, metrics, seed


def export_checkpoint(run: str | Path, out: str | Path) -> dict[str, Any]:
    """Read ``run/checkpoint.pt`` (or ``run`` itself if it is a file), write
    the v1 artifact to ``out`` and its manifest next to it; return the
    manifest dict.
    """
    run_path = Path(run)
    source = run_path if run_path.is_file() else run_path / "checkpoint.pt"
    ck = _read(source)
    cfg, state, step, metrics, seed = _normalize(ck, source)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(out_path, model=_Shim(cfg, state), step=step, metrics=metrics, seed=seed)

    # Digest of the bytes actually on disk (R2) — equals `sha256sum <out>`.
    data = out_path.read_bytes()
    manifest: dict[str, Any] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "format_version": FORMAT_VERSION,
        "config": dataclasses.asdict(cfg),
        "step": step,
        "metrics": metrics,
        "source": str(source),
        "torch_version": str(torch.__version__),
    }
    manifest_path = out_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_checkpoint",
        description="Export runs/<name>/checkpoint.pt to a v1 artifact + JSON "
        "manifest for release.",
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="RUN",
        help="run directory (reads RUN/checkpoint.pt) or a checkpoint file",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="artifact to write; the manifest goes to PATH with .json suffix",
    )
    args = parser.parse_args(argv)
    manifest = export_checkpoint(args.run, args.out)
    print(f"artifact   {Path(args.out)} ({manifest['size_bytes']} bytes)")
    print(f"manifest   {Path(args.out).with_suffix('.json')}")
    print(f"sha256     {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
