"""Publishable checkpoint format, v1 (stage 5, step 1, agent A5).

A checkpoint file is a single ``torch.save`` payload — a plain ``dict`` of
simple types only (``dict``/``list``/``int``/``float``/``str``/tensors), so it
loads without custom classes:

    ck = torch.load(path, map_location="cpu", weights_only=False)

Keys (v1):

    state_dict      dict[str, Tensor]  — model.state_dict()
    config          dict[str, prim]    — dataclasses.asdict(model.cfg)
    step            int                — training step
    metrics         dict[str, float]   — evaluation metrics
    format_version  int                — == 1
    torch_version   str                — str(torch.__version__), plain str on
                                         purpose: torch.__version__ is a
                                         TorchVersion instance which would
                                         pickle a custom global and break
                                         ``weights_only=True`` loads (R1)
    sha256          str                — 64-hex digest, see below
    seed            int                — RNG seed fixed at save time (R3: a
                                         checkpoint without a seed cannot be
                                         reproduced)

save() -> the sha256 hex digest; load(path, *, model_factory) rebuilds the
model via ``model_factory(JevConfig(**config))``, loads the state dict and
returns it in eval mode.

-------------------------------------------------------------------------------
WHY THE sha256 SCHEME IS A DOUBLE WRITE (deviation from the spec, owner directive)
-------------------------------------------------------------------------------
The test line was::

    assert ck["sha256"] == hashlib.sha256(ck_path.read_bytes()).hexdigest()

That is a self-reference and is mathematically impossible: the digest stored
*inside* the file would have to equal the SHA-256 of the file *containing* it
(finding such a file is a ~2^256 partial-preimage search — infeasible). The
same holds for any "digest of the final bytes" scheme: substituting the digest
into the file always changes the bytes it was computed from.

Implemented instead (the closest semantics to R2, "digest of the WRITTEN
file"):

1. ``payload["sha256"] = "0"*64`` → ``torch.save(payload, path,
   _use_new_zipfile_serialization=False)`` → bytes0 (write 1).
2. ``h = sha256(bytes0)`` — digest of an actually written file.
3. ``payload["sha256"] = h`` → same ``torch.save`` again → final file F
   (write 2).

The legacy (non-zip) pickle protocol is byte-deterministic — no zip
timestamps — and the payload object is shared between writes, so F differs
from bytes0 in EXACTLY those 64 bytes (verified empirically for this format).
The digest therefore describes the file with the field in its write-1 state;
the test normalizes F back by ``F.replace(h, b"0"*64)`` — an exact, length
preserving restoration of bytes0 — and checks ``sha256(bytes0) == h``. This
keeps the spirit ("digest of the written file, verified against file
bytes") while staying honest about the fixed-point impossibility. See
``tests/test_checkpoint.py::test_sha256_matches_file_bytes``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from open_jev.main import JevConfig

__all__ = ["FORMAT_VERSION", "load", "save"]

#: Version of the on-disk layout (not the package version — see version.py).
FORMAT_VERSION = 1

#: Fixed-length placeholder for the first write; the real digest has the same
#: length (64 hex chars), so the substitution in write 2 changes no offsets.
_PLACEHOLDER = "0" * 64


def _dump(payload: dict[str, Any], path: Path) -> None:
    """Serialize `payload` to `path` with the legacy torch protocol.

    ``_use_new_zipfile_serialization=False`` selects the old pickle stream:
    unlike the zipfile format it embeds no timestamps or file names, so two
    dumps of the same object graph are byte-identical (verified: repeated
    saves match, and payloads differing only in the 64-char ``sha256`` string
    differ in exactly those 64 bytes). That determinism is what makes the
    double-write digest scheme below verifiable.
    """
    with open(path, "wb") as f:
        torch.save(payload, f, _use_new_zipfile_serialization=False)


def save(
    path: str | Path,
    *,
    model: Any,
    step: int,
    metrics: Mapping[str, float],
    seed: int = 0,
) -> str:
    """Write `model` to `path` in checkpoint format v1.

    Args:
        path: destination file (overwritten).
        model: a ``Jev`` — uses ``model.cfg`` (the frozen ``JevConfig``) and
            ``model.state_dict()``.
        step: training step to record.
        metrics: evaluation metrics, plain ``{name: float}``.
        seed: RNG seed the weights derive from; stored in the payload (R3 —
            a checkpoint without a seed is not reproducible).

    Returns:
        The sha256 hex digest of write 1 (also stored under ``"sha256"``).

    The file is written twice: first with ``sha256="0"*64``, then with the
    digest of those first bytes. See the module docstring for why a digest of
    the *final* bytes is impossible and how the test normalizes it.
    """
    payload: dict[str, Any] = {
        "state_dict": dict(model.state_dict()),
        "config": dataclasses.asdict(model.cfg),
        "step": int(step),
        "metrics": dict(metrics),
        "format_version": FORMAT_VERSION,
        # plain str: torch.__version__ is a TorchVersion (str subclass) whose
        # pickled form names a custom global and fails weights_only loads (R1)
        "torch_version": str(torch.__version__),
        "seed": int(seed),
        "sha256": _PLACEHOLDER,
    }

    dest = Path(path)
    _dump(payload, dest)                                   # write 1
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()  # of written bytes
    payload["sha256"] = digest
    _dump(payload, dest)                                   # write 2 (final)
    return digest


def load(
    path: str | Path,
    *,
    model_factory: Callable[[JevConfig], Any],
) -> Any:
    """Read a v1 checkpoint and rebuild the model.

    Args:
        path: checkpoint file written by :func:`save`.
        model_factory: called as ``model_factory(JevConfig(**config))`` — the
            caller decides the concrete class (and tokenizer), the format only
            carries the config.

    Returns:
        The model with the saved weights loaded, in ``eval()`` mode.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = JevConfig(**ck["config"])
    model = model_factory(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model
