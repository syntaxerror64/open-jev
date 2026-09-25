"""Trained BPE tokenizer with the same contract as ``HashTokenizer``.

Stage 3 replaces the stand-in hash tokenizer with
a real, *trained* Byte-Pair Encoding model. This module is the inference half;
training lives in ``scripts/train_bpe.py``.

Artifact
--------
``BPETokenizer.load(path)`` reads the JSON produced by::

    python -m scripts.train_bpe --vocab-size 4096 --out cache/tokenizer.json

The artifact is self-contained: NFC + lowercase normalizer, whitespace
pre-tokenizer and the learned merges all travel inside the file, so encoding
is reproducible from the artifact alone. No network access is involved
(decision O1: the default corpus is a fixed list of repository text files) and
the artifact is rebuilt locally instead of being committed (decision O4:
``cache/`` stays out of git).

Contract (see ``open_jev.tokenizers.Tokenizer`` — every tokenizer obeys it):

* ``encode(text, max_len) -> list[int]`` with ``1 <= len(ids) <= max_len``
  (``max_len >= 1``), every id inside ``[1, vocab_size)``, deterministic;
* ``encode("", max_len) == [1]`` — never an empty list;
* id ``0`` (PAD) is produced only by ``encode_batch``;
* ``encode_batch(texts, max_len) -> (ids [B, L] long, pad_mask [B, L] bool)``
  — right-padded, mask True exactly at PAD (inherited from
  ``BatchEncodingMixin``, the same logic ``HashTokenizer`` uses);
* ``vocab_size <= cfg.vocab_size`` so the ids fit both embedding tables in
  ``main.py`` (demos use 4096, the config default is 32000).

``tokenizers`` (the training extra, see requirements-dev.txt) is imported
lazily: importing this class never pulls it in, only ``load()`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from open_jev.tokenizers import BatchEncodingMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tokenizers import Tokenizer

# Contract ids: 0 is PAD (encode_batch only), 1 doubles as "unknown/out of
# range" so a degenerate input can never emit 0.
PAD_ID = 0
UNK_ID = 1


class BPETokenizer(BatchEncodingMixin):
    """Byte-Pair Encoding tokenizer backed by the ``tokenizers`` package.

    Instances are normally built via :meth:`load` from a trained artifact;
    the constructor accepts a ready ``tokenizers.Tokenizer`` for tests and
    for callers that trained one in memory.
    """

    PAD: int = PAD_ID

    def __init__(self, tokenizer: "Tokenizer") -> None:
        self._tokenizer = tokenizer
        # Every id the model can emit is in [0, vocab_size); the contract
        # needs ids in [1, vocab_size), which encode() enforces.
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Rebuild a trained tokenizer from its JSON artifact.

        Raises ``FileNotFoundError`` with a training hint when the artifact
        has not been produced yet.
        """
        from tokenizers import Tokenizer

        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"BPE artifact not found: {artifact}. Train it first with "
                f"`python -m scripts.train_bpe --out {artifact}` "
                "(the default corpus is local — no download needed)."
            )
        return cls(Tokenizer.from_file(str(artifact)))

    def encode(self, text: str, max_len: int) -> list[int]:
        """Encode ``text`` into ids in ``[1, vocab_size)``, at most ``max_len``.

        The normalizer inside the artifact lowercases (NFC first), so casing
        never changes the output. Empty/whitespace-only input yields ``[1]``.
        """
        if not text.strip():
            return [1]
        raw = self._tokenizer.encode(text, add_special_tokens=False).ids
        # PAD is reserved for encode_batch; anything outside the contract
        # range (only PAD can be) degrades to the unknown id.
        ids = [i if 1 <= i < self.vocab_size else UNK_ID for i in raw]
        return ids[:max_len] or [1]
