"""
Tokenizer interfaces shared by every tokenizer implementation.

Stage 3 swaps the stand-in hash tokenizer for a
trained BPE tokenizer. Both must honour one contract so `Jev` can switch
between them without touching its call sites:

    * `encode(text, max_len) -> list[int]`  ids inside `[1, vocab_size)`,
      never empty (`encode("", n) == [1]`), deterministic, truncated to
      `max_len`.
    * `encode_batch(texts, max_len) -> (ids, pad_mask)`  `ids` is a
      `torch.long` [B, L] tensor, `pad_mask` a `torch.bool` [B, L] tensor that
      is True only at padding positions; PAD = 0 and every row keeps at least
      one real token.
    * `vocab_size`  must fit inside `cfg.vocab_size` (both embedding tables in
      `main.py` use `padding_idx=0`, so 0 stays reserved for PAD).
    * `load(path)`  classmethod that rebuilds a trained tokenizer from its
      saved artifact (e.g. `cache/tokenizer.json`).

Typing is *structural* (`typing.Protocol`): implementations do not inherit
anything and are never checked at runtime — a class satisfies the contract by
simply having the same members.

`BatchEncodingMixin` carries the one piece of logic that is identical for
every "just an `encode`" tokenizer: right-padded batch tensors.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch
from torch import Tensor


class Tokenizer(Protocol):
    """Structural contract every tokenizer must satisfy.

    `HashTokenizer` (main.py) and `BPETokenizer` (bpe_tokenizer.py) both fit
    this shape; nothing forces them to subclass it.
    """

    PAD: int
    vocab_size: int

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        """Rebuild a tokenizer from a saved artifact."""
        ...

    def encode(self, text: str, max_len: int) -> list[int]:
        """Encode one string into ids in `[1, vocab_size)`; `""` -> `[1]`."""
        ...

    def encode_batch(self, texts: Sequence[str], max_len: int) -> tuple[Tensor, Tensor]:
        """Return `(ids [B, L] long, pad_mask [B, L] bool)`, mask True at PAD."""
        ...


class BatchEncodingMixin:
    """Default `encode_batch` for tokenizers that define only `encode`.

    Inherit this and you get the exact batch tensors `HashTokenizer` produces
    today: right-padded to the longest sequence in the batch, `torch.long` ids,
    `torch.bool` mask that is True only where the PAD id (0) sits.

    The concrete class supplies `vocab_size` (and may override `PAD`).
    """

    PAD: int = 0
    vocab_size: int  # provided by the concrete tokenizer

    def encode(self, text: str, max_len: int) -> list[int]:
        """Concrete tokenizers override this with their real encoder."""
        raise NotImplementedError

    def encode_batch(self, texts: Sequence[str], max_len: int) -> tuple[Tensor, Tensor]:
        """Returns (ids [B, L], pad_mask [B, L]) where pad_mask is True at PAD."""
        seqs = [self.encode(t, max_len) for t in texts]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), self.PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return ids, ids.eq(self.PAD)
