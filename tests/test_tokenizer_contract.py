import pytest
import torch

def _hash():
    from open_jev.main import HashTokenizer
    return HashTokenizer(512)

def _bpe():
    # Ленивый импорт: пока класса нет — падает ТОЛЬКО этот параметр, а не весь файл
    from open_jev.bpe_tokenizer import BPETokenizer
    return BPETokenizer.load("cache/tokenizer.json")

@pytest.mark.parametrize("factory", [_hash, _bpe], ids=["hash", "bpe"])
def test_empty_string_encodes_to_single_pad_free_token(factory) -> None:
    tok = factory()
    assert tok.encode("", max_len=8) == [1]

@pytest.mark.parametrize("factory", [_hash, _bpe], ids=["hash", "bpe"])
def test_ids_stay_inside_vocab_and_respect_max_len(factory) -> None:
    tok = factory()
    ids = tok.encode("the customer wants a refund now", max_len=6)
    assert 1 <= len(ids) <= 6
    assert all(1 <= i < tok.vocab_size for i in ids)

@pytest.mark.parametrize("factory", [_hash, _bpe], ids=["hash", "bpe"])
def test_encode_batch_contract(factory) -> None:
    tok = factory()
    ids, mask = tok.encode_batch(["alpha", "", "gamma delta"], max_len=8)
    assert ids.dtype == torch.long
    assert mask.dtype == torch.bool
    assert ids.shape == mask.shape and ids.shape[0] == 3
    assert not mask.all(dim=1).any(), "ни одна строка не должна быть целиком PAD"
    assert (ids[mask] == 0).all(), "на позициях PAD должен лежать 0"

@pytest.mark.parametrize("factory", [_hash, _bpe], ids=["hash", "bpe"])
def test_encoding_is_deterministic(factory) -> None:
    tok = factory()
    assert tok.encode("repeatable text", 16) == tok.encode("repeatable text", 16)
