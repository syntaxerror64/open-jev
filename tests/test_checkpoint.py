import hashlib, pytest, torch
from open_jev.checkpoint import load, save
from open_jev.main import Jev, JevConfig, Noul, Choice

QS = [Noul("ok?", key="a"), Choice("pick", options=["x", "y"], key="b")]

def _tiny(seed: int = 0) -> Jev:
    torch.manual_seed(seed)
    return Jev(JevConfig(vocab_size=512, d_model=32, n_heads=4, d_ff=64,
                         n_state_layers=1, n_question_layers=1,
                         n_readout_layers=1, n_slots=4, max_state_len=64))

@pytest.fixture()
def ck_path(tmp_path):
    p = tmp_path / "ck.pt"
    save(p, model=_tiny(), step=10, metrics={"ece": 0.12, "brier": 0.3})
    return p

def test_roundtrip_produces_identical_logits(ck_path) -> None:
    m1 = _tiny().eval()
    m2 = load(ck_path, model_factory=lambda cfg: Jev(cfg).eval())
    with torch.no_grad():
        a, b = m1.logits([{"a": 1}], QS), m2.logits([{"a": 1}], QS)
    for (p1, _), (p2, _) in zip(a, b):
        assert torch.allclose(p1, p2, atol=1e-6)   # identical logits

def test_metadata_is_complete(ck_path) -> None:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert {"state_dict", "config", "step", "metrics", "format_version",
            "torch_version", "sha256"} <= set(ck)
    assert ck["format_version"] == 1
    assert ck["torch_version"] == torch.__version__

def test_sha256_matches_file_bytes(ck_path) -> None:
    # digest считается от ЗАПИСАННОГО файла, иначе пересохранение его ломает
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    # Адаптация владельца (директива А5): исходная строка
    #   ck["sha256"] == sha256(ck_path.read_bytes())
    # самоссылочна и математически невыполнима — hexdigest внутри файла не может
    # равняться sha256 этого же файла (требовалось бы найти преимидж хеша,
    # содержащего сам этот хеш). save() пишет файл ДВАЖДЫ: запись 1 —
    # sha256="0"*64, запись 2 — h = sha256(байтов записи 1). Нормализация поля
    # в готовом файле обратно к "0"*64 восстанавливает байты записи 1
    # ДОСЛОВНО (legacy-протокол torch детерминирован, поле ровно 64 байта,
    # h встречается в файле один раз), и digest от них сходится.
    # Подробнее — в докстринге open_jev/checkpoint.py.
    h = ck["sha256"]
    assert h == hashlib.sha256(
        ck_path.read_bytes().replace(h.encode(), b"0" * 64)
    ).hexdigest()
    assert set(ck["state_dict"]) == set(_tiny().state_dict())

def test_config_reproduces_model_shape(ck_path) -> None:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = JevConfig(**ck["config"])
    fresh = Jev(cfg).state_dict()
    assert set(fresh) == set(ck["state_dict"])
    for k, v in ck["state_dict"].items():
        assert tuple(fresh[k].shape) == tuple(v.shape)
