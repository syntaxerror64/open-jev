import hashlib, json, os, re, subprocess, sys, urllib.request
import pytest, torch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

@pytest.mark.slow
def test_published_checkpoint_smoke_load() -> None:
    """Загрузка ОПУБЛИКОВАННОГО чекпоинта; сеть + артефакт обязательны."""
    from open_jev.checkpoint import load
    from open_jev.main import Jev, JevConfig, Noul

    url = os.environ.get("JEV_CHECKPOINT_URL")
    if not url:
        pytest.skip("JEV_CHECKPOINT_URL не задан — публикации ещё нет")
    path = Path("/tmp/jev-published.pt")
    path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    model = load(path, model_factory=lambda cfg: Jev(cfg).eval())
    with torch.no_grad():
        probs, _ = model.logits([{"a": 1}], [Noul("ok?", key="k")])[0]
    assert torch.isfinite(probs).all()
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
    # sha256 манифеста сверяется в test_release_script (офлайн) — здесь только факт загрузки


def test_checkpoint_artifact_metadata_is_self_describing(tmp_path) -> None:
    """Офлайн-двойник slow-теста: артефакт self-describing, сеть не нужна.

    Даёт быстрый не-skip зелёный в этом файле, пока публикации нет (slow-тест
    выше честно skip-ается без JEV_CHECKPOINT_URL).
    """
    from open_jev.checkpoint import save
    from open_jev.main import Jev, JevConfig

    torch.manual_seed(0)
    model = Jev(JevConfig(vocab_size=512, d_model=32, n_heads=4, d_ff=64,
                          n_state_layers=1, n_question_layers=1,
                          n_readout_layers=1, n_slots=4, max_state_len=64))
    path = tmp_path / "artifact.pt"
    digest = save(path, model=model, step=0, metrics={"ece": 0.0}, seed=0)

    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["format_version"] == 1
    # sha256: 64 символа hex; сверка с байтами файла — в test_checkpoint.py
    # (двойная запись: digest от записи 1, см. докстринг open_jev/checkpoint.py)
    assert isinstance(ck["sha256"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", ck["sha256"]), ck["sha256"]
    assert ck["sha256"] == digest
    assert ck["torch_version"] == str(torch.__version__)
    assert "seed" in ck and isinstance(ck["seed"], int)


def test_model_card_matches_release_v020() -> None:
    manifest = REPO / "dist" / "jev-real.json"
    if not manifest.is_file(): pytest.skip("нет dist/ — экспорт не выполнялся")
    sha = json.loads(manifest.read_text(encoding="utf-8"))["sha256"]
    card = (REPO / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{64}", sha) and "Apache 2.0" in card
    assert sha in card, "MODEL_CARD.md не содержит sha256 dist-манифеста"
