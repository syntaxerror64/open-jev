"""Stage 5, step 2: export + release dry-run (agent B).

Код — Шаг 2 из stage 5 дословно, с одной
самодостаточной адаптацией: исходные предусловия (runs/tiny от этапа 4, ассет
dist/jev.pt в git-каталоге) в этой среде не гарантированы, поэтому тест сам
готовит вход там, где файла нет. Ни один тест не полагается на чужие артефакты
и не выходит в сеть (O2).
"""

import json, subprocess, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def _tiny_run(tmp_path: Path) -> Path:
    """Каталог run с checkpoint.pt для --run (предусловие теста 1).

    Если этап 4 уже положил runs/tiny/checkpoint.pt — берём его (тест проверяет
    реальный вход). Иначе создаём свой: open_jev.checkpoint.save пишет v1-формат
    (ключ state_dict), а вход export-скрипта для этапа 4 — dict с ключами
    {"model", "config", "step", "metrics"}, поэтому результат save()
    проецируется на эти ключи (ровно тот формат, который читает скрипт).
    """
    staged = REPO / "runs" / "tiny" / "checkpoint.pt"
    if staged.is_file():
        return staged.parent

    from open_jev.checkpoint import save
    from open_jev.main import Jev, JevConfig

    torch.manual_seed(0)
    model = Jev(JevConfig(vocab_size=512, d_model=32, n_heads=4, d_ff=64,
                          n_state_layers=1, n_question_layers=1,
                          n_readout_layers=1, n_slots=4, max_state_len=64))
    staging = tmp_path / "_staging.pt"
    save(staging, model=model, step=10,
         metrics={"ece": 0.12, "brier": 0.3}, seed=0)
    ck = torch.load(staging, map_location="cpu", weights_only=False)
    staging.unlink()

    run = tmp_path / "tiny"
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ck["state_dict"], "config": ck["config"],
                "step": ck["step"], "metrics": ck["metrics"]},
               run / "checkpoint.pt")
    return run


def test_export_script_writes_artifact_and_manifest(tmp_path) -> None:
    # предусловие: runs/tiny существует (создаёт этап 4 или conftest-фикстура)
    run = _tiny_run(tmp_path)
    out = tmp_path / "jev.pt"
    p = subprocess.run([sys.executable, "-m", "scripts.export_checkpoint",
                        "--run", str(run), "--out", str(out)],
                       cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert out.exists() and out.with_suffix(".json").exists()
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert {"sha256", "size_bytes", "format_version", "config"} <= set(manifest)


def test_release_script_dry_run_builds_manifest(tmp_path) -> None:
    # ассет dist/jev.pt в рабочей копии может отсутствовать — свой:
    asset = tmp_path / "jev.pt"
    asset.write_bytes(b"placeholder checkpoint bytes")
    p = subprocess.run([sys.executable, "scripts/release.py",
                        "--tag", "v0.0.0-test", "--asset", str(asset),
                        "--dry-run"], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    manifest = json.loads(p.stdout)          # что было бы сделано: тег, ассеты, sha256
    assert manifest["tag"] == "v0.0.0-test" and manifest["assets"]
