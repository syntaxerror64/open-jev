import json, subprocess, sys; from pathlib import Path
import pytest, torch
from pipeline.data import build_batch, load_jsonl
# модуля ещё нет — импорт внутри тестов Шага 1, чтобы Шаги 2-3 собирались
# и падали по своим причинам (returncode != 0 и т.п.), а не collection error
FIXTURE = Path(__file__).parent / "fixtures" / "alpaca_tiny.json"; REPO = Path(__file__).parent.parent

def test_transform_items_to_valid_rows() -> None:
    from scripts.fetch_data import transform               # модуля ещё нет
    rows = transform(json.loads(FIXTURE.read_text(encoding="utf-8"))[:3])
    for row in rows:
        assert set(row) == {"state", "questions", "targets"}
        assert set(row["state"]) == {"instruction", "input"}
        assert [q["type"] for q in row["questions"]] == ["choice", "noul", "score"]
        assert row["questions"] == rows[0]["questions"]     # data.py:153
    for t in build_batch(rows).targets:                   # внутри — validate_soft_targets
        assert t.dtype == torch.float32 and not (t >= 1.0 - 1e-6).any()   # не one-hot
        assert torch.allclose(t.sum(-1), torch.ones(len(rows)))           # uniform, сумма ≈1
def test_long_text_is_truncated_deterministically() -> None:
    from scripts.fetch_data import transform               # модуля ещё нет
    rows = transform([{"instruction": "x" * 900, "input": "y" * 700, "output": "z"}, {"instruction": "short", "input": "", "output": "ok"}])
    assert rows[0]["state"] == {"instruction": "x" * 600, "input": "y" * 600}
    assert rows[1]["state"] == {"instruction": "short", "input": ""}

def _run(out, seed=7, sample=8, train=6):                # subprocess, cwd=REPO
    return subprocess.run([sys.executable, "-m", "scripts.fetch_data", "--from-local",
        str(FIXTURE), "--out", str(out), "--seed", str(seed), "--sample-size",
        str(sample), "--train-size", str(train)], capture_output=True, text=True, cwd=REPO)

def test_cli_offline_same_seed_is_byte_identical(tmp_path) -> None:   # все флаги CLI, exit 0
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(a).returncode == 0 and _run(b).returncode == 0
    for name in ("train.jsonl", "holdout.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes()   # байт-в-байт
def test_sample_and_split_counts(tmp_path) -> None:       # 600/500 → 500/100
    out = tmp_path / "o"
    assert _run(out, seed=3, sample=6, train=5).returncode == 0
    tr, ho = load_jsonl(out / "train.jsonl"), load_jsonl(out / "holdout.jsonl")
    assert (len(tr), len(ho)) == (5, 1)
    key = lambda r: json.dumps(r["state"], sort_keys=True)
    assert not ({key(r) for r in tr} & {key(r) for r in ho})   # сплит без пересечений по state

def test_broken_json_fails_with_line_number(tmp_path) -> None:
    bad = tmp_path / "bad.json"; bad.write_text('[{"instruction": "x",}]', encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "scripts.fetch_data", "--from-local",
        str(bad), "--out", str(tmp_path / "o")], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0                          # exit ≠ 0
    assert "invalid JSON" in p.stderr                 # формат data.py:58: {path}:{lineno}:...
def test_data_real_is_gitignored() -> None:
    p = subprocess.run(["git", "check-ignore", "-q", "data/real/train.jsonl"], capture_output=True, cwd=REPO)
    if p.returncode not in (0, 1): pytest.skip("git недоступен")
    assert p.returncode == 0, "data/real/ обязан быть в .gitignore"
@pytest.mark.slow                                     # вне гейта; сеть + 52 MB
def test_real_alpaca_download_smoke(tmp_path) -> None:
    p = subprocess.run([sys.executable, "-m", "scripts.fetch_data", "--out", str(tmp_path), "--seed", "0",
        "--sample-size", "6", "--train-size", "5"], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0 and len(load_jsonl(tmp_path / "train.jsonl")) == 5
