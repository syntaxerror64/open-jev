import json
from eval.dataset import load_jsonl, write_jsonl
from eval.report import build_report

def test_jsonl_roundtrip(tmp_path) -> None:
    rows = [{"state": {"a": 1}, "questions": ["x"], "targets": [[0.5, 0.5]]}]
    p = tmp_path / "d.jsonl"
    write_jsonl(rows, p)
    assert load_jsonl(p) == rows

def test_report_is_valid_json_and_markdown(tmp_path) -> None:
    report = build_report(metrics={"ece": 0.1, "brier": 0.2, "nll": 0.7,
                                   "consistency": 0.05}, model="random")
    assert json.loads(report["json"])["metrics"]["ece"] == 0.1
    assert "| metric |" in report["markdown"] and "ece" in report["markdown"]
