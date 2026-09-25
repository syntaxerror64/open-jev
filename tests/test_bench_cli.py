"""Smoke test for the stage-1 benchmark CLI (stage 1, agent B).

Spec: stage 1, step 3. The CLI must:

* exist as `python -m benchmarks.cli`,
* finish a `--smoke` run well under 5 seconds (this is a push gate, so no
  `slow` marker),
* write artifacts exactly where the caller points it -- here `tmp_path`, never
  the repository root -- and emit valid JSON with the agreed schema
  (`meta` / `cases` / `speedup_vs_uncached`),
* pin the reproducibility knobs from risk R1 (fixed seed, >= 1 repeats,
  warmup, one torch thread) in `meta`,
* and mention `speedup` in the markdown, because the readiness criterion for
  the committed baseline is `grep -n speedup benchmarks/RESULTS.md`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_cli_smoke_is_fast_and_valid_json(tmp_path: Path) -> None:
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"

    t0 = time.perf_counter()
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.cli",
            "--smoke",
            "--json",
            str(json_path),
            "--md",
            str(md_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    assert time.perf_counter() - t0 < 5.0, "smoke должен укладываться в 5 c"

    report = json.loads(json_path.read_text())
    assert set(report) >= {"meta", "cases", "speedup_vs_uncached"}
    assert set(report["meta"]) >= {
        "seed",
        "repeats",
        "warmup",
        "torch",
        "device",
        "threads",
    }
    assert report["meta"]["threads"] == 1, "замеры должны идти в один поток (R1)"

    assert report["cases"], "smoke must produce at least one case"
    assert report["cases"][0]["encode_ms"]["n"] >= 1
    assert all(c["encode_ms"]["median"] >= 0 for c in report["cases"])
    # The state length actually measured by `encode_state` matches what meta
    # says was requested (make_state really produces T tokens).
    assert all(c["T"] == report["meta"]["T"] for c in report["cases"])
    assert isinstance(report["speedup_vs_uncached"], (int, float))
    assert report["speedup_vs_uncached"] > 0

    md = md_path.read_text()
    assert "speedup" in md, "readiness criterion greps for 'speedup' in the md"
