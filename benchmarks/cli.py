"""Stage-1 benchmark CLI: state-cache reuse and multi-question scaling.

Spec: stage 1 (contract + step 3), README todo #3.

Usage:
    python -m benchmarks.cli --T 256 --N 1,4,16,64 --B 1 --repeats 7 \
        --warmup 2 --json out.json --md RESULTS.md [--smoke]

`--smoke` is the push-gate run: T=64, N=[1, 2], B=1, repeats=1, warmup=0 --
seconds, not minutes, because `tests/test_bench_cli.py` asserts < 5 s.

WHAT IS MEASURED (one case = one T, one N, one B)
--------------------------------------------------
    encode_ms    Stats over `repeats` calls of `encode_state([state] * B)`:
                 O(T^2), paid once per request.
    readout_ms   Stats over `repeats` calls of `_readout(cache, questions)`:
                 ALL N questions in one call against the warm cache,
                 O(M * T * N).
    cached_ms    = encode.median + readout.median
                 -- the architecture's path: encode the state ONCE, answer all
                 N questions from that cache.
    uncached_ms  = N * encode.median + readout.median
                 -- no cache: each of the N questions re-encodes the state,
                 i.e. N full forward passes ~= N encodes + the same total
                 readout work (readout is ~linear in N, so N single-question
                 readouts cost about one N-question batch).
    speedup             = uncached_ms / cached_ms, per case.
    speedup_vs_uncached = sum(uncached_ms) / sum(cached_ms), whole report;
                          the headline "cache reuse" number (README todo #3).

cached/uncached are derived from the two measured medians rather than timed
directly: they are exactly the algebra under test (1 encode vs N encodes),
and deriving them keeps the report to the four fields of the agreed schema.

MEASUREMENT HYGIENE (risk R1)
----------------------------------
Fixed seed for weights and state, `torch.set_num_threads(1)` (recorded in
`meta`), `model.eval()`, `torch.no_grad()`, `warmup` untimed runs before
`repeats` timed runs (benchmarks.stats.repeat), and every comparison taken on
the median (benchmarks.stats.summarize / Stats.median).

MODEL
-----
The same TINY shape as tests/conftest.py (0.57M params; milliseconds per
forward on CPU, which keeps the smoke gate honest) with only `max_state_len`
raised to T -- that makes the requested T the measured T exactly. The
"T > 2048 is pointless" note applies to the default JevConfig's 2048-cap;
here the cap follows T, but states beyond the default 2048 still have no
architectural meaning.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from benchmarks.fixtures import make_questions, make_state
from benchmarks.stats import repeat, summarize
from open_jev.main import Jev, JevConfig

# TINY hyperparameters, mirroring tests/conftest.py; only max_state_len varies
# (per-T, see module docstring).
_BENCH_TINY = dict(
    vocab_size=512,
    d_model=64,
    n_heads=4,
    d_ff=128,
    n_state_layers=2,
    n_question_layers=1,
    n_readout_layers=2,
    n_slots=4,
)

SMOKE_T = 64
SMOKE_N = [1, 2]


@dataclass(frozen=True)
class BenchConfig:
    """Everything one benchmark run needs; also what the CLI flags map to."""

    T: int = 256
    N: list[int] = field(default_factory=lambda: [1, 4, 16, 64])
    B: int = 1
    repeats: int = 7  # >= 5 per risk R1
    warmup: int = 2  # >= 2 per risk R1
    seed: int = 0
    smoke: bool = False
    json_path: str | None = None
    md_path: str | None = None

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if not self.N:
            raise ValueError("N must contain at least one value")
        if any(n < 1 for n in self.N):
            raise ValueError(f"every N must be >= 1, got {self.N}")
        if self.B < 1:
            raise ValueError(f"B must be >= 1, got {self.B}")
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")


def run(config: BenchConfig) -> dict:
    """Run the benchmark grid and return the report dict (schema in module docstring)."""
    if config.smoke:
        config = replace(config, T=SMOKE_T, N=list(SMOKE_N), B=1, repeats=1, warmup=0)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # risk R1: one thread for the whole run
    try:
        torch.manual_seed(config.seed)  # deterministic weights
        model = Jev(JevConfig(**_BENCH_TINY, max_state_len=config.T)).eval()
        state = make_state(config.T, config.seed)

        cases: list[dict] = []
        with torch.no_grad():
            for n in config.N:
                questions = make_questions(n)
                batch = [state] * config.B

                enc = summarize(
                    repeat(
                        lambda: model.encode_state(batch),
                        n=config.repeats,
                        warmup=config.warmup,
                    )
                )
                cache = model.encode_state(batch)
                measured_T = int(cache.hidden.shape[1])
                ro = summarize(
                    repeat(
                        lambda: model._readout(cache, questions),
                        n=config.repeats,
                        warmup=config.warmup,
                    )
                )

                cached_ms = enc.median + ro.median
                uncached_ms = n * enc.median + ro.median
                cases.append(
                    {
                        "T": measured_T,
                        "N": n,
                        "B": config.B,
                        "encode_ms": asdict(enc),
                        "readout_ms": asdict(ro),
                        "cached_ms": cached_ms,
                        "uncached_ms": uncached_ms,
                        "speedup": _ratio(uncached_ms, cached_ms),
                    }
                )

        total_cached = sum(c["cached_ms"] for c in cases)
        total_uncached = sum(c["uncached_ms"] for c in cases)

        meta = {
            "seed": config.seed,
            "repeats": config.repeats,
            "warmup": config.warmup,
            "torch": torch.__version__,
            "device": str(next(model.parameters()).device),
            "threads": torch.get_num_threads(),  # == 1 (risk R1)
            "T": config.T,
            "N": list(config.N),
            "B": config.B,
            "smoke": config.smoke,
            "model": {
                "vocab_size": model.cfg.vocab_size,
                "d_model": model.cfg.d_model,
                "n_state_layers": model.cfg.n_state_layers,
                "n_question_layers": model.cfg.n_question_layers,
                "n_readout_layers": model.cfg.n_readout_layers,
                "n_slots": model.cfg.n_slots,
                "max_state_len": model.cfg.max_state_len,
            },
        }
        return {
            "meta": meta,
            "cases": cases,
            "speedup_vs_uncached": _ratio(total_uncached, total_cached),
        }
    finally:
        torch.set_num_threads(previous_threads)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def render_markdown(report: dict) -> str:
    """Human-readable form of a report; always contains the word `speedup`."""
    meta = report["meta"]
    model = meta["model"]
    model_bits = ", ".join(f"{k}={v}" for k, v in model.items())
    lines = [
        "# open_jev stage-1 benchmark report",
        "",
        "- reproducibility: "
        f"seed={meta['seed']}, repeats={meta['repeats']}, warmup={meta['warmup']}, "
        f"threads={meta['threads']}, torch={meta['torch']}, device={meta['device']}, "
        f"smoke={meta['smoke']}",
        f"- model: {model_bits}",
        "- paths: cached_ms = 1 encode + 1 readout of all N questions; "
        "uncached_ms = N encodes + the same readout work "
        "(every question re-encodes the state)",
        "",
        "| T | N | B | encode_ms median | readout_ms median "
        "| cached_ms | uncached_ms | speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report["cases"]:
        lines.append(
            f"| {c['T']} | {c['N']} | {c['B']} "
            f"| {c['encode_ms']['median']:.3f} | {c['readout_ms']['median']:.3f} "
            f"| {c['cached_ms']:.3f} | {c['uncached_ms']:.3f} "
            f"| {c['speedup']:.2f}x |"
        )
    lines += [
        "",
        f"**speedup_vs_uncached = {report['speedup_vs_uncached']:.2f}x** "
        "(sum of uncached_ms / sum of cached_ms across all cases)",
        "",
    ]
    return "\n".join(lines)


def _parse_n(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"--N must be a comma-separated list of ints, got {raw!r}")
    return [int(p) for p in parts]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.cli",
        description="Benchmark state-cache reuse and multi-question scaling.",
    )
    parser.add_argument("--T", type=int, default=256, help="state length in tokens")
    parser.add_argument(
        "--N", default="1,4,16,64", help="comma-separated question counts"
    )
    parser.add_argument("--B", type=int, default=1, help="state batch size")
    parser.add_argument("--repeats", type=int, default=7, help="timed samples per case")
    parser.add_argument("--warmup", type=int, default=2, help="untimed warmup runs")
    parser.add_argument("--seed", type=int, default=0, help="seed for weights + state")
    parser.add_argument("--json", dest="json_path", help="write the JSON report here")
    parser.add_argument("--md", dest="md_path", help="write the markdown report here")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"minimal run: T={SMOKE_T}, N=1,2, B=1, repeats=1, warmup=0",
    )
    return parser


def _write(path: str, text: str) -> None:
    target = Path(path)
    if target.parent != Path():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the markdown report, write the requested files."""
    args = _build_parser().parse_args(argv)
    try:
        config = BenchConfig(
            T=args.T,
            N=_parse_n(args.N),
            B=args.B,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
            smoke=args.smoke,
            json_path=args.json_path,
            md_path=args.md_path,
        )
    except ValueError as exc:
        print(f"benchmarks.cli: {exc}", file=sys.stderr)
        return 2

    report = run(config)
    md = render_markdown(report)
    print(md)
    if config.json_path:
        _write(config.json_path, json.dumps(report, indent=2) + "\n")
    if config.md_path:
        _write(config.md_path, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
