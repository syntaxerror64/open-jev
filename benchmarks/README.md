# Benchmarks

Reproducible CPU timings for the two architectural claims of this repository
(see the docstring of `open_jev/main.py`):

1. **State-cache reuse.** Encoding the state is paid once; `N` questions then
   read from the shared cache. `N` cached readouts are far cheaper than `N`
   full forwards that each re-encode the state.
2. **Multi-question scaling.** As `N` grows, `encode_state` cost stays constant
   while `_readout` grows roughly linearly — questions are folded into the
   batch dimension and cross-attend into one shared state representation.

The committed snapshot lives in [`RESULTS.md`](RESULTS.md). It is the baseline
for the stage-3 before/after comparison (trained tokenizer).

## Reproduce

```bash
.venv/bin/python -m benchmarks.cli --T 256 --N 1,4,16,64 --repeats 7 --warmup 2 --md benchmarks/RESULTS.md
```

For a quick sanity check (under 5 seconds):

```bash
.venv/bin/python -m benchmarks.cli --smoke --json /tmp/bench.json
```

The run is pinned down as follows:

- `seed=0` — weights and the synthetic state are deterministic.
- `warmup=2` — untimed warmup runs before sampling, so the first timed sample
  does not pay one-off costs.
- `repeats=7` — timed samples per case; the **median** (not the mean) is
  reported, which is robust against CPU timing noise on a shared machine.
- `torch.set_num_threads(1)` — single-threaded torch, recorded in the report
  `meta` (`threads`), so timings do not drift with machine load.
- `model.eval()` — disables the `0.1` dropout from `JevConfig`, otherwise the
  variance of the timings would be inflated.

## Reading the table

| Column | Meaning |
|---|---|
| `encode_ms` | median time of `Jev.encode_state` — the `O(T^2)` state encoding, paid once per cached path |
| `readout_ms` | median time of `Jev._readout` — all `N` questions answered from the cache, the `O(M·T·N)` part |
| `cached_ms` | one encode + one readout of all `N` questions: `encode_ms + readout_ms` |
| `uncached_ms` | the same readout work, but every question pays its own state encode: `N * encode_ms + readout_ms` |
| `speedup` | `uncached_ms / cached_ms` for that case — the cache-reuse win at this `N` |

`**speedup_vs_uncached**` at the bottom of the report is the same ratio taken
over the sums of all cases: total uncached work divided by total cached work.
It approaches `1.00x` at `N=1` (there is nothing to reuse yet) and grows with
`N`, which is exactly claim 1. Claim 2 is read off the `encode_ms` column:
it stays flat as `N` goes from 1 to 64 while `readout_ms` climbs.

## Caveat: the cache win depends on T and N (risk R3)

The advantage is algebraic: `O(T^2)` for one encode versus `O(M·T·N)` for `N`
re-encodes. It therefore holds for **large `T` and small `N`**. At small `T`
the constant overheads (cross-attention setup, Python call overhead) can
swamp the saved work and the win can shrink or vanish — `speedup ≈ 1.00x` at
small `T` is not a regression, it is the crossover point of the two costs.
The committed baseline uses `T=256`, which is comfortably on the winning side.

## Artifacts

- `benchmarks/RESULTS.md` — the committed baseline; regenerate it with the
  command above and commit the diff when timings legitimately change.
- `benchmarks/out/` — temporary JSON reports (`--json out/bench.json`); the
  directory is gitignored and safe to delete at any time.

Timings are CPU-only (`torch=…+cpu`, `device=cpu`); absolute milliseconds
vary from machine to machine, so compare ratios (`speedup`) across runs, not
raw times.
