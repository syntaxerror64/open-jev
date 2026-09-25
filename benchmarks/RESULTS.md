# open_jev stage-1 benchmark report

> Baseline captured 2026-09-25 on CPU-only torch 2.14.0+cpu (AMD Ryzen 9 5950X,
> threads=1, seed=0): **baseline BEFORE the trained tokenizer (stage 3)** —
> compare against this file after stage 3 lands.

- reproducibility: seed=0, repeats=7, warmup=2, threads=1, torch=2.14.0+cpu, device=cpu, smoke=False
- model: vocab_size=512, d_model=64, n_state_layers=2, n_question_layers=1, n_readout_layers=2, n_slots=4, max_state_len=256
- paths: cached_ms = 1 encode + 1 readout of all N questions; uncached_ms = N encodes + the same readout work (every question re-encodes the state)

| T | N | B | encode_ms median | readout_ms median | cached_ms | uncached_ms | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 1 | 8.954 | 2.653 | 11.608 | 11.608 | 1.00x |
| 256 | 4 | 1 | 9.645 | 3.949 | 13.594 | 42.528 | 3.13x |
| 256 | 16 | 1 | 9.510 | 14.478 | 23.988 | 166.642 | 6.95x |
| 256 | 64 | 1 | 7.009 | 60.937 | 67.946 | 509.532 | 7.50x |

**speedup_vs_uncached = 6.23x** (sum of uncached_ms / sum of cached_ms across all cases)
