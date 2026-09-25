# Model Card

**Open Jev** is an unofficial, from-first-principles PyTorch reconstruction of
the ideas behind [TypeSafe AI's Jev / System One
models](https://typesafe.ai/blog/introducing-system-one-models-and-jev). This
card describes what an exported artifact actually contains, what it can and
cannot be used for, and how to verify it. The repository overview lives in
[README.md](README.md); the design reasoning and tradeoffs live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Model details

| | |
|---|---|
| Name | open-jev |
| Version | 0.1.0 (`__version__` in [`open_jev/version.py`](open_jev/version.py)) |
| Architecture | System-One RLCD reconstruction (see below) |
| Checkpoint format | open-jev v1, `format_version = 1` ([`open_jev/checkpoint.py`](open_jev/checkpoint.py)) |
| Training state | Random weights until stage 4 (pretraining pipeline) lands |
| Framework | PyTorch, tested with `2.14.0+cpu` on CPU |
| License | Apache 2.0 ([LICENSE](LICENSE)) |

## Architecture

The model treats unstructured program state as input and produces typed,
probabilistic answers instead of free-form text. A bidirectional transformer
encodes the nested state once into a shared, cacheable representation;
questions are isolated from one another, folded into the batch dimension, and
cross-attend into that cache through learned query slots. Three typed heads
produce the outputs: `Noul` returns p(true), `Choice` softmaxes only over the
options declared at runtime, and `Score` applies ordinal thresholds to return
a distribution over rubric levels plus its expectation. An evidential head
models epistemic confidence separately from class probability, and `RLCDLoss`
combines soft-target NLL, Brier score, consistency, evidential, and ECE terms.
The full diagram and the list of architectural claims are in the
[Architecture section of the README](README.md#architecture) and the reasoning
behind each choice is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Configuration

Hyperparameters live in the frozen `JevConfig` dataclass
([`open_jev/main.py`](open_jev/main.py)): vocabulary size, `d_model`,
attention heads, feed-forward width, state- and question-encoder depth, readout
depth, query slots, and length limits. The demo configuration used by
[`example.py`](example.py) and [`forward.py`](forward.py) is
`vocab_size=4096, d_model=128, n_heads=4, d_ff=512, n_state_layers=3,
n_readout_layers=4, n_slots=8`; the CPU-oriented TINY shape used by tests and
the training pipeline is `vocab_size=512, d_model=64, ...`
([`tests/conftest.py`](tests/conftest.py), [`pipeline/config.py`](pipeline/config.py)).

Every exported artifact carries its own configuration: the `config` key inside
the `.pt` file and the `config` field of the sidecar `.json` manifest. Rebuild
the exact model shape with `JevConfig(**manifest["config"])` — no number from
this card has to be trusted by hand.

## Metrics

Metrics come from the stage-2 evaluation harness ([`eval/`](eval/)), which
reports ECE, Brier score, NLL, and consistency against soft targets on a
synthetic dataset, with `n_bins=10` pinned for comparability.

**Honest status:** the weights are still random, so the committed numbers are
evaluation *mechanics* only and carry no quality meaning. The harness says so
itself: [`eval/out/report.json`](eval/out/report.json) records
`"model": "random"` and an explicit warning alongside
`ece ≈ 0.127`, `brier ≈ 0.307`, `nll ≈ 0.905`, `consistency ≈ 1.1e-4`
(200 rows, seed 0). They demonstrate that the pipeline computes and reports
correctly, not that the model predicts well.

| metric | trained value | source |
|---|---|---|
| ECE | — placeholder: no trained run yet | `runs/tiny/metrics.json` (pending stage 4) |
| Brier | — placeholder: no trained run yet | `runs/tiny/metrics.json` (pending stage 4) |
| NLL | — placeholder: no trained run yet | `runs/tiny/metrics.json` (pending stage 4) |
| Consistency | — placeholder: no trained run yet | `runs/tiny/metrics.json` (pending stage 4) |

This tree contains no `runs/` directory at all: there are no trained weights
to publish yet, and any artifact exported today reproduces a seeded random
initialization. The table above is filled in from `runs/tiny/metrics.json`
(and re-checked through the harness) once stage 4 produces trained weights;
until then, treat every number in this card as a placeholder.

## Artifact and tensor data

- **Tokenizer.** The default is the deterministic hash tokenizer; the trained
  BPE tokenizer (built with `python -m scripts.train_bpe --vocab-size 4096
  --out cache/tokenizer.json` from repository text only) implements the same
  interface. Vocab size for the demos and the tokenizer artifact is **4096**
  ids (`PAD = 0`, ids in `[1, vocab_size)`); `JevConfig`'s default
  `vocab_size` is 32000 for the untuned hash tokenizer.
- **Tensors.** Weights are `float32` on CPU; the artifacts were produced with
  **torch `2.14.0+cpu`**, which is recorded in each file's `torch_version`
  field (risk R1: `torch.save` is not stable across torch versions, so load
  with a compatible release).
- **Contents (format v1).** `state_dict`, `config`, `step`, `metrics`,
  `format_version`, `torch_version`, `seed`, `sha256`. The `seed` field makes
  a random initialization reproducible; the `sha256` field follows the
  double-write scheme described in
  [`open_jev/checkpoint.py`](open_jev/checkpoint.py) and the digest of the
  published *file bytes* lives in the manifest.
- **Verification.** The manifest next to every artifact carries `sha256` and
  `size_bytes`:

  ```bash
  python -m scripts.export_checkpoint --run runs/tiny --out dist/jev-tiny.pt
  python -c "import json; print(json.load(open('dist/jev-tiny.json'))['sha256'])"
  sha256sum dist/jev-tiny.pt | cut -d' ' -f1   # must match the manifest
  ```

- **Loading.**

  ```python
  from open_jev.checkpoint import load
  from open_jev.main import Jev

  model = load("dist/jev-tiny.pt", model_factory=lambda cfg: Jev(cfg).eval())
  ```

## Intended use

Research and education: studying the architecture, exercising the training,
evaluation, and checkpoint tooling, and as a starting point for further
experiments on top of this codebase. The structural guarantees the suite
checks (typed answers that cannot violate their declared schema, normalized
distributions, isolated questions, cache reuse) hold regardless of weights.

## Limitations and not-intended use

- **Random weights.** Until stage 4 lands, every artifact is a random
  initialization. Predictions are not meaningful measurements of anything.
- **Research scope.** Even with trained weights, this is an independent
  reconstruction from public material — it does not reproduce TypeSafe AI's
  data, weights, or reported results, and makes no claim to match them.
- **CPU-only.** Development and tests target CPU (torch `2.14.0+cpu`); no GPU
  or distributed training path is provided or validated.
- **English only.** The corpora, synthetic data, and evaluation sets are
  English; behavior on other languages is untested.
- **No correctness guarantee.** Typed output prevents schema violations, not
  wrong answers; calibration must be re-measured under distribution shift
  before any unattended use.
- **Not intended for** production decisions, safety-critical or
  person-affecting systems, benchmarking against the real Jev, or any form of
  deployment where a wrong-but-well-typed answer would cause harm.

## License

Apache 2.0 — see [LICENSE](LICENSE). Attribution for the original ideas goes
to TypeSafe AI's public post on
[System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
as noted in the README acknowledgements.
