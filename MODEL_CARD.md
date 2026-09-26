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
| Version | **Release v0.2.0** — the wording used by this card and the GitHub release tag, deliberately separate from `__version__` |
| Package version | `0.1.0` (`__version__` in [`open_jev/version.py`](open_jev/version.py)) — unchanged for this release (integrator decision R7; the release tag, not the package version, identifies the artifact) |
| Architecture | System-One RLCD reconstruction (see below) |
| Checkpoint format | open-jev v1, `format_version = 1` ([`open_jev/checkpoint.py`](open_jev/checkpoint.py)) |
| Training state | **Trained by distillation** (stages 06–08, see [Training data and teacher](#training-data-and-teacher)) |
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
The full diagram is in [The idea section of the README](README.md#the-idea),
the list of architectural claims is in
[Highlights](README.md#highlights), and the reasoning behind each choice is in
`docs/ARCHITECTURE.md` (local development document).

## Configuration

Hyperparameters live in the frozen `JevConfig` dataclass
([`open_jev/main.py`](open_jev/main.py)): vocabulary size, `d_model`,
attention heads, feed-forward width, state- and question-encoder depth, readout
depth, query slots, and length limits. The released v0.2.0 artifact uses
`vocab_size=512, d_model=128, n_heads=4, d_ff=256, n_state_layers=2,
n_question_layers=1, n_readout_layers=2, n_slots=4, max_state_len=128` —
1,571,349 parameters — as pinned in
[`pipeline/examples/real.json`](pipeline/examples/real.json). For contrast, the
demo configuration used by [`example.py`](example.py) and
[`forward.py`](forward.py) is `vocab_size=4096, d_model=128, n_heads=4,
d_ff=512, n_state_layers=3, n_readout_layers=4, n_slots=8`, and the narrower
CPU-oriented TINY shape used by tests is `vocab_size=512, d_model=64, ...`
([`tests/conftest.py`](tests/conftest.py), [`pipeline/config.py`](pipeline/config.py)).

Every exported artifact carries its own configuration: the `config` key inside
the `.pt` file and the `config` field of the sidecar `.json` manifest. Rebuild
the exact model shape with `JevConfig(**manifest["config"])` — no number from
this card has to be trusted by hand.

## Training data and teacher

The v0.2.0 weights are the product of a supervised distillation run (stages
06–08):

- **Corpus.** `alpaca_data.json` (Alpaca-style instructions) → deterministic
  sample of 600 rows with seed 0 → **500 train / 100 holdout** rows. Each row
  is a state `{state: {instruction, input}}` paired with **3 template
  questions**: `choice` (K=2), `noul` (K=2), `score` (K=3).
- **Teacher.** `python -m pipeline.distill` with
  `{"kind": "hf_local", "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "mode": "logits"}`, seed 0 → `data/real/train.targets.jsonl` (500 rows of
  soft targets; `data/` is gitignored — the file exists locally only).
- **Training.** [`pipeline/examples/real.json`](pipeline/examples/real.json):
  200 steps, batch 4, `eval_every` 50, lr 0.001, seed 0, `d_model` 128,
  consuming the precomputed teacher targets (`"teacher": {"kind":
  "precomputed"}`) → `runs/real/` (gitignored — local run directory).

There is no ground-truth label anywhere in this pipeline: the student learns
the teacher's logits, not verified answers.

## Metrics

Numbers below come from the distillation run (`runs/real/metrics.json` —
gitignored, present locally) and from the random-vs-checkpoint comparison
committed as [eval/COMPARISON.md](eval/COMPARISON.md). Both use the stage-2
evaluation harness ([`eval/`](eval/)), which reports ECE, Brier score, NLL, and
consistency against soft targets with `n_bins=10` pinned for comparability.

**What these numbers measure.** The holdout has no ground-truth labels: ECE /
Brier / NLL are computed against the distilled targets of the teacher, so they
measure **agreement with the Qwen2.5-0.5B teacher's distilled targets on the
holdout — not ground truth**. In the wording fixed by the evaluation stage:

> ECE / Brier / NLL считаются против дистиллированных таргетов учителя, а не
> против ground truth: это ***teacher-consistency*** (согласованность студента
> с HFLocalTeacher), **не абсолютная правда** -- честнее uniform-заглушек, но и
> это не «качество модели».

### Training run (`runs/real/metrics.json`, local)

Training history: loss **1.0493 → 0.7880** over 200 steps (batch 4,
lr 0.001, `eval_every` 50, seed 0).

| block | brier | ece | nll | evidential | consistency | total |
|---|---|---|---|---|---|---|
| `final` @ step 200 | 0.2246 | 0.2370 | 0.6476 | 0.02195 | 0.0 | 0.7880 |
| `eval` @ step 200 | 0.09192 | 0.1184 | 0.7038 | — | — | — |

### Random vs checkpoint (holdout)

From [eval/COMPARISON.md](eval/COMPARISON.md) — holdout 100 rows, seed 0,
`n_bins` 10, step 200:

| metric | random | checkpoint |
|---|---|---|
| ece | 0.3073 | 0.1532 |
| brier | 0.3133 | 0.1509 |
| nll | 0.8850 | 0.6695 |
| consistency | 0.006497 | 0.0001031 |
| rows | 100 | 100 |
| questions | 3 | 3 |
| seed | 0 | 0 |
| n_bins | 10 | 10 |
| step | - | 200 |
| sha256 | - | 7be05cc3d23be4598e7d41773229d8c8ad32767a504b83ddfff902e908c720aa |

The `sha256` row is the evaluated checkpoint file `runs/real/checkpoint.pt`
(gitignored). The published release asset is a re-export of that checkpoint
and its file bytes hash differently (torch serialization is not byte-stable
across saves) — see the manifest digest in the next section.

## Artifact and tensor data

The released **v0.2.0** artifact is `dist/jev-real.pt` + `dist/jev-real.json`
(both gitignored; published as GitHub release assets):

- **sha256** (digest of the published file bytes, from the manifest; this is
  the value the `test_model_card_matches_release_v020` gate pins to this card):

  ```
  0306c8d473fbeae863f0134f0cd3ca616577a4a139cdea73c9e1e30dac101e8a
  ```

  It equals `sha256sum dist/jev-real.pt`.
- **size_bytes** 6304285 (~6.3 MB), **format_version** 1,
  **torch_version** `2.14.0+cpu`, **step** 200, **source**
  `runs/real/checkpoint.pt`.
- **config** (manifest field): `d_model 128, d_ff 256, n_heads 4,
  vocab_size 512, n_state_layers 2, n_question_layers 1, n_readout_layers 2,
  n_slots 4, max_state_len 128` — 1,571,349 parameters.
- **Local smoke (offline, verified):** `load('dist/jev-real.pt')` returns
  finite logits that sum to 1: `[[0.677, 0.323]]`.
- **Provenance in git:** the source checkpoint (`runs/`) does not enter git —
  the only public trace of the checkpoint is the release asset and the sha256
  above. On a clean clone `runs/real` does not exist.
- **v0.1.0 history:** the previous release shipped `jev-tiny.pt`, 2301415
  bytes, sha256
  `953f957abcee3efdd7b7dc5ee97bcaa1ab76ab3bbbd4d7e8860dcac2aa811858`
  (random initialization, superseded by v0.2.0).

- **Tokenizer.** The default is the deterministic hash tokenizer; the trained
  BPE tokenizer (built with `python -m scripts.train_bpe --vocab-size 4096
  --out cache/tokenizer.json` from repository text only) implements the same
  interface. Vocab size for the demos and the tokenizer artifact is **4096**
  ids (`PAD = 0`, ids in `[1, vocab_size)`); `JevConfig`'s default
  `vocab_size` is 32000 for the untuned hash tokenizer. The released v0.2.0
  artifact uses the deterministic hash tokenizer with `vocab_size=512`.
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
  python -m scripts.export_checkpoint --run runs/real --out dist/jev-real.pt
  python -c "import json; print(json.load(open('dist/jev-real.json'))['sha256'])"
  sha256sum dist/jev-real.pt | cut -d' ' -f1   # must match the manifest
  ```

- **Loading.**

  ```python
  from open_jev.checkpoint import load
  from open_jev.main import Jev

  model = load("dist/jev-real.pt", model_factory=lambda cfg: Jev(cfg).eval())
  ```

## Intended use

Research and education: studying the architecture, exercising the training,
evaluation, and checkpoint tooling, and as a starting point for further
experiments on top of this codebase. The structural guarantees the suite
checks (typed answers that cannot violate their declared schema, normalized
distributions, isolated questions, cache reuse) hold regardless of weights.

## Limitations and not-intended use

- **Small, distillation-trained model.** 1,571,349 parameters trained for 200
  steps by distillation from a 0.5B teacher (`Qwen/Qwen2.5-0.5B-Instruct`) on
  **500 training instructions**. The teacher's distilled targets are not
  ground truth, so **"trained" ≠ "good"**: the metrics above are
  teacher-consistency, not correctness.
- **No external benchmarks.** Nothing here is measured against external
  benchmarks or ground-truth labels; 500 instructions is a toy corpus, and
  calibration is unswept under distribution shift — re-measure before any
  unattended use.
- **Research scope.** This is an independent reconstruction from public
  material — it does not reproduce TypeSafe AI's data, weights, or reported
  results, and makes no claim to match them.
- **CPU-only.** Development and tests target CPU (torch `2.14.0+cpu`); no GPU
  or distributed training path is provided or validated.
- **English only.** The corpora, synthetic data, and evaluation sets are
  English; behavior on other languages is untested.
- **No correctness guarantee.** Typed output prevents schema violations, not
  wrong answers.
- **Not intended for** production decisions, safety-critical or
  person-affecting systems, benchmarking against the real Jev, or any form of
  deployment where a wrong-but-well-typed answer would cause harm.

## License

Apache 2.0 — see [LICENSE](LICENSE). Attribution for the original ideas goes
to TypeSafe AI's public post on
[System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
as noted in the README acknowledgements.
