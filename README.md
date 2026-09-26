# Open Jev

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/syntaxerror64/open-jev?sort=semver)](https://github.com/syntaxerror64/open-jev/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2%2B-ee4c2c)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/syntaxerror64/open-jev/pulls)

<div align="center">

<img src="https://img.shields.io/badge/-%F0%9F%87%AC%F0%9F%87%A7%20English-555555.svg" width="225" alt="🇬🇧 English">&nbsp;&nbsp;[<img src="https://img.shields.io/badge/-%F0%9F%87%B7%F0%9F%87%BA%20%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-555555.svg" width="243" alt="🇷🇺 Русский">](README.ru.md)&nbsp;&nbsp;[<img src="https://img.shields.io/badge/-%F0%9F%87%A8%F0%9F%87%B3%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-555555.svg" width="237" alt="🇨🇳 简体中文">](README.zh-CN.md)

</div>

An open-source, from-first-principles PyTorch reconstruction of the ideas
behind [TypeSafe AI's Jev / System One
models](https://typesafe.ai/blog/introducing-system-one-models-and-jev):
unstructured program state goes in, typed probabilistic decisions come out.

> [!IMPORTANT]
> This is an unofficial research implementation. The released checkpoint is
> trained by distillation — research-grade, not externally benchmarked. It is
> not the production Jev model, does not reproduce TypeSafe AI's training data
> or weights, and makes no claim of matching their published results.

## Contents

- [The idea](#the-idea)
- [Highlights](#highlights)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)
- [Released checkpoints](#released-checkpoints)
- [Evaluation](#evaluation)
- [Development](#development)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## The idea

Jev is presented as a *system one* model: unstructured program state goes in,
typed probabilistic decisions come out.

Instead of generating text one token at a time, this implementation encodes the
state once and answers every question through small typed readout heads.

```text
JSON-like state ──> bidirectional state encoder ──> shared state cache
                                                        │
                         question slots ── cross-attend ┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
                noul          choice         score
               p(true)      p(options)     p(levels)
```

Questions are isolated from one another and folded into the batch dimension.
They may attend to the shared state, but not to other questions. The result is
a single parallel forward pass with outputs constrained by their declared
types.

## Highlights

- **Typed outputs.** `Noul` returns p(true), `Choice` softmaxes only over the
  options declared at runtime, `Score` returns a distribution over rubric
  levels plus its expectation — a `Choice` answer can never be something you
  did not offer.
- **One encode, many questions.** A bidirectional transformer encodes the
  nested state once into a shared, cacheable representation; every question
  cross-attends into that cache through learned query slots. Cache reuse and
  multi-question scaling are benchmarked in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
- **Confidence you can reason about.** An evidential head models epistemic
  confidence separately from class probability, with a spread-based alternative
  selectable per configuration.
- **Soft-target training.** `RLCDLoss` combines soft-target NLL, Brier score,
  consistency, evidential, and ECE terms against full distributions — never
  one-hot labels — so disagreement and ambiguity stay visible.
- **Reproducible artifacts.** Every release ships a `.pt` checkpoint plus a
  `sha256` JSON manifest; the model config travels inside the artifact, so
  `JevConfig(**manifest["config"])` rebuilds the exact shape — no number has
  to be trusted by hand.
- **Tested by construction.** A 150-test suite checks the structural
  guarantees (typed answers, normalized distributions, question isolation,
  cache reuse) and gates every `git push`.

The implementation is intentionally small and legible. The default tokenizer
is a deterministic hash tokenizer; a trained BPE tokenizer (built with
`python -m scripts.train_bpe` from the repository's own text) implements the
same interface and can be passed to `Jev` as an opt-in.

## Installation

```bash
git clone https://github.com/syntaxerror64/open-jev.git
cd open-jev
python -m pip install "torch>=2.0"
```

Python 3.10 or newer is recommended.

## Quick start

```python
import torch

from open_jev.main import Choice, Jev, JevConfig, Noul, Score

model = Jev(
    JevConfig(
        vocab_size=4096,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_state_layers=3,
        n_readout_layers=4,
    )
).eval()

state = {
    "customer": {"tier": "enterprise", "tenure_months": 34},
    "message": "This is the third duplicate charge. Please fix it.",
    "policy": "Duplicate charges are refundable within 60 days.",
}

questions = [
    Noul("The customer is requesting a refund.", key="wants_refund"),
    Choice(
        "Which team should handle this?",
        options=["billing", "technical", "account"],
        key="route",
    ),
    Score(
        "How frustrated is the customer?",
        labels=["calm", "annoyed", "frustrated", "very frustrated"],
        key="frustration",
    ),
]

with torch.no_grad():
    answers = model([state], questions)[0]

for answer in answers:
    print(answer)
```

This snippet builds a fresh, randomly initialized model and never loads a
checkpoint, so the values above are not meaningful; the released checkpoint
(see [Released checkpoints](#released-checkpoints)) is trained by
distillation. The useful guarantee is structural: a `Choice` answer can only
be one of the options that were declared.

To run the complete demo, including one calibration-oriented training step:

```bash
python example.py
```

`forward.py` is the minimal example — inference without a training step.

## Training

`RLCDLoss` expects probability distributions rather than one-hot labels. This
allows disagreement and ambiguity to remain visible instead of forcing every
example toward certainty.

```python
from open_jev.main import RLCDLoss

targets = [
    torch.tensor([[0.08, 0.92]]),
    torch.tensor([[0.80, 0.05, 0.15]]),
    torch.tensor([[0.02, 0.10, 0.48, 0.40]]),
]

loss_fn = RLCDLoss()
loss = loss_fn(model, [state], questions, targets)
loss.backward()
```

For consistency training, pass semantically equivalent states through
`augmented_states` — for example, paraphrases or shuffled dictionary keys.

The training pipeline wraps this loss in a resumable loop with a teacher
layer supplying soft targets (never labels from the dataset):

```bash
python -m pipeline.train --config pipeline/examples/tiny.json --steps 30
python -m pipeline.train --config pipeline/examples/tiny.json --steps 10 \
       --resume runs/tiny/checkpoint.pt
```

## Released checkpoints

Trained runs are exported to the versioned v1 format (`state_dict`, config,
step, metrics, `sha256`) and published as GitHub Release assets:

| Release | Artifacts | Status |
|---|---|---|
| [v0.2.0](https://github.com/syntaxerror64/open-jev/releases/tag/v0.2.0) | [`jev-real.pt`](https://github.com/syntaxerror64/open-jev/releases/download/v0.2.0/jev-real.pt) + [`jev-real.json`](https://github.com/syntaxerror64/open-jev/releases/download/v0.2.0/jev-real.json) (6.3 MB) | Trained by distillation: Qwen2.5-0.5B teacher, 500 Alpaca rows, 200 steps |
| [v0.1.0](https://github.com/syntaxerror64/open-jev/releases/tag/v0.1.0) | `jev-tiny.pt` + `jev-tiny.json` (2.3 MB) | Random initialization — superseded |

Verify a download against its manifest, or load it straight from the release:

```bash
sha256sum jev-real.pt                       # must equal "sha256" in jev-real.json
JEV_CHECKPOINT_URL=<asset-url> pytest tests/test_checkpoint_release.py -v
```

To export and publish a run yourself:

```bash
python -m scripts.export_checkpoint --run runs/tiny --out dist/jev-tiny.pt
python scripts/release.py --tag v0.1.0 --asset dist/jev-tiny.pt --dry-run
```

`--dry-run` prints the release manifest as JSON without touching the network; the
real command runs `gh release create` when a trained artifact exists.
[MODEL_CARD.md](MODEL_CARD.md) describes what the artifact contains and its
current limitations.

## Evaluation

Holdout comparison of the random initialization against the released v0.2.0
checkpoint (100 rows, seed 0, `n_bins` 10 — full details in
[eval/COMPARISON.md](eval/COMPARISON.md)):

| Metric | Random | Checkpoint (step 200) |
|---|---|---|
| ECE | 0.3073 | 0.1532 |
| Brier | 0.3133 | 0.1509 |
| NLL | 0.8850 | 0.6695 |
| Consistency (sym. KL, lower is better) | 0.006497 | 0.0001031 |

**What these numbers mean.** The holdout has no ground-truth labels: ECE,
Brier, and NLL are computed against the distilled targets of the teacher, so
they measure *agreement with the teacher on this holdout* — not correctness,
and not an external benchmark. See [MODEL_CARD.md](MODEL_CARD.md) for the
exact wording and provenance.

## Development

```bash
python -m venv .venv
.venv/bin/pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The suite lives in `tests/` and checks structural guarantees only (typed
answers, normalized distributions, question isolation, cache reuse) —
properties that hold for any weights, trained or random. Tests marked `slow`
(the `example.py` demo) are skipped by the push gate below; run them with a
plain `pytest`.

Pushes are gated on that suite: enable the bundled hook once per clone and
`git push` refuses to upload anything unless the tests pass.

```bash
git config core.hooksPath .githooks
```

For an emergency push use `git push --no-verify`.

## Documentation

- [MODEL_CARD.md](MODEL_CARD.md) — what the published artifact contains,
  metrics provenance, and limitations
- [eval/COMPARISON.md](eval/COMPARISON.md) — random vs. checkpoint comparison
  on the holdout
- [benchmarks/RESULTS.md](benchmarks/RESULTS.md) — state-cache reuse and
  multi-question scaling measurements
- `docs/ARCHITECTURE.md` — the full reasoning and trade-offs behind the
  reconstruction (local development document)

## Limitations

- **Typed output prevents schema violations, not wrong answers.** Calibration
  must be re-measured under distribution shift before any unattended use.
- **Small, distillation-trained model.** 1.57M parameters, 200 steps, 500
  training instructions; "trained" ≠ "good" — the metrics above are
  teacher-consistency, not correctness. No external benchmarks.
- **Research scope.** An independent reconstruction from public material — it
  does not reproduce TypeSafe AI's data, weights, or reported results.
- **CPU-only, English-only.** Developed and tested with `torch` CPU builds on
  English data; other languages and accelerators are untested.

## Contributing

Issues and pull requests are welcome. Before pushing, run
`.venv/bin/python -m pytest` — the bundled pre-push hook
(`git config core.hooksPath .githooks`, once per clone) enforces the same
suite on every push.

## License

Licensed under the [Apache License 2.0](LICENSE).

In short: you can do essentially whatever you want with this code — use it,
modify it, redistribute it, sell it, run it privately — for commercial or
personal purposes. The only conditions are keeping the license and its
notices and marking the files you changed. There is no warranty.

## Acknowledgements

- Appreciation to [TypeSafe AI](https://typesafe.ai) for introducing
  [System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  and for sharing enough public detail to inspire independent experimentation.
