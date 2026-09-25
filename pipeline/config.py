"""Train-loop configuration: one stdlib dataclass, one JSON file, no deps.

Why JSON and nothing else (Декомпозиция, agent А: "сериализация в JSON —
без новых зависимостей"): the repo has no YAML/TOML parser anywhere (
"Текущее состояние": `ls *.yaml *.toml` -> NONE), and a config that must be
hand-edited for `pipeline/examples/*.json`, diffed in review, and round-tripped
by `--config`/`--resume` does not need a schema library -- stdlib ``json``
covers it. Keeping every import standard-library also means the config module
loads before torch does, so a broken config fails fast on its own terms.

Field choices follow the loop contract: steps/lr/seed (the
reproducibility triple, risk R3), batch_size and eval_every (periodic
evaluation through the stage-2 harness), out_dir (checkpoint.pt/metrics.json),
AdamW + schedule + clipping knobs (weight_decay/warmup_steps/grad_clip),
teacher_repeats (10 samples splitting 7/3 -> 0.7, open_jev/main.py:810-812),
data_path (the stage-4 JSONL, risk O1), and the TINY model shape lifted from
tests/conftest.py:12-22 so the default run stays milliseconds-per-step on CPU
(risk R2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

__all__ = ["TrainConfig"]

#: Fields of TrainConfig that map 1:1 onto open_jev.main.JevConfig; everything
#: else in the config belongs to the training loop, not to the model. Keeping
#: the split explicit lets callers write ``JevConfig(**cfg.model_kwargs())``
#: instead of filtering ``asdict(cfg)`` by hand.
MODEL_FIELDS: tuple[str, ...] = (
    "vocab_size",
    "d_model",
    "n_heads",
    "d_ff",
    "n_state_layers",
    "n_question_layers",
    "n_readout_layers",
    "n_slots",
    "max_state_len",
)

#: Sub-keys accepted inside ``TrainConfig.teacher`` (stage 7, Шаг 1).
#: Anything else in the block is a typo and fails loudly -- same reasoning as
#: ``load()`` rejecting unknown top-level fields: a dropped key would make the
#: run silently use different teacher settings than the file claims.
_TEACHER_KEYS: frozenset[str] = frozenset(
    {
        "kind",
        "model_id",
        "mode",
        "max_new_tokens",
        "temperature",
        "seed",
        "repeats",
        "device",
        "cache_dir",
    }
)

#: Backends ``train.py`` can build from the block; "precomputed" reads row
#: targets verbatim without calling any teacher.
_VALID_KINDS: frozenset[str] = frozenset({"stub", "api", "hf_local", "precomputed"})

#: HFLocalTeacher modes: "logits" (one forward -> first-token softmax) and
#: "sample" (the votes path).
_VALID_MODES: frozenset[str] = frozenset({"logits", "sample"})


@dataclass
class TrainConfig:
    """Everything one training run needs; ``save``/``load`` speak JSON only.

    Defaults are the TINY CPU config from tests/conftest.py:12-22 plus a
    30-step loop, matching the success criteria (`--steps 30`, loss
    falls, artifacts land in ``runs/tiny/``). The dataclass stays mutable on
    purpose: a CLI such as ``python -m pipeline.train --steps N`` can override
    a loaded config in place before the run starts, and ``__eq__`` (needed by
    the config-roundtrip test) is the plain dataclass one.
    """

    # --- loop ---
    steps: int = 30
    lr: float = 1e-4
    seed: int = 0
    batch_size: int = 2
    eval_every: int = 10
    out_dir: str = "runs/tiny"

    # --- optimization: AdamW + LR schedule + clipping ---
    weight_decay: float = 0.01
    warmup_steps: int = 0
    grad_clip: float = 1.0  # <= 0 disables clip_grad_norm_

    # --- data / teacher ---
    data_path: str = "data/train/synth.jsonl"
    teacher_repeats: int = 10
    #: Which teacher feeds the training loop, as a JSON object:
    #: ``{"kind": "stub"|"api"|"hf_local"|"precomputed", ...}``. Configs
    #: written before this field existed simply omit it and get the stub, so
    #: old runs stay byte-for-byte identical (stage 7, Шаг 1).
    teacher: dict = field(default_factory=lambda: {"kind": "stub"})

    # --- TINY model, mirroring tests/conftest.py:12-22 ---
    vocab_size: int = 512
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_state_layers: int = 2
    n_question_layers: int = 1
    n_readout_layers: int = 2
    n_slots: int = 4
    max_state_len: int = 128

    def __post_init__(self) -> None:
        """Reject values that would break the loop halfway through a run."""
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.teacher_repeats < 1:
            raise ValueError(
                f"teacher_repeats must be >= 1, got {self.teacher_repeats}"
            )
        self._validate_teacher()

    def _validate_teacher(self) -> None:
        """Check the ``teacher`` block: known sub-keys, known ``kind``/``mode``.

        Runs from ``__post_init__``, i.e. for direct construction *and* for
        ``load()`` (which ends in ``cls(**data)``), so a typo'd JSON sub-key
        raises ``ValueError`` here rather than being silently dropped. The
        top-level unknown-key filter in ``load`` never sees inside the block.
        """
        if not isinstance(self.teacher, dict):
            raise ValueError(
                f"teacher must be a JSON object, got {type(self.teacher).__name__}"
            )
        unknown = sorted(set(self.teacher) - _TEACHER_KEYS)
        if unknown:
            raise ValueError(f"unknown teacher config key(s): {unknown}")
        kind = self.teacher.get("kind", "stub")
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown teacher kind: {kind!r} (expected one of "
                f"{sorted(_VALID_KINDS)})"
            )
        mode = self.teacher.get("mode")
        if mode is not None and mode not in _VALID_MODES:
            raise ValueError(
                f"unknown teacher mode: {mode!r} (expected one of "
                f"{sorted(_VALID_MODES)})"
            )

    def model_kwargs(self) -> dict[str, int]:
        """The JevConfig-shaped subset: ``JevConfig(**cfg.model_kwargs())``."""
        return {name: getattr(self, name) for name in MODEL_FIELDS}

    def save(self, path: str | Path) -> None:
        """Write the config as indented, key-sorted JSON (creates parents)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        """Read a config written by :meth:`save`; unknown keys are an error.

        Silently ignoring an extra key would let a typo'd field look like it
        took effect, so the load fails loudly instead -- same reasoning as
        eval/dataset.py failing on a non-object line rather than coercing it.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a JSON object, got {type(data).__name__}"
            )
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                f"{path}: unknown TrainConfig field(s): {sorted(unknown)}"
            )
        return cls(**data)
