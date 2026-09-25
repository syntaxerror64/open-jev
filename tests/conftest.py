"""Shared fixtures: one tiny, seeded model reused across the whole suite."""

from __future__ import annotations

import pytest
import torch

from open_jev.main import Choice, Jev, JevConfig, Noul, Score

# Small enough that a full forward pass is milliseconds, big enough that every
# head has room to produce non-degenerate outputs.
TINY = dict(
    vocab_size=512,
    d_model=64,
    n_heads=4,
    d_ff=128,
    n_state_layers=2,
    n_question_layers=1,
    n_readout_layers=2,
    n_slots=4,
    max_state_len=128,
)


@pytest.fixture(scope="session")
def cfg() -> JevConfig:
    return JevConfig(**TINY)


@pytest.fixture(scope="session")
def model(cfg: JevConfig) -> Jev:
    torch.manual_seed(0)
    return Jev(cfg).eval()


@pytest.fixture(scope="session")
def state() -> dict:
    return {
        "customer": {"tier": "enterprise", "tenure_months": 34},
        "message": "This is the third duplicate charge. Please fix it.",
        "transactions": [
            {"id": "t_1", "amount": 89.0, "status": "captured"},
            {"id": "t_2", "amount": 89.0, "status": "captured"},
        ],
        "policy": "Duplicate charges are refundable within 60 days.",
    }


@pytest.fixture(scope="session")
def questions() -> list:
    return [
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
