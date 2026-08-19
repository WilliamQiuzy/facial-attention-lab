from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.response_statistic_shared_router_v7 import (
    ResponseStatisticSharedRouterV7, candidate_registry_v7,
)
from src.preprocessing.shared_response_statistics_v7 import dense_response_statistics_v7


def test_response_statistics_are_view_commutative_and_velocity_uses_seconds(c):
    original = np.zeros((1, 1, 32, 478, 3), dtype=np.float32)
    mirrored = np.zeros_like(original)
    original[0, 0, :, 0, 0] = np.arange(32)
    mirrored[0, 0, :, 0, 0] = 2 * np.arange(32)
    times = np.arange(32, dtype=np.float64)[None, None, :] / 10.0
    available = np.ones((1, 1), dtype=bool)
    first = dense_response_statistics_v7(original, mirrored, times, available)
    second = dense_response_statistics_v7(mirrored, original, times, available)
    c.true(np.array_equal(first, second))
    c.eq(first.shape, (1, 1, 478 * 3 * 5 * 2))
    c.true(np.isclose(first[0, 0, 4], 15.0))


def test_registry_and_shared_model_contract(c):
    registry = candidate_registry_v7()
    c.eq(len(registry), 4)
    c.eq({item.pca_dim for item in registry}, {64, 128})
    c.eq({item.head_mode for item in registry}, {"linear", "small_mlp"})
    model = ResponseStatisticSharedRouterV7(registry[0])
    batch, actions = 6, 3
    clinical = torch.randn(batch, actions, 110)
    dense = torch.randn(batch, actions, registry[0].pca_dim)
    available = torch.ones(batch, actions, dtype=torch.bool)
    mask = torch.ones(batch, actions, dtype=torch.bool)
    codes = torch.arange(actions)[None].repeat(batch, 1)
    tokens = model.shared_action_tokens(clinical, clinical, dense, available, mask, codes)
    c.eq(tuple(tokens.shape), (batch, actions, 64))
    tasks = torch.tensor([0, 1, 2, 0, 1, 2])
    c.eq(tuple(model.routed_logits(tokens, mask, tasks).shape), (batch,))


if __name__ == "__main__":
    run_all("test_response_statistic_shared_router_v7", dict(globals()))
