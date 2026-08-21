from __future__ import annotations

import numpy as np
import torch

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from scripts.freeze_shared_v8_deployment_v1 import (
    FROZEN_CANDIDATE_ID,
    fit_full_dataset,
)


def _state(model):
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }


def test_full_dataset_fit_is_shared_complete_and_deterministic(c):
    dataset = _dataset()
    first = fit_full_dataset(dataset, epochs=1, seed=0, device="cpu")
    second = fit_full_dataset(dataset, epochs=1, seed=0, device="cpu")
    c.eq(FROZEN_CANDIDATE_ID, "RSR8-001")
    c.eq(first.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.eq(first.training_examples, 12)
    c.eq(first.epochs, 1)
    c.true(np.isfinite(first.final_loss) and first.final_loss > 0.0)
    c.eq(first.scaler_mean.shape, (110,))
    c.eq(first.scaler_scale.shape, (110,))
    c.true(np.all(first.scaler_scale > 0.0))
    for name, value in _state(first.model).items():
        c.true(torch.equal(torch.from_numpy(value), torch.from_numpy(_state(second.model)[name])))


def test_fit_rejects_nonfrozen_training_configuration(c):
    dataset = _dataset()
    c.raises(lambda: fit_full_dataset(dataset, epochs=0, seed=0, device="cpu"), ValueError)
    c.raises(lambda: fit_full_dataset(dataset, epochs=1, seed=1, device="cpu"), ValueError)
    c.raises(lambda: fit_full_dataset(dataset, epochs=1, seed=0, device="meta"), ValueError)


if __name__ == "__main__":
    run_all("test_freeze_shared_v8_deployment_v1", dict(globals()))
