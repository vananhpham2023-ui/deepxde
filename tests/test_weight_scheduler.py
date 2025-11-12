import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

from examples.pinn_inverse.force_estimation import RESIDUAL_COMPONENT_NAMES
from examples.pinn_inverse.weight_scheduler import AdaptiveWeightScheduler


class _DummyTrainState:
    def __init__(self):
        self.step = 0
        self.loss_train = []


class _DummyModel:
    def __init__(self, loss_weights):
        self.loss_weights = loss_weights
        self.train_state = _DummyTrainState()


def test_adaptive_scheduler_updates_weights_inverse_rms():
    base = [1.0] * len(RESIDUAL_COMPONENT_NAMES)
    model = _DummyModel(loss_weights=base + [1.0, 1.0])
    scheduler = AdaptiveWeightScheduler(
        component_names=RESIDUAL_COMPONENT_NAMES,
        base_weights=base,
        period=1,
        ema=0.0,
        min_weight=0.1,
        max_weight=10.0,
    )
    scheduler.set_model(model)
    scheduler.on_train_begin()
    model.train_state.step = 1
    model.train_state.loss_train = [4.0] * 6 + [1.0] + [0.5, 0.5]
    scheduler.on_epoch_end()
    expected = [0.875] * 6 + [1.75]
    assert np.allclose(model.loss_weights[: len(RESIDUAL_COMPONENT_NAMES)], expected)


def test_adaptive_scheduler_skips_when_losses_missing():
    base = [1.0] * len(RESIDUAL_COMPONENT_NAMES)
    model = _DummyModel(loss_weights=[1.0, 1.0])
    scheduler = AdaptiveWeightScheduler(
        component_names=RESIDUAL_COMPONENT_NAMES,
        base_weights=base,
        period=1,
        ema=0.0,
        min_weight=0.1,
        max_weight=10.0,
    )
    scheduler.set_model(model)
    scheduler.on_train_begin()
    model.train_state.step = 1
    model.train_state.loss_train = [1.0, 2.0]
    scheduler.on_epoch_end()
    assert model.loss_weights == [1.0, 1.0]
