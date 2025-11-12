from __future__ import annotations

import math
from typing import Sequence

import numpy as np

import deepxde as dde


class AdaptiveWeightScheduler(dde.callbacks.Callback):
    def __init__(
        self,
        *,
        component_names: Sequence[str],
        base_weights: Sequence[float],
        period: int = 200,
        ema: float = 0.5,
        min_weight: float = 1e-3,
        max_weight: float = 10.0,
        initial_rms: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if len(component_names) != len(base_weights):
            raise ValueError("component_names 与 base_weights 长度必须一致。")
        self.component_names = list(component_names)
        self.base_weights = np.asarray(base_weights, dtype=np.float64)
        if np.any(self.base_weights <= 0):
            raise ValueError("base_weights 必须为正数。")
        self.period = max(1, int(period))
        self.ema = float(np.clip(ema, 0.0, 0.999))
        self.min_weight = float(max(min_weight, 1e-8))
        self.max_weight = float(max(max_weight, self.min_weight))
        self.initial_rms = (
            np.asarray(initial_rms, dtype=np.float64).reshape(-1)
            if initial_rms is not None
            else None
        )
        if self.initial_rms is not None and len(self.initial_rms) != len(self.base_weights):
            raise ValueError("initial_rms 长度必须与 base_weights 匹配。")
        self._running_rms = None
        self._last_step = -1
        self._warned_short = False

    def on_train_begin(self):
        if self.initial_rms is not None:
            self._running_rms = np.copy(self.initial_rms)
        else:
            self._running_rms = np.ones_like(self.base_weights, dtype=np.float64)
        self._last_step = -1

    def on_epoch_end(self):
        if self.model is None or self._running_rms is None:
            return
        step = getattr(self.model.train_state, "step", None)
        if step is None or step <= 0 or step == self._last_step:
            return
        if step % self.period != 0:
            return
        unweighted = self._extract_unweighted_losses()
        if unweighted is None:
            return
        rms = np.sqrt(np.maximum(unweighted, 0.0))
        self._running_rms = self.ema * self._running_rms + (1.0 - self.ema) * rms
        inv = np.reciprocal(np.maximum(self._running_rms, 1e-9))
        target = self.base_weights * inv
        target_sum = float(np.sum(target))
        base_sum = float(np.sum(self.base_weights))
        if target_sum > 0 and base_sum > 0:
            target *= base_sum / target_sum
        new_weights = np.clip(target, self.min_weight, self.max_weight)
        self._apply_weights(new_weights)
        self._last_step = step
        formatted = ", ".join(
            f"{name}={weight:.3e}" for name, weight in zip(self.component_names, new_weights)
        )
        print(f"[AdaptiveWeightScheduler] step={step} -> {formatted}")

    def _extract_unweighted_losses(self) -> np.ndarray | None:
        losses = getattr(self.model.train_state, "loss_train", None)
        weights = getattr(self.model, "loss_weights", None)
        if losses is None or weights is None:
            return None
        comp = min(len(self.base_weights), len(losses), len(weights))
        if comp < len(self.base_weights):
            if not self._warned_short:
                print(
                    "[AdaptiveWeightScheduler] loss_train/weights 维度不足，无法调度动态权重。"
                )
                self._warned_short = True
            return None
        values = []
        for i in range(len(self.base_weights)):
            weight = float(weights[i])
            loss_val = float(losses[i])
            if not math.isfinite(loss_val):
                return None
            if weight > 0:
                values.append(loss_val / weight)
            else:
                values.append(loss_val)
        return np.asarray(values, dtype=np.float64)

    def _apply_weights(self, new_weights: np.ndarray) -> None:
        for i, value in enumerate(new_weights.tolist()):
            self.model.loss_weights[i] = float(value)


__all__ = ["AdaptiveWeightScheduler"]

