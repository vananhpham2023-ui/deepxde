from __future__ import annotations

import csv
import math
import os
from typing import Callable

import deepxde as dde


def compute_global_grad_norm(model) -> float:
    if model is None:
        return float("nan")
    try:
        from deepxde.backend import backend_name
    except Exception:
        return float("nan")
    if backend_name != "pytorch":
        return float("nan")
    try:
        import torch

        train_state = getattr(model, "train_state", None)
        if train_state is None:
            return float("nan")
        X = getattr(train_state, "X_train", None)
        y = getattr(train_state, "y_train", None)
        aux = getattr(train_state, "train_aux_vars", None)
        if X is None or y is None:
            return float("nan")
        _, losses_tensor = model.outputs_losses_train(X, y, aux)
        total = torch.sum(losses_tensor)
        params = [p for p in model.net.parameters() if p.requires_grad]
        if not params:
            return float("nan")
        grads = torch.autograd.grad(total, params, allow_unused=True)
        sq = 0.0
        for g in grads:
            if g is None:
                continue
            sq += float(g.detach().pow(2).sum().item())
        return math.sqrt(sq) if sq > 0 else 0.0
    except Exception:
        return float("nan")


class GradientNormMonitor(dde.callbacks.Callback):
    def __init__(
        self,
        *,
        log_path: str,
        period: int = 200,
        min_norm: float = 1e-6,
        max_norm: float = 1e3,
        patience: int = 0,
        compute_fn: Callable[[dde.callbacks.Callback], float] | None = None,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.period = max(1, int(period))
        self.min_norm = float(min_norm)
        self.max_norm = float(max_norm) if max_norm >= min_norm else float(min_norm)
        self.patience = max(0, int(patience))
        self._compute_fn = compute_fn
        self._violations = 0
        self._last_step = -1

    def on_train_begin(self):
        if not self.log_path:
            return
        parent = os.path.dirname(os.path.abspath(self.log_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.log_path, "w", newline="") as fp:
            csv.writer(fp).writerow(["step", "iteration", "grad_l2"])
        self._violations = 0
        self._last_step = -1

    def on_epoch_end(self):
        if self.model is None:
            return
        step = getattr(self.model.train_state, "step", None)
        if step is None or step == self._last_step or step <= 0:
            return
        if step % self.period != 0:
            return
        grad_norm = self._compute_grad()
        if not math.isfinite(grad_norm):
            return
        iteration = getattr(self.model.train_state, "iteration", step)
        if self.log_path:
            with open(self.log_path, "a", newline="") as fp:
                csv.writer(fp).writerow([step, iteration, grad_norm])
        if grad_norm < self.min_norm or grad_norm > self.max_norm:
            self._violations += 1
            status = "vanishing" if grad_norm < self.min_norm else "exploding"
            print(
                "[GradientNormMonitor] step={} grad_l2={:.3e} ({}) outside [{:.1e}, {:.1e}]".format(
                    step,
                    grad_norm,
                    status,
                    self.min_norm,
                    self.max_norm,
                )
            )
            if self.patience > 0 and self._violations >= self.patience:
                print(
                    "[GradientNormMonitor] violation count {} reached patience {}; stopping training.".format(
                        self._violations,
                        self.patience,
                    )
                )
                self.model.stop_training = True
        self._last_step = step

    def _compute_grad(self) -> float:
        if self._compute_fn is not None:
            return float(self._compute_fn(self))
        return compute_global_grad_norm(self.model)


__all__ = ["GradientNormMonitor", "compute_global_grad_norm"]

