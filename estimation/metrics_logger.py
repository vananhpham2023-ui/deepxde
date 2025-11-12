from __future__ import annotations

import csv
import math
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np

import deepxde as dde

try:  # pragma: no cover
    from .gradient_monitor import compute_global_grad_norm  # type: ignore
except ImportError:  # pragma: no cover
    from gradient_monitor import compute_global_grad_norm  # type: ignore


class MetricsLogger(dde.callbacks.Callback):
    """Training metrics logger for force_estimation PINN.

    Logs per-step metrics to a CSV file:
    - step, iteration, optimizer phase
    - total train/test loss
    - per-component MSE and RMS (unweighted) for PDE residuals
    - per-component loss weights
    - optional global gradient L2 norm (PyTorch only)
    """

    def __init__(
        self,
        *,
        csv_path: str = os.path.join("logs", "training_metrics.csv"),
        component_names: Sequence[str],
        period: int = 100,
        record_gradients: bool = False,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.component_names = list(component_names)
        self.period = max(1, int(period))
        self.record_gradients = bool(record_gradients)

        self._header_written = False
        self._last_logged_step: int = -1

    def on_train_begin(self) -> None:
        if not self.csv_path:
            return
        parent = os.path.dirname(os.path.abspath(self.csv_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def on_epoch_end(self) -> None:
        step = getattr(self.model.train_state, "step", None)
        if step is None or step == self._last_logged_step:
            return
        if step % self.period != 0:
            return
        self._write_current_row()

    def on_train_end(self) -> None:
        step = getattr(self.model.train_state, "step", None)
        if step is None or step == self._last_logged_step:
            return
        self._write_current_row()

    def _ensure_header(self, extra_bc: int) -> None:
        if self._header_written or not self.csv_path:
            return
        fields: List[str] = [
            "step",
            "iteration",
            "phase",
            "loss_train_total",
            "loss_test_total",
        ]
        for name in self.component_names:
            fields.append(f"weight_{name}")
        for name in self.component_names:
            fields.append(f"mse_{name}")
        for name in self.component_names:
            fields.append(f"rms_{name}")
        if extra_bc > 0:
            fields.append("bc_loss_total")
        fields.append("grad_l2")

        with open(self.csv_path, "w", newline="") as fp:
            csv.writer(fp).writerow(fields)
        self._header_written = True

    @staticmethod
    def _safe_sum(values: Iterable[float]) -> float:
        return float(np.nansum(values))

    @staticmethod
    def _to_list(values) -> List[float]:
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return [float(v) for v in values]
        if isinstance(values, np.ndarray):
            return [float(v) for v in values.reshape(-1)]
        return [float(values)]

    def _current_phase(self) -> str:
        return str(getattr(self.model, "opt_name", "")).lower()

    def _read_losses(self) -> Tuple[List[float], List[float]]:
        losses = getattr(self.model.train_state, "loss_train", None)
        tests = getattr(self.model.train_state, "loss_test", None)
        return self._to_list(losses), self._to_list(tests)

    def _read_weights(self) -> List[float]:
        lw = getattr(self.model, "loss_weights", None)
        return self._to_list(lw)

    def _compute_unweighted_mse(self, weighted_losses: List[float], weights: List[float]) -> List[float]:
        n = min(len(self.component_names), len(weighted_losses), len(weights))
        mse = []
        for i in range(n):
            w = weights[i] if i < len(weights) else 1.0
            v = weighted_losses[i]
            if w == 0 or not np.isfinite(v):
                mse.append(float("nan"))
            else:
                mse.append(float(v) / float(w))
        return mse

    def _compute_grad_norm(self) -> float:
        if not self.record_gradients:
            return float("nan")
        return compute_global_grad_norm(self.model)

    def _write_current_row(self) -> None:
        if not self.csv_path:
            return
        loss_train, loss_test = self._read_losses()
        weights = self._read_weights()
        comp_count = len(self.component_names)
        extra_bc = max(0, len(loss_train) - comp_count)
        self._ensure_header(extra_bc)

        total_train = self._safe_sum(loss_train)
        total_test = self._safe_sum(loss_test)
        mse_vals = self._compute_unweighted_mse(loss_train[:comp_count], weights[:comp_count])
        rms_vals = [math.sqrt(v) if np.isfinite(v) and v >= 0 else float("nan") for v in mse_vals]
        bc_total = self._safe_sum(loss_train[comp_count:]) if extra_bc > 0 else None
        grad_l2 = self._compute_grad_norm()

        row: List[object] = [
            getattr(self.model.train_state, "step", 0),
            getattr(self.model.train_state, "iteration", 0),
            self._current_phase(),
            total_train,
            total_test,
        ]
        row.extend(weights[:comp_count])
        row.extend(mse_vals)
        row.extend(rms_vals)
        if extra_bc > 0:
            row.append(bc_total)
        row.append(grad_l2)

        with open(self.csv_path, "a", newline="") as fp:
            csv.writer(fp).writerow(row)
        self._last_logged_step = int(getattr(self.model.train_state, "step", 0))


__all__ = ["MetricsLogger"]
