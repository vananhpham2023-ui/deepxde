import os
import math
import csv

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

from examples.pinn_inverse.metrics_logger import MetricsLogger


class _DummyTrainState:
    def __init__(self, step: int, iteration: int, loss_train, loss_test):
        self.step = step
        self.iteration = iteration
        self.loss_train = loss_train
        self.loss_test = loss_test


class _DummyModel:
    def __init__(self, loss_weights, train_state, opt_name: str = "adam"):
        self.loss_weights = loss_weights
        self.train_state = train_state
        self.opt_name = opt_name


def test_metrics_logger_writes_csv_and_rms(tmp_path):
    comp_names = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z", "r3")
    # Build synthetic weighted losses: mse * weight
    unweighted_mse = np.array([1.0, 4.0, 9.0, 16.0, 0.25, 0.0, 100.0], dtype=float)
    weights = np.array([2.0, 0.5, 1.0, 4.0, 2.0, 1.0, 0.1], dtype=float)
    weighted = (unweighted_mse * weights).tolist()
    # add two BC terms to the tail
    loss_train = weighted + [3.0, 5.0]
    loss_test = [sum(loss_train)]

    ts = _DummyTrainState(step=100, iteration=10, loss_train=loss_train, loss_test=loss_test)
    model = _DummyModel(loss_weights=weights.tolist() + [1.0, 1.0], train_state=ts)

    out_csv = tmp_path / "metrics.csv"
    cb = MetricsLogger(csv_path=str(out_csv), component_names=comp_names, period=1, record_gradients=False)
    cb.set_model(model)
    cb.on_train_begin()
    cb.on_epoch_end()  # triggers write for step=100

    assert os.path.exists(out_csv)
    with open(out_csv, "r", newline="") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 2  # header + 1 data row
    header = rows[0]
    row = rows[1]

    # Validate RMS columns
    # Locate the first rms column index
    rms_start = header.index("rms_" + comp_names[0])
    rms_vals = [float(v) for v in row[rms_start : rms_start + len(comp_names)]]
    expected_rms = [math.sqrt(v) for v in unweighted_mse]
    for a, b in zip(rms_vals, expected_rms):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    # Check bc_total exists and equals sum of the tail two weighted BC losses
    assert "bc_loss_total" in header
    bc_idx = header.index("bc_loss_total")
    assert math.isfinite(float(row[bc_idx]))
    assert math.isclose(float(row[bc_idx]), 8.0, rel_tol=1e-12)
