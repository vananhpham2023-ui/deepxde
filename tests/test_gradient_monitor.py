import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import csv

from examples.pinn_inverse.gradient_monitor import GradientNormMonitor


class _DummyTrainState:
    def __init__(self):
        self.step = 0
        self.iteration = 0


class _DummyModel:
    def __init__(self):
        self.train_state = _DummyTrainState()
        self.stop_training = False


def test_gradient_monitor_logs_and_warns(tmp_path, capsys):
    log_path = tmp_path / "grad.csv"

    class _MockMonitor(GradientNormMonitor):
        def __init__(self):
            super().__init__(
                log_path=str(log_path),
                period=1,
                min_norm=1e-3,
                max_norm=10.0,
                patience=2,
            )
            self.sequence = iter([1e-4, 5.0, 1e2])

        def _compute_grad(self):
            return float(next(self.sequence))

    monitor = _MockMonitor()
    model = _DummyModel()
    monitor.set_model(model)
    monitor.on_train_begin()
    for step in range(1, 4):
        model.train_state.step = step
        monitor.on_epoch_end()

    with open(log_path, "r", newline="") as fp:
        rows = list(csv.reader(fp))
    # header + 3 rows
    assert len(rows) == 4
    assert rows[1][0] == "1"
    captured = capsys.readouterr().out
    assert "GradientNormMonitor" in captured
    assert model.stop_training  # patience reached after two violations

