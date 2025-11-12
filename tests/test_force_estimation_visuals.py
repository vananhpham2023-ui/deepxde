import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import pandas as pd

from examples.pinn_inverse.force_estimation import (
    AUXILIARY_COLUMNS,
    FEATURE_COLUMNS,
    RESIDUAL_COMPONENT_NAMES,
    compute_residual_components_numpy,
    compute_residual_metrics_numpy,
    generate_residual_distribution_plots,
    save_training_phase_outputs,
)


def _make_dummy_df(rows: int = 8) -> pd.DataFrame:
    data = {"time": np.linspace(0, 1, rows)}
    for idx, col in enumerate(FEATURE_COLUMNS):
        data[col] = np.linspace(idx, idx + 0.5, rows)
    for idx, col in enumerate(AUXILIARY_COLUMNS):
        base = 10.0 + idx
        data[col] = np.linspace(base, base + 0.2, rows)
    return pd.DataFrame(data)


class _DummyLossHistory:
    def __init__(self):
        self.steps = [0, 1, 2]
        self.loss_train = [[4.0], [1.0], [0.5]]
        self.loss_test = [[5.0], [1.2], [0.6]]
        self.metrics_test = [[0.0]] * len(self.steps)


class _DummyTrainState:
    def __init__(self):
        self.X_train = np.zeros((3, 1))
        self.X_test = np.zeros((3, 1))
        self.y_train = np.zeros((3, 1))
        self.y_test = np.zeros((3, 1))
        self.best_y = np.zeros((3, 1))
        self.best_ystd = np.zeros((3, 1))


def test_compute_residual_metrics_numpy_matches_expected():
    df = _make_dummy_df(6)
    preds = np.random.randn(len(df), 6)
    residual_matrix = compute_residual_components_numpy(df, preds)
    metrics = compute_residual_metrics_numpy(
        df,
        preds,
        residual_matrix=residual_matrix,
    ).set_index("metric")
    for idx, name in enumerate(RESIDUAL_COMPONENT_NAMES):
        vector = residual_matrix[:, idx]
        assert np.isclose(metrics.loc["MAE", name], np.mean(np.abs(vector)))
        assert np.isclose(metrics.loc["RMSE", name], np.sqrt(np.mean(vector**2)))
        assert np.isclose(metrics.loc["Mean", name], np.mean(vector))
        assert np.isclose(metrics.loc["Std", name], np.std(vector))


def test_save_training_phase_outputs_emits_files(tmp_path):
    history = _DummyLossHistory()
    train_state = _DummyTrainState()
    data_root = tmp_path / "train_logs"
    png_path = tmp_path / "plots" / "train_plot_adam.png"
    saved = save_training_phase_outputs(
        "adam",
        history,
        train_state,
        data_root=str(data_root),
        plot_path=str(png_path),
    )
    assert saved == str(png_path)
    assert png_path.exists()
    phase_dir = data_root / "adam"
    for name in ("loss.dat", "train.dat", "test.dat"):
        assert (phase_dir / name).exists()


def test_generate_residual_distribution_plots(tmp_path):
    residual_matrix = np.random.randn(32, len(RESIDUAL_COMPONENT_NAMES))
    output_dir = tmp_path / "residual_plots"
    paths = generate_residual_distribution_plots(
        residual_matrix,
        str(output_dir),
    )
    assert len(paths) == 2
    for path in paths:
        assert os.path.exists(path)
