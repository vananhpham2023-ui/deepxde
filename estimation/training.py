from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence

import deepxde as dde
import numpy as np
import pandas as pd


def build_network(config: Dict[str, object], input_dim: int, output_dim: int) -> dde.nn.FNN:
    depth = int(config["depth"])
    width = int(config["width"])
    activation = str(config["activation"])
    initializer = str(config["initializer"])
    layer_sizes = [input_dim] + [width] * depth + [output_dim]
    return dde.nn.FNN(layer_sizes, activation, initializer)


def build_observation_bcs(
    df: pd.DataFrame,
    anchors: np.ndarray,
    observation_columns: Dict[str, Dict[str, List[str] | int]],
    *,
    batch_size: int | None = None,
    dtype: np.dtype = np.float64,
) -> List[dde.icbc.PointSetBC]:
    bcs: List[dde.icbc.PointSetBC] = []
    for name, meta in observation_columns.items():
        cols: List[str] = meta["cols"]  # type: ignore[assignment]
        if not set(cols).issubset(df.columns):
            continue
        values = df[cols].to_numpy(dtype=dtype)
        offset = int(meta["first_component"])  # type: ignore[arg-type]
        for i in range(values.shape[1]):
            kwargs = {}
            if batch_size:
                kwargs["batch_size"] = batch_size
            bcs.append(
                dde.icbc.PointSetBC(
                    anchors,
                    values[:, i : i + 1],
                    component=offset + i,
                    **kwargs,
                )
            )
    return bcs


def build_loss_weight_vector(
    pde_weights: Sequence[float],
    bc_weight: float,
    num_bcs: int,
) -> List[float]:
    weights = list(pde_weights)
    if num_bcs > 0:
        weights.extend([bc_weight] * num_bcs)
    return weights


def train_model(
    model: dde.Model,
    loss_weights: Sequence[float],
    training_config: Dict[str, object],
    *,
    extra_callbacks: List[dde.callbacks.Callback] | None = None,
) -> dde.Model:
    adam_lr = float(training_config["adam_lr"])
    adam_iters = int(training_config["adam_iterations"])

    model.compile("adam", lr=adam_lr, loss_weights=loss_weights)

    callbacks = list(extra_callbacks or [])
    checkpoint_path = str(training_config.get("checkpoint_path", ""))
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            dde.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_better_only=True,
                period=1000,
            )
        )

    model.train(
        iterations=adam_iters,
        callbacks=callbacks if callbacks else None,
    )

    if training_config.get("use_lbfgs", True):
        model.compile("L-BFGS", loss_weights=loss_weights)
        model.train(callbacks=callbacks if callbacks else None)

    return model


def evaluate_model(
    model: dde.Model,
    anchors: np.ndarray,
    df: pd.DataFrame,
    out_dir: str,
    observation_columns: Dict[str, Dict[str, List[str] | int]],
) -> np.ndarray:
    preds = model.predict(anchors)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "force_estimation_prediction.csv")
    pred_columns = [
        "fQ_hat_x",
        "fQ_hat_y",
        "fQ_hat_z",
        "fL_hat_x",
        "fL_hat_y",
        "fL_hat_z",
    ]
    export_df = pd.DataFrame(preds, columns=pred_columns)
    if "time" in df.columns:
        export_df.insert(0, "time", df["time"].values)
    for name, meta in observation_columns.items():
        cols: List[str] = meta["cols"]  # type: ignore[assignment]
        if set(cols).issubset(df.columns):
            for col in cols:
                export_df[f"{col}_true"] = df[col].values
    export_df.to_csv(out_path, index=False)
    print(f"预测结果已导出至 {out_path}")
    return preds


__all__ = [
    "build_network",
    "build_observation_bcs",
    "build_loss_weight_vector",
    "train_model",
    "evaluate_model",
]
