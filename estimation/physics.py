from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import deepxde.backend as bkd

RESIDUAL_COMPONENT_NAMES: Tuple[str, ...] = (
    "r1_x",
    "r1_y",
    "r1_z",
    "r2_x",
    "r2_y",
    "r2_z",
    "r3",
)

DEFAULT_RESIDUAL_NORM_EPS = 1e-6


@dataclass(frozen=True)
class ResidualContext:
    idx_aq: slice
    idx_al: slice
    idx_rho: slice
    idx_ez: slice
    idx_thrust: int
    rho_columns: Sequence[str]
    mass_columns: Tuple[str, str]
    gravity_column: str
    lever_column: str | None = None


class ResidualScaler:
    def __init__(
        self,
        component_scales: Sequence[float],
        group_scales: Dict[str, float],
        mode: str,
        eps: float = DEFAULT_RESIDUAL_NORM_EPS,
    ) -> None:
        self.scales = np.asarray(component_scales, dtype=np.float64).reshape(-1)
        self._component_map = {
            name: float(scale)
            for name, scale in zip(RESIDUAL_COMPONENT_NAMES, self.scales)
        }
        self.group_scales = dict(group_scales)
        self.mode = mode
        self.eps = float(eps)

    def normalize_tensors(self, tensors):
        normalized = []
        for idx, tensor in enumerate(tensors):
            scale = float(self.scales[idx]) if idx < len(self.scales) else 1.0
            if scale <= self.eps:
                normalized.append(tensor)
            else:
                normalized.append(tensor / scale)
        return normalized

    def normalize_numpy(self, matrix: np.ndarray) -> np.ndarray:
        safe = np.where(self.scales <= self.eps, 1.0, self.scales)
        return matrix / safe.reshape(1, -1)

    def component_scale(self, name: str) -> float:
        return float(self._component_map.get(name, 1.0))

    def summary(self) -> str:
        pairs = ", ".join(
            f"{key}={self.group_scales.get(key, 1.0):.3e}"
            for key in ("r1", "r2", "r3")
        )
        return f"{self.mode} ({pairs})"


def cross_product(vec_a, vec_b):
    ax, ay, az = vec_a[:, 0:1], vec_a[:, 1:2], vec_a[:, 2:3]
    bx, by, bz = vec_b[:, 0:1], vec_b[:, 1:2], vec_b[:, 2:3]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return bkd.concat([cx, cy, cz], axis=1)


def _reduce_statistic(values: np.ndarray, mode: str, eps: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return eps
    arr = np.abs(arr)
    if mode == "median":
        stat = float(np.median(arr))
    elif mode == "mean":
        stat = float(np.mean(arr))
    else:
        raise ValueError(f"Unsupported residual_norm_mode '{mode}'.")
    if not np.isfinite(stat):
        return eps
    return max(stat, eps)


def _apply_scale_override(base: float, override: float | None, eps: float) -> float:
    if override is None:
        return max(base, eps)
    if override <= 0:
        raise ValueError("Residual scale overrides必须为正数。")
    return max(float(override), eps)


def build_residual_scaler(
    df: pd.DataFrame,
    *,
    mode: str,
    overrides: Dict[str, float | None] | None,
    context: ResidualContext,
    eps: float = DEFAULT_RESIDUAL_NORM_EPS,
) -> ResidualScaler | None:
    normalized_mode = (mode or "off").strip().lower()
    if normalized_mode in {"off", "none", "disable", "disabled"}:
        return None
    if normalized_mode not in {"median", "mean"}:
        raise ValueError(
            f"residual_norm_mode 仅支持 median/mean/off，收到 {mode!r}."
        )
    overrides = overrides or {}
    m_q_col, m_l_col = context.mass_columns
    g_col = context.gravity_column
    m_q = df[m_q_col].to_numpy(dtype=np.float64, copy=False)
    m_l = df[m_l_col].to_numpy(dtype=np.float64, copy=False)
    g_vals = df[g_col].to_numpy(dtype=np.float64, copy=False)
    total_mass = m_q + m_l
    force_scale = _reduce_statistic(total_mass * g_vals, normalized_mode, eps)

    rho_vectors = df[list(context.rho_columns)].to_numpy(dtype=np.float64, copy=False)
    rho_norm = np.linalg.norm(rho_vectors, axis=1)
    lever_candidates = rho_norm
    if context.lever_column and context.lever_column in df.columns:
        lever_vals = df[context.lever_column].to_numpy(dtype=np.float64, copy=False)
        lever_candidates = np.where(np.isfinite(lever_vals), lever_vals, lever_candidates)
    lever_scale = _reduce_statistic(lever_candidates, normalized_mode, eps)
    torque_scale = lever_scale * force_scale

    scale_r1 = _apply_scale_override(force_scale, overrides.get("r1"), eps)
    scale_r2 = _apply_scale_override(torque_scale, overrides.get("r2"), eps)
    scale_r3 = _apply_scale_override(torque_scale, overrides.get("r3"), eps)

    component_scales = np.array(
        [scale_r1, scale_r1, scale_r1, scale_r2, scale_r2, scale_r2, scale_r3],
        dtype=np.float64,
    )
    group_scales = {"r1": scale_r1, "r2": scale_r2, "r3": scale_r3}
    return ResidualScaler(component_scales, group_scales, normalized_mode, eps=eps)


def build_force_estimation_residual(
    normalizer,
    const_mgr,
    *,
    context: ResidualContext,
    residual_scaler: ResidualScaler | None = None,
):
    def residual(x, y, aux_vars):
        features = normalizer.inverse_tensor(x)
        if aux_vars is None:
            raise ValueError("ConstantManager auxiliary vars missing.")
        consts = aux_vars

        f_Q = y[:, 0:3]
        f_L = y[:, 3:6]
        rho = features[:, context.idx_rho]

        acc_q = features[:, context.idx_aq]
        acc_l = features[:, context.idx_al]
        thrust = features[:, context.idx_thrust : context.idx_thrust + 1]
        m_q = const_mgr.slice_tensor(consts, context.mass_columns[0])
        m_l = const_mgr.slice_tensor(consts, context.mass_columns[1])
        g_scalar = const_mgr.slice_tensor(consts, context.gravity_column)

        ez_world = features[:, context.idx_ez]

        zero_like = bkd.zeros_like(g_scalar)
        gravity_vector = bkd.concat([zero_like, zero_like, g_scalar], axis=1)

        r1 = (
            f_Q
            + f_L
            - m_q * (gravity_vector + acc_q)
            - m_l * (gravity_vector + acc_l)
            + thrust * ez_world
        )

        aero_term = thrust * ez_world - m_q * gravity_vector - m_q * acc_q
        r2 = cross_product(rho, f_Q) - cross_product(rho, aero_term)

        r3 = bkd.sum(f_L * rho, 1, keepdims=True)

        residuals = [
            r1[:, 0:1],
            r1[:, 1:2],
            r1[:, 2:3],
            r2[:, 0:1],
            r2[:, 1:2],
            r2[:, 2:3],
            r3,
        ]

        if residual_scaler is not None:
            residuals = residual_scaler.normalize_tensors(residuals)

        return residuals

    return residual


def compute_residual_components_numpy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    feature_columns: Sequence[str],
    context: ResidualContext,
    residual_scaler: ResidualScaler | None = None,
) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[1] != 6:
        raise ValueError("predictions 必须为形状 (N, 6) 的数组。")
    features = df[feature_columns].to_numpy(dtype=np.float64, copy=False)
    preds = np.asarray(predictions, dtype=np.float64)
    f_q = preds[:, 0:3]
    f_l = preds[:, 3:6]
    acc_q = features[:, context.idx_aq]
    acc_l = features[:, context.idx_al]
    rho = features[:, context.idx_rho]
    ez_world = features[:, context.idx_ez]
    thrust = features[:, context.idx_thrust].reshape(-1, 1)

    m_q_col, m_l_col = context.mass_columns
    g_col = context.gravity_column
    m_q = df[m_q_col].to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)
    m_l = df[m_l_col].to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)
    g_scalar = df[g_col].to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)
    zero = np.zeros_like(g_scalar)
    gravity_vector = np.concatenate([zero, zero, g_scalar], axis=1)

    r1 = (
        f_q
        + f_l
        - m_q * (gravity_vector + acc_q)
        - m_l * (gravity_vector + acc_l)
        + thrust * ez_world
    )

    aero_term = thrust * ez_world - m_q * gravity_vector - m_q * acc_q
    r2 = np.cross(rho, f_q) - np.cross(rho, aero_term)

    r3 = np.sum(f_l * rho, axis=1, keepdims=True)

    residual_matrix = np.hstack([r1, r2, r3])
    if residual_scaler is not None:
        residual_matrix = residual_scaler.normalize_numpy(residual_matrix)
    return residual_matrix


def compute_residual_variance_table(
    df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    feature_columns: Sequence[str],
    context: ResidualContext,
    residual_scaler: ResidualScaler | None = None,
) -> pd.DataFrame:
    residual_matrix = compute_residual_components_numpy(
        df,
        predictions,
        feature_columns=feature_columns,
        context=context,
        residual_scaler=residual_scaler,
    )
    means = residual_matrix.mean(axis=0)
    stds = residual_matrix.std(axis=0)
    variances = residual_matrix.var(axis=0)
    mae = np.mean(np.abs(residual_matrix), axis=0)
    rmse = np.sqrt(np.mean(residual_matrix**2, axis=0))

    data = {
        "component": RESIDUAL_COMPONENT_NAMES,
        "mean": means,
        "std": stds,
        "variance": variances,
        "mae": mae,
        "rmse": rmse,
    }
    if residual_scaler is not None:
        data["scale"] = [
            residual_scaler.component_scale(name) for name in RESIDUAL_COMPONENT_NAMES
        ]
    return pd.DataFrame(data)


__all__ = [
    "RESIDUAL_COMPONENT_NAMES",
    "DEFAULT_RESIDUAL_NORM_EPS",
    "ResidualContext",
    "ResidualScaler",
    "build_residual_scaler",
    "build_force_estimation_residual",
    "compute_residual_components_numpy",
    "compute_residual_variance_table",
]
