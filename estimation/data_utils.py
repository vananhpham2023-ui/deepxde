from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import deepxde.backend as bkd


@dataclass
class Normalizer:
    minimum: np.ndarray
    range: np.ndarray
    eps: float = 1e-8

    @classmethod
    def from_array(cls, data: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        scale = maximum - minimum
        scale = np.where(np.abs(scale) < eps, 1.0, scale)
        dtype = data.dtype
        return cls(minimum.astype(dtype), scale.astype(dtype), eps)

    def transform(self, data: np.ndarray) -> np.ndarray:
        scaled = (data - self.minimum) / (self.range + self.eps)
        return scaled * 2.0 - 1.0

    def inverse(self, data: np.ndarray) -> np.ndarray:
        scaled = (data + 1.0) * 0.5
        return scaled * (self.range + self.eps) + self.minimum

    def inverse_tensor(self, tensor):
        if not hasattr(self, "_min_tensor"):
            self._min_tensor = bkd.as_tensor(self.minimum.reshape(1, -1))
            self._range_tensor = bkd.as_tensor(self.range.reshape(1, -1))
        scaled = (tensor + 1.0) * 0.5
        return scaled * (self._range_tensor + self.eps) + self._min_tensor


class ConstantManager:
    def __init__(
        self,
        df: pd.DataFrame,
        columns: List[str],
        reference_inputs: np.ndarray,
        seed: int | None = None,
        precision: int = 6,
        match_tol: float | None = 5e-4,
    ) -> None:
        if not set(columns).issubset(df.columns):
            missing = sorted(set(columns) - set(df.columns))
            raise ValueError(f"Constant columns missing: {missing}")
        self.columns = list(columns)
        self.values = df[self.columns].to_numpy(dtype=np.float64)
        self.precision = precision
        self._match_tol = None if match_tol is None else float(match_tol)
        self._rng = np.random.default_rng(seed)
        self._reference_inputs = np.asarray(reference_inputs, dtype=np.float64)
        if self._reference_inputs.shape[0] != self.values.shape[0]:
            raise ValueError("reference_inputs 与常量行数不一致。")
        self._table = {
            self._make_key(row): self.values[i]
            for i, row in enumerate(self._reference_inputs)
        }
        self._column_indices = {name: idx for idx, name in enumerate(self.columns)}
        self._fallback = 0

    def _make_key(self, row: np.ndarray) -> tuple:
        return tuple(np.round(row, self.precision))

    def _sample(self, n: int) -> np.ndarray:
        idx = self._rng.integers(0, len(self.values), size=max(n, 1))
        return self.values[idx]

    def auxiliary(self, inputs: np.ndarray) -> np.ndarray:
        arr = np.asarray(inputs, dtype=np.float64)
        out = np.empty((arr.shape[0], len(self.columns)), dtype=np.float64)
        missing = 0
        for i, row in enumerate(arr):
            val = self._table.get(self._make_key(row))
            if val is None:
                idx = self._find_close(row)
                if idx is not None:
                    val = self.values[idx]
                else:
                    missing += 1
                    val = self._sample(1)[0]
            out[i] = val
        if missing:
            self._fallback += missing
            print(
                f"[force_estimation][ConstantManager] fallback samples used {missing} (total {self._fallback})."
            )
        return out.astype(inputs.dtype if hasattr(inputs, "dtype") else np.float64, copy=False)

    def slice_tensor(self, tensor, name: str):
        idx = self._column_indices[name]
        return tensor[:, idx : idx + 1]

    def get_columns(self) -> List[str]:
        return self.columns[:]

    def _find_close(self, row: np.ndarray) -> int | None:
        if self._match_tol is None:
            return None
        diffs = np.max(np.abs(self._reference_inputs - row), axis=1)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= self._match_tol:
            return idx
        return None


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"数据文件缺少必要列: {missing}")


def apply_quantile_clipping(
    df: pd.DataFrame,
    columns: List[str],
    lower: float,
    upper: float,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("Quantile bounds must satisfy 0 <= lower < upper <= 1.")
    targets = [col for col in columns if col in df.columns]
    if not targets:
        return df, {}
    quantiles = df[targets].quantile([lower, upper])
    lower_bounds = quantiles.loc[lower]
    upper_bounds = quantiles.loc[upper]
    df.loc[:, targets] = df[targets].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    stats = {
        col: (float(lower_bounds[col]), float(upper_bounds[col])) for col in targets
    }
    print(
        "[force_estimation] Applied quantile clipping "
        f"(lower={lower}, upper={upper}) to {len(targets)} columns."
    )
    return df, stats


def apply_quantile_bounds(
    df: pd.DataFrame,
    stats: Dict[str, Tuple[float, float]] | None,
) -> pd.DataFrame:
    """Apply pre-computed quantile bounds to a dataframe."""
    if not stats:
        return df
    available = [col for col in stats if col in df.columns]
    if not available:
        return df
    lower = {col: stats[col][0] for col in available}
    upper = {col: stats[col][1] for col in available}
    df.loc[:, available] = df[available].clip(lower=lower, upper=upper, axis=1)
    return df


def _expand_cache_path(path: str | None) -> str | None:
    if not path:
        return None
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def _read_pickle(path: str | None, label: str, fallback: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception as exc:
        print(f"[force_estimation] 无法读取{label} {path}: {exc}. {fallback}")
        return None


def _write_pickle(path: str | None, label: str, payload: Any) -> None:
    if not path:
        return
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    try:
        with open(path, "wb") as fp:
            pickle.dump(payload, fp)
        print(f"[force_estimation] Saved {label} to {path}")
    except Exception as exc:
        print(f"[force_estimation] 无法写入{label} {path}: {exc}")


def _load_normalizer_cache(
    cache_path: str | None,
    dtype: np.dtype,
    feature_columns: Sequence[str],
) -> Normalizer | None:
    stats = _read_pickle(cache_path, "归一化缓存", "将重新计算统计量。")
    if stats is None:
        return None
    feature_cols = stats.get("feature_columns")
    if feature_cols and list(feature_cols) != list(feature_columns):
        print(
            "[force_estimation] 缓存的特征列与当前定义不一致，"
            "将忽略缓存并重新计算统计量。"
        )
        return None
    try:
        minimum = np.asarray(stats["minimum"], dtype=dtype)
        ranges = np.asarray(stats["range"], dtype=dtype)
    except KeyError as exc:
        print(
            f"[force_estimation] 归一化缓存缺少字段 {exc}. 将重新计算统计量。"
        )
        return None
    eps = float(stats.get("eps", 1e-8))
    print(f"[force_estimation] Loaded cached normalizer stats from {cache_path}")
    return Normalizer(minimum, ranges, eps=eps)


def _save_normalizer_cache(cache_path: str | None, normalizer: Normalizer, *, feature_columns: Sequence[str]) -> None:
    stats = {
        "minimum": normalizer.minimum.tolist(),
        "range": normalizer.range.tolist(),
        "eps": normalizer.eps,
        "feature_columns": list(feature_columns),
    }
    _write_pickle(cache_path, "normalizer stats", stats)


__all__ = [
    "Normalizer",
    "ConstantManager",
    "ensure_columns",
    "apply_quantile_clipping",
    "_expand_cache_path",
    "_load_normalizer_cache",
    "_save_normalizer_cache",
]
