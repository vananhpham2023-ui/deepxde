from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import deepxde as dde


@dataclass(frozen=True)
class SamplingPlotSpec:
    name: str
    columns: Tuple[str, ...]

    def __post_init__(self):
        if len(self.columns) not in (2, 3):
            raise ValueError(
                f"SamplingPlotSpec '{self.name}' 需要 2 或 3 个列名，实际为 {len(self.columns)}."
            )


def resolve_sample_plot_specs(
    raw: str | None,
    *,
    feature_columns: Sequence[str],
    default_specs: Sequence[SamplingPlotSpec],
) -> List[SamplingPlotSpec]:
    if not raw:
        return list(default_specs)
    entries = [entry.strip() for entry in raw.split(";") if entry.strip()]
    if not entries:
        return list(default_specs)
    resolved: List[SamplingPlotSpec] = []
    for idx, entry in enumerate(entries):
        name_part, cols_part = (entry.split("=", 1) + [""])[:2]
        if cols_part == "":
            cols_part = name_part
            name_part = f"spec_{idx+1}"
        columns = tuple(col.strip() for col in cols_part.split(",") if col.strip())
        if len(columns) not in (2, 3):
            raise ValueError(
                f"采样可视化配置 '{entry}' 需要 2 或 3 个列名，实际为 {len(columns)}."
            )
        missing = [col for col in columns if col not in feature_columns]
        if missing:
            raise ValueError(
                f"采样可视化配置 '{entry}' 包含未知列: {missing}. "
                "请确保列名存在于 FEATURE_COLUMNS 中。"
            )
        name = name_part.strip() or "_".join(columns)
        resolved.append(SamplingPlotSpec(name, columns))
    return resolved


def generate_sampling_visualizations(
    domain_samples: np.ndarray,
    reference_samples: np.ndarray | None,
    specs: Sequence[SamplingPlotSpec],
    output_dir: str,
    *,
    feature_columns: Sequence[str],
) -> List[str]:
    if not specs or domain_samples.size == 0:
        return []
    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        try:
            plt.switch_backend("Agg")
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        print(f"[force_estimation] matplotlib 不可用，跳过采样可视化: {exc}")
        return []

    saved_paths: List[str] = []
    domain_arr = np.asarray(domain_samples, dtype=np.float64)
    domain_df = pd.DataFrame(domain_arr, columns=feature_columns)
    reference_df = (
        pd.DataFrame(np.asarray(reference_samples, dtype=np.float64), columns=feature_columns)
        if reference_samples is not None and len(reference_samples) > 0
        else None
    )

    for spec in specs:
        if any(col not in domain_df.columns for col in spec.columns):
            print(f"[force_estimation] 跳过采样可视化 '{spec.name}'，列不存在。")
            continue
        if len(spec.columns) == 2:
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            if reference_df is not None:
                ax.scatter(
                    reference_df[spec.columns[0]],
                    reference_df[spec.columns[1]],
                    s=6,
                    alpha=0.15,
                    color="#B0BEC5",
                    label="dataset",
                )
            ax.scatter(
                domain_df[spec.columns[0]],
                domain_df[spec.columns[1]],
                s=10,
                alpha=0.75,
                color="#1f77b4",
                label="domain_sample",
            )
            ax.set_xlabel(spec.columns[0])
            ax.set_ylabel(spec.columns[1])
            ax.set_title(f"{spec.name} (n={len(domain_df)})")
            if reference_df is not None:
                ax.legend(loc="best")
            fig.tight_layout()
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(5.2, 4.6))
            ax = fig.add_subplot(111, projection="3d")
            if reference_df is not None:
                ax.scatter(
                    reference_df[spec.columns[0]],
                    reference_df[spec.columns[1]],
                    reference_df[spec.columns[2]],
                    s=8,
                    alpha=0.12,
                    color="#B0BEC5",
                    label="dataset",
                )
            ax.scatter(
                domain_df[spec.columns[0]],
                domain_df[spec.columns[1]],
                domain_df[spec.columns[2]],
                s=12,
                alpha=0.8,
                color="#1f77b4",
                label="domain_sample",
            )
            ax.set_xlabel(spec.columns[0])
            ax.set_ylabel(spec.columns[1])
            ax.set_zlabel(spec.columns[2])
            ax.set_title(f"{spec.name} (n={len(domain_df)})")
            if reference_df is not None:
                ax.legend(loc="best")
            fig.tight_layout()
        out_path = os.path.join(output_dir, f"{spec.name}.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


class MixedFeatureSampler:
    def __init__(
        self,
        df: pd.DataFrame,
        normalizer,
        feature_columns: Sequence[str],
        *,
        time_column: str = "time",
        mix_ratio: float = 0.6,
        jitter_scale: float = 0.02,
        seed: int | None = None,
    ):
        if time_column not in df.columns:
            raise ValueError(f"数据集中缺少 {time_column} 列，无法构建混合几何采样器。")
        self._normalizer = normalizer
        self._feature_columns = feature_columns
        self._rng = np.random.default_rng(seed)
        self._dtype = normalizer.minimum.dtype
        mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        self._mix_ratio = mix_ratio
        jitter_scale = float(max(0.0, jitter_scale))
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        time_values = df_sorted[time_column].to_numpy(dtype=np.float64)
        unique_times, unique_idx = np.unique(time_values, return_index=True)
        self._allow_interpolation = len(unique_times) >= 2
        if self._allow_interpolation:
            self._times = unique_times
            self._feature_history = (
                df_sorted.iloc[unique_idx][feature_columns].to_numpy(dtype=np.float64)
            )
            self._time_geom = dde.geometry.TimeDomain(
                float(self._times[0]), float(self._times[-1])
            )
        else:
            self._times = np.array([], dtype=np.float64)
            self._feature_history = np.empty((0, len(feature_columns)), dtype=np.float64)
            self._time_geom = None
        raw_features = df[feature_columns].to_numpy(dtype=np.float64)
        self._raw_features = raw_features
        feature_min = df[feature_columns].min().to_numpy(dtype=np.float64)
        feature_max = df[feature_columns].max().to_numpy(dtype=np.float64)
        self._feature_span = np.where(feature_max - feature_min < 1e-6, 1.0, feature_max - feature_min)
        self._feature_min = feature_min
        self._jitter_scale = jitter_scale

    def sample(self, n: int) -> np.ndarray:
        samples = np.empty((n, len(self._feature_columns)), dtype=self._dtype)
        if not self._allow_interpolation or self._mix_ratio <= 0.0:
            indices = self._rng.integers(0, len(self._raw_features), size=n)
            samples = self._raw_features[indices]
        else:
            num_interp = int(round(self._mix_ratio * n))
            num_emp = n - num_interp
            if num_emp > 0:
                indices = self._rng.integers(0, len(self._raw_features), size=num_emp)
                samples[:num_emp] = self._raw_features[indices]
            if num_interp > 0 and self._time_geom is not None:
                times = self._time_geom.random_points(num_interp)
                interp = np.empty((num_interp, len(self._feature_columns)), dtype=np.float64)
                for i, t in enumerate(times):
                    right_idx = np.searchsorted(self._times, t)
                    left_idx = max(0, min(right_idx - 1, len(self._times) - 1))
                    right_idx = min(len(self._times) - 1, left_idx + 1)
                    if left_idx == right_idx:
                        interp[i] = self._feature_history[left_idx]
                    else:
                        t_left = self._times[left_idx]
                        t_right = self._times[right_idx]
                        weight = (t - t_left) / (t_right - t_left)
                        interp[i] = (
                            (1 - weight) * self._feature_history[left_idx]
                            + weight * self._feature_history[right_idx]
                        )
                jitter = self._rng.standard_normal(interp.shape) * (
                    self._feature_span * self._jitter_scale
                )
                samples[num_emp:] = interp + jitter
        return self._normalizer.transform(samples)


class MixedGeometry(dde.geometry.geometry_nd.Hypercube):
    def __init__(self, lower, upper, sampler=None):
        super().__init__(lower, upper)
        self._sampler = sampler

    def random_points(self, n, random=None):
        if self._sampler is not None:
            return self._sampler.sample(n)
        return super().random_points(n, random=random)


def build_geometry(
    features_norm: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    normalizer=None,
    feature_columns: Sequence[str],
    seed: int | None = None,
    mix_ratio: float | None = None,
    jitter_scale: float | None = None,
    enable_mixed: bool = True,
) -> dde.geometry.geometry_nd.Hypercube:
    xmin = features_norm.min(axis=0)
    xmax = features_norm.max(axis=0)
    dx = np.where(np.isclose(xmin, xmax), 1e-3, 0.0)
    lower = (xmin - dx).tolist()
    upper = (xmax + dx).tolist()
    sampler = None
    if enable_mixed and df is not None and normalizer is not None and "time" in df.columns:
        ratio = mix_ratio if mix_ratio is not None else 0.6
        jitter = jitter_scale if jitter_scale is not None else 0.02
        try:
            sampler = MixedFeatureSampler(
                df,
                normalizer,
                feature_columns,
                mix_ratio=ratio,
                jitter_scale=jitter,
                seed=seed,
            )
        except Exception as exc:
            print(
                f"[force_estimation] 构建混合几何采样器失败({exc})，将回退至 Hypercube 采样。"
            )
            sampler = None
    return MixedGeometry(lower, upper, sampler=sampler)


def resolve_effective_num_domain(
    explicit: int | None,
    *,
    data_size: int,
    auto_enabled: bool,
    ratio: float,
    min_points: int,
    max_points: int,
) -> Tuple[int, str]:
    def _safe_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    explicit_val = _safe_int(explicit, 0)
    if explicit_val > 0:
        return explicit_val, "explicit"
    if not auto_enabled:
        return 0, "disabled"
    ratio = float(max(ratio, 0.0))
    min_pts = max(_safe_int(min_points, 0), 0)
    max_pts = max(_safe_int(max_points, min_pts), min_pts)
    estimated = int(round(float(data_size) * ratio))
    estimated = max(estimated, min_pts)
    estimated = min(estimated, max_pts)
    estimated = max(estimated, 0)
    mode = "auto"
    return estimated, mode


__all__ = [
    "SamplingPlotSpec",
    "MixedFeatureSampler",
    "MixedGeometry",
    "resolve_sample_plot_specs",
    "generate_sampling_visualizations",
    "resolve_effective_num_domain",
    "build_geometry",
]
