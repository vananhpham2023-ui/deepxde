"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Physics-informed neural network (PINN) for estimating the interaction forces
between a quadrotor UAV and a suspended payload. The network ingests 37 features
covering UAV/payload states, cable direction, thrust, and environmental terms
and predicts the six components of the unknown contact forces ``f_Q`` and
``f_L`` while enforcing three physics residuals.
"""
from __future__ import annotations

import os
import sys
import glob
import argparse
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepxde.backend as bkd

try:  # pragma: no cover
    from .metrics_logger import MetricsLogger  # type: ignore
except ImportError:  # pragma: no cover
    from metrics_logger import MetricsLogger  # type: ignore

try:  # pragma: no cover
    from .weight_scheduler import AdaptiveWeightScheduler  # type: ignore
except ImportError:  # pragma: no cover
    from weight_scheduler import AdaptiveWeightScheduler  # type: ignore

try:  # pragma: no cover
    from .gradient_monitor import GradientNormMonitor  # type: ignore
except ImportError:  # pragma: no cover
    from gradient_monitor import GradientNormMonitor  # type: ignore

try:  # pragma: no cover
    from .export_utils import (  # type: ignore
        export_model_metadata,
        export_normalizer_to_json,
        export_onnx_model,
        export_torchscript_model,
    )
except ImportError:  # pragma: no cover
    from export_utils import (  # type: ignore
        export_model_metadata,
        export_normalizer_to_json,
        export_onnx_model,
        export_torchscript_model,
    )

try:  # pragma: no cover
    from .physics import (  # type: ignore
        RESIDUAL_COMPONENT_NAMES,
        ResidualContext,
        ResidualScaler,
        build_force_estimation_residual as physics_build_force_estimation_residual,
        build_residual_scaler as physics_build_residual_scaler,
        compute_residual_components_numpy as physics_compute_residual_components_numpy,
        compute_residual_variance_table as physics_compute_residual_variance_table,
        compute_residual_metrics_numpy as physics_compute_residual_metrics_numpy,
    )
except ImportError:  # pragma: no cover
    from physics import (  # type: ignore
        RESIDUAL_COMPONENT_NAMES,
        ResidualContext,
        ResidualScaler,
        build_force_estimation_residual as physics_build_force_estimation_residual,
        build_residual_scaler as physics_build_residual_scaler,
        compute_residual_components_numpy as physics_compute_residual_components_numpy,
        compute_residual_variance_table as physics_compute_residual_variance_table,
        compute_residual_metrics_numpy as physics_compute_residual_metrics_numpy,
    )

try:  # pragma: no cover
    from .data_utils import (  # type: ignore
        Normalizer,
        ConstantManager,
        ensure_columns,
        apply_quantile_clipping,
        _expand_cache_path,
        _load_normalizer_cache,
        _save_normalizer_cache,
    )
except ImportError:  # pragma: no cover
    from data_utils import (  # type: ignore
        Normalizer,
        ConstantManager,
        ensure_columns,
        apply_quantile_clipping,
        _expand_cache_path,
        _load_normalizer_cache,
        _save_normalizer_cache,
    )

try:  # pragma: no cover
    from .geometry_utils import (  # type: ignore
        SamplingPlotSpec,
        MixedFeatureSampler,
        MixedGeometry,
        resolve_sample_plot_specs as geom_resolve_sample_plot_specs,
        generate_sampling_visualizations as geom_generate_sampling_visualizations,
        resolve_effective_num_domain as geom_resolve_effective_num_domain,
        build_geometry as geom_build_geometry,
    )
except ImportError:  # pragma: no cover
    from geometry_utils import (  # type: ignore
        SamplingPlotSpec,
        MixedFeatureSampler,
        MixedGeometry,
        resolve_sample_plot_specs as geom_resolve_sample_plot_specs,
        generate_sampling_visualizations as geom_generate_sampling_visualizations,
        resolve_effective_num_domain as geom_resolve_effective_num_domain,
        build_geometry as geom_build_geometry,
    )

# --------------------------------------------------------------------------------------
# 数据与模型配置 —— 调试时优先调整以下参数
# --------------------------------------------------------------------------------------

# 数据文件路径（默认指向 examples/pinn_inverse/datasets/force_estimation_train.csv）
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "force_estimation_train.csv"
)

# 原始数据列（保持与 CSV 一致，37 维）
FULL_FEATURE_COLUMNS: List[str] = [
    "time",
    "xQ_x",
    "xQ_y",
    "xQ_z",
    "vQ_x",
    "vQ_y",
    "vQ_z",
    "accQ_x",
    "accQ_y",
    "accQ_z",
    "omega_b_x",
    "omega_b_y",
    "omega_b_z",
    "xL_x",
    "xL_y",
    "xL_z",
    "vL_x",
    "vL_y",
    "vL_z",
    "accL_x",
    "accL_y",
    "accL_z",
    "rho_x",
    "rho_y",
    "rho_z",
    "ez_world_x",
    "ez_world_y",
    "ez_world_z",
    "thrust",
    "m_Q",
    "m_L",
    "g",
    "l_length",
    "sqrt_kf",
    "wind_x",
    "wind_y",
    "wind_z",
]

# 网络输入特征列（最小充要集动态部分）
FEATURE_COLUMNS: List[str] = [
    "accQ_x",
    "accQ_y",
    "accQ_z",
    "accL_x",
    "accL_y",
    "accL_z",
    "rho_x",
    "rho_y",
    "rho_z",
    "ez_world_x",
    "ez_world_y",
    "ez_world_z",
    "thrust",
]

# 常量/缓慢变量列（通过 ConstantManager 提供，不进入网络）
AUXILIARY_COLUMNS: List[str] = ["m_Q", "m_L", "g", "l_length"]

# 最小充要特征集（供调试/验证用：动态输入 + 常量）
MINIMAL_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + AUXILIARY_COLUMNS
DEFAULT_SAMPLE_PLOT_DIR = os.path.join("plots", "sampling")
DEFAULT_RESIDUAL_VARIANCE_PATH = os.path.join("logs", "residual_variance.csv")
DEFAULT_RESIDUAL_METRICS_PATH = os.path.join("logs", "force_estimation_residual_metrics.csv")
DEFAULT_RESIDUAL_PLOT_DIR = os.path.join("logs", "residual_plots")
DEFAULT_TRAIN_PLOT_DIR = os.path.join("logs")
DEFAULT_TRAIN_PLOT_PREFIX = "train_plot"

# 可选观测列，用于 PointSetBC 监督（存在时自动启用）
OBSERVATION_COLUMNS: Dict[str, Dict[str, List[str] | int]] = {
    "f_Q": {"cols": ["fQ_x", "fQ_y", "fQ_z"], "first_component": 0},
    "f_L": {"cols": ["fL_x", "fL_y", "fL_z"], "first_component": 3},
}

# 损失权重，r1/r2/r3 对应各自残差
LOSS_WEIGHTS: Dict[str, float] = {
    "r1": 1.0,
    "r2": 1.0,
    "r3": 1.0,
}

SUPERVISION_WEIGHT = 1.0

# 网络结构配置（可根据数据规模调节）
NETWORK_CONFIG: Dict[str, int | str] = {
    "width": 128,
    "depth": 5,
    "activation": "tanh",
    "initializer": "Glorot uniform",
}

# 训练超参数
TRAINING_CONFIG: Dict[str, object] = {
    "adam_iterations": 20000,
    "adam_lr": 1e-3,
    "use_lbfgs": True,
    "checkpoint_path": "checkpoints/force_estimation",
}

MIXED_GEOMETRY_CONFIG: Dict[str, float] = {
    "mix_ratio": 0.6,
    "jitter_scale": 0.02,
}

DOMAIN_SAMPLING_CONFIG: Dict[str, float | int] = {
    "ratio": 0.35,
    "min_points": 1024,
    "max_points": 8192,
}

# --------------------------------------------------------------------------------------
# 常量与索引
# --------------------------------------------------------------------------------------

INPUT_DIM = len(FEATURE_COLUMNS)
OUTPUT_DIM = 6  # f_Q(3) + f_L(3)

POSITION_COLUMNS = [
    "xQ_x",
    "xQ_y",
    "xQ_z",
    "xL_x",
    "xL_y",
    "xL_z",
]
VELOCITY_COLUMNS = [
    "vQ_x",
    "vQ_y",
    "vQ_z",
    "vL_x",
    "vL_y",
    "vL_z",
]
ACCELERATION_COLUMNS = [
    "accQ_x",
    "accQ_y",
    "accQ_z",
    "accL_x",
    "accL_y",
    "accL_z",
]
RHO_COLUMNS = ["rho_x", "rho_y", "rho_z"]
EZ_COLUMNS = ["ez_world_x", "ez_world_y", "ez_world_z"]

# FULL 列索引（用于构造演示数据等）
IDX_TIME_FULL = FULL_FEATURE_COLUMNS.index("time")
IDX_XQ_FULL = slice(
    FULL_FEATURE_COLUMNS.index("xQ_x"), FULL_FEATURE_COLUMNS.index("xQ_z") + 1
)
IDX_VQ_FULL = slice(
    FULL_FEATURE_COLUMNS.index("vQ_x"), FULL_FEATURE_COLUMNS.index("vQ_z") + 1
)
IDX_AQ_FULL = slice(
    FULL_FEATURE_COLUMNS.index("accQ_x"), FULL_FEATURE_COLUMNS.index("accQ_z") + 1
)
IDX_OMEGA_FULL = slice(
    FULL_FEATURE_COLUMNS.index("omega_b_x"), FULL_FEATURE_COLUMNS.index("omega_b_z") + 1
)
IDX_XL_FULL = slice(
    FULL_FEATURE_COLUMNS.index("xL_x"), FULL_FEATURE_COLUMNS.index("xL_z") + 1
)
IDX_VL_FULL = slice(
    FULL_FEATURE_COLUMNS.index("vL_x"), FULL_FEATURE_COLUMNS.index("vL_z") + 1
)
IDX_AL_FULL = slice(
    FULL_FEATURE_COLUMNS.index("accL_x"), FULL_FEATURE_COLUMNS.index("accL_z") + 1
)
IDX_RHO_FULL = slice(
    FULL_FEATURE_COLUMNS.index("rho_x"), FULL_FEATURE_COLUMNS.index("rho_z") + 1
)
IDX_EZ_FULL = slice(
    FULL_FEATURE_COLUMNS.index("ez_world_x"), FULL_FEATURE_COLUMNS.index("ez_world_z") + 1
)
IDX_THRUST_FULL = FULL_FEATURE_COLUMNS.index("thrust")
IDX_MQ_FULL = FULL_FEATURE_COLUMNS.index("m_Q")
IDX_ML_FULL = FULL_FEATURE_COLUMNS.index("m_L")
IDX_G_FULL = FULL_FEATURE_COLUMNS.index("g")
IDX_L_LENGTH_FULL = FULL_FEATURE_COLUMNS.index("l_length")
IDX_SQRT_KF_FULL = FULL_FEATURE_COLUMNS.index("sqrt_kf")
IDX_WIND_FULL = slice(
    FULL_FEATURE_COLUMNS.index("wind_x"), FULL_FEATURE_COLUMNS.index("wind_z") + 1
)

# 网络输入特征索引
IDX_AQ = slice(FEATURE_COLUMNS.index("accQ_x"), FEATURE_COLUMNS.index("accQ_z") + 1)
IDX_AL = slice(FEATURE_COLUMNS.index("accL_x"), FEATURE_COLUMNS.index("accL_z") + 1)
IDX_RHO = slice(FEATURE_COLUMNS.index("rho_x"), FEATURE_COLUMNS.index("rho_z") + 1)
IDX_EZ = slice(
    FEATURE_COLUMNS.index("ez_world_x"), FEATURE_COLUMNS.index("ez_world_z") + 1
)
IDX_THRUST = FEATURE_COLUMNS.index("thrust")

RESIDUAL_CONTEXT = ResidualContext(
    idx_aq=IDX_AQ,
    idx_al=IDX_AL,
    idx_rho=IDX_RHO,
    idx_ez=IDX_EZ,
    idx_thrust=IDX_THRUST,
    rho_columns=RHO_COLUMNS,
    mass_columns=("m_Q", "m_L"),
    gravity_column="g",
    lever_column="l_length",
)


def build_residual_scaler(
    df: pd.DataFrame,
    *,
    mode: str,
    overrides: Dict[str, float | None] | None = None,
) -> ResidualScaler | None:
    return physics_build_residual_scaler(
        df,
        mode=mode,
        overrides=overrides,
        context=RESIDUAL_CONTEXT,
    )


def build_force_estimation_residual(
    normalizer,
    const_mgr,
    residual_scaler: ResidualScaler | None = None,
):
    return physics_build_force_estimation_residual(
        normalizer,
        const_mgr,
        context=RESIDUAL_CONTEXT,
        residual_scaler=residual_scaler,
    )


def compute_residual_components_numpy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    residual_scaler: ResidualScaler | None = None,
) -> np.ndarray:
    return physics_compute_residual_components_numpy(
        df,
        predictions,
        feature_columns=FEATURE_COLUMNS,
        context=RESIDUAL_CONTEXT,
        residual_scaler=residual_scaler,
    )


def compute_residual_variance_table(
    df: pd.DataFrame,
    predictions: np.ndarray,
    residual_scaler: ResidualScaler | None = None,
    residual_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    return physics_compute_residual_variance_table(
        df,
        predictions,
        feature_columns=FEATURE_COLUMNS,
        context=RESIDUAL_CONTEXT,
        residual_scaler=residual_scaler,
        residual_matrix=residual_matrix,
    )


def compute_residual_metrics_numpy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    residual_scaler: ResidualScaler | None = None,
    residual_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    return physics_compute_residual_metrics_numpy(
        df,
        predictions,
        feature_columns=FEATURE_COLUMNS,
        context=RESIDUAL_CONTEXT,
        residual_scaler=residual_scaler,
        residual_matrix=residual_matrix,
    )

CLIP_EXCLUDE_COLUMNS = {"time"}
DEFAULT_CLIP_QUANTILES = (0.01, 0.99)
CLIP_TARGET_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CLIP_EXCLUDE_COLUMNS]
UNIT_META_KEYS = [
    "unit_profile_selected",
    "pos_unit_selected",
    "vel_unit_selected",
    "acc_unit_selected",
    "thrust_unit_selected",
    "thrust_source",
    "fix_unit_vectors_applied",
    "rho_recomputed",
]

UNIT_PROFILE_DEFAULTS = {
    "si": {
        "pos": "m",
        "vel": "mps",
        "acc": "mps2",
    },
    "cgs": {
        "pos": "cm",
        "vel": "cmps",
        "acc": "cmps2",
    },
}

UNIT_SCALE = {
    "pos": {"m": 1.0, "cm": 0.01},
    "vel": {"mps": 1.0, "cmps": 0.01},
    "acc": {"mps2": 1.0, "cmps2": 0.01},
    "thrust": {"n": 1.0, "gf": 0.00980665},
}

UNIT_AUTO_THRESHOLDS = {
    "pos": 1e3,
    "vel": 200.0,
    "acc": 200.0,
}
CLI_ARGUMENT_SPECS: Tuple[Tuple[str, Dict[str, object]], ...] = (
    (
        "--data",
        {
            "type": str,
            "default": lambda: os.environ.get("FORCE_ESTIMATION_DATA", DATA_PATH),
            "help": (
                "数据文件或目录路径。若为目录，将在其中搜寻包含必要列的 CSV 并合并。"
                "也可通过环境变量 FORCE_ESTIMATION_DATA 指定。"
            ),
        },
    ),
    ("--demo", {"action": "store_true", "help": "启用演示数据（忽略 --data）"}),
    ("--float32", {"action": "store_true", "help": "使用 float32 精度训练（默认 float64）。"}),
    (
        "--num-domain",
        {
            "type": int,
            "default": 0,
            "help": "每次迭代采样的 PDE 域内点数量（显式指定时关闭自动推断）。",
        },
    ),
    ("--disable-auto-domain", {"action": "store_true", "help": "禁用基于数据规模的域内点自动推断。"}),
    (
        "--domain-ratio",
        {
            "type": float,
            "default": None,
            "help": "自动推断模式下，域内点数量 = 数据行数 * ratio（默认 0.35）。",
        },
    ),
    (
        "--domain-min-points",
        {
            "type": int,
            "default": None,
            "help": "自动模式域内点数量的下界（默认 1024）。",
        },
    ),
    (
        "--domain-max-points",
        {
            "type": int,
            "default": None,
            "help": "自动模式域内点数量的上界（默认 8192）。",
        },
    ),
    (
        "--num-test",
        {
            "type": int,
            "default": None,
            "help": (
                "PDE 测试阶段随机采样的域内点数量。默认 None 表示复用训练点；"
                "0 表示与 num_domain 相同，负值表示复用训练点，正数表示显式指定。"
            ),
        },
    ),
    ("--visualize-samples", {"action": "store_true", "help": "生成域内采样点的 2D/3D 可视化（Task 2.4）。"}),
    (
        "--sample-plot-dir",
        {
            "type": str,
            "default": DEFAULT_SAMPLE_PLOT_DIR,
            "help": "采样可视化输出目录（默认 plots/sampling）。",
        },
    ),
    (
        "--sample-plot-count",
        {
            "type": int,
            "default": 4096,
            "help": "用于可视化的域内采样点数量上限（<=0 则根据数据规模自动推断）。",
        },
    ),
    (
        "--sample-plot-specs",
        {
            "type": str,
            "default": None,
            "help": (
                "自定义投影列，使用 ';' 分隔多组，格式为 'name=col1,col2[,col3]'. "
                "默认绘制 accQ_xy/accL_xy/rho_xyz/ez_world_xyz。"
            ),
        },
    ),
    (
        "--residual-variance-path",
        {
            "type": str,
            "default": DEFAULT_RESIDUAL_VARIANCE_PATH,
            "help": "残差方差统计输出 CSV 路径（留空以禁用）。",
        },
    ),
    (
        "--residual-metrics-path",
        {
            "type": str,
            "default": DEFAULT_RESIDUAL_METRICS_PATH,
            "help": "残差评估指标输出 CSV 路径（留空以禁用）。",
        },
    ),
    (
        "--residual-plot-dir",
        {
            "type": str,
            "default": DEFAULT_RESIDUAL_PLOT_DIR,
            "help": "残差分布图（直方图/箱线图）输出目录（留空以禁用）。",
        },
    ),
    (
        "--train-plot-dir",
        {
            "type": str,
            "default": DEFAULT_TRAIN_PLOT_DIR,
            "help": "保存 loss.dat/train.dat/test.dat 及训练曲线 PNG 的目录（留空以禁用）。",
        },
    ),
    (
        "--train-plot-prefix",
        {
            "type": str,
            "default": DEFAULT_TRAIN_PLOT_PREFIX,
            "help": "训练曲线 PNG 前缀（默认 train_plot_<phase>.png）。",
        },
    ),
    (
        "--export-torchscript",
        {
            "type": str,
            "default": "",
            "help": "TorchScript 模型输出路径（留空以禁用）。",
        },
    ),
    (
        "--export-onnx",
        {
            "type": str,
            "default": "",
            "help": "ONNX 模型输出路径（留空以禁用，需要安装 onnx 包）。",
        },
    ),
    (
        "--export-metadata",
        {
            "type": str,
            "default": "",
            "help": "模型元数据 JSON 输出路径（留空以禁用）。",
        },
    ),
    (
        "--export-normalizer",
        {
            "type": str,
            "default": "",
            "help": "归一化参数 JSON 输出路径（留空以禁用）。",
        },
    ),
    (
        "--residual-norm-mode",
        {
            "type": str,
            "choices": ["off", "median", "mean"],
            "default": "median",
            "help": "残差归一化策略 (Task 3.1)。设置为 off 关闭归一化。",
        },
    ),
    (
        "--residual-scale-r1",
        {
            "type": float,
            "default": None,
            "help": "覆盖 r1 (力平衡) 残差的归一尺度，默认自动按数据推断。",
        },
    ),
    (
        "--residual-scale-r2",
        {
            "type": float,
            "default": None,
            "help": "覆盖 r2 (力矩平衡) 残差的归一尺度，默认自动按数据推断。",
        },
    ),
    (
        "--residual-scale-r3",
        {
            "type": float,
            "default": None,
            "help": "覆盖 r3 (平行约束) 残差的归一尺度，默认自动按数据推断。",
        },
    ),
    (
        "--loss-weight-mode",
        {
            "type": str,
            "choices": ["manual", "inverse_residual"],
            "default": "manual",
            "help": "静态权重模式：manual 或根据残差尺度反比设置 (Task 3.2)。",
        },
    ),
    (
        "--loss-weight-r1",
        {
            "type": float,
            "default": None,
            "help": "手动指定 r1 分支的 LOSS_WEIGHT（<=0 将报错）。",
        },
    ),
    (
        "--loss-weight-r2",
        {
            "type": float,
            "default": None,
            "help": "手动指定 r2 分支的 LOSS_WEIGHT（<=0 将报错）。",
        },
    ),
    (
        "--loss-weight-r3",
        {
            "type": float,
            "default": None,
            "help": "手动指定 r3 分支的 LOSS_WEIGHT（<=0 将报错）。",
        },
    ),
    (
        "--bc-loss-weight",
        {
            "type": float,
            "default": None,
            "help": "PointSetBC 的监督权重，默认 1.0（Task 3.2）。",
        },
    ),
    (
        "--adaptive-weights",
        {
            "action": "store_true",
            "help": "启用动态权重调度器 AdaptiveWeightScheduler (Task 3.3)。",
        },
    ),
    (
        "--adaptive-weight-period",
        {
            "type": int,
            "default": 500,
            "help": "动态权重更新周期（步数，默认 500）。",
        },
    ),
    (
        "--adaptive-weight-alpha",
        {
            "type": float,
            "default": 0.5,
            "help": "EMA 平滑系数 (0-1)，越大越平滑（默认 0.5）。",
        },
    ),
    (
        "--adaptive-weight-min",
        {
            "type": float,
            "default": 1e-3,
            "help": "动态权重下限（默认 1e-3）。",
        },
    ),
    (
        "--adaptive-weight-max",
        {
            "type": float,
            "default": 10.0,
            "help": "动态权重上限（默认 10）。",
        },
    ),
    (
        "--grad-monitor",
        {
            "action": "store_true",
            "help": "启用梯度范数监控 (Task 3.4)。",
        },
    ),
    (
        "--grad-monitor-period",
        {
            "type": int,
            "default": 200,
            "help": "梯度范数监控周期（步数，默认 200）。",
        },
    ),
    (
        "--grad-monitor-log",
        {
            "type": str,
            "default": os.path.join("logs", "gradient_norm.csv"),
            "help": "梯度范数监控 CSV 输出路径。",
        },
    ),
    (
        "--grad-norm-min",
        {
            "type": float,
            "default": 1e-6,
            "help": "梯度范数告警下界（默认 1e-6）。",
        },
    ),
    (
        "--grad-norm-max",
        {
            "type": float,
            "default": 1e3,
            "help": "梯度范数告警上界（默认 1e3）。",
        },
    ),
    (
        "--grad-monitor-patience",
        {
            "type": int,
            "default": 0,
            "help": "梯度范数违反阈值后的停止耐心 (0 表示仅警告)。",
        },
    ),
    (
        "--metrics-csv",
        {
            "type": str,
            "default": os.path.join("logs", "training_metrics.csv"),
            "help": "训练监控 CSV 输出路径（留空以禁用，Task 3.0）。",
        },
    ),
    (
        "--metrics-period",
        {
            "type": int,
            "default": 100,
            "help": "监控写入周期（步数，默认 100，Task 3.0）。",
        },
    ),
    (
        "--metrics-gradients",
        {
            "action": "store_true",
            "help": "在日志中记录梯度范数（可能较慢，PyTorch 后端可用，Task 3.0）。",
        },
    ),
    ("--resample-period", {"type": int, "default": 0, "help": "PDE/BC 训练点重采样周期（0 表示禁用重采样）。"}),
    ("--bc-batch", {"type": int, "default": 0, "help": "PointSetBC 的 mini-batch 大小（仅对 PyTorch/Paddle 生效，0 表示使用全部样本）。"}),
    ("--max-points", {"type": int, "default": None, "help": "若提供，将随机抽取最多该数量的数据行用于训练。"}),
    (
        "--sample-size",
        {
            "type": int,
            "default": None,
            "help": "显式指定用于训练的随机子样本大小（<=数据量时生效，与 --seed 配合以保证可复现）。",
        },
    ),
    ("--seed", {"type": int, "default": 42, "help": "控制数据采样与训练初始化的随机种子。"}),
    ("--norm-cache", {"type": str, "default": "normalizer_stats.pkl", "help": "归一化统计缓存文件路径（Task 1.2 将启用）。"}),
    ("--dataset-cache", {"type": str, "default": None, "help": "数据集缓存文件路径，用于复用随机采样后的训练数据。"}),
    (
        "--clip-quantiles",
        {
            "type": float,
            "nargs": 2,
            "metavar": ("LOWER", "UPPER"),
            "default": DEFAULT_CLIP_QUANTILES,
            "help": "分位数截断范围 (默认 0.01 0.99)。设置为 0 1 可等效关闭。",
        },
    ),
    (
        "--unit-profile",
        {
            "type": str,
            "choices": ["auto", "si", "cgs"],
            "default": "auto",
            "help": "单位配置：auto/si/cgs。可被 --pos-unit/--vel-unit/--acc-unit 覆盖。",
        },
    ),
    (
        "--pos-unit",
        {
            "type": str,
            "choices": ["auto", "m", "cm"],
            "default": "auto",
            "help": "位置列原始单位（auto/m/cm）。",
        },
    ),
    (
        "--vel-unit",
        {
            "type": str,
            "choices": ["auto", "mps", "cmps"],
            "default": "auto",
            "help": "速度列原始单位（auto/mps/cmps）。",
        },
    ),
    (
        "--acc-unit",
        {
            "type": str,
            "choices": ["auto", "mps2", "cmps2"],
            "default": "auto",
            "help": "加速度列原始单位（auto/mps2/cmps2）。",
        },
    ),
    (
        "--disable-fix-unit-vectors",
        {
            "action": "store_true",
            "help": "禁用方向向量单位化（默认启用）。",
        },
    ),
    (
        "--recompute-rho",
        {
            "action": "store_true",
            "help": "在单位化前根据 (xL - xQ) 重算 rho。",
        },
    ),
    (
        "--thrust-unit",
        {
            "type": str,
            "choices": ["auto", "n", "gf"],
            "default": "auto",
            "help": "推力列原始单位（auto/N/gf），默认为自动推断。",
        },
    ),
    (
        "--thrust-from",
        {
            "type": str,
            "default": "thrust",
            "help": "作为推力来源的列名，默认使用 `thrust` 列，可指定其他列。",
        },
    ),
    (
        "--sqrt-kf-col",
        {
            "type": str,
            "default": "sqrt_kf",
            "help": "当推力来自电机指令时，指定 sqrt_kf 列用于平方放大。",
        },
    ),
    (
        "--motor-cmd-col",
        {
            "type": str,
            "default": None,
            "help": "电机指令/控制量列名，与 sqrt_kf 联合用于推力派生。",
        },
    ),
    (
        "--thrust-balance-tol",
        {
            "type": float,
            "default": 0.25,
            "help": "推力与重量比的容差阈值（默认 0.25）。",
        },
    ),
    ("--skip-unit-check", {"action": "store_true", "help": "跳过单位与量纲检查（Task 1.4）。"}),
    ("--skip-clip", {"action": "store_true", "help": "跳过分位数截断（Task 1.5）。"}),
    ("--early-patience", {"type": int, "default": 0, "help": "EarlyStopping 的耐心轮数（0 表示禁用早停）。"}),
    ("--early-min-delta", {"type": float, "default": 0.0, "help": "EarlyStopping 判定改进的最小提升幅度。"}),
    (
        "--adam-iters",
        {"type": int, "default": None, "help": "覆盖默认的 Adam 迭代次数（用于快速对比实验）。"},
    ),
    (
        "--adam-lr",
        {"type": float, "default": None, "help": "覆盖默认的 Adam 学习率。"},
    ),
    ("--disable-lbfgs", {"action": "store_true", "help": "禁用 L-BFGS 微调阶段。"}),
    ("--disable-mixed-geometry", {"action": "store_true", "help": "禁用混合几何采样，回退至高维 Hypercube。"}),
    (
        "--geometry-mix-ratio",
        {
            "type": float,
            "default": None,
            "help": "混合几何使用插值样本的比例（0-1，默认 0.6）。",
        },
    ),
    (
        "--geometry-jitter-scale",
        {
            "type": float,
            "default": None,
            "help": "插值样本抖动幅度，相对于原始列范围（默认 0.02）。",
        },
    ),
)

# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"数据文件缺少必要列: {missing}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN force estimation for UAV-payload system")
    for flag, kwargs in CLI_ARGUMENT_SPECS:
        params = dict(kwargs)
        default = params.get("default")
        if callable(default):
            params["default"] = default()
        parser.add_argument(flag, **params)
    return parser


def _print_config(pairs: Iterable[Tuple[str, object]]) -> None:
    summary = ", ".join(f"{key}={value}" for key, value in pairs)
    print(f"[force_estimation] {summary}")


DEFAULT_SAMPLE_PLOT_SPECS: Tuple[SamplingPlotSpec, ...] = (
    SamplingPlotSpec("accQ_xy", ("accQ_x", "accQ_y")),
    SamplingPlotSpec("accL_xy", ("accL_x", "accL_y")),
    SamplingPlotSpec("rho_xyz", ("rho_x", "rho_y", "rho_z")),
    SamplingPlotSpec("ez_world_xyz", ("ez_world_x", "ez_world_y", "ez_world_z")),
)


def _summarize_losses(sequences) -> np.ndarray:
    if not sequences:
        return np.zeros((0,), dtype=np.float64)
    totals: List[float] = []
    for entry in sequences:
        arr = np.asarray(entry, dtype=np.float64).reshape(-1)
        totals.append(float(np.sum(arr))) if arr.size else totals.append(0.0)
    return np.asarray(totals, dtype=np.float64)


def _plot_loss_history_png(loss_history, output_path: str, title: str) -> None:
    if not output_path:
        return
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    steps = np.asarray(getattr(loss_history, "steps", []), dtype=np.float64)
    train_loss = _summarize_losses(getattr(loss_history, "loss_train", []))
    test_loss = _summarize_losses(getattr(loss_history, "loss_test", []))
    if steps.size == 0 or steps.size != train_loss.size:
        steps = np.arange(train_loss.size, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if train_loss.size:
        ax.semilogy(steps[: train_loss.size], train_loss, label="Train loss")
    if test_loss.size:
        test_steps = steps if test_loss.size == steps.size else np.arange(test_loss.size, dtype=np.float64)
        ax.semilogy(test_steps, test_loss, label="Test loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def save_training_phase_outputs(
    phase: str,
    loss_history,
    train_state,
    *,
    data_root: str,
    plot_path: str,
) -> str:
    if loss_history is None or train_state is None:
        return ""
    phase_slug = (phase or "phase").lower()
    saved_plot = ""
    if data_root:
        phase_dir = os.path.join(os.path.abspath(data_root), phase_slug)
        os.makedirs(phase_dir, exist_ok=True)
        dde.saveplot(
            loss_history,
            train_state,
            issave=True,
            isplot=False,
            output_dir=phase_dir,
        )
    if plot_path:
        phase_title = f"{phase_slug.upper()} loss history"
        _plot_loss_history_png(loss_history, plot_path, phase_title)
        print(f"[force_estimation] 训练曲线已保存: {plot_path}")
        saved_plot = plot_path
    return saved_plot


def _extract_group_samples(residual_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    if residual_matrix.size == 0:
        return {"r1": np.zeros((0,)), "r2": np.zeros((0,)), "r3": np.zeros((0,))}
    groups = {
        "r1": residual_matrix[:, 0:3].reshape(-1),
        "r2": residual_matrix[:, 3:6].reshape(-1),
        "r3": residual_matrix[:, 6:7].reshape(-1),
    }
    clean = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.float64)
        clean[key] = arr[np.isfinite(arr)]
    return clean


def generate_residual_distribution_plots(
    residual_matrix: np.ndarray,
    output_dir: str,
    *,
    bins: int = 50,
) -> List[str]:
    if residual_matrix.size == 0 or not output_dir:
        return []
    os.makedirs(output_dir, exist_ok=True)
    groups = _extract_group_samples(residual_matrix)
    paths: List[str] = []

    hist_path = os.path.join(output_dir, "residual_hist.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (label, values) in zip(axes, groups.items()):
        if values.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.hist(values, bins=bins, color="#1f77b4", alpha=0.85)
        ax.set_title(f"{label.upper()} residuals")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=240)
    plt.close(fig)
    paths.append(hist_path)

    box_path = os.path.join(output_dir, "residual_box.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [groups["r1"], groups["r2"], groups["r3"]],
        labels=["R1", "R2", "R3"],
        showfliers=False,
    )
    ax.set_ylabel("Residual")
    ax.set_title("Residual distribution")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(box_path, dpi=240)
    plt.close(fig)
    paths.append(box_path)
    return paths

def _sanitize_positive(value: float, label: str) -> float:
    val = float(value)
    if val <= 0:
        raise ValueError(f"{label} 必须为正数，收到 {value}.")
    return val


def resolve_loss_weight_dict(
    args,
    residual_scaler: ResidualScaler | None,
) -> Dict[str, float]:
    weights = dict(LOSS_WEIGHTS)
    overrides = {
        "r1": args.loss_weight_r1,
        "r2": args.loss_weight_r2,
        "r3": args.loss_weight_r3,
    }
    for key, override in overrides.items():
        if override is not None:
            weights[key] = _sanitize_positive(override, f"--loss-weight-{key}")
    mode = (args.loss_weight_mode or "manual").lower()
    if mode == "inverse_residual":
        if residual_scaler is None:
            print(
                "[force_estimation] residual_norm_mode=off，无法依据残差尺度自动设置 LOSS_WEIGHTS，回退 manual。"
            )
        else:
            inv = {
                key: 1.0 / residual_scaler.group_scales[key] for key in weights
            }
            inv_sum = sum(inv.values())
            base_sum = sum(weights.values())
            scale = base_sum / inv_sum if inv_sum > 0 else 1.0
            for key in weights:
                weights[key] = inv[key] * scale
    return weights


def resolve_bc_loss_weight(default_weight: float, override: float | None) -> float:
    if override is None:
        return default_weight
    return _sanitize_positive(override, "--bc-loss-weight")


def expand_pde_loss_weights(weight_dict: Dict[str, float]) -> List[float]:
    return (
        [weight_dict["r1"]] * 3
        + [weight_dict["r2"]] * 3
        + [weight_dict["r3"]]
    )


def build_loss_weight_vector(
    num_bcs: int,
    pde_weights: Sequence[float],
    bc_weight: float,
) -> List[float]:
    weights = list(pde_weights)
    if num_bcs > 0:
        weights.extend([bc_weight] * num_bcs)
    return weights


def resolve_sample_plot_specs(raw: str | None) -> List[SamplingPlotSpec]:
    return geom_resolve_sample_plot_specs(
        raw,
        feature_columns=FEATURE_COLUMNS,
        default_specs=DEFAULT_SAMPLE_PLOT_SPECS,
    )


def generate_sampling_visualizations(
    domain_samples: np.ndarray,
    reference_samples: np.ndarray | None,
    specs: Sequence[SamplingPlotSpec],
    output_dir: str,
) -> List[str]:
    return geom_generate_sampling_visualizations(
        domain_samples,
        reference_samples,
        specs,
        output_dir,
        feature_columns=FEATURE_COLUMNS,
    )


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
    cache_path: str | None, dtype: np.dtype
) -> Normalizer | None:
    stats = _read_pickle(cache_path, "归一化缓存", "将重新计算统计量。")
    if stats is None:
        return None
    feature_cols = stats.get("feature_columns")
    if feature_cols and list(feature_cols) != FEATURE_COLUMNS:
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


def _save_normalizer_cache(cache_path: str | None, normalizer: Normalizer) -> None:
    stats = {
        "minimum": normalizer.minimum.tolist(),
        "range": normalizer.range.tolist(),
        "eps": normalizer.eps,
        "feature_columns": FEATURE_COLUMNS,
    }
    _write_pickle(cache_path, "normalizer stats", stats)


def _describe_data_source(path: str, use_demo: bool) -> str:
    if use_demo or not os.path.exists(path):
        return "__demo__"
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        return abs_path
    csv_files = sorted(glob.glob(os.path.join(abs_path, "*.csv")))
    if csv_files:
        return "|".join(os.path.abspath(fp) for fp in csv_files)
    return abs_path


def _normalize_optional_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_dataset_cache_frame(
    cache_path: str | None,
    *,
    expected_meta: Dict[str, object],
    seed: int | None,
) -> Tuple[pd.DataFrame, Dict[str, object]] | None:
    payload = _read_pickle(cache_path, "数据缓存", "将重新构建数据集。")
    if payload is None:
        return None
    meta = payload.get("meta") or {}
    comparators: Dict[str, Tuple[Callable[[object, object], bool], str]] = {
        "data_source": (lambda cur, exp: cur == exp, "数据源不同"),
        "feature_columns": (
            lambda cur, exp: list(cur or []) == exp,
            "特征列不匹配",
        ),
        "sample_size": (
            lambda cur, exp: _normalize_optional_int(cur) == exp,
            "sample_size 不一致",
        ),
        "max_points": (
            lambda cur, exp: _normalize_optional_int(cur) == exp,
            "max_points 不一致",
        ),
        "use_demo": (lambda cur, exp: bool(cur) == bool(exp), "demo 模式不同"),
        "clip_range": (
            lambda cur, exp: (tuple(cur) if cur else None) == exp,
            "clip_range 不一致",
        ),
        "unit_profile_arg": (
            lambda cur, exp: (cur or "auto") == exp,
            "unit_profile 不一致",
        ),
        "pos_unit_arg": (
            lambda cur, exp: (cur or "auto") == exp,
            "pos_unit 不一致",
        ),
        "vel_unit_arg": (
            lambda cur, exp: (cur or "auto") == exp,
            "vel_unit 不一致",
        ),
        "acc_unit_arg": (
            lambda cur, exp: (cur or "auto") == exp,
            "acc_unit 不一致",
        ),
        "fix_unit_vectors": (
            lambda cur, exp: bool(cur) == bool(exp),
            "fix_unit_vectors 不一致",
        ),
        "recompute_rho": (
            lambda cur, exp: bool(cur) == bool(exp),
            "recompute_rho 不一致",
        ),
        "thrust_unit_arg": (
            lambda cur, exp: (cur or "auto") == exp,
            "thrust_unit 不一致",
        ),
        "thrust_from": (
            lambda cur, exp: (cur or "thrust") == exp,
            "thrust_from 不一致",
        ),
        "sqrt_kf_col": (
            lambda cur, exp: (cur or None) == exp,
            "sqrt_kf_col 不一致",
        ),
        "motor_cmd_col": (
            lambda cur, exp: (cur or None) == exp,
            "motor_cmd_col 不一致",
        ),
    }
    mismatches = [
        reason
        for key, (cmp_fn, reason) in comparators.items()
        if key in expected_meta and not cmp_fn(meta.get(key), expected_meta[key])
    ]
    if bool(meta.get("sampling_method")):
        cached_seed = _normalize_optional_int(meta.get("seed"))
        if cached_seed != _normalize_optional_int(seed):
            mismatches.append("seed 不一致")
    if mismatches:
        print(
            "[force_estimation] 数据缓存元数据不匹配({})，将重新构建数据集。".format(
                "，".join(mismatches)
            )
        )
        return None
    df = payload.get("dataframe")
    if not isinstance(df, pd.DataFrame):
        print(
            f"[force_estimation] 数据缓存 {cache_path} 内容无效，将重新构建数据集。"
        )
        return None
    print(f"[force_estimation] Loaded cached dataset from {cache_path}")
    return df.copy(deep=True), dict(meta)


def _save_dataset_cache_frame(
    cache_path: str | None,
    df: pd.DataFrame,
    *,
    meta: Dict[str, object],
) -> None:
    payload = {
        "meta": meta,
        "dataframe": df.copy(deep=True),
    }
    _write_pickle(cache_path, "dataset cache", payload)


def _auto_detect_unit(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float,
    alt_unit: str,
    default_unit: str,
) -> str:
    available = [col for col in columns if col in df.columns]
    if not available:
        return default_unit
    values = df[available].to_numpy(dtype=np.float64, copy=False)
    if values.size == 0:
        return default_unit
    max_abs = float(np.nanmax(np.abs(values)))
    return alt_unit if max_abs > threshold else default_unit


def _resolve_unit_selection(
    df: pd.DataFrame,
    unit_profile: str,
    pos_unit: str,
    vel_unit: str,
    acc_unit: str,
) -> Dict[str, str]:
    profile = (unit_profile or "auto").lower()
    pos_arg = (pos_unit or "auto").lower()
    vel_arg = (vel_unit or "auto").lower()
    acc_arg = (acc_unit or "auto").lower()
    defaults = UNIT_PROFILE_DEFAULTS.get(profile, {})

    def resolve(arg: str, kind: str, columns: List[str]) -> str:
        if arg != "auto":
            return arg
        profile_default = defaults.get(kind)
        if profile_default and profile_default != "auto":
            return profile_default
        threshold = UNIT_AUTO_THRESHOLDS[kind]
        alt = "cm" if kind == "pos" else ("cmps" if kind == "vel" else "cmps2")
        default = "m" if kind == "pos" else ("mps" if kind == "vel" else "mps2")
        return _auto_detect_unit(df, columns, threshold, alt, default)

    selected = {
        "profile": profile,
        "pos": resolve(pos_arg, "pos", POSITION_COLUMNS),
        "vel": resolve(vel_arg, "vel", VELOCITY_COLUMNS),
        "acc": resolve(acc_arg, "acc", ACCELERATION_COLUMNS),
        "pos_arg": pos_arg,
        "vel_arg": vel_arg,
        "acc_arg": acc_arg,
        "profile_arg": profile,
    }
    return selected


def apply_unit_conversions(
    df: pd.DataFrame,
    unit_selection: Dict[str, str],
    *,
    thrust_unit: str,
    thrust_source: str,
    sqrt_kf_col: str | None,
    motor_cmd_col: str | None,
) -> Dict[str, object]:
    resolved_thrust_unit = (thrust_unit or "auto").lower()
    meta: Dict[str, object] = {
        "unit_profile_selected": unit_selection["profile"],
        "pos_unit_selected": unit_selection["pos"],
        "vel_unit_selected": unit_selection["vel"],
        "acc_unit_selected": unit_selection["acc"],
        "thrust_unit_selected": resolved_thrust_unit,
        "thrust_source": thrust_source,
    }

    def scale_columns(columns: List[str], kind: str) -> None:
        unit = unit_selection[kind]
        scale = UNIT_SCALE[kind][unit]
        if abs(scale - 1.0) < 1e-12:
            return
        available = [col for col in columns if col in df.columns]
        if not available:
            return
        df.loc[:, available] = df[available].mul(scale)

    scale_columns(POSITION_COLUMNS, "pos")
    scale_columns(VELOCITY_COLUMNS, "vel")
    scale_columns(ACCELERATION_COLUMNS, "acc")

    def update_thrust(new_values: np.ndarray) -> None:
        if "thrust" not in df.columns:
            return
        df.loc[:, "thrust"] = new_values

    if resolved_thrust_unit in ("n", "newton"):
        resolved_thrust_unit = "n"
    elif resolved_thrust_unit == "gf":
        scale = UNIT_SCALE["thrust"]["gf"]
        if "thrust" in df.columns:
            df.loc[:, "thrust"] = df["thrust"].to_numpy(dtype=np.float64) * scale
        print("[force_estimation] Converted thrust from gf to N (scale=0.00980665)")
    elif resolved_thrust_unit == "auto":
        if "thrust" in df.columns:
            thrust_vals = df["thrust"].to_numpy(dtype=np.float64)
            max_abs = np.nanmax(np.abs(thrust_vals)) if thrust_vals.size else 0.0
            if max_abs > 1e3:
                df.loc[:, "thrust"] = thrust_vals * UNIT_SCALE["thrust"]["gf"]
                print(
                    "[force_estimation] thrust auto-detected as gf; converted to N"
                )
                resolved_thrust_unit = "gf"
            else:
                resolved_thrust_unit = "n"
    else:
        print(
            f"[force_estimation] 未知推力单位 {thrust_unit}，将假定已在 N 单位。"
        )
        resolved_thrust_unit = "unknown"

    derived = None
    source_lower = (thrust_source or "thrust").lower()
    if source_lower != "thrust":
        if thrust_source not in df.columns:
            raise ValueError(
                f"推力来源列 {thrust_source} 不存在，无法派生推力。"
            )
        base = df[thrust_source].to_numpy(dtype=np.float64)
        if motor_cmd_col and motor_cmd_col in df.columns:
            cmd = df[motor_cmd_col].to_numpy(dtype=np.float64)
        else:
            cmd = base
        if sqrt_kf_col and sqrt_kf_col in df.columns:
            sqrt_kf = df[sqrt_kf_col].to_numpy(dtype=np.float64)
            derived = (sqrt_kf * cmd) ** 2
        else:
            derived = cmd ** 2
        update_thrust(derived)
        meta["thrust_source"] = thrust_source
        resolved_thrust_unit = "derived"

    print(
        "[force_estimation] Unit profile resolved: pos={}, vel={}, acc={}, thrust_unit={}".format(
            unit_selection["pos"],
            unit_selection["vel"],
            unit_selection["acc"],
            resolved_thrust_unit,
        )
    )
    meta["thrust_unit_selected"] = resolved_thrust_unit
    return meta


def _recompute_rho_from_positions(df: pd.DataFrame) -> np.ndarray | None:
    required = ["xL_x", "xL_y", "xL_z", "xQ_x", "xQ_y", "xQ_z"]
    if not set(required).issubset(df.columns):
        return None
    xl = df[["xL_x", "xL_y", "xL_z"]].to_numpy(dtype=np.float64, copy=False)
    xq = df[["xQ_x", "xQ_y", "xQ_z"]].to_numpy(dtype=np.float64, copy=False)
    return xl - xq


def normalize_vector_columns(
    df: pd.DataFrame,
    columns: List[str],
    *,
    recompute_values: np.ndarray | None = None,
    eps: float = 1e-8,
) -> None:
    if not set(columns).issubset(df.columns):
        return
    values = (
        recompute_values
        if recompute_values is not None
        else df[columns].to_numpy(dtype=np.float64, copy=False)
    )
    if values.size == 0:
        return
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    normalized = values / norms
    df.loc[:, columns] = normalized


class MixedFeatureSampler:
    """Generate domain samples via low-dimensional (time) geometry + empirical mixing."""

    def __init__(
        self,
        df: pd.DataFrame,
        normalizer: Normalizer,
        feature_columns: List[str],
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
        self._feature_min = feature_min
        self._feature_max = feature_max
        feature_range = np.maximum(feature_max - feature_min, 1e-8)
        self._jitter_amplitude = feature_range * jitter_scale

    def sample(self, n: int, random: str = "pseudo") -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0, len(self._feature_columns)), dtype=self._dtype)
        n_geom = int(np.round(self._mix_ratio * n)) if self._allow_interpolation else 0
        n_geom = min(n_geom, n)
        n_empirical = n - n_geom
        batches: List[np.ndarray] = []
        if n_geom > 0:
            times = self._sample_times(n_geom, random)
            interpolated = self._interpolate(times)
            interpolated = self._apply_jitter(interpolated)
            batches.append(interpolated)
        if n_empirical > 0:
            idx = self._rng.integers(0, len(self._raw_features), size=n_empirical)
            batches.append(self._raw_features[idx].astype(self._dtype, copy=False))
        mixed = (
            np.vstack(batches).astype(self._dtype, copy=False)
            if len(batches) > 1
            else batches[0].astype(self._dtype, copy=False)
        )
        self._rng.shuffle(mixed)
        return self._normalizer.transform(mixed.astype(self._dtype, copy=False))

    def _sample_times(self, n: int, random: str) -> np.ndarray:
        if self._time_geom is None:
            raise RuntimeError("混合几何插值不可用时不应调用 _sample_times。")
        samples = self._time_geom.random_points(n, random=random)
        return samples.reshape(-1)

    def _interpolate(self, times: np.ndarray) -> np.ndarray:
        out = np.empty((len(times), len(self._feature_columns)), dtype=np.float64)
        for col_idx in range(self._feature_history.shape[1]):
            out[:, col_idx] = np.interp(times, self._times, self._feature_history[:, col_idx])
        return out

    def _apply_jitter(self, values: np.ndarray) -> np.ndarray:
        if not np.any(self._jitter_amplitude > 0):
            return values.astype(np.float64, copy=False)
        noise = self._rng.normal(
            loc=0.0, scale=self._jitter_amplitude, size=values.shape
        )
        jittered = values + noise
        jittered = np.clip(jittered, self._feature_min, self._feature_max)
        return jittered


class MixedGeometry(dde.geometry.geometry_nd.Hypercube):
    """Hypercube geometry with customizable random sampling."""

    def __init__(
        self,
        lower: List[float],
        upper: List[float],
        sampler: MixedFeatureSampler | None = None,
    ):
        super().__init__(lower, upper)
        self._sampler = sampler

    def random_points(self, n, random="pseudo"):
        if self._sampler is None or int(n) <= 0:
            return super().random_points(n, random=random)
        return self._sampler.sample(int(n), random=random)


def _fill_feature_matrix(
    matrix: np.ndarray,
    assignments: Iterable[Tuple[int | slice, np.ndarray | float]],
) -> None:
    for idx, values in assignments:
        arr = np.asarray(values, dtype=matrix.dtype)
        if arr.ndim == 1:
            arr = arr[:, None]
        selector = idx if isinstance(idx, slice) else slice(idx, idx + 1)
        matrix[:, selector] = arr


def _apply_sampling(
    df: pd.DataFrame,
    sample_size: int | None,
    max_points: int | None,
    seed: int | None,
) -> Tuple[pd.DataFrame, str | None]:
    effective_seed = seed if seed is not None else 42
    sort_reset = lambda frame: frame.sort_values("time").reset_index(drop=True)
    sampling_method: str | None = None
    if sample_size and sample_size > 0:
        n_rows = min(int(sample_size), len(df))
        if n_rows < len(df):
            df = sort_reset(df.sample(n=n_rows, random_state=effective_seed))
            sampling_method = "sample_size"
        else:
            df = sort_reset(df)
    elif max_points and len(df) > max_points:
        df = sort_reset(df.sample(n=int(max_points), random_state=effective_seed))
        sampling_method = "max_points"
    else:
        df = sort_reset(df)
    return df, sampling_method


def _format_metric(metric: object) -> str:
    if isinstance(metric, tuple):
        return f"min={metric[0]:.3f}, max={metric[1]:.3f}"
    if isinstance(metric, float):
        return f"{metric:.6g}"
    return str(metric)


def _nan_safe_max(values: np.ndarray) -> float:
    try:
        return float(np.nanmax(values))
    except ValueError:
        return float("nan")


def _check_max_abs(
    df: pd.DataFrame, columns: List[str], limit: float, label: str
) -> Dict[str, object]:
    array = df[columns].to_numpy(dtype=np.float64, copy=False)
    metric = _nan_safe_max(np.abs(array))
    passed = metric <= limit
    return {
        "name": label,
        "metric": metric,
        "threshold": f"<= {limit}",
        "passed": bool(passed),
    }


def _check_unit_vector(
    df: pd.DataFrame, columns: List[str], tol: float, label: str
) -> Dict[str, object]:
    array = df[columns].to_numpy(dtype=np.float64, copy=False)
    norms = np.linalg.norm(array, axis=1)
    metric = _nan_safe_max(np.abs(norms - 1.0))
    passed = metric <= tol
    return {
        "name": label,
        "metric": metric,
        "threshold": f"|norm-1| <= {tol}",
        "passed": bool(passed),
    }


def _check_range(
    series: pd.Series, lower: float, upper: float, label: str
) -> Dict[str, object]:
    min_val = float(series.min())
    max_val = float(series.max())
    passed = (min_val >= lower) and (max_val <= upper)
    return {
        "name": label,
        "metric": (min_val, max_val),
        "threshold": f"[{lower}, {upper}]",
        "passed": bool(passed),
    }


def _check_gravity(series: pd.Series, target: float, tol: float) -> Dict[str, object]:
    mean_val = float(series.mean())
    delta = abs(mean_val - target)
    passed = delta <= tol
    return {
        "name": "Gravity (g)",
        "metric": mean_val,
        "threshold": f"{target}±{tol}",
        "passed": bool(passed),
    }


def _check_thrust_balance(df: pd.DataFrame, tol: float) -> Dict[str, object]:
    thrust = df["thrust"].to_numpy(dtype=np.float64, copy=False)
    m_total = (df["m_Q"] + df["m_L"]).to_numpy(dtype=np.float64, copy=False)
    gravity = df["g"].to_numpy(dtype=np.float64, copy=False)
    expected = m_total * gravity
    mask = expected != 0
    ratio = np.zeros_like(thrust)
    ratio[mask] = thrust[mask] / expected[mask]
    metric = _nan_safe_max(np.abs(ratio[mask] - 1.0)) if np.any(mask) else float("nan")
    passed = metric <= tol if not np.isnan(metric) else False
    return {
        "name": "Thrust vs (m_Q+m_L)g",
        "metric": metric,
        "threshold": f"|ratio-1| <= {tol}",
        "passed": bool(passed),
    }


BASE_UNIT_CHECKS: Tuple[Callable[[pd.DataFrame], Dict[str, object]], ...] = (
    lambda df: _check_max_abs(
        df, ["xQ_x", "xQ_y", "xQ_z"], 10.0, "Quadrotor position (m)"
    ),
    lambda df: _check_max_abs(
        df, ["xL_x", "xL_y", "xL_z"], 10.0, "Payload position (m)"
    ),
    lambda df: _check_max_abs(
        df, ["vQ_x", "vQ_y", "vQ_z", "vL_x", "vL_y", "vL_z"], 12.0, "Velocity (m/s)"
    ),
    lambda df: _check_max_abs(
        df,
        ["accQ_x", "accQ_y", "accQ_z", "accL_x", "accL_y", "accL_z"],
        60.0,
        "Acceleration (m/s^2)",
    ),
    lambda df: _check_range(df["m_Q"], 0.1, 10.0, "m_Q (kg)"),
    lambda df: _check_range(df["m_L"], 0.01, 10.0, "m_L (kg)"),
    lambda df: _check_gravity(df["g"], 9.81, 0.5),
)


def run_unit_consistency_checks(
    df: pd.DataFrame, tol_unit_vector: float = 5e-3, thrust_tol: float = 0.25
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = [builder(df) for builder in BASE_UNIT_CHECKS]
    for cols, label in (
        (["rho_x", "rho_y", "rho_z"], "rho unit vector"),
        (["ez_world_x", "ez_world_y", "ez_world_z"], "e_z unit vector"),
    ):
        checks.append(_check_unit_vector(df, cols, tol_unit_vector, label))
    checks.append(_check_thrust_balance(df, thrust_tol))
    passed = all(item["passed"] for item in checks)
    status = "PASS" if passed else "WARN"
    print(f"[force_estimation][unit-check] status={status}")
    for item in checks:
        label = "PASS" if item["passed"] else "WARN"
        print(
            f"  - {label:<4} {item['name']}: {_format_metric(item['metric'])} (expect {item['threshold']})"
        )
    if not passed:
        print(
            "[force_estimation][unit-check] 警告: 检测到潜在的单位或量纲问题，请检查数据来源。"
        )
    return {"passed": passed, "checks": checks}


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


def build_observation_bcs(
    df: pd.DataFrame,
    anchors: np.ndarray,
    batch_size: int | None = None,
    dtype: np.dtype = np.float64,
):
    """根据可选观测列生成 PointSetBC."""
    bcs: List[dde.icbc.PointSetBC] = []
    for name, meta in OBSERVATION_COLUMNS.items():
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


def load_dataset(
    path: str,
    use_demo: bool,
    max_points: int | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    norm_cache: str | None = None,
    dataset_cache: str | None = None,
    clip_quantiles: Tuple[float, float] | None = DEFAULT_CLIP_QUANTILES,
    unit_profile: str = "auto",
    pos_unit: str = "auto",
    vel_unit: str = "auto",
    acc_unit: str = "auto",
    fix_unit_vectors: bool = True,
    recompute_rho: bool = False,
    thrust_unit: str = "auto",
    thrust_from: str = "thrust",
    sqrt_kf_col: str | None = "sqrt_kf",
    motor_cmd_col: str | None = None,
    dtype: np.dtype = np.float64,
) -> tuple[pd.DataFrame, np.ndarray, Normalizer]:
    dtype = np.dtype(dtype)
    norm_cache_path = _expand_cache_path(norm_cache)
    dataset_cache_path = _expand_cache_path(dataset_cache)
    use_demo_mode = use_demo or not os.path.exists(path)
    data_signature = _describe_data_source(path, use_demo_mode)
    clip_range = tuple(clip_quantiles) if clip_quantiles else None
    expected_meta = {
        "data_source": data_signature,
        "sample_size": _normalize_optional_int(sample_size),
        "max_points": _normalize_optional_int(max_points),
        "use_demo": bool(use_demo_mode),
        "feature_columns": FEATURE_COLUMNS,
        "clip_range": clip_range,
        "unit_profile_arg": unit_profile,
        "pos_unit_arg": pos_unit,
        "vel_unit_arg": vel_unit,
        "acc_unit_arg": acc_unit,
        "fix_unit_vectors": bool(fix_unit_vectors),
        "recompute_rho": bool(recompute_rho),
        "thrust_unit_arg": thrust_unit,
        "thrust_from": thrust_from,
        "sqrt_kf_col": sqrt_kf_col,
        "motor_cmd_col": motor_cmd_col,
    }
    cached_meta: Dict[str, object] | None = None
    cache_result = _load_dataset_cache_frame(
        dataset_cache_path,
        expected_meta=expected_meta,
        seed=seed,
    )
    if cache_result is not None:
        df, cached_meta = cache_result
    else:
        df = None
    sampling_method: str | None = None
    unit_conversion_meta: Dict[str, object] | None = None
    if df is None:
        if use_demo_mode:
            n = 256  # 采样点数，可调
            t = np.linspace(0.0, 12.0, n, dtype=np.float64)
            dt = t[1] - t[0]

            x_q = np.stack(
                [
                    0.8 * np.sin(0.5 * t),
                    0.8 * np.cos(0.5 * t),
                    2.5 + 0.15 * np.sin(t),
                ],
                axis=1,
            )
            v_q = np.gradient(x_q, dt, axis=0, edge_order=2)
            acc_q = np.gradient(v_q, dt, axis=0, edge_order=2)

            omega_b = np.stack(
                [
                    0.4 * np.sin(0.3 * t),
                    0.6 * np.cos(0.4 * t),
                    -0.8 * np.sin(0.2 * t),
                ],
                axis=1,
            )

            x_l = x_q + np.array([0.0, 0.0, -0.8]) + 0.1 * np.sin(0.8 * t)[:, None]
            v_l = np.gradient(x_l, dt, axis=0, edge_order=2)
            acc_l = np.gradient(v_l, dt, axis=0, edge_order=2)

            cable_vec = x_l - x_q
            cable_norm = np.linalg.norm(cable_vec, axis=1, keepdims=True)
            cable_norm = np.where(cable_norm < 1e-6, 1.0, cable_norm)
            rho = cable_vec / cable_norm

            yaw = 0.2 * np.sin(0.5 * t)
            pitch = 0.1 * np.cos(0.4 * t)
            roll = 0.05 * np.sin(0.3 * t)
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            ez_world = np.stack(
                [
                    cy * cp,
                    sy * cp,
                    -sp,
                ],
                axis=1,
            )

            thrust = 14.0 + 1.5 * np.sin(0.7 * t)
            m_q = np.full(n, 1.15)
            m_l = np.full(n, 0.285)
            gravity = np.full(n, 9.81)
            l_length = np.full(n, 0.68)
            sqrt_kf = np.full(n, 9.5e-05)

            wind = np.stack(
                [
                    0.3 * np.sin(0.2 * t),
                    0.4 * np.cos(0.3 * t),
                    -0.2 * np.sin(0.4 * t),
                ],
                axis=1,
            )

            feature_matrix = np.zeros((n, len(FULL_FEATURE_COLUMNS)), dtype=np.float64)
            _fill_feature_matrix(
                feature_matrix,
                [
                    (IDX_TIME_FULL, t),
                    (IDX_XQ_FULL, x_q),
                    (IDX_VQ_FULL, v_q),
                    (IDX_AQ_FULL, acc_q),
                    (IDX_OMEGA_FULL, omega_b),
                    (IDX_XL_FULL, x_l),
                    (IDX_VL_FULL, v_l),
                    (IDX_AL_FULL, acc_l),
                    (IDX_RHO_FULL, rho),
                    (IDX_EZ_FULL, ez_world),
                    (IDX_THRUST_FULL, thrust),
                    (IDX_MQ_FULL, m_q),
                    (IDX_ML_FULL, m_l),
                    (IDX_G_FULL, gravity),
                    (IDX_L_LENGTH_FULL, l_length),
                    (IDX_SQRT_KF_FULL, sqrt_kf),
                    (IDX_WIND_FULL, wind),
                ],
            )
            feature_matrix = feature_matrix.astype(dtype, copy=False)
            df = pd.DataFrame(feature_matrix, columns=FULL_FEATURE_COLUMNS)
            f_q = (0.01 * np.sin(0.5 * t[:, None])).astype(dtype, copy=False)
            f_l = (0.01 * np.cos(0.6 * t[:, None])).astype(dtype, copy=False)
            df[["fQ_x", "fQ_y", "fQ_z"]] = np.tile(f_q, (1, 3))
            df[["fL_x", "fL_y", "fL_z"]] = np.tile(f_l, (1, 3))
        else:
            if os.path.isdir(path):
                csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
                if not csv_files:
                    raise FileNotFoundError(f"目录中未找到 CSV 文件: {path}")
                dfs: List[pd.DataFrame] = []
                missing_reports: List[str] = []
                for fp in csv_files:
                    try:
                        tmp = pd.read_csv(fp)
                    except Exception as e:
                        missing_reports.append(f"读取失败 {fp}: {e}")
                        continue
                    if set(FULL_FEATURE_COLUMNS).issubset(tmp.columns):
                        dfs.append(tmp)
                    else:
                        miss = sorted(set(FULL_FEATURE_COLUMNS) - set(tmp.columns))
                        missing_reports.append(f"列缺失 {fp}: {miss}")
                if not dfs:
                    detail = "\n".join(missing_reports[:10])
                    raise ValueError(
                        "目录内没有任何 CSV 同时包含所需特征列。\n"
                        f"目录: {path}\n"
                        f"必要列: {FULL_FEATURE_COLUMNS}\n"
                        f"检查信息(最多展示10条):\n{detail}"
                    )
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.read_csv(path)

        unit_selection = _resolve_unit_selection(
            df,
            unit_profile,
            pos_unit,
            vel_unit,
            acc_unit,
        )
        unit_conversion_meta = apply_unit_conversions(
            df,
            unit_selection,
            thrust_unit=thrust_unit,
            thrust_source=thrust_from,
            sqrt_kf_col=sqrt_kf_col,
            motor_cmd_col=motor_cmd_col,
        )

        if fix_unit_vectors:
            rho_recomputed = None
            if recompute_rho:
                rho_recomputed = _recompute_rho_from_positions(df)
            normalize_vector_columns(
                df,
                RHO_COLUMNS,
                recompute_values=rho_recomputed,
            )
            normalize_vector_columns(df, EZ_COLUMNS)
            meta_flags = {
                "fix_unit_vectors_applied": True,
                "rho_recomputed": bool(rho_recomputed is not None),
            }
            if unit_conversion_meta is None:
                unit_conversion_meta = meta_flags
            else:
                unit_conversion_meta.update(meta_flags)
        else:
            if unit_conversion_meta is None:
                unit_conversion_meta = {"fix_unit_vectors_applied": False}
            else:
                unit_conversion_meta.setdefault("fix_unit_vectors_applied", False)

        df, sampling_method = _apply_sampling(
            df.reset_index(drop=True), sample_size, max_points, seed
        )

        if clip_range is not None:
            df, _ = apply_quantile_clipping(df, CLIP_TARGET_COLUMNS, clip_range[0], clip_range[1])

        cache_meta = dict(expected_meta)
        cache_meta.update(
            {
                "seed": _normalize_optional_int(seed),
                "sampling_method": sampling_method,
            }
        )
        if unit_conversion_meta:
            cache_meta.update(unit_conversion_meta)
        _save_dataset_cache_frame(
            dataset_cache_path,
            df,
            meta=cache_meta,
        )
    else:
        unit_conversion_meta = cached_meta
    ensure_columns(df, FULL_FEATURE_COLUMNS)
    unit_meta_filtered = None
    source_meta = unit_conversion_meta or cached_meta
    if source_meta:
        unit_meta_filtered = {
            key: source_meta.get(key)
            for key in UNIT_META_KEYS
            if source_meta.get(key) is not None
        }
        if unit_meta_filtered:
            df.attrs["unit_meta"] = unit_meta_filtered
    features = df[FEATURE_COLUMNS].to_numpy(dtype=dtype)
    normalizer = _load_normalizer_cache(norm_cache_path, dtype)
    if normalizer is None:
        normalizer = Normalizer.from_array(features)
        _save_normalizer_cache(norm_cache_path, normalizer)
    features_norm = normalizer.transform(features)
    return df, features_norm, normalizer


def _resolve_geometry_hparams(
    mix_ratio: float | None, jitter_scale: float | None
) -> Tuple[float, float]:
    default_ratio = float(MIXED_GEOMETRY_CONFIG["mix_ratio"])
    default_jitter = float(MIXED_GEOMETRY_CONFIG["jitter_scale"])
    ratio = default_ratio if mix_ratio is None else float(mix_ratio)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    jitter = default_jitter if jitter_scale is None else float(jitter_scale)
    jitter = float(max(jitter, 0.0))
    return ratio, jitter


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


def build_geometry(
    features_norm: np.ndarray,
    df: pd.DataFrame | None = None,
    normalizer: Normalizer | None = None,
    *,
    seed: int | None = None,
    mix_ratio: float | None = None,
    jitter_scale: float | None = None,
    enable_mixed: bool = True,
) -> dde.geometry.geometry_nd.Hypercube:
    xmin = features_norm.min(axis=0)
    xmax = features_norm.max(axis=0)
    # 防止边界重合导致几何体非法
    dx = np.where(np.isclose(xmin, xmax), 1e-3, 0.0)
    lower = (xmin - dx).tolist()
    upper = (xmax + dx).tolist()
    sampler = None
    if enable_mixed and df is not None and normalizer is not None and "time" in df.columns:
        ratio, jitter = _resolve_geometry_hparams(mix_ratio, jitter_scale)
        try:
            sampler = MixedFeatureSampler(
                df,
                normalizer,
                FEATURE_COLUMNS,
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


def build_network() -> dde.nn.FNN:
    depth = int(NETWORK_CONFIG["depth"])
    width = int(NETWORK_CONFIG["width"])
    activation = str(NETWORK_CONFIG["activation"])
    initializer = str(NETWORK_CONFIG["initializer"])
    layer_sizes = [INPUT_DIM] + [width] * depth + [OUTPUT_DIM]
    return dde.nn.FNN(layer_sizes, activation, initializer)


def train_model(
    model: dde.Model,
    loss_weights: List[float],
    extra_callbacks: List[dde.callbacks.Callback] | None = None,
) -> Tuple[dde.Model, Dict[str, Tuple[object, object]]]:
    adam_lr = float(TRAINING_CONFIG["adam_lr"])
    adam_iters = int(TRAINING_CONFIG["adam_iterations"])

    model.compile("adam", lr=adam_lr, loss_weights=loss_weights)

    base_callbacks = list(extra_callbacks or [])
    checkpoint_path = str(TRAINING_CONFIG["checkpoint_path"])
    adam_checkpoint = None
    lbfgs_checkpoint = None
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        adam_checkpoint = dde.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_better_only=True,
            period=1000,
        )
        lbfgs_checkpoint = dde.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_better_only=True,
            period=1,
        )

    def _build_callbacks(phase_callback):
        callbacks = list(base_callbacks)
        if phase_callback is not None:
            callbacks.append(phase_callback)
        return callbacks if callbacks else None

    training_artifacts: Dict[str, Tuple[object, object]] = {}
    adam_history = model.train(
        iterations=adam_iters,
        callbacks=_build_callbacks(adam_checkpoint),
    )
    if isinstance(adam_history, tuple) and len(adam_history) == 2:
        training_artifacts["adam"] = adam_history

    if TRAINING_CONFIG.get("use_lbfgs", True):
        model.compile("L-BFGS", loss_weights=loss_weights)
        lbfgs_history = model.train(callbacks=_build_callbacks(lbfgs_checkpoint))
        if isinstance(lbfgs_history, tuple) and len(lbfgs_history) == 2:
            training_artifacts["lbfgs"] = lbfgs_history

    return model, training_artifacts


def evaluate_model(
    model: dde.Model, anchors: np.ndarray, df: pd.DataFrame, out_dir: str
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
    export_df.insert(0, "time", df["time"].values)
    for name, meta in OBSERVATION_COLUMNS.items():
        cols: List[str] = meta["cols"]  # type: ignore[assignment]
        if set(cols).issubset(df.columns):
            for col in cols:
                export_df[f"{col}_true"] = df[col].values
    export_df.to_csv(out_path, index=False)
    print(f"预测结果已导出至 {out_path}")
    return preds


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def main():
    dde.config.set_default_float("float64")

    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    try:
        sample_plot_specs = resolve_sample_plot_specs(args.sample_plot_specs)
    except ValueError as exc:
        raise SystemExit(f"[force_estimation] 采样可视化配置错误: {exc}") from exc
    residual_variance_target = (args.residual_variance_path or "").strip()
    residual_metrics_target = (args.residual_metrics_path or "").strip()
    residual_plot_dir = (args.residual_plot_dir or "").strip()
    train_plot_dir = (args.train_plot_dir or "").strip()
    train_plot_prefix = (args.train_plot_prefix or DEFAULT_TRAIN_PLOT_PREFIX).strip() or DEFAULT_TRAIN_PLOT_PREFIX

    if args.adam_iters is not None:
        TRAINING_CONFIG["adam_iterations"] = max(1, int(args.adam_iters))
    if args.adam_lr is not None:
        TRAINING_CONFIG["adam_lr"] = float(args.adam_lr)
    if args.disable_lbfgs:
        TRAINING_CONFIG["use_lbfgs"] = False

    data_path = str(args.data)
    use_demo = bool(args.demo)
    if args.float32:
        dde.config.set_default_float("float32")

    dtype = np.float32 if args.float32 else np.float64

    clip_quantiles = (
        tuple(float(v) for v in args.clip_quantiles)
        if args.clip_quantiles and not args.skip_clip
        else None
    )
    df, features_norm, normalizer = load_dataset(
        data_path,
        use_demo,
        max_points=args.max_points,
        sample_size=args.sample_size,
        seed=args.seed,
        norm_cache=args.norm_cache,
        dataset_cache=args.dataset_cache,
        clip_quantiles=clip_quantiles,
        unit_profile=args.unit_profile,
        pos_unit=args.pos_unit,
        vel_unit=args.vel_unit,
        acc_unit=args.acc_unit,
        fix_unit_vectors=not args.disable_fix_unit_vectors,
        recompute_rho=args.recompute_rho,
        thrust_unit=args.thrust_unit,
        thrust_from=args.thrust_from,
        sqrt_kf_col=args.sqrt_kf_col,
        motor_cmd_col=args.motor_cmd_col,
        dtype=dtype,
    )
    if not args.skip_unit_check:
        run_unit_consistency_checks(df, thrust_tol=args.thrust_balance_tol)

    for attr in ("_min_tensor", "_range_tensor"):
        if hasattr(normalizer, attr):
            delattr(normalizer, attr)
    demo_used = use_demo or not os.path.exists(data_path)

    anchors = features_norm.astype(dtype, copy=False)
    resolved_mix_ratio, resolved_jitter = _resolve_geometry_hparams(
        args.geometry_mix_ratio, args.geometry_jitter_scale
    )
    use_mixed_geometry = (not args.disable_mixed_geometry) and ("time" in df.columns)
    domain_ratio = (
        float(args.domain_ratio)
        if args.domain_ratio is not None
        else float(DOMAIN_SAMPLING_CONFIG["ratio"])
    )
    domain_min_points = (
        int(args.domain_min_points)
        if args.domain_min_points is not None
        else int(DOMAIN_SAMPLING_CONFIG["min_points"])
    )
    domain_max_points = (
        int(args.domain_max_points)
        if args.domain_max_points is not None
        else int(DOMAIN_SAMPLING_CONFIG["max_points"])
    )
    effective_num_domain, domain_mode = resolve_effective_num_domain(
        args.num_domain,
        data_size=len(df),
        auto_enabled=not args.disable_auto_domain,
        ratio=domain_ratio,
        min_points=domain_min_points,
        max_points=domain_max_points,
    )
    if domain_mode == "auto":
        print(
            "[force_estimation] Auto domain sampling enabled: "
            f"num_domain={effective_num_domain} (ratio={domain_ratio}, "
            f"min={domain_min_points}, max={domain_max_points})."
        )
    elif domain_mode == "explicit":
        print(f"[force_estimation] Using explicit num_domain={effective_num_domain}.")
    else:
        print("[force_estimation] Domain sampling disabled (num_domain=0).")

    effective_num_test: int | None = None
    num_test_mode = "train_points"
    raw_num_test = args.num_test
    if raw_num_test is None:
        num_test_mode = "train_points"
    else:
        requested_num_test = int(raw_num_test)
        if requested_num_test < 0:
            num_test_mode = "train_points"
        elif requested_num_test == 0:
            if effective_num_domain > 0:
                effective_num_test = max(1, int(effective_num_domain))
                num_test_mode = "match_domain"
        else:
            effective_num_test = max(1, int(requested_num_test))
            num_test_mode = "explicit"

    if effective_num_test is not None:
        print(
            "[force_estimation] Independent PDE test sampling enabled: "
            f"num_test={effective_num_test} ({num_test_mode})."
        )
    else:
        print("[force_estimation] Test loss shares training points (num_test disabled).")

    geom = build_geometry(
        anchors,
        df if use_mixed_geometry else None,
        normalizer if use_mixed_geometry else None,
        seed=args.seed,
        mix_ratio=args.geometry_mix_ratio,
        jitter_scale=args.geometry_jitter_scale,
        enable_mixed=use_mixed_geometry,
    )
    sampling_plot_paths: List[str] = []
    if args.visualize_samples:
        plot_count = int(args.sample_plot_count or 0)
        if plot_count <= 0:
            inferred = max(effective_num_domain, min(len(df), 4096))
            plot_count = max(inferred, 512)
        plot_count = max(1, min(plot_count, 20000))
        try:
            sample_points_norm = geom.random_points(plot_count)
            sample_points = normalizer.inverse(sample_points_norm)
        except Exception as exc:
            print(
                f"[force_estimation] 采样可视化生成失败（random_points 异常）: {exc}"
            )
            sample_points = None
        if sample_points is not None:
            plot_dir = os.path.abspath(args.sample_plot_dir or DEFAULT_SAMPLE_PLOT_DIR)
            reference_points = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
            sampling_plot_paths = generate_sampling_visualizations(
                sample_points,
                reference_points,
                sample_plot_specs,
                plot_dir,
            )
            for path in sampling_plot_paths:
                print(f"[force_estimation] 采样可视化已保存: {path}")

    bc_batch = args.bc_batch if args.bc_batch > 0 else None
    bcs = build_observation_bcs(df, anchors, batch_size=bc_batch, dtype=dtype)

    const_mgr = ConstantManager(
        df,
        AUXILIARY_COLUMNS,
        anchors,
        seed=args.seed,
    )

    residual_scaler = build_residual_scaler(
        df,
        mode=args.residual_norm_mode,
        overrides={
            "r1": args.residual_scale_r1,
            "r2": args.residual_scale_r2,
            "r3": args.residual_scale_r3,
        },
    )
    if residual_scaler is None:
        print("[force_estimation] Residual normalization disabled.")
    else:
        print(
            "[force_estimation] Residual normalization enabled: "
            f"{residual_scaler.summary()}"
        )

    loss_weight_dict = resolve_loss_weight_dict(args, residual_scaler)
    bc_loss_weight = resolve_bc_loss_weight(SUPERVISION_WEIGHT, args.bc_loss_weight)
    pde_loss_weights = expand_pde_loss_weights(loss_weight_dict)
    loss_weights = build_loss_weight_vector(len(bcs), pde_loss_weights, bc_loss_weight)
    print(
        "[force_estimation] Loss weights -> "
        f"r1={loss_weight_dict['r1']:.3e}, "
        f"r2={loss_weight_dict['r2']:.3e}, "
        f"r3={loss_weight_dict['r3']:.3e}, "
        f"bc={bc_loss_weight:.3e} "
        f"(mode={args.loss_weight_mode})"
    )

    residual = build_force_estimation_residual(normalizer, const_mgr, residual_scaler)

    data = dde.data.PDE(
        geom,
        residual,
        bcs,
        num_domain=effective_num_domain,
        num_boundary=0,
        num_test=effective_num_test,
        anchors=anchors,
        auxiliary_var_function=const_mgr.auxiliary,
    )

    net = build_network()
    model = dde.Model(data, net)

    callbacks: List[dde.callbacks.Callback] = []
    if args.resample_period > 0 and (effective_num_domain > 0 or bc_batch):
        callbacks.append(
            dde.callbacks.PDEPointResampler(
                period=args.resample_period,
                pde_points=effective_num_domain > 0,
                bc_points=bc_batch is not None,
            )
        )
    if args.early_patience > 0:
        callbacks.append(
            dde.callbacks.EarlyStopping(
                min_delta=args.early_min_delta,
                patience=args.early_patience,
                monitor="loss_train",
            )
        )

    # Task 3.0: 监控基础设施（MetricsLogger）
    metrics_csv = (args.metrics_csv or "").strip()
    if metrics_csv:
        try:
            callbacks.append(
                MetricsLogger(
                    csv_path=metrics_csv,
                    component_names=RESIDUAL_COMPONENT_NAMES,
                    period=max(1, int(args.metrics_period)),
                    record_gradients=bool(args.metrics_gradients),
                )
            )
        except Exception as exc:
            print(f"[force_estimation] 初始化 MetricsLogger 失败: {exc}")

    if args.grad_monitor:
        try:
            callbacks.append(
                GradientNormMonitor(
                    log_path=args.grad_monitor_log,
                    period=max(1, int(args.grad_monitor_period)),
                    min_norm=float(args.grad_norm_min),
                    max_norm=float(args.grad_norm_max),
                    patience=int(args.grad_monitor_patience),
                )
            )
        except Exception as exc:
            print(f"[force_estimation] 初始化 GradientNormMonitor 失败: {exc}")

    if args.adaptive_weights:
        try:
            weight_min = _sanitize_positive(args.adaptive_weight_min, "--adaptive-weight-min")
            weight_max = _sanitize_positive(args.adaptive_weight_max, "--adaptive-weight-max")
            if weight_max < weight_min:
                weight_max = weight_min
            scheduler = AdaptiveWeightScheduler(
                component_names=RESIDUAL_COMPONENT_NAMES,
                base_weights=pde_loss_weights,
                period=max(1, int(args.adaptive_weight_period)),
                ema=float(args.adaptive_weight_alpha),
                min_weight=weight_min,
                max_weight=weight_max,
                initial_rms=(residual_scaler.scales if residual_scaler is not None else None),
            )
            callbacks.append(scheduler)
        except Exception as exc:
            print(f"[force_estimation] 初始化 AdaptiveWeightScheduler 失败: {exc}")

    dtype_name = np.dtype(dtype).name
    unit_meta = getattr(df, "attrs", {}).get("unit_meta", {})
    num_test_summary = (
        f"{effective_num_test} ({num_test_mode})"
        if effective_num_test is not None
        else "train_points"
    )
    config_pairs = [
        ("dtype", dtype_name),
        ("adam_iters", TRAINING_CONFIG["adam_iterations"]),
        ("adam_lr", TRAINING_CONFIG["adam_lr"]),
        ("use_lbfgs", "yes" if TRAINING_CONFIG["use_lbfgs"] else "no"),
        (
            "num_domain",
            f"{effective_num_domain} ({domain_mode})"
            if effective_num_domain > 0
            else f"{effective_num_domain} ({domain_mode})",
        ),
        ("num_test", num_test_summary),
        ("bc_batch", bc_batch if bc_batch is not None else "all"),
        ("resample_period", args.resample_period if args.resample_period > 0 else "disabled"),
        ("max_points", args.max_points or "full"),
        ("sample_size", args.sample_size or "full"),
        ("seed", args.seed),
        ("norm_cache", args.norm_cache),
        ("dataset_cache", args.dataset_cache or "disabled"),
        (
            "domain_auto",
            "disabled"
            if args.disable_auto_domain
            else f"ratio={domain_ratio:.2f}, min={domain_min_points}, max={domain_max_points}",
        ),
        (
            "sample_plots",
            os.path.abspath(args.sample_plot_dir)
            if args.visualize_samples
            else "disabled",
        ),
        (
            "residual_variance",
            os.path.abspath(residual_variance_target)
            if residual_variance_target
            else "disabled",
        ),
        (
            "residual_metrics",
            os.path.abspath(residual_metrics_target)
            if residual_metrics_target
            else "disabled",
        ),
        (
            "residual_plots",
            os.path.abspath(residual_plot_dir)
            if residual_plot_dir
            else "disabled",
        ),
        (
            "train_plots",
            os.path.abspath(train_plot_dir)
            if train_plot_dir
            else "disabled",
        ),
        (
            "residual_norm",
            residual_scaler.summary() if residual_scaler else "disabled",
        ),
        (
            "clip_quantiles",
            "disabled"
            if clip_quantiles is None
            else f"[{clip_quantiles[0]:.3f}, {clip_quantiles[1]:.3f}]",
        ),
        (
            "unit_profile_selected",
            unit_meta.get("unit_profile_selected", args.unit_profile),
        ),
        (
            "pos_unit_selected",
            unit_meta.get("pos_unit_selected", args.pos_unit),
        ),
        (
            "vel_unit_selected",
            unit_meta.get("vel_unit_selected", args.vel_unit),
        ),
        (
            "acc_unit_selected",
            unit_meta.get("acc_unit_selected", args.acc_unit),
        ),
        (
            "thrust_unit_selected",
            unit_meta.get("thrust_unit_selected", args.thrust_unit),
        ),
        (
            "thrust_source",
            unit_meta.get("thrust_source", args.thrust_from),
        ),
        (
            "fix_unit_vectors",
            "disabled" if args.disable_fix_unit_vectors else "enabled",
        ),
        (
            "rho_recomputed",
            "yes" if args.recompute_rho else "no",
        ),
        ("unit_check", "skipped" if args.skip_unit_check else "enabled"),
        (
            "early_stopping",
            f"patience={args.early_patience}, min_delta={args.early_min_delta}"
            if args.early_patience > 0
            else "disabled",
        ),
        (
            "mixed_geometry",
            "disabled"
            if args.disable_mixed_geometry
            else (
                f"ratio={resolved_mix_ratio:.2f}, jitter={resolved_jitter:.3f}"
                if "time" in df.columns
                else "unavailable"
            ),
        ),
        (
            "loss_weights",
            "r1={:.3f}, r2={:.3f}, r3={:.3f}, bc={:.3f}".format(
                loss_weight_dict["r1"],
                loss_weight_dict["r2"],
                loss_weight_dict["r3"],
                bc_loss_weight,
            ),
        ),
        (
            "adaptive_weights",
            f"period={args.adaptive_weight_period}, alpha={args.adaptive_weight_alpha:.2f}"
            if args.adaptive_weights
            else "disabled",
        ),
        (
            "grad_monitor",
            "range=[{:.1e}, {:.1e}], period={}, patience={}".format(
                args.grad_norm_min,
                args.grad_norm_max,
                args.grad_monitor_period,
                args.grad_monitor_patience,
            )
            if args.grad_monitor
            else "disabled",
        ),
    ]
    _print_config(config_pairs)

    model, training_artifacts = train_model(model, loss_weights, extra_callbacks=callbacks)
    if train_plot_dir:
        for phase, artifact in training_artifacts.items():
            loss_history, train_state = artifact
            png_name = f"{train_plot_prefix}_{phase}.png"
            plot_path = os.path.join(train_plot_dir, png_name)
            save_training_phase_outputs(
                phase,
                loss_history,
                train_state,
                data_root=train_plot_dir,
                plot_path=plot_path,
            )

    if demo_used:
        out_dir = os.getcwd()
    elif os.path.isdir(data_path):
        out_dir = data_path
    else:
        out_dir = os.path.dirname(data_path) or os.getcwd()
    preds = evaluate_model(model, anchors, df, out_dir)

    need_residual_matrix = bool(
        residual_variance_target or residual_metrics_target or residual_plot_dir
    )
    residual_matrix = None
    if need_residual_matrix:
        residual_matrix = compute_residual_components_numpy(
            df,
            preds,
            residual_scaler=residual_scaler,
        )

    if residual_variance_target:
        residual_table = compute_residual_variance_table(
            df,
            preds,
            residual_scaler=residual_scaler,
            residual_matrix=residual_matrix,
        )
        residual_abs_path = os.path.abspath(residual_variance_target)
        residual_dir = os.path.dirname(residual_abs_path)
        if residual_dir:
            os.makedirs(residual_dir, exist_ok=True)
        residual_table.to_csv(residual_abs_path, index=False)
        print(f"[force_estimation] 残差方差统计已保存: {residual_abs_path}")
        top_summary = ", ".join(
            f"{row.component}: {row.variance:.4e}"
            for row in residual_table.itertuples()
        )
        print(f"[force_estimation] residual variance summary -> {top_summary}")

    if residual_metrics_target:
        metrics_table = compute_residual_metrics_numpy(
            df,
            preds,
            residual_scaler=residual_scaler,
            residual_matrix=residual_matrix,
        )
        metrics_abs_path = os.path.abspath(residual_metrics_target)
        metrics_dir = os.path.dirname(metrics_abs_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        metrics_table.to_csv(metrics_abs_path, index=False)
        print(f"[force_estimation] 残差评估指标已保存: {metrics_abs_path}")

    if residual_plot_dir and residual_matrix is not None:
        plot_dir_abs = os.path.abspath(residual_plot_dir)
        plot_paths = generate_residual_distribution_plots(residual_matrix, plot_dir_abs)
        for path in plot_paths:
            print(f"[force_estimation] 残差分布图已保存: {path}")

    export_tasks: List[Tuple[str, str, Callable[[str], None]]] = []

    torchscript_path = (args.export_torchscript or "").strip()
    if torchscript_path:
        export_tasks.append(
            (
                "TorchScript",
                torchscript_path,
                lambda path: export_torchscript_model(model, path, INPUT_DIM),
            )
        )

    onnx_path = (args.export_onnx or "").strip()
    if onnx_path:
        export_tasks.append(
            (
                "ONNX",
                onnx_path,
                lambda path: export_onnx_model(model, path, INPUT_DIM),
            )
        )

    metadata_path = (args.export_metadata or "").strip()
    if metadata_path:
        export_tasks.append(
            (
                "metadata",
                metadata_path,
                lambda path: export_model_metadata(
                    path,
                    feature_columns=FEATURE_COLUMNS,
                    auxiliary_columns=AUXILIARY_COLUMNS,
                    residual_components=RESIDUAL_COMPONENT_NAMES,
                    network_config=NETWORK_CONFIG,
                    input_dim=INPUT_DIM,
                    output_dim=OUTPUT_DIM,
                    dtype=dtype_name,
                ),
            )
        )

    normalizer_path = (args.export_normalizer or "").strip()
    if normalizer_path:
        export_tasks.append(
            (
                "normalizer",
                normalizer_path,
                lambda path: export_normalizer_to_json(normalizer, FEATURE_COLUMNS, path),
            )
        )

    for label, target_path, fn in export_tasks:
        try:
            fn(target_path)
        except Exception as exc:  # pragma: no cover - runtime/env dependent
            print(f"[force_estimation] 导出 {label} 失败: {exc}")


if __name__ == "__main__":
    main()
