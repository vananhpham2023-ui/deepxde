"""
force_estimation_easy.py
========================

这是一个基于 DeepXDE 的轻量级演示脚本，用于快速验证无人机-载荷受力估计 PINN
的建模流程。与完整版 `force_estimation.py` 相比，本脚本内置演示数据生成逻辑，
无需外部文件即可运行，便于检查网络结构、残差定义及训练流程是否正常。

核心特性
--------
- 输入特征包含时间、无人机/载荷位置-速度-加速度、姿态第三列、推力、质量、重力等
  共 32 个维度；
- 网络输出九维向量 `[f_Q(3), f_L(3), ρ(3)]`；
- 残差函数实现动力学平衡、力矩平衡与载荷约束三类物理约束，并可选启用缆绳单位向量罚项；
- 采用内置的合成数据（正弦/余弦轨迹）作为训练样本，默认使用 Adam + L-BFGS 组合优化器。
"""
from __future__ import annotations

import math
from typing import List

import deepxde as dde
import numpy as np
import deepxde.backend as bkd

# --------------------------------------------------------------------------------------
# 配置区域：训练规模、网络和损失超参数，可根据需要自行调整
# --------------------------------------------------------------------------------------

NUM_SAMPLES = 128  # 采样点数量
LOSS_WEIGHTS = {"r1": 1.0, "r2": 1.0, "r3": 1.0, "rho_unit": 0.05}
NETWORK_SPEC = {"width": 64, "depth": 4, "activation": "tanh", "initializer": "Glorot uniform"}
TRAINING_SPEC = {"adam_lr": 1e-3, "adam_iters": 2000, "use_lbfgs": True}

# --------------------------------------------------------------------------------------
# 特征列定义，与完整版保持一致，方便后续迁移或对比
# --------------------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
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
    "xL_x",
    "xL_y",
    "xL_z",
    "vL_x",
    "vL_y",
    "vL_z",
    "accL_x",
    "accL_y",
    "accL_z",
    "R_00",
    "R_01",
    "R_02",
    "R_10",
    "R_11",
    "R_12",
    "R_20",
    "R_21",
    "R_22",
    "thrust",
    "m_Q",
    "m_L",
    "g",
]

INPUT_DIM = len(FEATURE_COLUMNS)
OUTPUT_DIM = 9

IDX_TIME = FEATURE_COLUMNS.index("time")
IDX_AQ = slice(FEATURE_COLUMNS.index("accQ_x"), FEATURE_COLUMNS.index("accQ_z") + 1)
IDX_AL = slice(FEATURE_COLUMNS.index("accL_x"), FEATURE_COLUMNS.index("accL_z") + 1)
IDX_R = slice(FEATURE_COLUMNS.index("R_00"), FEATURE_COLUMNS.index("R_22") + 1)
IDX_THRUST = FEATURE_COLUMNS.index("thrust")
IDX_MQ = FEATURE_COLUMNS.index("m_Q")
IDX_ML = FEATURE_COLUMNS.index("m_L")
IDX_G = FEATURE_COLUMNS.index("g")

# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------


def cross_product(vec_a, vec_b):
    ax, ay, az = vec_a[:, 0:1], vec_a[:, 1:2], vec_a[:, 2:3]
    bx, by, bz = vec_b[:, 0:1], vec_b[:, 1:2], vec_b[:, 2:3]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return bkd.concat([cx, cy, cz], axis=1)


def rho_unit_enabled() -> bool:
    return LOSS_WEIGHTS.get("rho_unit", 0.0) > 0.0


# --------------------------------------------------------------------------------------
# 合成数据构造
# --------------------------------------------------------------------------------------


def generate_demo_dataset(num_samples: int) -> np.ndarray:
    """生成一个满足基本物理量级的合成数据集，方便快速测试。"""
    t = np.linspace(0.0, 2 * math.pi, num_samples, dtype=np.float64)
    dt = t[1] - t[0]

    # 无人机位置/速度/加速度（简单双频正弦）
    x_q = np.stack(
        [
            0.5 * np.sin(t),
            0.5 * np.cos(t),
            0.25 * np.sin(2 * t),
        ],
        axis=1,
    )
    v_q = np.gradient(x_q, dt, axis=0, edge_order=2)
    acc_q = np.gradient(v_q, dt, axis=0, edge_order=2)

    # 载荷比无人机向下偏移 0.8m，并附加轻微相位差
    x_l = x_q + np.array([0.0, 0.0, -0.8]) + 0.05 * np.sin(t)[:, None]
    v_l = np.gradient(x_l, dt, axis=0, edge_order=2)
    acc_l = np.gradient(v_l, dt, axis=0, edge_order=2)

    # 姿态矩阵取常量绕 z 轴轻微摆动，保证 Re_z^w 为单位向量
    yaw = 0.1 * np.sin(t)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rotation = np.stack(
        [
            cos_y,
            -sin_y,
            np.zeros_like(t),
            sin_y,
            cos_y,
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.ones_like(t),
        ],
        axis=1,
    )

    thrust = np.full((num_samples, 1), 12.0)
    mass_q = np.full((num_samples, 1), 1.6)
    mass_l = np.full((num_samples, 1), 0.9)
    gravity = np.full((num_samples, 1), 9.81)

    feature_matrix = np.zeros((num_samples, INPUT_DIM), dtype=np.float64)
    feature_matrix[:, IDX_TIME] = t
    feature_matrix[:, FEATURE_COLUMNS.index("xQ_x") : FEATURE_COLUMNS.index("xQ_z") + 1] = x_q
    feature_matrix[:, FEATURE_COLUMNS.index("vQ_x") : FEATURE_COLUMNS.index("vQ_z") + 1] = v_q
    feature_matrix[:, IDX_AQ] = acc_q
    feature_matrix[:, FEATURE_COLUMNS.index("xL_x") : FEATURE_COLUMNS.index("xL_z") + 1] = x_l
    feature_matrix[:, FEATURE_COLUMNS.index("vL_x") : FEATURE_COLUMNS.index("vL_z") + 1] = v_l
    feature_matrix[:, IDX_AL] = acc_l
    feature_matrix[:, IDX_R] = rotation
    feature_matrix[:, IDX_THRUST : IDX_THRUST + 1] = thrust
    feature_matrix[:, IDX_MQ : IDX_MQ + 1] = mass_q
    feature_matrix[:, IDX_ML : IDX_ML + 1] = mass_l
    feature_matrix[:, IDX_G : IDX_G + 1] = gravity

    return feature_matrix


# --------------------------------------------------------------------------------------
# 残差函数
# --------------------------------------------------------------------------------------


def Force_estimation(x, y):
    """计算三组残差（可选加入 ρ 单位向量约束）。"""
    features = x
    f_q = y[:, 0:3]
    f_l = y[:, 3:6]
    rho = y[:, 6:9]

    acc_q = features[:, IDX_AQ]
    acc_l = features[:, IDX_AL]
    thrust = features[:, IDX_THRUST : IDX_THRUST + 1]
    m_q = features[:, IDX_MQ : IDX_MQ + 1]
    m_l = features[:, IDX_ML : IDX_ML + 1]
    g_scalar = features[:, IDX_G : IDX_G + 1]

    rotation_flat = features[:, IDX_R]
    ez_world = bkd.concat(
        [
            rotation_flat[:, 2:3],
            rotation_flat[:, 5:6],
            rotation_flat[:, 8:9],
        ],
        axis=1,
    )

    zero = g_scalar * 0.0
    gravity_vec = bkd.concat([zero, zero, g_scalar], axis=1)
    if gravity_vec is None:
        raise RuntimeError("gravity_vec is None; concat operation failed.")

    r1 = f_q + f_l - m_q * (gravity_vec + acc_q) + m_l * (gravity_vec + acc_l)

    aero_term = thrust * ez_world - m_q * gravity_vec - m_q * acc_q
    r2 = cross_product(rho, f_q) - cross_product(rho, aero_term)

    r3 = bkd.sum(f_l * rho, dim=1, keepdims=True)

    residuals = [
        r1[:, 0:1],
        r1[:, 1:2],
        r1[:, 2:3],
        r2[:, 0:1],
        r2[:, 1:2],
        r2[:, 2:3],
        r3,
    ]

    if rho_unit_enabled():
        rho_norm_res = bkd.sum(rho * rho, dim=1, keepdims=True) - 1.0
        residuals.append(rho_norm_res)

    return residuals


# --------------------------------------------------------------------------------------
# 模型构建与训练流程
# --------------------------------------------------------------------------------------


def build_geometry(features_array: np.ndarray) -> dde.geometry.geometry_nd.Hypercube:
    xmin = features_array.min(axis=0)
    xmax = features_array.max(axis=0)
    padding = np.where(np.isclose(xmin, xmax), 1e-3, 0.0)
    return dde.geometry.geometry_nd.Hypercube((xmin - padding).tolist(), (xmax + padding).tolist())


def build_network() -> dde.nn.FNN:
    width = int(NETWORK_SPEC["width"])
    depth = int(NETWORK_SPEC["depth"])
    activation = NETWORK_SPEC["activation"]
    initializer = NETWORK_SPEC["initializer"]
    layer_sizes = [INPUT_DIM] + [width] * depth + [OUTPUT_DIM]
    return dde.nn.FNN(layer_sizes, activation, initializer)


def build_loss_weights() -> List[float]:
    weights = [LOSS_WEIGHTS["r1"]] * 3 + [LOSS_WEIGHTS["r2"]] * 3 + [LOSS_WEIGHTS["r3"]]
    if rho_unit_enabled():
        weights.append(LOSS_WEIGHTS["rho_unit"])
    return weights


def train(model: dde.Model) -> None:
    loss_weights = build_loss_weights()
    model.compile("adam", lr=TRAINING_SPEC["adam_lr"], loss_weights=loss_weights)
    model.train(iterations=TRAINING_SPEC["adam_iters"])
    if TRAINING_SPEC.get("use_lbfgs", True):
        model.compile("L-BFGS", loss_weights=loss_weights)
        model.train()


def main():
    dde.config.disable_xla_jit()
    dde.config.set_default_float("float64")

    features = generate_demo_dataset(NUM_SAMPLES)
    anchors = features.astype(np.float64)
    geom = build_geometry(features)

    data = dde.data.PDE(
        geom,
        Force_estimation,
        [],
        num_domain=0,
        num_boundary=0,
        anchors=anchors,
    )

    net = build_network()
    model = dde.Model(data, net)

    train(model)

    preds = model.predict(anchors[:5])
    print("示例输出 (前 5 个样本)：")
    print(preds)


if __name__ == "__main__":
    main()
