from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Columns required by estimation/cli.py
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

FORCE_COLUMNS = [
    "fQ_x",
    "fQ_y",
    "fQ_z",
    "fL_x",
    "fL_y",
    "fL_z",
]

DROP_COLUMNS = [
    "fl_est_x",
    "fl_est_y",
    "fl_est_z",
    "fq_est_x",
    "fq_est_y",
    "fq_est_z",
]

COLUMN_MAP = {
    "timestamp": "time",
    "quad_pos_x": "xQ_x",
    "quad_pos_y": "xQ_y",
    "quad_pos_z": "xQ_z",
    "quad_vel_x": "vQ_x",
    "quad_vel_y": "vQ_y",
    "quad_vel_z": "vQ_z",
    "quad_acc_x": "accQ_x",
    "quad_acc_y": "accQ_y",
    "quad_acc_z": "accQ_z",
    "quad_omega_x": "omega_b_x",
    "quad_omega_y": "omega_b_y",
    "quad_omega_z": "omega_b_z",
    "load_pos_x": "xL_x",
    "load_pos_y": "xL_y",
    "load_pos_z": "xL_z",
    "load_vel_x": "vL_x",
    "load_vel_y": "vL_y",
    "load_vel_z": "vL_z",
    "load_acc_x": "accL_x",
    "load_acc_y": "accL_y",
    "load_acc_z": "accL_z",
    "cable_dir_x": "rho_x",
    "cable_dir_y": "rho_y",
    "cable_dir_z": "rho_z",
    "quad_rot_r13": "ez_world_x",
    "quad_rot_r23": "ez_world_y",
    "quad_rot_r33": "ez_world_z",
    "thrust_total": "thrust",
    "mass_quad": "m_Q",
    "mass_load": "m_L",
}

FORCE_MAP = {
    "fq_true_x": "fQ_x",
    "fq_true_y": "fQ_y",
    "fq_true_z": "fQ_z",
    "fl_true_x": "fL_x",
    "fl_true_y": "fL_y",
    "fl_true_z": "fL_z",
}


def convert_dataset(
    src: Path,
    dst: Path,
    gravity: float = 9.81,
    relative_time: bool = True,
) -> None:
    df = pd.read_csv(src)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    rename_map = {}
    rename_map.update(COLUMN_MAP)
    rename_map.update(FORCE_MAP)
    df = df.rename(columns=rename_map)

    if "time" not in df.columns:
        raise ValueError("源数据缺少 timestamp 列，无法生成 time。")

    if relative_time:
        df["time"] = df["time"] - float(df["time"].iloc[0])

    df["g"] = gravity

    missing = [col for col in FULL_FEATURE_COLUMNS + FORCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"源数据缺少必要列: {missing}")

    ordered = df[FULL_FEATURE_COLUMNS + FORCE_COLUMNS].copy()
    ordered = ordered.astype(np.float64, copy=False)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(dst, index=False)
    print(f"[convert_clean_dataset] Wrote {len(ordered)} rows to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 clean_dataset.csv 转换为 force_estimation 训练集格式。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("clean_dataset.csv"),
        help="源 CSV 文件路径 (默认: clean_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("clean_dataset_force_estimation.csv"),
        help="输出 CSV 文件路径 (默认: clean_dataset_force_estimation.csv)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="重力常数 g 的数值 (默认: 9.81)",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="保留原始时间戳，不转换为相对时间。",
    )
    args = parser.parse_args()
    convert_dataset(
        args.input,
        args.output,
        gravity=args.gravity,
        relative_time=not args.absolute_time,
    )


if __name__ == "__main__":
    main()
