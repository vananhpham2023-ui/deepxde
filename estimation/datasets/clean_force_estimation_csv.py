#!/usr/bin/env python3
"""
清洗 force_estimation_* 数据集，移除包含 NaN/Inf 或非法推力的样本。

用途：
- 训练 PINN 之前，对 `force_estimation_task9.csv` 之类的数据做一次“保底清洗”，
  防止归一化统计和训练过程被 NaN 搞崩。

默认行为：
- 从 `--input` 读取 CSV（默认 `force_estimation_task9.csv`）；
- 仅保留在「网络输入 + 力输出」这些列上全为有限数值的行；
- 额外丢弃推力 `thrust` <= 0 或 > 150 的明显异常样本；
- 将结果写入 `--output`，未指定时覆盖原文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover
    # 与 convert_clean_dataset 保持列名一致
    from .convert_clean_dataset import FULL_FEATURE_COLUMNS, FORCE_COLUMNS
except ImportError:  # pragma: no cover
    from convert_clean_dataset import FULL_FEATURE_COLUMNS, FORCE_COLUMNS  # type: ignore


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("force_estimation_task9.csv"),
        help="待清洗的 CSV 文件路径（默认：force_estimation_task9.csv）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="清洗后的输出路径（默认：覆盖 --input 文件）",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output or input_path

    if not input_path.is_file():
        raise SystemExit(f"[clean_force_estimation] 输入文件不存在: {input_path}")

    df = pd.read_csv(input_path)
    before = len(df)

    # 只在实际存在的列上做检查，兼容定制数据。
    cols_to_check = [c for c in list(FULL_FEATURE_COLUMNS) + list(FORCE_COLUMNS) if c in df.columns]
    if not cols_to_check:
        raise SystemExit(
            "[clean_force_estimation] 输入文件中找不到任何预期的特征/力列，"
            f"请确认来源是否为 convert_clean_dataset.py 生成的数据: {input_path}"
        )

    values = df[cols_to_check].to_numpy(dtype=np.float64, copy=False)
    finite_mask = np.isfinite(values).all(axis=1)

    # 推力合法性检查（若存在 thrust 列）
    thrust_mask = np.ones(len(df), dtype=bool)
    if "thrust" in df.columns:
        thrust = df["thrust"].to_numpy(dtype=np.float64, copy=False)
        thrust_mask = (thrust > 0.0) & (thrust <= 150.0) & np.isfinite(thrust)

    keep_mask = finite_mask & thrust_mask
    cleaned = df.loc[keep_mask].reset_index(drop=True)
    removed = before - len(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(
        f"[clean_force_estimation] 输入 {before} 行，移除 {removed} 行非法样本，"
        f"保留 {len(cleaned)} 行 → {output_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

