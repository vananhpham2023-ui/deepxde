#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("DDE_BACKEND", "pytorch")
try:
    from .force_estimation import RESIDUAL_COMPONENT_NAMES
except ImportError:  # pragma: no cover
    from force_estimation import RESIDUAL_COMPONENT_NAMES


def _contains_flag(args_list: Sequence[str], flag: str) -> bool:
    return any(
        token == flag or token.startswith(f"{flag}=") for token in args_list
    )


def build_comparison_table(
    hyper_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    *,
    baseline_label: str,
    mixed_label: str,
    min_threshold: float,
    ideal_threshold: float,
) -> pd.DataFrame:
    def _df_to_map(frame: pd.DataFrame) -> Dict[str, float]:
        if "component" not in frame.columns or "variance" not in frame.columns:
            raise ValueError("Residual variance CSV 必须包含 component 与 variance 列。")
        return {
            str(row.component): float(row.variance)
            for row in frame.itertuples()
        }

    baseline_map = _df_to_map(hyper_df)
    mixed_map = _df_to_map(mixed_df)

    records: List[Dict[str, object]] = []

    def _status(value: float) -> str:
        if not np.isfinite(value) or value < min_threshold:
            return "⚠ 未达标"
        if value >= ideal_threshold:
            return "✓ 达到理想目标"
        return "✓ 达到最低门槛"

    def _append_row(metric: str, base_val: float, mixed_val: float):
        if base_val == 0:
            improvement = float("nan")
        else:
            improvement = (base_val - mixed_val) / base_val
        records.append(
            {
                "Metric": metric,
                baseline_label: base_val,
                mixed_label: mixed_val,
                "Improvement": improvement * 100.0 if np.isfinite(improvement) else np.nan,
                "Status": _status(improvement) if np.isfinite(improvement) else "⚠ 未达标",
            }
        )

    for comp in RESIDUAL_COMPONENT_NAMES:
        if comp not in baseline_map or comp not in mixed_map:
            raise ValueError(f"残差组件 {comp} 未在 CSV 中找到。")
        _append_row(comp, baseline_map[comp], mixed_map[comp])

    total_baseline = sum(baseline_map.values())
    total_mixed = sum(mixed_map.values())
    _append_row("total_variance", total_baseline, total_mixed)

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Hypercube vs Mixed geometry residual variance."
    )
    default_force_script = os.path.join(
        os.path.dirname(__file__), "force_estimation.py"
    )
    parser.add_argument(
        "--force-script",
        type=str,
        default=default_force_script,
        help="force_estimation.py 路径（默认指向当前目录）。",
    )
    parser.add_argument(
        "--force-args",
        type=str,
        default="",
        help="传递给两次 force_estimation 运行的公共参数（使用 shell 风格字符串）。",
    )
    parser.add_argument(
        "--baseline-extra-args",
        type=str,
        default="",
        help="仅用于 Hypercube 运行的附加参数。",
    )
    parser.add_argument(
        "--mixed-extra-args",
        type=str,
        default="",
        help="仅用于 Mixed_Geom 运行的附加参数。",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="geometry_comparison_runs",
        help="临时输出目录，用于保存中间残差 CSV。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="geometry_comparison.csv",
        help="最终比较结果 CSV 路径。",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.3,
        help="最低达标提升比例（默认 0.3，即 30%%）。",
    )
    parser.add_argument(
        "--ideal-threshold",
        type=float,
        default=0.5,
        help="理想目标提升比例（默认 0.5，即 50%%）。",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Hypercube",
        help="基线列名称（默认 Hypercube）。",
    )
    parser.add_argument(
        "--mixed-label",
        type=str,
        default="Mixed_Geom",
        help="混合几何列名称（默认 Mixed_Geom）。",
    )
    return parser.parse_args()


def _run_force_estimation(
    label: str,
    *,
    base_cmd: List[str],
    extra_args: List[str],
    residual_path: str,
) -> pd.DataFrame:
    cmd = base_cmd + extra_args + ["--residual-variance-path", residual_path]
    print(f"[geometry_comparison] Running {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    df = pd.read_csv(residual_path)
    if df.empty:
        raise RuntimeError(f"{label} 的残差 CSV 为空: {residual_path}")
    return df


def main() -> None:
    args = parse_args()
    base_args = shlex.split(args.force_args)
    baseline_extra = shlex.split(args.baseline_extra_args)
    mixed_extra = shlex.split(args.mixed_extra_args)

    if _contains_flag(base_args, "--residual-variance-path"):
        raise SystemExit("请勿在 --force-args 中指定 --residual-variance-path。")

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    force_script = os.path.abspath(args.force_script)

    base_cmd = [sys.executable, force_script] + base_args

    baseline_residual_path = os.path.join(work_dir, f"{args.baseline_label}_residual.csv")
    mixed_residual_path = os.path.join(work_dir, f"{args.mixed_label}_residual.csv")

    hyper_df = _run_force_estimation(
        args.baseline_label,
        base_cmd=base_cmd,
        extra_args=baseline_extra + ["--disable-mixed-geometry"],
        residual_path=baseline_residual_path,
    )
    mixed_df = _run_force_estimation(
        args.mixed_label,
        base_cmd=base_cmd,
        extra_args=mixed_extra,
        residual_path=mixed_residual_path,
    )

    comparison_df = build_comparison_table(
        hyper_df,
        mixed_df,
        baseline_label=args.baseline_label,
        mixed_label=args.mixed_label,
        min_threshold=float(args.min_threshold),
        ideal_threshold=float(args.ideal_threshold),
    )
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"[geometry_comparison] 对比结果已写入 {output_path}")


if __name__ == "__main__":
    main()
