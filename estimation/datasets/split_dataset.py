from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _compute_split_counts(n_rows: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must be positive.")
    normed = np.array(ratios, dtype=np.float64) / total
    raw_counts = normed * n_rows
    counts = np.floor(raw_counts).astype(int)
    remainder = n_rows - int(counts.sum())
    if remainder > 0:
        fractional = raw_counts - counts
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def split_dataset(
    src_path: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    out_prefix: Path | None = None,
) -> List[Path]:
    df = pd.read_csv(src_path)
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("输入数据为空，无法划分。")
    train_count, val_count, test_count = _compute_split_counts(
        n_rows, (train_ratio, val_ratio, test_ratio)
    )
    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError("划分后某个子集为空，请调整比例或使用更大的数据。")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]

    stem = out_prefix or src_path.with_suffix("")
    output_paths = []
    for name, idx in zip(
        ("train", "val", "test"),
        (train_idx, val_idx, test_idx),
    ):
        subset = df.iloc[idx].reset_index(drop=True)
        out_path = stem.with_name(f"{stem.name}_{name}.csv")
        subset.to_csv(out_path, index=False)
        output_paths.append(out_path)
        print(f"[split_dataset] {name} -> {len(subset)} 行，已保存至 {out_path}")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按比例拆分 force_estimation 数据集为 train/val/test。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("clean_dataset_force_estimation.csv"),
        help="待划分的 CSV 文件（默认：clean_dataset_force_estimation.csv）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.55,
        help="训练集比例（默认 0.55）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="验证集比例（默认 0.20）",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.25,
        help="测试集比例（默认 0.25）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，确保划分可复现（默认 42）",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help="输出文件名前缀（默认沿用输入名，附加 _train/_val/_test 后缀）",
    )
    args = parser.parse_args()
    split_dataset(
        args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        out_prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
