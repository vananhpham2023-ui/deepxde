"""Utilities for converting raw UAV-load experiment logs into PINN-ready datasets.

This script bridges between the AutoTrans logging format and the feature schema
used by `examples/pinn_inverse/force_estimation.py`. It reads CSV logs produced
under `examples/dataset/pinn-plot/`, performs column renaming/aggregation,
normalises time stamps, fills missing constants (e.g., gravity) and exports a
single CSV with the canonical 37 feature columns plus the 6 supervision columns.

Usage
------
python3 examples/pinn_inverse/pinn_data_recorder.py \
    --src examples/dataset/pinn-plot/pinn_dataset_circle-c1_low_20251104182953_2_9978.csv \
    --dst examples/pinn_inverse/datasets/force_estimation_sample.csv

Production runs can point `--src` to a directory; the recorder will merge
multiple CSV files in chronological order.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

# Canonical schema expected by the PINN model.
FEATURE_FIELDS: Sequence[str] = (
    "time",
    "xQ_x", "xQ_y", "xQ_z",
    "vQ_x", "vQ_y", "vQ_z",
    "accQ_x", "accQ_y", "accQ_z",
    "omega_b_x", "omega_b_y", "omega_b_z",
    "xL_x", "xL_y", "xL_z",
    "vL_x", "vL_y", "vL_z",
    "accL_x", "accL_y", "accL_z",
    "rho_x", "rho_y", "rho_z",
    "ez_world_x", "ez_world_y", "ez_world_z",
    "thrust", "m_Q", "m_L", "g",
    "l_length", "sqrt_kf",
    "wind_x", "wind_y", "wind_z",
)

SUPERVISION_FIELDS: Sequence[str] = (
    "fQ_x", "fQ_y", "fQ_z",
    "fL_x", "fL_y", "fL_z",
)

ALL_FIELDS = tuple(FEATURE_FIELDS) + tuple(SUPERVISION_FIELDS)

# Mapping from raw CSV header -> canonical field name. Fields absent here are copied
# verbatim if they already match canonical names.
RAW_TO_CANONICAL = {
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
    "fl_true_x": "fL_x",
    "fl_true_y": "fL_y",
    "fl_true_z": "fL_z",
    "fq_true_x": "fQ_x",
    "fq_true_y": "fQ_y",
    "fq_true_z": "fQ_z",
}

DEFAULT_CONSTANTS = {
    "g": 9.81,
}

@dataclass
class FieldStat:
    minimum: float
    maximum: float


def _iter_csv_rows(src_paths: Iterable[str]) -> Iterable[dict[str, str]]:
    """Yield rows from one or multiple CSV files preserving order."""
    for path in src_paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row


def _normalise_time(rows: List[dict[str, float]]) -> None:
    if not rows:
        return
    t0 = rows[0]["time"]
    for row in rows:
        row["time"] -= t0


def _convert_rows(src_paths: Iterable[str]) -> List[dict[str, float]]:
    converted: List[dict[str, float]] = []
    for raw in _iter_csv_rows(src_paths):
        canon: dict[str, float] = {}
        for key, value in raw.items():
            if value is None or value == "" or value.lower() == "nan":
                continue
            try:
                val = float(value)
            except ValueError:
                continue
            target = RAW_TO_CANONICAL.get(key, key)
            if target in ALL_FIELDS:
                canon[target] = val
        for const_key, const_val in DEFAULT_CONSTANTS.items():
            canon.setdefault(const_key, const_val)
        if all(field in canon for field in FEATURE_FIELDS):
            converted.append(canon)
    _normalise_time(converted)
    return converted


def _write_csv(rows: List[dict[str, float]], dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_FIELDS)
        writer.writeheader()
        for row in rows:
            record = {field: row.get(field, "") for field in ALL_FIELDS}
            writer.writerow(record)


def _compute_stats(rows: List[dict[str, float]]) -> dict[str, FieldStat]:
    stats: dict[str, FieldStat] = {}
    for field in ALL_FIELDS:
        values = [row[field] for row in rows if field in row]
        if not values:
            continue
        stats[field] = FieldStat(minimum=min(values), maximum=max(values))
    return stats


def _format_stats(stats: dict[str, FieldStat]) -> str:
    lines = ["field,min,max"]
    for field in ALL_FIELDS:
        stat = stats.get(field)
        if stat:
            lines.append(f"{field},{stat.minimum},{stat.maximum}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw UAV-load dataset to PINN schema")
    parser.add_argument("--src", required=True, nargs="+", help="Input CSV file(s) or directory")
    parser.add_argument("--dst", required=True, help="Output CSV path")
    parser.add_argument("--print-stats", action="store_true", help="Print min/max stats to stdout")
    args = parser.parse_args()

    src_entries: List[str] = []
    for entry in args.src:
        if os.path.isdir(entry):
            for name in sorted(os.listdir(entry)):
                if name.endswith(".csv"):
                    src_entries.append(os.path.join(entry, name))
        else:
            src_entries.append(entry)

    if not src_entries:
        raise FileNotFoundError("No CSV input files provided")

    rows = _convert_rows(src_entries)
    if not rows:
        raise RuntimeError("No valid rows found in input dataset; check schema compatibility")

    _write_csv(rows, args.dst)

    if args.print_stats:
        print(_format_stats(_compute_stats(rows)))


if __name__ == "__main__":
    main()
