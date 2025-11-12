import math

import pandas as pd
import pytest

from examples.pinn_inverse.force_estimation import RESIDUAL_COMPONENT_NAMES
from examples.pinn_inverse.geometry_comparison import build_comparison_table


def _make_variance_df(scale: float) -> pd.DataFrame:
    data = [
        {"component": comp, "variance": (idx + 1) * scale}
        for idx, comp in enumerate(RESIDUAL_COMPONENT_NAMES)
    ]
    return pd.DataFrame(data)


def test_build_comparison_table_basic():
    hyper = _make_variance_df(1.0)
    mixed = _make_variance_df(0.4)
    table = build_comparison_table(
        hyper,
        mixed,
        baseline_label="Hyper",
        mixed_label="Mixed",
        min_threshold=0.3,
        ideal_threshold=0.5,
    )
    assert "total_variance" in table["Metric"].values
    r1_row = table.loc[table["Metric"] == RESIDUAL_COMPONENT_NAMES[0]].iloc[0]
    assert r1_row["Status"] == "✓ 达到理想目标"
    total_row = table.loc[table["Metric"] == "total_variance"].iloc[0]
    assert math.isclose(total_row["Improvement"], 60.0)


def test_build_comparison_table_requires_components():
    hyper = _make_variance_df(1.0)
    mixed = pd.DataFrame(
        [
            {"component": "r1_x", "variance": 1.0},
        ]
    )
    with pytest.raises(ValueError):
        build_comparison_table(
            hyper,
            mixed,
            baseline_label="Hyper",
            mixed_label="Mixed",
            min_threshold=0.3,
            ideal_threshold=0.5,
        )
