import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import pandas as pd
import pytest

from examples.pinn_inverse.force_estimation import (
    AUXILIARY_COLUMNS,
    FEATURE_COLUMNS,
    ConstantManager,
    MixedFeatureSampler,
    MixedGeometry,
    Normalizer,
    build_residual_scaler,
    build_geometry,
    compute_residual_components_numpy,
    compute_residual_variance_table,
    generate_sampling_visualizations,
    RESIDUAL_COMPONENT_NAMES,
    resolve_effective_num_domain,
    resolve_sample_plot_specs,
)


def _make_dummy_df(num_rows: int = 6) -> pd.DataFrame:
    time = np.linspace(0.0, 1.0, num_rows)
    data = {"time": time}
    for idx, col in enumerate(FEATURE_COLUMNS):
        data[col] = np.linspace(idx, idx + 1.0, num_rows)
    for idx, col in enumerate(AUXILIARY_COLUMNS):
        start = 10.0 + idx
        data[col] = np.linspace(start, start + 0.5, num_rows)
    return pd.DataFrame(data)


def _make_normalizer(df: pd.DataFrame) -> Normalizer:
    return Normalizer.from_array(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))


def test_mixed_feature_sampler_interpolation_range():
    df = _make_dummy_df()
    normalizer = _make_normalizer(df)
    sampler = MixedFeatureSampler(
        df,
        normalizer,
        FEATURE_COLUMNS,
        mix_ratio=1.0,
        jitter_scale=0.0,
        seed=0,
    )
    samples = sampler.sample(8)
    assert samples.shape == (8, len(FEATURE_COLUMNS))
    denorm = normalizer.inverse(samples)
    mins = df[FEATURE_COLUMNS].min().to_numpy()
    maxs = df[FEATURE_COLUMNS].max().to_numpy()
    assert np.all(denorm >= mins - 1e-12)
    assert np.all(denorm <= maxs + 1e-12)


def test_constant_manager_nearest_neighbor_stitching():
    df = _make_dummy_df()
    normalizer = _make_normalizer(df)
    anchors = normalizer.transform(df[FEATURE_COLUMNS].to_numpy())
    const_mgr = ConstantManager(
        df,
        AUXILIARY_COLUMNS,
        anchors,
        seed=0,
        match_tol=0.5,
    )
    perturbed = anchors[0].copy()
    perturbed += 0.1  # 无精确匹配，触发最近邻逻辑
    aux = const_mgr.auxiliary(np.vstack([perturbed]))
    expected = df[AUXILIARY_COLUMNS].iloc[0].to_numpy()
    assert aux.shape == (1, len(AUXILIARY_COLUMNS))
    assert np.allclose(aux[0], expected)


def test_build_geometry_returns_mixed_geometry_when_enabled():
    df = _make_dummy_df()
    normalizer = _make_normalizer(df)
    features_norm = normalizer.transform(df[FEATURE_COLUMNS].to_numpy())
    geom = build_geometry(
        features_norm,
        df=df,
        normalizer=normalizer,
        seed=0,
        mix_ratio=1.0,
        jitter_scale=0.0,
        enable_mixed=True,
    )
    assert isinstance(geom, MixedGeometry)
    samples = geom.random_points(4)
    assert samples.shape == (4, len(FEATURE_COLUMNS))
    denorm = normalizer.inverse(samples)
    mins = df[FEATURE_COLUMNS].min().to_numpy()
    maxs = df[FEATURE_COLUMNS].max().to_numpy()
    assert np.all(denorm >= mins - 1e-12)
    assert np.all(denorm <= maxs + 1e-12)


def test_resolve_effective_num_domain_auto_mode():
    count, mode = resolve_effective_num_domain(
        explicit=0,
        data_size=2000,
        auto_enabled=True,
        ratio=0.1,
        min_points=256,
        max_points=1024,
    )
    assert mode == "auto"
    assert count == 256  # ratio gives 200, but min_points enforces 256


def test_resolve_effective_num_domain_explicit_and_disabled():
    explicit_count, explicit_mode = resolve_effective_num_domain(
        explicit=1500,
        data_size=100,
        auto_enabled=True,
        ratio=0.5,
        min_points=64,
        max_points=512,
    )
    assert explicit_mode == "explicit"
    assert explicit_count == 1500

    disabled_count, disabled_mode = resolve_effective_num_domain(
        explicit=0,
        data_size=1000,
        auto_enabled=False,
        ratio=0.5,
        min_points=64,
        max_points=512,
    )
    assert disabled_mode == "disabled"
    assert disabled_count == 0


def test_resolve_sample_plot_specs_custom_and_invalid():
    specs = resolve_sample_plot_specs("plane=accQ_x,accQ_y;vec=rho_x,rho_y,rho_z")
    assert len(specs) == 2
    assert specs[0].name == "plane"
    assert specs[0].columns == ("accQ_x", "accQ_y")
    assert specs[1].columns == ("rho_x", "rho_y", "rho_z")

    with pytest.raises(ValueError):
        resolve_sample_plot_specs("bad=foo_x,foo_y")


def test_generate_sampling_visualizations_creates_png(tmp_path):
    specs = resolve_sample_plot_specs("plane=accQ_x,accQ_y")
    samples = np.random.rand(256, len(FEATURE_COLUMNS))
    reference = np.random.rand(128, len(FEATURE_COLUMNS))
    output_dir = tmp_path / "plots"
    paths = generate_sampling_visualizations(
        samples,
        reference,
        specs,
        str(output_dir),
    )
    assert len(paths) == 1
    for path in paths:
        assert os.path.exists(path)


def test_compute_residual_variance_table_matches_numpy():
    df = _make_dummy_df(8)
    preds = np.random.randn(len(df), 6)
    residual_matrix = compute_residual_components_numpy(df, preds)
    summary = compute_residual_variance_table(df, preds)
    assert list(summary["component"]) == list(RESIDUAL_COMPONENT_NAMES)
    assert np.allclose(summary["variance"].to_numpy(), residual_matrix.var(axis=0))
    assert np.allclose(summary["mean"].to_numpy(), residual_matrix.mean(axis=0))
    assert np.allclose(summary["std"].to_numpy(), residual_matrix.std(axis=0))


def test_residual_scaler_uses_force_and_length_medians():
    df = _make_dummy_df(10)
    scaler = build_residual_scaler(df, mode="median")
    assert scaler is not None
    total_mass = (df["m_Q"] + df["m_L"]).to_numpy()
    force_scale_expected = np.median(np.abs(total_mass * df["g"].to_numpy()))
    assert np.isclose(scaler.group_scales["r1"], force_scale_expected, rtol=1e-12)
    if "l_length" in df.columns:
        torque_scale_expected = (
            np.median(np.abs(df["l_length"].to_numpy())) * force_scale_expected
        )
        assert np.isclose(scaler.group_scales["r2"], torque_scale_expected, rtol=1e-12)
    rho_norm = np.linalg.norm(df[["rho_x", "rho_y", "rho_z"]].to_numpy(), axis=1)
    assert np.isclose(scaler.group_scales["r3"], np.median(rho_norm), rtol=1e-12)


def test_residual_components_are_normalized_when_scaler_provided():
    df = _make_dummy_df(8)
    preds = np.random.randn(len(df), 6)
    raw = compute_residual_components_numpy(df, preds, residual_scaler=None)
    scaler = build_residual_scaler(df, mode="median")
    scaled = compute_residual_components_numpy(df, preds, residual_scaler=scaler)
    assert np.allclose(
        scaled[:, 0], raw[:, 0] / scaler.group_scales["r1"]
    )
    assert np.allclose(
        scaled[:, 3], raw[:, 3] / scaler.group_scales["r2"]
    )
    assert np.allclose(
        scaled[:, -1], raw[:, -1] / scaler.group_scales["r3"]
    )


def test_residual_variance_table_contains_scale_column_when_normalized():
    df = _make_dummy_df(8)
    preds = np.random.randn(len(df), 6)
    scaler = build_residual_scaler(df, mode="median")
    summary = compute_residual_variance_table(
        df,
        preds,
        residual_scaler=scaler,
    )
    assert "scale" in summary.columns
    assert np.allclose(summary["scale"].to_numpy(), scaler.scales)
