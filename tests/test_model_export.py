import json
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import pytest

from examples.pinn_inverse.force_estimation import (
    AUXILIARY_COLUMNS,
    FEATURE_COLUMNS,
    INPUT_DIM,
    NETWORK_CONFIG,
    OUTPUT_DIM,
    RESIDUAL_COMPONENT_NAMES,
    build_network,
)
from estimation.export_utils import (
    export_model_metadata,
    export_normalizer_to_json,
    export_onnx_model,
    export_torchscript_model,
)
from estimation.data_utils import Normalizer


class _DummyModel:
    def __init__(self):
        # Use the real network builder to stay close to production configuration.
        self.net = build_network()


def test_export_normalizer_and_metadata(tmp_path):
    data = np.linspace(-1.0, 1.0, len(FEATURE_COLUMNS) * 4, dtype=np.float64).reshape(
        -1, len(FEATURE_COLUMNS)
    )
    normalizer = Normalizer.from_array(data)
    norm_path = tmp_path / "normalizer_params.json"
    meta_path = tmp_path / "metadata.json"

    export_normalizer_to_json(normalizer, FEATURE_COLUMNS, str(norm_path))
    export_model_metadata(
        str(meta_path),
        feature_columns=FEATURE_COLUMNS,
        auxiliary_columns=AUXILIARY_COLUMNS,
        residual_components=RESIDUAL_COMPONENT_NAMES,
        network_config=NETWORK_CONFIG,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        dtype="float64",
    )

    with open(norm_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    assert payload["feature_columns"] == list(FEATURE_COLUMNS)
    assert len(payload["minimum"]) == len(FEATURE_COLUMNS)
    assert len(payload["range"]) == len(FEATURE_COLUMNS)

    with open(meta_path, "r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    assert metadata["input_dim"] == INPUT_DIM
    assert metadata["output_dim"] == OUTPUT_DIM
    assert metadata["network"]["width"] == NETWORK_CONFIG["width"]


def test_export_torchscript_and_onnx(tmp_path):
    model = _DummyModel()
    ts_path = tmp_path / "model.pt"
    onnx_path = tmp_path / "model.onnx"

    export_torchscript_model(model, str(ts_path), INPUT_DIM)
    pytest.importorskip("onnx")
    export_onnx_model(model, str(onnx_path), INPUT_DIM)

    assert ts_path.exists() and ts_path.stat().st_size > 0
    assert onnx_path.exists() and onnx_path.stat().st_size > 0
