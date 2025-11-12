from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Sequence

import numpy as np

try:  # pragma: no cover
    from deepxde.backend import backend_name, torch
except Exception:  # pragma: no cover
    backend_name = ""
    torch = None


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def export_normalizer_to_json(
    normalizer,
    feature_columns: Sequence[str],
    path: str,
) -> str:
    if not path:
        return ""
    _ensure_parent_dir(path)
    payload = {
        "feature_columns": list(feature_columns),
        "minimum": np.asarray(normalizer.minimum).tolist(),
        "range": np.asarray(normalizer.range).tolist(),
        "eps": float(getattr(normalizer, "eps", 1e-8)),
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(f"[force_estimation] 归一化参数已保存: {os.path.abspath(path)}")
    return path


def _resolve_torch_module(model):
    if backend_name != "pytorch":
        raise RuntimeError("仅支持 PyTorch backend 导出模型。")
    if torch is None:
        raise RuntimeError("未检测到 PyTorch，无法导出。")
    module = getattr(model, "net", model)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("模型不包含 PyTorch 模块，无法导出。")
    return module


def _infer_dtype_device(module):
    params = list(module.parameters())
    if not params:
        return torch.float32, torch.device("cpu")
    sample = params[0]
    return sample.dtype, sample.device


def export_torchscript_model(model, path: str, input_dim: int) -> str:
    if not path:
        return ""
    module = _resolve_torch_module(model)
    dtype, device = _infer_dtype_device(module)
    module.eval()
    target_device = torch.device("cpu")
    if device != target_device:
        module = module.to(target_device)
    example = torch.zeros(1, input_dim, dtype=dtype, device=target_device)
    with torch.no_grad():
        scripted = torch.jit.trace(module, example)
    _ensure_parent_dir(path)
    scripted.save(path)
    if device != target_device:
        module.to(device)
    print(f"[force_estimation] TorchScript 模型已保存: {os.path.abspath(path)}")
    return path


def export_onnx_model(
    model,
    path: str,
    input_dim: int,
    *,
    opset: int = 12,
) -> str:
    if not path:
        return ""
    try:
        import onnx  # noqa: F401  # pylint: disable=unused-import
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ONNX 导出需要安装 onnx 包 (pip install onnx)。") from exc
    module = _resolve_torch_module(model)
    dtype, device = _infer_dtype_device(module)
    module.eval()
    target_device = torch.device("cpu")
    if device != target_device:
        module = module.to(target_device)
    example = torch.zeros(1, input_dim, dtype=dtype, device=target_device)
    _ensure_parent_dir(path)
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            path,
            input_names=["features"],
            output_names=["forces"],
            dynamic_axes={
                "features": {0: "batch"},
                "forces": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    if device != target_device:
        module.to(device)
    print(f"[force_estimation] ONNX 模型已保存: {os.path.abspath(path)}")
    return path


def export_model_metadata(
    path: str,
    *,
    feature_columns: Sequence[str],
    auxiliary_columns: Sequence[str],
    residual_components: Sequence[str],
    network_config: Dict[str, Any],
    input_dim: int,
    output_dim: int,
    dtype: str,
) -> str:
    if not path:
        return ""
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend_name or "unknown",
        "dtype": dtype,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "feature_columns": list(feature_columns),
        "auxiliary_columns": list(auxiliary_columns),
        "residual_components": list(residual_components),
        "network": {
            "width": int(network_config["width"]),
            "depth": int(network_config["depth"]),
            "activation": str(network_config["activation"]),
            "initializer": str(network_config["initializer"]),
        },
    }
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(f"[force_estimation] 模型元数据已保存: {os.path.abspath(path)}")
    return path


__all__ = [
    "export_normalizer_to_json",
    "export_torchscript_model",
    "export_onnx_model",
    "export_model_metadata",
]
