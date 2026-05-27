import argparse
import json
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from safetensors import safe_open


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIXED_POINT_SCALE = 1 << 12


LAYER0_TENSORS = {
    "input_layernorm.weight": "input_layernorm.weight.bin",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight.bin",
    "self_attn.q_proj.weight": "self_attn.q_proj.weight.bin",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight.bin",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight.bin",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight.bin",
    "mlp.up_proj.weight": "mlp.up_proj.weight.bin",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight.bin",
    "mlp.down_proj.weight": "mlp.down_proj.weight.bin",
}


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _quantize_fixed_point(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    scaled = arr.astype(np.float64) * FIXED_POINT_SCALE
    return np.where(
        scaled >= 0,
        np.floor(scaled + 0.5),
        np.ceil(scaled - 0.5),
    ).astype(np.int32)


def _load_weight_map(model_dir: str) -> Dict[str, str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        return index.get("weight_map", {})

    single_file = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_file):
        with safe_open(single_file, framework="pt", device="cpu") as f:
            return {key: os.path.basename(single_file) for key in f.keys()}

    weight_map = {}
    for filename in sorted(os.listdir(model_dir)):
        if not filename.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = filename
    return weight_map


def _iter_required_tensors(layer: int) -> Iterable[Tuple[str, str]]:
    prefix = f"model.layers.{layer}."
    for suffix, output_name in LAYER0_TENSORS.items():
        yield prefix + suffix, output_name


def _expected_output_files(output_dir: str, layer: int) -> Iterable[str]:
    prefix = f"model.layers.{layer}."
    for suffix, output_name in LAYER0_TENSORS.items():
        yield os.path.join(output_dir, output_name)
        yield os.path.join(output_dir, f"{prefix}{suffix}.bin")
    yield os.path.join(output_dir, "manifest.json")


def export_layer_weights(model_dir: str, output_dir: str, layer: int, force: bool = False) -> None:
    model_dir = _resolve_project_path(model_dir)
    output_dir = _resolve_project_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    expected_files = list(_expected_output_files(output_dir, layer))
    if not force and all(os.path.exists(path) and os.path.getsize(path) > 0 for path in expected_files):
        print(f"zk weights already exist, skip export: {output_dir}")
        print("use --force to regenerate")
        return

    weight_map = _load_weight_map(model_dir)
    missing = [key for key, _ in _iter_required_tensors(layer) if key not in weight_map]
    if missing:
        missing_lines = "\n".join(f"  - {key}" for key in missing)
        raise FileNotFoundError(f"模型 safetensors 中缺少必要张量:\n{missing_lines}")

    manifest = {
        "model_dir": model_dir,
        "layer": layer,
        "scale": FIXED_POINT_SCALE,
        "files": {},
    }

    for key, output_name in _iter_required_tensors(layer):
        shard_path = os.path.join(model_dir, weight_map[key])
        output_path = os.path.join(output_dir, output_name)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)

        quantized = _quantize_fixed_point(tensor)
        quantized.tofile(output_path)

        # Also write the full HuggingFace key name; verify.py supports both forms.
        full_name_path = os.path.join(output_dir, f"{key}.bin")
        if full_name_path != output_path:
            quantized.tofile(full_name_path)

        manifest["files"][key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "source": shard_path,
            "output": output_path,
        }
        print(f"exported {key} -> {output_path} shape={tuple(tensor.shape)}")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export layer-0 zkhook int32 weights from safetensors.")
    parser.add_argument("--model-dir", required=True, help="HuggingFace safetensors model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for int32 .bin files")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to export")
    parser.add_argument("--force", action="store_true", help="Regenerate files even if zk_weights already exist")
    args = parser.parse_args()

    export_layer_weights(args.model_dir, args.output_dir, args.layer, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
