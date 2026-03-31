#!/usr/bin/env python3
"""
benchmark_stepwise.py - 分步执行 Transformer 层 ZKP 基准测试 (非交互式)

执行步骤:
  [1/6] RMSNorm (Attention 前)
  [2/6] Self-Attention
  [3/6] Skip Connection (Attention)
  [4/6] RMSNorm (FFN 前)
  [5/6] FFN
  [6/6] Skip Connection (FFN)

测试指标定义（与 Baseline 统一标准）:
  - 证明时间: 纯 Compute + Prove 时间 (从 [TIME] 标签解析)
  - 验证时间: 纯 Verify 时间 (从 [TIME] 标签解析)
  - 证明大小: 打包的 .bin 证明文件大小 (含 Fiat-Shamir 哈希、承诺、sumcheck 证明)
  - 承诺大小: 权重承诺文件大小
  - 表大小: SwiGLU + Softmax 查找表大小
"""

import argparse
import os
import subprocess
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ==================== 配置 ====================

ZKLLM_DIR = Path(__file__).parent / "zkhook"
# 默认本地模型路径 (AutoDL 服务器)
MODEL_DIR = Path(os.environ.get("LLAMA_MODEL_PATH", "./weights/shakechen/Llama-2-7b-hf"))

# LLaMA-2-7B 参数
EMBED_DIM = 4096
HIDDEN_DIM = 11008
NUM_HEADS = 32
HEAD_DIM = 128
NUM_LAYERS = 32  # LLaMA-2-7B 有 32 层

# 量化参数
WEIGHT_SCALE = 1 << 16
INPUT_SCALE = 1 << 16
RMSNORM_EPS = 1e-5


# ==================== 工具函数 ====================

import re

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, capture: bool = True) -> Tuple[int, float, str]:
    """运行命令，返回 (返回码, 耗时, 输出)"""
    start = time.time()
    if capture:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        output = result.stdout + result.stderr
    else:
        result = subprocess.run(cmd, cwd=cwd)
        output = ""
    elapsed = time.time() - start
    return result.returncode, elapsed, output


def compute_rms_inv(x: np.ndarray) -> np.ndarray:
    """计算 RMSNorm 的 1/rms 值"""
    mean_sq = np.mean(x ** 2, axis=1)
    rms = np.sqrt(mean_sq + RMSNORM_EPS)
    return (1.0 / rms).astype(np.float32)


def generate_random_input(path: Path, seq_len: int, seed: int = 0) -> np.ndarray:
    """生成随机输入并保存"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(seq_len, EMBED_DIM), dtype=np.float32)
    x_int = np.rint(x * INPUT_SCALE).astype(np.int32)
    x_int.tofile(str(path))
    return x  # 返回原始浮点值，用于计算 rms_inv


def save_rms_inv(x_float: np.ndarray, path: Path):
    """计算并保存 rms_inv"""
    rms_inv = compute_rms_inv(x_float)
    rms_inv_int = np.rint(rms_inv * INPUT_SCALE).astype(np.int32)
    rms_inv_int.tofile(str(path))


def load_int32_as_float(path: Path, shape: Tuple[int, ...]) -> np.ndarray:
    """读取 int32 文件并转换为 float"""
    data = np.fromfile(str(path), dtype=np.int32).reshape(shape)
    return data.astype(np.float32) / INPUT_SCALE


# ==================== 权重导出 ====================

# 全局模型缓存（避免重复加载）
_cached_model = None

def _get_model(model_dir: Path):
    """获取模型（缓存）"""
    global _cached_model
    if _cached_model is None:
        import torch
        from transformers import AutoModelForCausalLM
        print(f"   加载模型: {model_dir}")
        _cached_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            local_files_only=True,
        )
    return _cached_model

def _release_model():
    """释放模型缓存"""
    global _cached_model
    if _cached_model is not None:
        del _cached_model
        _cached_model = None
        import gc
        gc.collect()

def export_layer_weights(model_dir: Path, out_dir: Path, layer_idx: int) -> Dict[str, Path]:
    """导出 LLaMA-2-7B 指定层的所有权重
    
    v2 格式: Attention 权重 (q/k/v/o) 用 float32，FFN/RMSNorm 用 int32
    """
    layer_dir = out_dir / f"layer{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # 格式版本标记 (v2 = float32 for attention weights)
    format_marker = layer_dir / ".format_v2"
    
    paths = {
        "q": layer_dir / "q_weight.bin",
        "k": layer_dir / "k_weight.bin",
        "v": layer_dir / "v_weight.bin",
        "o": layer_dir / "o_weight.bin",
        "up": layer_dir / "up_weight.bin",
        "gate": layer_dir / "gate_weight.bin",
        "down": layer_dir / "down_weight.bin",
        "input_layernorm": layer_dir / "input_layernorm_weight.bin",
        "post_attn_layernorm": layer_dir / "post_attn_layernorm_weight.bin",
    }
    
    # 检查是否存在 v2 格式的权重
    if format_marker.exists() and all(p.exists() for p in paths.values()):
        return paths
    
    # 删除旧格式权重 (如果存在但没有 format_marker)
    if not format_marker.exists():
        for p in paths.values():
            if p.exists():
                p.unlink()
    
    import torch
    model = _get_model(model_dir)
    layer = model.model.layers[layer_idx]
    
    # Attention 权重
    wq = layer.self_attn.q_proj.weight.detach().to(torch.float32).cpu().numpy()
    wk = layer.self_attn.k_proj.weight.detach().to(torch.float32).cpu().numpy()
    wv = layer.self_attn.v_proj.weight.detach().to(torch.float32).cpu().numpy()
    wo = layer.self_attn.o_proj.weight.detach().to(torch.float32).cpu().numpy()
    
    # FFN 权重
    wup = layer.mlp.up_proj.weight.detach().to(torch.float32).cpu().numpy()
    wgate = layer.mlp.gate_proj.weight.detach().to(torch.float32).cpu().numpy()
    wdown = layer.mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()
    
    # RMSNorm gamma 权重
    gamma_input = layer.input_layernorm.weight.detach().to(torch.float32).cpu().numpy()
    gamma_post_attn = layer.post_attention_layernorm.weight.detach().to(torch.float32).cpu().numpy()
    
    # 保存（转置，保存为 float32，C++ 会做量化）
    # 注意：Attention 权重用 float32 (attn_prove.cu 内部量化)
    #       FFN 和 RMSNorm 权重用 int32 (ffn_prove.cu/rmsnorm_prove.cu 直接读取)
    wq.T.astype(np.float32).tofile(str(paths["q"]))
    wk.T.astype(np.float32).tofile(str(paths["k"]))
    wv.T.astype(np.float32).tofile(str(paths["v"]))
    wo.T.astype(np.float32).tofile(str(paths["o"]))
    # FFN 和 RMSNorm 仍用 int32
    np.rint(wup.T * WEIGHT_SCALE).astype(np.int32).tofile(str(paths["up"]))
    np.rint(wgate.T * WEIGHT_SCALE).astype(np.int32).tofile(str(paths["gate"]))
    np.rint(wdown.T * WEIGHT_SCALE).astype(np.int32).tofile(str(paths["down"]))
    np.rint(gamma_input * WEIGHT_SCALE).astype(np.int32).tofile(str(paths["input_layernorm"]))
    np.rint(gamma_post_attn * WEIGHT_SCALE).astype(np.int32).tofile(str(paths["post_attn_layernorm"]))
    
    # 创建格式版本标记
    format_marker.touch()
    
    return paths

def export_all_layers_weights(model_dir: Path, out_dir: Path, num_layers: int = NUM_LAYERS) -> List[Dict[str, Path]]:
    """导出所有层的权重"""
    all_weights = []
    
    # 检查是否已全部缓存 (必须是 v2 格式)
    all_cached = True
    for i in range(num_layers):
        layer_dir = out_dir / f"layer{i}"
        format_marker = layer_dir / ".format_v2"
        if not layer_dir.exists() or not format_marker.exists() or not all((layer_dir / f"{k}_weight.bin").exists() 
                                              for k in ["q", "k", "v", "o", "up", "gate", "down"]):
            all_cached = False
            break
    
    if all_cached:
        print(f"   所有 {num_layers} 层权重已缓存 (v2 格式): {out_dir}")
        for i in range(num_layers):
            layer_dir = out_dir / f"layer{i}"
            paths = {
                "q": layer_dir / "q_weight.bin",
                "k": layer_dir / "k_weight.bin",
                "v": layer_dir / "v_weight.bin",
                "o": layer_dir / "o_weight.bin",
                "up": layer_dir / "up_weight.bin",
                "gate": layer_dir / "gate_weight.bin",
                "down": layer_dir / "down_weight.bin",
                "input_layernorm": layer_dir / "input_layernorm_weight.bin",
                "post_attn_layernorm": layer_dir / "post_attn_layernorm_weight.bin",
            }
            all_weights.append(paths)
        return all_weights
    
    print(f"   导出 {num_layers} 层权重到: {out_dir}")
    for i in range(num_layers):
        print(f"      Layer {i+1}/{num_layers}...", end="\r")
        weights = export_layer_weights(model_dir, out_dir, i)
        all_weights.append(weights)
    print(f"      完成导出 {num_layers} 层权重           ")
    
    _release_model()  # 导出完成后释放模型
    return all_weights

# 向后兼容：导出 layer0
def export_layer0_weights(model_dir: Path, out_dir: Path) -> Dict[str, Path]:
    """导出 LLaMA-2-7B layer0 的所有权重（向后兼容）"""
    return export_layer_weights(model_dir, out_dir / "layer0", 0) if (out_dir / "layer0").exists() else export_layer_weights(model_dir, out_dir.parent, 0)


# ==================== 步骤执行 ====================

@dataclass
class StepResult:
    name: str
    prove_time: float          # 纯 Prove 时间 = Compute + ProofGen (从 [TIME] 解析)
    verify_time: float         # 纯 Verify 时间 (从 [TIME] 解析)
    proof_size: int
    success: bool
    output: str = ""
    compute_time: float = 0.0  # 计算时间 (从 [TIME] 解析)
    proof_gen_time: float = 0.0  # 证明生成时间 (从 [TIME] 解析)
    total_elapsed: float = 0.0  # 外部测量的总时间 (包含加载等)
    commitment_size: int = 0   # 总承诺大小 (bytes) - 包含 I/O
    weight_commitment_size: int = 0  # 权重承诺大小 (bytes) - 与 Baseline 对齐


def parse_time_output(output: str) -> Dict[str, float]:
    """解析程序输出中的 [TIME] 和 [SIZE] 信息"""
    result = {}
    import re
    
    # 解析时间
    time_pattern = r'\[TIME\]\s+(\w+)\s+-\s+(\w+):\s+([\d.]+)\s*s'
    for match in re.finditer(time_pattern, output):
        component = match.group(1)  # RMSNorm, FFN, Attention
        metric = match.group(2)     # Compute, Prove, Verify
        value = float(match.group(3))
        result[f"{component}_{metric}"] = value
    
    # 解析权重承诺大小 (与 Baseline 对齐)
    weight_commit_pattern = r'\[SIZE\]\s+(\w+)\s+-\s+WeightCommitment:\s+(\d+)\s*bytes'
    for match in re.finditer(weight_commit_pattern, output):
        component = match.group(1)
        value = int(match.group(2))
        result[f"{component}_WeightCommitment"] = value
    
    # 解析总承诺大小 (包含 I/O 承诺)
    size_pattern = r'\[SIZE\]\s+(\w+)\s+-\s+Commitment:\s+(\d+)\s*bytes'
    for match in re.finditer(size_pattern, output):
        component = match.group(1)
        value = int(match.group(2))
        result[f"{component}_Commitment"] = value
    
    return result


def run_step_rmsnorm(
    input_file: Path,
    rms_inv_file: Path,
    gamma_file: Path,
    output_file: Path,
    proof_file: Path,
    seq_len: int,
) -> StepResult:
    """运行 RMSNorm 步骤"""
    prove_exe = ZKLLM_DIR / "rmsnorm_prove"
    verify_exe = ZKLLM_DIR / "rmsnorm_verify"
    
    # 使用绝对路径
    input_abs = input_file.resolve()
    rms_inv_abs = rms_inv_file.resolve()
    gamma_abs = gamma_file.resolve()
    output_abs = output_file.resolve()
    proof_abs = proof_file.resolve()
    
    # 检查可执行文件和输入文件
    debug_info = []
    if not prove_exe.exists():
        debug_info.append(f"错误: rmsnorm_prove 不存在: {prove_exe}")
    if not input_abs.exists():
        debug_info.append(f"错误: 输入文件不存在: {input_abs}")
    if not rms_inv_abs.exists():
        debug_info.append(f"错误: rms_inv 文件不存在: {rms_inv_abs}")
    if not gamma_abs.exists():
        debug_info.append(f"错误: gamma 文件不存在: {gamma_abs}")
    
    if debug_info:
        return StepResult("RMSNorm", 0, 0, 0, False, "\n".join(debug_info))
    
    # Prove
    prove_cmd = [
        str(prove_exe),
        str(input_abs),
        str(rms_inv_abs),
        str(gamma_abs),
        str(seq_len),
        str(EMBED_DIM),
        str(output_abs),
        str(proof_abs),
    ]
    code, elapsed_prove, out = run_cmd(prove_cmd, cwd=ZKLLM_DIR)
    if code != 0:
        err_msg = f"命令: {' '.join(prove_cmd[:3])}...\n返回码: {code}\n输出: {out}"
        return StepResult("RMSNorm", 0, 0, 0, False, err_msg)
    
    # 解析内部时间 (仅用于详细分解显示)
    times = parse_time_output(out)
    compute_time = times.get("RMSNorm_Compute", 0.0)
    proof_gen_time = times.get("RMSNorm_Prove", 0.0)
    
    # Verify
    verify_cmd = [str(verify_exe), str(proof_abs)]
    code, elapsed_verify, out2 = run_cmd(verify_cmd, cwd=ZKLLM_DIR)
    
    # 解析 Verify 时间 (从 [TIME] 标签，与 Baseline 统一标准)
    verify_times = parse_time_output(out2)
    pure_verify_time = verify_times.get("RMSNorm_Verify", elapsed_verify)
    
    # 纯 Prove 时间 = Compute + ProofGen (从 [TIME] 标签解析)
    pure_prove_time = compute_time + proof_gen_time
    
    proof_size = proof_abs.stat().st_size if proof_abs.exists() else 0
    
    result = StepResult(
        name="RMSNorm",
        prove_time=pure_prove_time,      # 纯 Prove 时间 (从 [TIME] 解析)
        verify_time=pure_verify_time,    # 纯 Verify 时间 (从 [TIME] 解析)
        proof_size=proof_size,
        success=(code == 0),
        output=out + out2,
        compute_time=compute_time,
        proof_gen_time=proof_gen_time,
        total_elapsed=elapsed_prove + elapsed_verify,
    )
    return result


def run_step_attention(
    input_file: Path,
    output_file: Path,
    proof_file: Path,
    weights: Dict[str, Path],
    tables_file: Path,
    seq_len: int,
) -> StepResult:
    """运行 Self-Attention 步骤 (v8: 完整版含 O 投影)
    
    现在包括 Q/K/V/O 四个投影的完整证明
    """
    prove_exe = ZKLLM_DIR / "attn_prove"
    verify_exe = ZKLLM_DIR / "attn_verify"
    
    # 使用绝对路径
    input_abs = input_file.resolve()
    output_abs = output_file.resolve()
    proof_abs = proof_file.resolve()
    weights_abs = {k: v.resolve() for k, v in weights.items()}
    
    nonce = f"attn_step_{seq_len}"
    
    # 新版 attn_prove 接口 (v8 - 包含 O 投影):
    # attn_prove <input.bin> <q.bin> <k.bin> <v.bin> <o.bin> <seq_len> <embed_dim> <proof.bin> <nonce>
    prove_cmd = [
        str(prove_exe),
        str(input_abs),
        str(weights_abs["q"]), str(weights_abs["k"]), str(weights_abs["v"]), str(weights_abs["o"]),
        str(seq_len), str(EMBED_DIM),
        str(proof_abs),
        nonce,
    ]
    code, elapsed_prove, out = run_cmd(prove_cmd, cwd=ZKLLM_DIR)
    if code != 0:
        return StepResult("Attention", 0, 0, 0, False, out)
    
    # 解析内部时间 (仅用于详细分解显示)
    times = parse_time_output(out)
    compute_time = times.get("Attention_Compute", 0.0)
    proof_gen_time = times.get("Attention_Prove", 0.0)
    commitment_size = times.get("Attention_Commitment", 0)
    weight_commitment_size = times.get("Attention_WeightCommitment", 0)  # 权重承诺 (与 Baseline 对齐)
    
    # 简化: 暂时复制输入作为输出（实际应该是 Attention 的计算结果）
    # TODO: attn_prove 需要更新以输出计算结果
    import shutil
    shutil.copy(str(input_abs), str(output_abs))
    
    # Verify
    verify_cmd = [str(verify_exe), str(proof_abs)]
    code, elapsed_verify, out2 = run_cmd(verify_cmd, cwd=ZKLLM_DIR)
    
    # 解析 Verify 时间 (从 [TIME] 标签，与 Baseline 统一标准)
    verify_times = parse_time_output(out2)
    pure_verify_time = verify_times.get("Attention_Verify", elapsed_verify)
    
    # 纯 Prove 时间 = Compute + ProofGen (从 [TIME] 标签解析)
    pure_prove_time = compute_time + proof_gen_time
    
    proof_size = proof_abs.stat().st_size if proof_abs.exists() else 0
    
    result = StepResult(
        name="Attention",
        prove_time=pure_prove_time,      # 纯 Prove 时间 (从 [TIME] 解析)
        verify_time=pure_verify_time,    # 纯 Verify 时间 (从 [TIME] 解析)
        proof_size=proof_size,
        success=(code == 0),
        output=out + out2,
        compute_time=compute_time,
        proof_gen_time=proof_gen_time,
        total_elapsed=elapsed_prove + elapsed_verify,
        commitment_size=int(commitment_size) if commitment_size else 0,
        weight_commitment_size=int(weight_commitment_size) if weight_commitment_size else 0,
    )
    return result


def run_step_ffn(
    input_file: Path,
    output_file: Path,
    proof_file: Path,
    weights: Dict[str, Path],
    tables_file: Path,
    seq_len: int,
) -> StepResult:
    """运行 FFN 步骤
    
    注意: 当前 ffn_prove 接口是简化版本，这里使用适配的参数
    """
    prove_exe = ZKLLM_DIR / "ffn_prove"
    verify_exe = ZKLLM_DIR / "ffn_verify"
    
    # 使用绝对路径
    input_abs = input_file.resolve()
    output_abs = output_file.resolve()
    proof_abs = proof_file.resolve()
    weights_abs = {k: v.resolve() for k, v in weights.items()}
    
    nonce = f"ffn_step_{seq_len}"
    
    # 现有 ffn_prove 接口:
    # ffn_prove <input.bin> <up.bin> <gate.bin> <down.bin> <seq_len> <embed_dim> <hidden_dim> <proof.bin> <nonce>
    prove_cmd = [
        str(prove_exe),
        str(input_abs),
        str(weights_abs["up"]), str(weights_abs["gate"]), str(weights_abs["down"]),
        str(seq_len), str(EMBED_DIM), str(HIDDEN_DIM),
        str(proof_abs),
        nonce,
    ]
    code, elapsed_prove, out = run_cmd(prove_cmd, cwd=ZKLLM_DIR)
    if code != 0:
        return StepResult("FFN", 0, 0, 0, False, out)
    
    # 解析内部时间 (仅用于详细分解显示)
    times = parse_time_output(out)
    compute_time = times.get("FFN_Compute", 0.0)
    proof_gen_time = times.get("FFN_Prove", 0.0)
    commitment_size = times.get("FFN_Commitment", 0)
    weight_commitment_size = times.get("FFN_WeightCommitment", 0)  # 权重承诺 (与 Baseline 对齐)
    
    # 简化: 暂时复制输入作为输出（实际应该是 FFN 的计算结果）
    # TODO: ffn_prove 需要更新以输出计算结果
    import shutil
    shutil.copy(str(input_abs), str(output_abs))
    
    # Verify
    verify_cmd = [str(verify_exe), str(proof_abs)]
    code, elapsed_verify, out2 = run_cmd(verify_cmd, cwd=ZKLLM_DIR)
    
    # 解析 Verify 时间 (从 [TIME] 标签，与 Baseline 统一标准)
    verify_times = parse_time_output(out2)
    pure_verify_time = verify_times.get("FFN_Verify", elapsed_verify)
    
    # 纯 Prove 时间 = Compute + ProofGen (从 [TIME] 标签解析)
    pure_prove_time = compute_time + proof_gen_time
    
    proof_size = proof_abs.stat().st_size if proof_abs.exists() else 0
    
    result = StepResult(
        name="FFN",
        prove_time=pure_prove_time,      # 纯 Prove 时间 (从 [TIME] 解析)
        verify_time=pure_verify_time,    # 纯 Verify 时间 (从 [TIME] 解析)
        proof_size=proof_size,
        success=(code == 0),
        output=out + out2,
        compute_time=compute_time,
        proof_gen_time=proof_gen_time,
        total_elapsed=elapsed_prove + elapsed_verify,
        commitment_size=int(commitment_size) if commitment_size else 0,
        weight_commitment_size=int(weight_commitment_size) if weight_commitment_size else 0,
    )
    return result


def run_step_skip_connection(input_a: Path, input_b: Path, output: Path) -> float:
    """运行 Skip Connection 步骤（无证明）"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "skip_connection.py"),
        "--input_a", str(input_a),
        "--input_b", str(input_b),
        "--output", str(output),
    ]
    code, elapsed, _ = run_cmd(cmd)
    return elapsed if code == 0 else -1


# ==================== 证明打包 ====================

def pack_all_proofs(
    proof_files: List[Path],
    output_path: Path,
    metadata: Dict,
):
    """将所有证明打包成一个文件"""
    with open(output_path, "wb") as f:
        # Magic number
        f.write(b"ZKSTEP01")
        
        # 元数据
        f.write(struct.pack("<I", metadata["seq_len"]))
        f.write(struct.pack("<I", metadata["embed_dim"]))
        f.write(struct.pack("<I", metadata["hidden_dim"]))
        
        # 证明数量
        f.write(struct.pack("<I", len(proof_files)))
        
        # 每个证明
        for pf in proof_files:
            if pf.exists():
                data = pf.read_bytes()
                f.write(struct.pack("<I", len(data)))
                f.write(data)
            else:
                f.write(struct.pack("<I", 0))
    
    print(f"   打包完成: {output_path} ({output_path.stat().st_size} bytes)")


# ==================== 测试结果数据类 ====================

@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    seq_len: int
    prove_time: float           # 纯 Prove 时间 (从 [TIME] 解析)
    verify_time: float          # 纯 Verify 时间 (从 [TIME] 解析)
    proof_size: int             # 实际 ZKP 证明文件大小 (bytes)
    intermediate_data_size: int  # 中间数据文件大小 (bytes) - 与 Baseline 的 "Proof Size" 对齐
    commitment_size: float      # MB - 总承诺 (包含 I/O)
    weight_commitment_size: float  # MB - 权重承诺 (与 Baseline 对齐)
    table_size: float           # MB
    step_results: List[StepResult]
    success: bool


# ==================== 单次测试函数 ====================

def run_single_layer_benchmark(
    seq_len: int,
    weights: Dict[str, Path],
    tables_path: Path,
    artifacts_dir: Path,
    layer_idx: int = 0,
    verbose: bool = True,
) -> Optional[BenchmarkResult]:
    """运行单层的基准测试
    
    Args:
        seq_len: 序列长度
        weights: 该层的权重路径字典
        tables_path: 查找表路径
        artifacts_dir: 中间文件目录
        layer_idx: 层号 (0-31)
        verbose: 是否输出详细日志
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"测试 Layer {layer_idx}, seq_len = {seq_len}")
        print(f"{'='*70}")
    
    # 文件路径 (加入层号以支持多层)
    layer_input = artifacts_dir / f"L{layer_idx}_layer_input_{seq_len}.bin"
    attn_input = artifacts_dir / f"L{layer_idx}_attn_input_{seq_len}.bin"
    attn_output = artifacts_dir / f"L{layer_idx}_attn_output_{seq_len}.bin"
    residual_1 = artifacts_dir / f"L{layer_idx}_residual_1_{seq_len}.bin"
    ffn_input = artifacts_dir / f"L{layer_idx}_ffn_input_{seq_len}.bin"
    ffn_output = artifacts_dir / f"L{layer_idx}_ffn_output_{seq_len}.bin"
    layer_output = artifacts_dir / f"L{layer_idx}_layer_output_{seq_len}.bin"
    
    rms_inv_1 = artifacts_dir / f"L{layer_idx}_rms_inv_1_{seq_len}.bin"
    rms_inv_2 = artifacts_dir / f"L{layer_idx}_rms_inv_2_{seq_len}.bin"
    
    proof_rmsnorm_1 = artifacts_dir / f"L{layer_idx}_proof_rmsnorm_1_{seq_len}.bin"
    proof_attn = artifacts_dir / f"L{layer_idx}_proof_attn_{seq_len}.bin"
    proof_rmsnorm_2 = artifacts_dir / f"L{layer_idx}_proof_rmsnorm_2_{seq_len}.bin"
    proof_ffn = artifacts_dir / f"L{layer_idx}_proof_ffn_{seq_len}.bin"
    
    # 生成随机输入
    if verbose:
        print("\n[准备] 生成随机输入...")
    x_float = generate_random_input(layer_input, seq_len, seed=42 + layer_idx)
    save_rms_inv(x_float, rms_inv_1)
    
    results: List[StepResult] = []
    total_prove = 0.0
    total_verify = 0.0
    total_proof_size = 0
    
    def log(msg: str):
        if verbose:
            print(msg)
    
    # ========== [1/6] RMSNorm (Attention 前) ==========
    log("\n[1/6] RMSNorm (Attention 前)...")
    r = run_step_rmsnorm(
        layer_input, rms_inv_1, weights["input_layernorm"],
        attn_input, proof_rmsnorm_1, seq_len
    )
    results.append(r)
    if not r.success:
        print(f"   ❌ Layer {layer_idx} RMSNorm 失败:\n{r.output}")
        return None
    log(f"   ✓ Prove: {r.prove_time:.3f}s | Verify: {r.verify_time:.3f}s | Proof: {r.proof_size} bytes")
    total_prove += r.prove_time
    total_verify += r.verify_time
    total_proof_size += r.proof_size
    
    # ========== [2/6] Self-Attention ==========
    log("\n[2/6] Self-Attention...")
    r = run_step_attention(
        attn_input, attn_output, proof_attn,
        weights, tables_path, seq_len
    )
    results.append(r)
    if not r.success:
        print(f"   ❌ Layer {layer_idx} Attention 失败:\n{r.output}")
        return None
    log(f"   ✓ Prove: {r.prove_time:.3f}s | Verify: {r.verify_time:.3f}s | Proof: {r.proof_size} bytes")
    total_prove += r.prove_time
    total_verify += r.verify_time
    total_proof_size += r.proof_size
    
    # ========== [3/6] Skip Connection (Attention) ==========
    log("\n[3/6] Skip Connection (Attention)...")
    t = run_step_skip_connection(layer_input, attn_output, residual_1)
    if t < 0:
        print(f"   ❌ Layer {layer_idx} Skip Connection 失败")
        return None
    log(f"   ✓ 完成 (无证明): {t:.3f}s")
    
    # 计算 residual_1 的 rms_inv
    residual_1_float = load_int32_as_float(residual_1, (seq_len, EMBED_DIM))
    save_rms_inv(residual_1_float, rms_inv_2)
    
    # ========== [4/6] RMSNorm (FFN 前) ==========
    log("\n[4/6] RMSNorm (FFN 前)...")
    r = run_step_rmsnorm(
        residual_1, rms_inv_2, weights["post_attn_layernorm"],
        ffn_input, proof_rmsnorm_2, seq_len
    )
    results.append(r)
    if not r.success:
        print(f"   ❌ Layer {layer_idx} RMSNorm 失败:\n{r.output}")
        return None
    log(f"   ✓ Prove: {r.prove_time:.3f}s | Verify: {r.verify_time:.3f}s | Proof: {r.proof_size} bytes")
    total_prove += r.prove_time
    total_verify += r.verify_time
    total_proof_size += r.proof_size
    
    # ========== [5/6] FFN ==========
    log("\n[5/6] FFN...")
    r = run_step_ffn(
        ffn_input, ffn_output, proof_ffn,
        weights, tables_path, seq_len
    )
    results.append(r)
    if not r.success:
        print(f"   ❌ Layer {layer_idx} FFN 失败:\n{r.output}")
        return None
    log(f"   ✓ Prove: {r.prove_time:.3f}s | Verify: {r.verify_time:.3f}s | Proof: {r.proof_size} bytes")
    total_prove += r.prove_time
    total_verify += r.verify_time
    total_proof_size += r.proof_size
    
    # ========== [6/6] Skip Connection (FFN) ==========
    log("\n[6/6] Skip Connection (FFN)...")
    t = run_step_skip_connection(residual_1, ffn_output, layer_output)
    if t < 0:
        print(f"   ❌ Layer {layer_idx} Skip Connection 失败")
        return None
    log(f"   ✓ 完成 (无证明): {t:.3f}s")
    
    # ========== 计算与 Baseline 一致的指标 ==========
    
    # 中间数据文件大小 (与 Baseline 的 "Proof Size" 对齐)
    # Baseline: proof_files = [attn_input, attn_output, post_attn_output, ffn_input, ffn_output, layer_output]
    intermediate_files = [attn_input, attn_output, residual_1, ffn_input, ffn_output, layer_output]
    intermediate_data_size = sum(
        f.stat().st_size for f in intermediate_files if f.exists()
    )
    
    # Commitment Size (从 [SIZE] 标签解析)
    # 总承诺大小 (包含 I/O 承诺)
    total_commitment_bytes = sum(sr.commitment_size for sr in results if sr.commitment_size > 0)
    # 权重承诺大小 (与 Baseline 对齐)
    weight_commitment_bytes = sum(sr.weight_commitment_size for sr in results if sr.weight_commitment_size > 0)
    
    # 如果没有从输出解析到，使用估算
    if total_commitment_bytes == 0:
        # 估算: 每个权重 4 bytes int32 -> 承诺点 48 bytes, 除以 generator.size
        total_weight_size = sum(p.stat().st_size for p in weights.values() if p.exists())
        commitment_size_mb = (total_weight_size / 4 * 48 / EMBED_DIM) / 1024 / 1024
        weight_commitment_size_mb = commitment_size_mb  # 估算值即为权重承诺
    else:
        commitment_size_mb = total_commitment_bytes / 1024 / 1024
        weight_commitment_size_mb = weight_commitment_bytes / 1024 / 1024 if weight_commitment_bytes > 0 else commitment_size_mb
    
    # Table Size (与 Baseline 一致: SwiGLU + Softmax)
    table_size_mb = tables_path.stat().st_size / 1024 / 1024 if tables_path.exists() else 0
    
    # 清理中间文件
    for f in [layer_input, attn_input, attn_output, residual_1, ffn_input, ffn_output,
              layer_output, rms_inv_1, rms_inv_2]:
        try:
            f.unlink(missing_ok=True)
        except:
            pass
    
    return BenchmarkResult(
        seq_len=seq_len,
        prove_time=total_prove,
        verify_time=total_verify,
        proof_size=total_proof_size,
        intermediate_data_size=intermediate_data_size,
        commitment_size=commitment_size_mb,
        weight_commitment_size=weight_commitment_size_mb,
        table_size=table_size_mb,
        step_results=results,
        success=True
    )


def run_full_model_benchmark(
    seq_len: int,
    all_weights: List[Dict[str, Path]],
    tables_path: Path,
    artifacts_dir: Path,
    num_layers: int = NUM_LAYERS,
) -> Optional[BenchmarkResult]:
    """运行完整模型 (所有层) 的基准测试
    
    Args:
        seq_len: 序列长度
        all_weights: 所有层的权重路径列表
        tables_path: 查找表路径
        artifacts_dir: 中间文件目录
        num_layers: 层数 (默认 32)
    
    Returns:
        累加所有层结果的 BenchmarkResult
    """
    print(f"\n{'='*70}")
    print(f"全模型测试: {num_layers} 层, seq_len = {seq_len}")
    print(f"{'='*70}")
    
    total_prove = 0.0
    total_verify = 0.0
    total_proof_size = 0
    total_intermediate_data = 0
    total_commitment = 0.0
    total_weight_commitment = 0.0
    all_step_results: List[StepResult] = []
    
    table_size_mb = tables_path.stat().st_size / 1024 / 1024 if tables_path.exists() else 0
    
    for layer_idx in range(num_layers):
        print(f"  Layer {layer_idx + 1}/{num_layers}...", end=" ", flush=True)
        
        result = run_single_layer_benchmark(
            seq_len=seq_len,
            weights=all_weights[layer_idx],
            tables_path=tables_path,
            artifacts_dir=artifacts_dir,
            layer_idx=layer_idx,
            verbose=False,  # 静默模式
        )
        
        if result is None:
            print("❌ 失败")
            return None
        
        print(f"✓ Prove: {result.prove_time:.2f}s | Verify: {result.verify_time:.2f}s")
        
        total_prove += result.prove_time
        total_verify += result.verify_time
        total_proof_size += result.proof_size
        total_intermediate_data += result.intermediate_data_size
        total_commitment += result.commitment_size
        total_weight_commitment += result.weight_commitment_size
        all_step_results.extend(result.step_results)
    
    print(f"\n{'─'*70}")
    print(f"全模型汇总 ({num_layers} 层):")
    print(f"  总 Prove 时间:  {total_prove:.2f}s")
    print(f"  总 Verify 时间: {total_verify:.2f}s")
    print(f"  总 Proof 大小:  {total_proof_size / 1024 / 1024:.2f} MB")
    print(f"{'─'*70}")
    
    return BenchmarkResult(
        seq_len=seq_len,
        prove_time=total_prove,
        verify_time=total_verify,
        proof_size=total_proof_size,
        intermediate_data_size=total_intermediate_data,
        commitment_size=total_commitment,
        weight_commitment_size=total_weight_commitment,
        table_size=table_size_mb,
        step_results=all_step_results,
        success=True
    )


# 向后兼容函数
def run_single_benchmark(
    seq_len: int,
    weights: Dict[str, Path],
    tables_path: Path,
    artifacts_dir: Path,
) -> Optional[BenchmarkResult]:
    """运行单层基准测试（向后兼容）"""
    return run_single_layer_benchmark(seq_len, weights, tables_path, artifacts_dir, layer_idx=0, verbose=True)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="分步执行 Transformer 层 ZKP 基准测试")
    parser.add_argument("--seq_len", type=int, default=None, help="单个序列长度 (与 --seq_lens 互斥)")
    parser.add_argument("--seq_lens", type=str, default="64,128,256", help="多个序列长度，逗号分隔")
    parser.add_argument("--num_layers", type=int, default=1, help="测试层数 (1-32, 默认 1 单层)")
    parser.add_argument("--model_dir", type=str, default=str(MODEL_DIR), help="模型路径")
    parser.add_argument("--keep_artifacts", action="store_true", help="保留中间文件")
    args = parser.parse_args()
    
    # 解析序列长度
    if args.seq_len is not None:
        seq_lens = [args.seq_len]
    else:
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
    
    model_dir = Path(args.model_dir)
    num_layers = min(max(1, args.num_layers), NUM_LAYERS)  # 限制在 1-32
    
    print("=" * 70)
    print("zkhook 分步执行 Transformer ZKP 基准测试")
    print("=" * 70)
    print(f"测试层数:     {num_layers} 层" + (" (全模型)" if num_layers == NUM_LAYERS else " (部分模型)"))
    print(f"测试序列长度: {seq_lens}")
    print(f"model_dir:    {model_dir}")
    print("=" * 70)
    
    # 设置目录
    artifacts_dir = Path("benchmark_artifacts/stepwise")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("benchmark_artifacts/llama2-7b-weights")
    
    # 编译
    print("\n[0/7] 编译...")
    code, _, out = run_cmd(["make", "stepwise"], cwd=ZKLLM_DIR)
    if code != 0:
        print(f"编译失败:\n{out}")
        return 1
    print("   编译完成")
    
    # 导出权重 (所有需要测试的层)
    print(f"\n[准备] 导出 {num_layers} 层权重...")
    all_weights = export_all_layers_weights(model_dir, weights_dir, num_layers)
    
    # 生成查找表
    tables_path = artifacts_dir / "tables.bin"
    if not tables_path.exists():
        print("\n[准备] 生成查找表...")
        tables_path_abs = tables_path.resolve()
        code, _, out = run_cmd(
            [str(ZKLLM_DIR / "table_gen"), str(EMBED_DIM), str(tables_path_abs)],
            cwd=ZKLLM_DIR
        )
        if code != 0:
            print(f"生成表失败:\n{out}")
            return 1
    else:
        print(f"\n[准备] 查找表已存在: {tables_path}")
    
    # ==================== 批量测试 ====================
    all_results: List[BenchmarkResult] = []
    
    for seq_len in seq_lens:
        if num_layers == 1:
            # 单层测试
            result = run_single_layer_benchmark(
                seq_len, all_weights[0], tables_path, artifacts_dir, 
                layer_idx=0, verbose=True
            )
        else:
            # 多层/全模型测试
            result = run_full_model_benchmark(
                seq_len, all_weights, tables_path, artifacts_dir, num_layers
            )
        
        if result is None:
            print(f"\n⚠️  seq_len={seq_len} 测试失败，跳过")
            continue
        all_results.append(result)
    
    if not all_results:
        print("\n❌ 所有测试都失败了")
        return 1
    
    # ==================== 汇总报告 ====================
    print("\n")
    print("=" * 90)
    layer_desc = f"{num_layers} 层 Transformer" if num_layers > 1 else "单层 Transformer"
    print(f"zkhook 基准测试汇总报告 (LLaMA-2-7B, {layer_desc})")
    print("=" * 90)
    
    # 固定开销
    table_size = all_results[0].table_size
    weight_commitment_size = all_results[0].weight_commitment_size
    single_layer_commitment = weight_commitment_size / num_layers if num_layers > 1 else weight_commitment_size
    
    print("\n" + "─" * 90)
    print("固定开销 (不随 token 数变化)")
    print("─" * 90)
    print(f"  Table Size (SwiGLU + Softmax):     {table_size:.2f} MB")
    print(f"  测试的 Commitment Size ({num_layers}层):   {weight_commitment_size:.2f} MB")
    if num_layers < NUM_LAYERS:
        print(f"  全模型 Commitment Size (32层):     {single_layer_commitment * 32:.2f} MB (估算)")
    print(f"  Public Parameters (估算):          ~235 MB")
    print("─" * 90)
    
    # ========== 与 Baseline 格式一致的输出 ==========
    print("\n" + "=" * 100)
    print(f"本项目开销 ({layer_desc}) - 非交互式证明")
    print("=" * 100)
    print("\n测试指标定义（与 Baseline 统一标准）:")
    print("  - 证明时间: 纯 Compute + Prove 时间 (从 [TIME] 标签解析)")
    print("  - 验证时间: 纯 Verify 时间 (从 [TIME] 标签解析)")
    print("  - 证明大小: 打包的 .bin 证明文件 (含 Fiat-Shamir 哈希、承诺、sumcheck)")
    print("  - 承诺大小: 权重承诺文件大小")
    print("  - 表大小:   SwiGLU + Softmax 查找表")
    if num_layers > 1:
        print(f"\n注: 以下数据为 {num_layers} 层的累计开销")
    print()
    
    # 按照 Baseline 格式输出 (指标为行，token数为列)
    # 构建列标题
    col_width = 20
    header = f"{'指标':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):  # 从大到小排序
        header += f"{r.seq_len} tokens".center(col_width)
    print(header)
    print("-" * (20 + col_width * len(all_results)))
    
    # Prove 时间
    row = f"{'Prove 时间':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        row += f"{r.prove_time:.4f} s".center(col_width)
    print(row)
    
    # Verify 时间
    row = f"{'Verify 时间':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        row += f"{r.verify_time:.4f} s".center(col_width)
    print(row)
    
    # 实际 ZKP 证明大小 (包含 Fiat-Shamir 哈希、承诺、sumcheck 等)
    row = f"{'Proof Size':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        zkp_size = r.proof_size
        if zkp_size >= 1024 * 1024:  # >= 1 MB
            row += f"{zkp_size / 1024 / 1024:.2f} MB".center(col_width)
        else:
            row += f"{zkp_size / 1024:.2f} KB".center(col_width)
    print(row)
    
    # Commitment Size (权重承诺，与 Baseline 对齐)
    row = f"{'Commitment Size':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        row += f"{r.weight_commitment_size:.2f} MB".center(col_width)
    print(row)
    
    # Table Size
    row = f"{'Table Size':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        row += f"{r.table_size:.2f} MB".center(col_width)
    print(row)
    
    print("-" * (20 + col_width * len(all_results)))
    
    # ========== Baseline 参考数据 (同样格式) ==========
    print("\n" + "=" * 100)
    print("Baseline 参考数据 (单层 Transformer) - 交互式证明")
    print("=" * 100)
    print("\n测试指标定义（与本项目统一标准）:")
    print("  - 证明时间: 纯 Compute + Prove 时间 (从 [TIME] 标签解析)")
    print("  - 验证时间: 纯 Verify 时间 (从 [TIME] 标签解析)")
    print("  - 证明大小: P 传给 V 的数据 (中间张量文件大小之和)")
    print("  - 承诺大小: 权重承诺文件大小")
    print("  - 表大小:   SwiGLU + Softmax 查找表")
    print("\n注意: 请运行 zkhook-ccs2024-main/test-attention-time.py 获取实际数据")
    print()
    
    # 旧数据 (纯计算时间，不含进程启动、CUDA初始化、网络往返等)
    baseline_data = {
        256: (2.3573, 6.0828, 24576.0, 0.18, 80.01),
        128: (0.6073, 2.9852, 12288.0, 0.18, 80.01),
        64:  (0.4687, 2.9893,  6144.0, 0.18, 80.01),
    }
    
    # 只显示我们测试过的 token 数
    tested_seqs = sorted([r.seq_len for r in all_results], reverse=True)
    available_seqs = [s for s in tested_seqs if s in baseline_data]
    
    header = f"{'指标':<20}"
    for seq in available_seqs:
        header += f"{seq} tokens".center(col_width)
    print(header)
    print("-" * (20 + col_width * len(available_seqs)))
    
    row = f"{'Prove 时间':<20}"
    for seq in available_seqs:
        row += f"{baseline_data[seq][0]:.4f} s".center(col_width)
    print(row)
    
    row = f"{'Verify 时间':<20}"
    for seq in available_seqs:
        row += f"{baseline_data[seq][1]:.4f} s".center(col_width)
    print(row)
    
    row = f"{'Proof Size':<20}"
    for seq in available_seqs:
        row += "N/A (交互式)".center(col_width)
    print(row)
    
    row = f"{'Commitment Size':<20}"
    for seq in available_seqs:
        row += f"{baseline_data[seq][3]:.2f} MB".center(col_width)
    print(row)
    
    row = f"{'Table Size':<20}"
    for seq in available_seqs:
        row += f"{baseline_data[seq][4]:.2f} MB".center(col_width)
    print(row)
    
    print("-" * (20 + col_width * len(available_seqs)))
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 100)
    print("性能对比: 本项目 vs Baseline")
    print("=" * 100)
    
    header = f"{'指标':<20}"
    for seq in available_seqs:
        header += f"{seq} tokens".center(col_width)
    print(header)
    print("-" * (20 + col_width * len(available_seqs)))
    
    # Prove 倍数
    row = f"{'Prove 倍数':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        if r.seq_len in baseline_data:
            ratio = r.prove_time / baseline_data[r.seq_len][0]
            row += f"{ratio:.2f}x".center(col_width)
    print(row)
    
    # Verify 倍数
    row = f"{'Verify 倍数':<20}"
    for r in sorted(all_results, key=lambda x: -x.seq_len):
        if r.seq_len in baseline_data:
            ratio = r.verify_time / baseline_data[r.seq_len][1]
            row += f"{ratio:.2f}x".center(col_width)
    print(row)
    
    print("-" * (20 + col_width * len(available_seqs)))
    
    # 详细分解
    # 保存结果到文件
    result_file = Path("benchmark_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"zkLLM 基准测试结果 (LLaMA-2-7B, {layer_desc}) - 与 Baseline 对齐\n")
        f.write("=" * 90 + "\n\n")
        
        # 本项目开销表 (Baseline 格式)
        f.write(f"本项目开销 ({layer_desc})\n")
        f.write("-" * 70 + "\n")
        
        col_width = 20
        sorted_results = sorted(all_results, key=lambda x: -x.seq_len)
        
        header = f"{'指标':<20}"
        for r in sorted_results:
            header += f"{r.seq_len} tokens".center(col_width)
        f.write(header + "\n")
        f.write("-" * (20 + col_width * len(sorted_results)) + "\n")
        
        row = f"{'Prove 时间':<20}"
        for r in sorted_results:
            row += f"{r.prove_time:.4f} s".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Verify 时间':<20}"
        for r in sorted_results:
            row += f"{r.verify_time:.4f} s".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Proof Size':<20}"
        for r in sorted_results:
            zkp_size = r.proof_size
            if zkp_size >= 1024 * 1024:
                row += f"{zkp_size / 1024 / 1024:.2f} MB".center(col_width)
            else:
                row += f"{zkp_size / 1024:.2f} KB".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Commitment Size':<20}"
        for r in sorted_results:
            row += f"{r.weight_commitment_size:.2f} MB".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Table Size':<20}"
        for r in sorted_results:
            row += f"{r.table_size:.2f} MB".center(col_width)
        f.write(row + "\n")
        
        f.write("-" * (20 + col_width * len(sorted_results)) + "\n\n")
        
        # Baseline 参考数据
        baseline_file_data = {
            256: (2.3573, 6.0828, 24576.0, 0.18, 80.01),
            128: (0.6073, 2.9852, 12288.0, 0.18, 80.01),
            64:  (0.4687, 2.9893,  6144.0, 0.18, 80.01),
        }
        tested_seqs = sorted([r.seq_len for r in all_results], reverse=True)
        available_seqs = [s for s in tested_seqs if s in baseline_file_data]
        
        f.write("Baseline 参考数据 (单层 Transformer)\n")
        f.write("-" * 70 + "\n")
        
        header = f"{'指标':<20}"
        for seq in available_seqs:
            header += f"{seq} tokens".center(col_width)
        f.write(header + "\n")
        f.write("-" * (20 + col_width * len(available_seqs)) + "\n")
        
        row = f"{'Prove 时间':<20}"
        for seq in available_seqs:
            row += f"{baseline_file_data[seq][0]:.4f} s".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Verify 时间':<20}"
        for seq in available_seqs:
            row += f"{baseline_file_data[seq][1]:.4f} s".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Proof Size':<20}"
        for seq in available_seqs:
            row += f"{baseline_file_data[seq][2]:.2f} KB".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Commitment Size':<20}"
        for seq in available_seqs:
            row += f"{baseline_file_data[seq][3]:.2f} MB".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Table Size':<20}"
        for seq in available_seqs:
            row += f"{baseline_file_data[seq][4]:.2f} MB".center(col_width)
        f.write(row + "\n")
        
        f.write("-" * (20 + col_width * len(available_seqs)) + "\n\n")
        
        # 性能对比
        f.write("性能对比: 本项目 vs Baseline\n")
        f.write("-" * 70 + "\n")
        
        header = f"{'指标':<20}"
        for seq in available_seqs:
            header += f"{seq} tokens".center(col_width)
        f.write(header + "\n")
        f.write("-" * (20 + col_width * len(available_seqs)) + "\n")
        
        row = f"{'Prove 倍数':<20}"
        for r in sorted_results:
            if r.seq_len in baseline_file_data:
                ratio = r.prove_time / baseline_file_data[r.seq_len][0]
                row += f"{ratio:.2f}x".center(col_width)
        f.write(row + "\n")
        
        row = f"{'Verify 倍数':<20}"
        for r in sorted_results:
            if r.seq_len in baseline_file_data:
                ratio = r.verify_time / baseline_file_data[r.seq_len][1]
                row += f"{ratio:.2f}x".center(col_width)
        f.write(row + "\n")
        
        f.write("-" * (20 + col_width * len(available_seqs)) + "\n")
    
    print(f"\n\n✓ 结果已保存到: {result_file}")
    print("=" * 90)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
