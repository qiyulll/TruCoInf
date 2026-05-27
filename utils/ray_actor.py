import inspect
import os
import time

import ray
import torch
from accelerate import dispatch_model, infer_auto_device_map
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .gac_gen_utils import *


import subprocess
import numpy as np

# Evidence root dir for zk artifacts. Keep consistent with verify/test scripts.
# 默认为项目根目录下的 zk_evidence 目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_ROOT_DIR = os.environ.get("ZK_EVIDENCE_ROOT_DIR", os.path.join(_PROJECT_ROOT, "zk_evidence"))
ZK_WEIGHTS_ROOT = os.environ.get("ZK_WEIGHTS_ROOT", os.path.join(_PROJECT_ROOT, "zk_weights"))

# ================= [ 全局账本 ] =================
# 专门用来记录 Layer 0 的犯罪现场（证据路径）
# 格式: [{"layer": 0, "type": "q_proj", "input": "...", "weight": "..."}]
EVIDENCE_LOG = []

# ================= [ 完整 Attention 层证据收集器 (v7) ] =================
# 用于收集完整的 Attention 层输入/输出，供 attn_prove 使用
# v7: 默认启用，支持完整零知识证明（输入、权重、输出均承诺）
ATTN_EVIDENCE_COLLECTOR = {
    "enabled": os.environ.get("ZK_ATTN_EVIDENCE", "1") in ("1", "true", "True"),  # v7: 默认启用
    "current_request_id": None,
    "current_model_size": None,
    "layer_0_input": None,      # 完整的 Attention 层输入 (seq_len, embed_dim)
    "layer_0_q_output": None,   # Q 投影输出
    "layer_0_k_output": None,   # K 投影输出
    "layer_0_v_output": None,   # V 投影输出
    "layer_0_attn_output": None,  # Attention 输出
}

# ================= [ 完整 FFN 层证据收集器 (v1) ] =================
# 用于收集完整的 FFN 层输入/输出，供 ffn_prove 使用
# FFN 结构: SwiGLU(gate_proj(x)) * up_proj(x) -> down_proj
FFN_EVIDENCE_COLLECTOR = {
    "enabled": os.environ.get("ZK_FFN_EVIDENCE", "1") in ("1", "true", "True"),  # 默认启用
    "current_request_id": None,
    "current_model_size": None,
    "layer_0_ffn_input": None,    # FFN 层输入 (seq_len, embed_dim)
    "layer_0_up_output": None,    # up_proj 输出
    "layer_0_gate_output": None,  # gate_proj 输出
    "layer_0_down_output": None,  # down_proj 输出
}


FIXED_POINT_SCALE = 1 << 12


def _safe_evidence_model_id(model_name):
    """Convert a model name into a stable evidence directory component."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)


def _quantize_fixed_point(arr: np.ndarray) -> np.ndarray:
    """Quantize float values to zkhook int32 fixed-point format."""
    scaled = arr.astype(np.float64) * FIXED_POINT_SCALE
    return np.where(
        scaled >= 0,
        np.floor(scaled + 0.5),
        np.ceil(scaled - 0.5),
    ).astype(np.int32)


def _first_existing_path(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return next((path for path in paths if path), "")


def _zk_weight_file(model_id: str, proj_type: str) -> str:
    """Return the long-lived zk_weights file for a layer-0 projection."""
    name_map = {
        "q_proj": ("self_attn.q_proj.weight.bin", "model.layers.0.self_attn.q_proj.weight.bin"),
        "k_proj": ("self_attn.k_proj.weight.bin", "model.layers.0.self_attn.k_proj.weight.bin"),
        "v_proj": ("self_attn.v_proj.weight.bin", "model.layers.0.self_attn.v_proj.weight.bin"),
        "o_proj": ("self_attn.o_proj.weight.bin", "model.layers.0.self_attn.o_proj.weight.bin"),
        "up_proj": ("mlp.up_proj.weight.bin", "model.layers.0.mlp.up_proj.weight.bin"),
        "gate_proj": ("mlp.gate_proj.weight.bin", "model.layers.0.mlp.gate_proj.weight.bin"),
        "down_proj": ("mlp.down_proj.weight.bin", "model.layers.0.mlp.down_proj.weight.bin"),
    }
    short_name, full_name = name_map.get(proj_type, ("", ""))
    weights_dir = os.path.join(ZK_WEIGHTS_ROOT, model_id)
    return _first_existing_path(
        os.path.join(weights_dir, short_name),
        os.path.join(weights_dir, full_name),
    )


def zk_rmsnorm_hook_logic(module, inputs, outputs):
    """Save layer-0 RMSNorm inputs and reciprocal RMS evidence for transformer_prove."""
    layer_idx = getattr(module, "layer_idx", "unknown")
    norm_type = getattr(module, "proj_type", "unknown")
    model_size = getattr(module, "model_size", "unknown")
    request_id = getattr(module, "request_id", "no_request_id")

    if layer_idx != 0 or norm_type != "input_layernorm":
        return outputs
    if not request_id or request_id == "no_request_id" or len(inputs) == 0:
        return outputs

    try:
        x = inputs[0].detach().float()
        eps = getattr(module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(module, "eps", None)
        if eps is None:
            eps = getattr(module, "rms_norm_eps", 1e-6)

        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        x_np = x.cpu().numpy().astype(np.float32)
        rms_inv_np = rms_inv.cpu().numpy().astype(np.float32)

        base_dir = os.path.join(EVIDENCE_ROOT_DIR, request_id, model_size, f"layer_{layer_idx}")
        rms_dir = os.path.join(base_dir, "rmsnorm")
        os.makedirs(rms_dir, exist_ok=True)
        timestamp = int(time.time() * 1000000)

        input_float_path = os.path.join(rms_dir, f"input_{timestamp}.bin")
        input_int_path = os.path.join(rms_dir, f"input_int_{timestamp}.bin")
        rms_inv_float_path = os.path.join(rms_dir, f"rms_inv_{timestamp}.bin")
        rms_inv_int_path = os.path.join(rms_dir, f"rms_inv_int_{timestamp}.bin")

        x_np.tofile(input_float_path)
        _quantize_fixed_point(x_np.flatten()).tofile(input_int_path)
        rms_inv_np.tofile(rms_inv_float_path)
        _quantize_fixed_point(rms_inv_np.flatten()).tofile(rms_inv_int_path)

        # Stable filenames point to the latest forward pass and are what verify.py uses.
        _quantize_fixed_point(x_np.flatten()).tofile(os.path.join(base_dir, "input_int.bin"))
        _quantize_fixed_point(rms_inv_np.flatten()).tofile(os.path.join(base_dir, "rms_inv.bin"))

        meta_path = os.path.join(rms_dir, f"rmsnorm_meta_{timestamp}.txt")
        input_shape = x_np.shape
        seq_len = input_shape[-2] if len(input_shape) >= 2 else 1
        embed_dim = input_shape[-1]
        with open(meta_path, "w") as f:
            f.write(f"request_id={request_id}\n")
            f.write(f"model_size={model_size}\n")
            f.write(f"norm_type={norm_type}\n")
            f.write(f"seq_len={seq_len}\n")
            f.write(f"embed_dim={embed_dim}\n")
            f.write(f"eps={eps}\n")
            f.write(f"input_int_file={os.path.basename(input_int_path)}\n")
            f.write(f"rms_inv_int_file={os.path.basename(rms_inv_int_path)}\n")
            f.write(f"timestamp={timestamp}\n")

        print(f"🔔 [ZK_RMSNORM] layer={layer_idx}, model={model_size}, rid={request_id}")
        print(f"   💾 input_int: {os.path.join(base_dir, 'input_int.bin')}")
        print(f"   💾 rms_inv: {os.path.join(base_dir, 'rms_inv.bin')}")
    except Exception as e:
        print(f"   ⚠️ [RMSNorm] 保存证据失败: {e}")

    return outputs


# ================= [ 1. 极速版 Hook (只存证，不卡顿) ] =================
def zk_hook_logic_fast(module, inputs, outputs):
    global EVIDENCE_LOG, ATTN_EVIDENCE_COLLECTOR, FFN_EVIDENCE_COLLECTOR
    
    # 1. 获取身份（新增 model_size）
    layer_idx = getattr(module, 'layer_idx', 'unknown')
    proj_type = getattr(module, 'proj_type', 'unknown')
    model_size = getattr(module, 'model_size', 'unknown')  # 新增：获取模型维度
    request_id = getattr(module, "request_id", "no_request_id")
    
    # 🚫【核心过滤】非第 0 层直接放行
    if layer_idx != 0:
        return outputs

    # 【调试】打印 hook 被触发
    print(f"🔔 [ZK_HOOK] layer={layer_idx}, proj={proj_type}, model={model_size}, rid={request_id}")
    
    # ===== 收集完整 Attention 层证据 =====
    if ATTN_EVIDENCE_COLLECTOR["enabled"] and proj_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        try:
            ATTN_EVIDENCE_COLLECTOR["current_request_id"] = request_id
            ATTN_EVIDENCE_COLLECTOR["current_model_size"] = model_size
            
            # 保存完整输入（只在 q_proj 时保存，因为 q/k/v 共享相同输入）
            if proj_type == "q_proj" and len(inputs) > 0:
                x_full = inputs[0].detach().cpu().numpy().astype(np.float32)
                ATTN_EVIDENCE_COLLECTOR["layer_0_input"] = x_full
                print(f"   📦 [ATTN] 收集完整输入: shape={x_full.shape}")
            
            # 保存完整投影输出
            if proj_type in ["q_proj", "k_proj", "v_proj"]:
                y = outputs
                if isinstance(y, (tuple, list)):
                    y = y[0]
                y_full = y.detach().cpu().numpy().astype(np.float32)
                
                if proj_type == "q_proj":
                    ATTN_EVIDENCE_COLLECTOR["layer_0_q_output"] = y_full
                elif proj_type == "k_proj":
                    ATTN_EVIDENCE_COLLECTOR["layer_0_k_output"] = y_full
                elif proj_type == "v_proj":
                    ATTN_EVIDENCE_COLLECTOR["layer_0_v_output"] = y_full
                print(f"   📦 [ATTN] 收集 {proj_type} 输出: shape={y_full.shape}")
            
            # 当 o_proj 被调用时，保存完整证据到文件
            if proj_type == "o_proj":
                _save_attn_evidence()
                
        except Exception as e:
            print(f"   ⚠️ [ATTN] 收集完整证据失败: {e}")
    
    # ===== 收集完整 FFN 层证据 =====
    if FFN_EVIDENCE_COLLECTOR["enabled"] and proj_type in ["up_proj", "gate_proj", "down_proj"]:
        try:
            FFN_EVIDENCE_COLLECTOR["current_request_id"] = request_id
            FFN_EVIDENCE_COLLECTOR["current_model_size"] = model_size
            
            # 保存 FFN 输入（只在 up_proj 时保存，因为 up/gate 共享相同输入）
            if proj_type == "up_proj" and len(inputs) > 0:
                x_full = inputs[0].detach().cpu().numpy().astype(np.float32)
                FFN_EVIDENCE_COLLECTOR["layer_0_ffn_input"] = x_full
                print(f"   📦 [FFN] 收集完整输入: shape={x_full.shape}")
            
            # 保存投影输出
            y = outputs
            if isinstance(y, (tuple, list)):
                y = y[0]
            y_full = y.detach().cpu().numpy().astype(np.float32)
            
            if proj_type == "up_proj":
                FFN_EVIDENCE_COLLECTOR["layer_0_up_output"] = y_full
                print(f"   📦 [FFN] 收集 up_proj 输出: shape={y_full.shape}")
            elif proj_type == "gate_proj":
                FFN_EVIDENCE_COLLECTOR["layer_0_gate_output"] = y_full
                print(f"   📦 [FFN] 收集 gate_proj 输出: shape={y_full.shape}")
            elif proj_type == "down_proj":
                FFN_EVIDENCE_COLLECTOR["layer_0_down_output"] = y_full
                print(f"   📦 [FFN] 收集 down_proj 输出: shape={y_full.shape}")
                # down_proj 是 FFN 的最后一步，保存证据
                _save_ffn_evidence()
                
        except Exception as e:
            print(f"   ⚠️ [FFN] 收集完整证据失败: {e}")

    # 2. 准备目录 (新增 model_size 层级)
    # 路径变为: <project_root>/zk_evidence/<request_id>/<model_size>/layer_0/q_proj
    evidence_dir = os.path.join(EVIDENCE_ROOT_DIR, request_id, model_size, f"layer_{layer_idx}", proj_type)
    print(f"   -> 证据目录: {evidence_dir}")
    os.makedirs(evidence_dir, exist_ok=True)
    
    # 时间戳 (微秒级，防止生成太快文件名冲突)
    timestamp = int(time.time() * 1000000)
    
    # 3. 极速保存【输入 X】
    input_save_path = os.path.join(evidence_dir, f"input_{timestamp}.bin")
    try:
        if len(inputs) > 0:
            x = inputs[0]
            # Save a single hidden vector to keep files small:
            # (bs, seq, hidden) -> (hidden,)
            while hasattr(x, "ndim") and x.ndim > 1:
                x = x[0]
            data = x.detach().cpu().numpy().astype(np.float32)
            data.tofile(input_save_path)
            print(f"   -> 输入已保存: {input_save_path}")
    except Exception as e:
        print(f"   ❌ 保存输入失败: {e}")
        return outputs 

    # 4. 权重不再随每次请求写入 evidence。证明阶段从长期 zk_weights 读取。
    weight_ref_path = _zk_weight_file(model_size, proj_type)
    if os.path.exists(weight_ref_path):
        print(f"   -> 权重引用: {weight_ref_path}")
    else:
        print(f"   ⚠️ 未找到 zk 权重引用: {weight_ref_path}")

    # 5. 保存【输出 Y】
    output_save_path = os.path.join(evidence_dir, f"output_{timestamp}.bin")
    try:
        y = outputs
        if isinstance(y, (tuple, list)):
            if len(y) == 0:
                return outputs
            y = y[0]
        while hasattr(y, "ndim") and y.ndim > 1:
            y = y[0]
        y_data = y.detach().cpu().numpy().astype(np.float32)
        y_data.tofile(output_save_path)
    except Exception:
        return outputs

    # 6. 【记账】新增 model_size 字段
    audit_record = {
        "model_size": model_size,  # 新增：记录模型维度
        "layer": layer_idx,     
        "type": proj_type,      
        "input_path": input_save_path,
        "weight_path": weight_ref_path,
        "output_path": output_save_path,
        "request_id": request_id,
        "timestamp": timestamp
    }
    EVIDENCE_LOG.append(audit_record)
    
    # 7. 立刻放行
    return outputs

# ================= [ 1.5 保存完整 Attention 层证据 (v7) ] =================
def _save_attn_evidence():
    """
    当 Attention 层处理完成时，保存完整的证据供 attn_prove 使用
    
    v7: 保存量化后的输入 (int.bin 格式)，供完整零知识证明使用
    """
    global ATTN_EVIDENCE_COLLECTOR
    
    collector = ATTN_EVIDENCE_COLLECTOR
    request_id = collector["current_request_id"]
    model_size = collector["current_model_size"]
    
    if not request_id or request_id == "no_request_id":
        print("   ⚠️ [ATTN] 无 request_id，跳过保存")
        return
    
    if collector["layer_0_input"] is None:
        print("   ⚠️ [ATTN] 缺少输入数据，跳过保存")
        return
    
    # 创建 Attention 证据目录
    attn_evidence_dir = os.path.join(
        EVIDENCE_ROOT_DIR, request_id, model_size, "layer_0", "attention"
    )
    os.makedirs(attn_evidence_dir, exist_ok=True)
    
    timestamp = int(time.time() * 1000000)
    
    try:
        # 保存完整输入 (float32 格式)
        input_path = os.path.join(attn_evidence_dir, f"attn_input_{timestamp}.bin")
        collector["layer_0_input"].tofile(input_path)
        print(f"   💾 [ATTN] 保存输入 (float): {input_path}")
        
        # v7: 保存量化后的输入 (int32 格式) - 供 attn_prove 使用
        input_int_path = os.path.join(attn_evidence_dir, f"attn_input_int_{timestamp}.bin")
        input_flat = collector["layer_0_input"].flatten()
        input_int = _quantize_fixed_point(input_flat)
        input_int.tofile(input_int_path)
        print(f"   💾 [ATTN] 保存输入 (int): {input_int_path}")
        
        # 保存 Q 投影输出
        if collector["layer_0_q_output"] is not None:
            q_path = os.path.join(attn_evidence_dir, f"q_output_{timestamp}.bin")
            collector["layer_0_q_output"].tofile(q_path)
        
        # 保存 K 投影输出
        if collector["layer_0_k_output"] is not None:
            k_path = os.path.join(attn_evidence_dir, f"k_output_{timestamp}.bin")
            collector["layer_0_k_output"].tofile(k_path)
        
        # 保存 V 投影输出
        if collector["layer_0_v_output"] is not None:
            v_path = os.path.join(attn_evidence_dir, f"v_output_{timestamp}.bin")
            collector["layer_0_v_output"].tofile(v_path)
        
        # 获取维度信息
        input_shape = collector["layer_0_input"].shape
        seq_len = input_shape[-2] if len(input_shape) >= 2 else 1
        embed_dim = input_shape[-1]
        
        # 保存元数据（供 attn_prove 读取）
        meta_path = os.path.join(attn_evidence_dir, f"attn_meta_{timestamp}.txt")
        with open(meta_path, "w") as f:
            f.write(f"request_id={request_id}\n")
            f.write(f"model_size={model_size}\n")
            f.write(f"seq_len={seq_len}\n")
            f.write(f"embed_dim={embed_dim}\n")
            f.write(f"input_file=attn_input_{timestamp}.bin\n")
            f.write(f"input_int_file=attn_input_int_{timestamp}.bin\n")
            f.write(f"timestamp={timestamp}\n")
            f.write(f"version=v7\n")
        
        print(f"   ✅ [ATTN v7] 完整 Attention 证据已保存:")
        print(f"      目录: {attn_evidence_dir}")
        print(f"      维度: seq_len={seq_len}, embed_dim={embed_dim}")
        print(f"      格式: float + int (量化)")
        
        # 清空收集器，准备下一次
        collector["layer_0_input"] = None
        collector["layer_0_q_output"] = None
        collector["layer_0_k_output"] = None
        collector["layer_0_v_output"] = None
        
    except Exception as e:
        print(f"   ❌ [ATTN] 保存证据失败: {e}")


# ================= [ 1.6 保存完整 FFN 层证据 (v1) ] =================
def _save_ffn_evidence():
    """
    当 FFN 层处理完成时，保存完整的证据供 ffn_prove 使用
    
    FFN 结构: SwiGLU(gate_proj(x)) * up_proj(x) -> down_proj
    """
    global FFN_EVIDENCE_COLLECTOR
    
    collector = FFN_EVIDENCE_COLLECTOR
    request_id = collector["current_request_id"]
    model_size = collector["current_model_size"]
    
    if not request_id or request_id == "no_request_id":
        print("   ⚠️ [FFN] 无 request_id，跳过保存")
        return
    
    if collector["layer_0_ffn_input"] is None:
        print("   ⚠️ [FFN] 缺少输入数据，跳过保存")
        return
    
    # 创建 FFN 证据目录
    ffn_evidence_dir = os.path.join(
        EVIDENCE_ROOT_DIR, request_id, model_size, "layer_0", "ffn"
    )
    os.makedirs(ffn_evidence_dir, exist_ok=True)
    
    timestamp = int(time.time() * 1000000)
    
    try:
        # 保存完整输入 (float32 格式)
        input_path = os.path.join(ffn_evidence_dir, f"ffn_input_{timestamp}.bin")
        collector["layer_0_ffn_input"].tofile(input_path)
        print(f"   💾 [FFN] 保存输入 (float): {input_path}")
        
        # 保存量化后的输入 (int32 格式)
        input_int_path = os.path.join(ffn_evidence_dir, f"ffn_input_int_{timestamp}.bin")
        input_flat = collector["layer_0_ffn_input"].flatten()
        input_int = _quantize_fixed_point(input_flat)
        input_int.tofile(input_int_path)
        print(f"   💾 [FFN] 保存输入 (int): {input_int_path}")
        
        # 保存 up_proj 输出
        if collector["layer_0_up_output"] is not None:
            up_path = os.path.join(ffn_evidence_dir, f"up_output_{timestamp}.bin")
            collector["layer_0_up_output"].tofile(up_path)
        
        # 保存 gate_proj 输出
        if collector["layer_0_gate_output"] is not None:
            gate_path = os.path.join(ffn_evidence_dir, f"gate_output_{timestamp}.bin")
            collector["layer_0_gate_output"].tofile(gate_path)
        
        # 保存 down_proj 输出
        if collector["layer_0_down_output"] is not None:
            down_path = os.path.join(ffn_evidence_dir, f"down_output_{timestamp}.bin")
            collector["layer_0_down_output"].tofile(down_path)
        
        # 获取维度信息
        input_shape = collector["layer_0_ffn_input"].shape
        seq_len = input_shape[-2] if len(input_shape) >= 2 else 1
        embed_dim = input_shape[-1]
        
        # 从 up_proj 输出推断 hidden_dim
        hidden_dim = embed_dim  # 默认值
        if collector["layer_0_up_output"] is not None:
            up_shape = collector["layer_0_up_output"].shape
            hidden_dim = up_shape[-1] if len(up_shape) >= 1 else embed_dim
        
        # 保存元数据（供 ffn_prove 读取）
        meta_path = os.path.join(ffn_evidence_dir, f"ffn_meta_{timestamp}.txt")
        with open(meta_path, "w") as f:
            f.write(f"request_id={request_id}\n")
            f.write(f"model_size={model_size}\n")
            f.write(f"seq_len={seq_len}\n")
            f.write(f"embed_dim={embed_dim}\n")
            f.write(f"hidden_dim={hidden_dim}\n")
            f.write(f"input_file=ffn_input_{timestamp}.bin\n")
            f.write(f"input_int_file=ffn_input_int_{timestamp}.bin\n")
            f.write(f"timestamp={timestamp}\n")
            f.write(f"version=v1\n")
        
        print(f"   ✅ [FFN v1] 完整 FFN 证据已保存:")
        print(f"      目录: {ffn_evidence_dir}")
        print(f"      维度: seq_len={seq_len}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")
        print(f"      格式: float + int (量化)")
        
        # 清空收集器，准备下一次
        collector["layer_0_ffn_input"] = None
        collector["layer_0_up_output"] = None
        collector["layer_0_gate_output"] = None
        collector["layer_0_down_output"] = None
        
    except Exception as e:
        print(f"   ❌ [FFN] 保存证据失败: {e}")


# ================= [ 2. 事后追责主程序 ] =================
def run_layer0_audit():
    if not EVIDENCE_LOG:
        print("\n⚠️ 账本为空，没有检测到 Layer 0 的推理记录。")
        return

    # 按模型维度分组统计
    from collections import defaultdict
    model_evidence = defaultdict(list)
    for record in EVIDENCE_LOG:
        model_evidence[record['model_size']].append(record)
    
    print(f"\n👮‍♂️ [Layer 0 专项审计] 启动! 涉及模型: {list(model_evidence.keys())}")
    print("="*60)
    
    zk_exe_path = "./zkhook/zk_prover"
    total_success = 0
    total_fail = 0
    
    # 按模型维度分别审计
    for model_size, records in model_evidence.items():
        print(f"\n📌 开始审计 {model_size} 模型 (共 {len(records)} 条证据):")
        success_count = 0
        fail_count = 0
        t_start = time.time()
        
        for i, record in enumerate(records):
            p_type = record['type']
            p_input = record['input_path']
            p_weight = record['weight_path']
            p_output = record.get('output_path')
            p_nonce = record.get('request_id', 'no_request_id')

            if not p_output:
                print("❌ 缺少 output 证据，跳过")
                fail_count += 1
                total_fail += 1
                continue

            cmd = [zk_exe_path, "0", p_input, p_weight, p_output, p_nonce]
            
            print(f"🔍 [{i+1}/{len(records)}] 审计 {p_type} ... ", end="", flush=True)
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                if "SUCCESS" in result.stdout:
                    print("✅ 通过")
                    success_count += 1
                    total_success += 1
                else:
                    print("❌ 驳回 (验证失败)")
                    fail_count += 1
                    total_fail += 1
                    
            except subprocess.CalledProcessError as e:
                print("❌ 程序崩溃 (C++ Error)")
                fail_count += 1
                total_fail += 1
        
        # 打印单个模型的审计结果
        print(f"\n📊 {model_size} 审计报告:")
        print(f"   - 通过率: {success_count}/{len(records)}")
        print(f"   - 耗时: {time.time() - t_start:.2f}s")
        print("-"*40)
    
    # 打印总报告
    print("\n📊 全局审计报告 (Layer 0 Only):")
    print(f"   - 总通过率: {total_success}/{len(EVIDENCE_LOG)}")
    print(f"   - 总失败数: {total_fail}")
    print("="*60)
    
    # 清空账本
    EVIDENCE_LOG.clear()

# ================= [ 3. 注入器 (扩展支持 FFN) ] =================
class ZkQwenWrapperFast:
    @staticmethod
    def inject(model, model_size, evidence_model_id=None):  # 新增参数：model_size (1.8B/7B)
        """
        为模型注入 ZK 证据收集钩子
        
        支持:
        - Attention 层: q_proj, k_proj, v_proj, o_proj
        - FFN 层: up_proj, gate_proj, down_proj
        """
        evidence_model_id = evidence_model_id or model_size
        layers = getattr(model, 'model', model).layers
        
        for i, layer in enumerate(layers):
            # ===== RMSNorm 钩子 =====
            rmsnorm_targets = {}
            if hasattr(layer, "input_layernorm"):
                rmsnorm_targets["input_layernorm"] = layer.input_layernorm
            if hasattr(layer, "post_attention_layernorm"):
                rmsnorm_targets["post_attention_layernorm"] = layer.post_attention_layernorm

            for name, module in rmsnorm_targets.items():
                module.layer_idx = i
                module.proj_type = name
                module.model_size = evidence_model_id
                module.register_forward_hook(zk_rmsnorm_hook_logic)

            # ===== Attention 层钩子 =====
            attn = layer.self_attn
            attn_targets = {
                'q_proj': attn.q_proj, 
                'k_proj': attn.k_proj, 
                'v_proj': attn.v_proj, 
                'o_proj': attn.o_proj
            }
            
            for name, module in attn_targets.items():
                module.layer_idx = i
                module.proj_type = name
                module.model_size = evidence_model_id
                module.register_forward_hook(zk_hook_logic_fast)
            
            # ===== FFN 层钩子 (MLP) =====
            mlp = layer.mlp
            
            # 检测 FFN 结构 (不同模型可能有不同的属性名)
            ffn_targets = {}
            
            # Qwen/LLaMA 风格: up_proj, gate_proj, down_proj
            if hasattr(mlp, 'up_proj'):
                ffn_targets['up_proj'] = mlp.up_proj
            if hasattr(mlp, 'gate_proj'):
                ffn_targets['gate_proj'] = mlp.gate_proj
            if hasattr(mlp, 'down_proj'):
                ffn_targets['down_proj'] = mlp.down_proj
            
            # 备选: 一些模型使用 fc1, fc2
            if hasattr(mlp, 'fc1') and 'up_proj' not in ffn_targets:
                ffn_targets['up_proj'] = mlp.fc1  # 映射 fc1 -> up_proj
            if hasattr(mlp, 'fc2') and 'down_proj' not in ffn_targets:
                ffn_targets['down_proj'] = mlp.fc2  # 映射 fc2 -> down_proj
            
            # 注入 FFN 钩子
            for name, module in ffn_targets.items():
                module.layer_idx = i
                module.proj_type = name
                module.model_size = evidence_model_id
                module.register_forward_hook(zk_hook_logic_fast)
            
            # 打印注入信息 (只打印第一层)
            if i == 0:
                print(f"💉 [Layer {i}] 注入 ZK 钩子:")
                print(f"   Model size: {model_size}, evidence id: {evidence_model_id}")
                print(f"   RMSNorm: {list(rmsnorm_targets.keys())}")
                print(f"   Attention: {list(attn_targets.keys())}")
                print(f"   FFN: {list(ffn_targets.keys())}")
        
        return model

def get_remote_model_generator_class(num_gpus):
    return ray.remote(num_gpus=num_gpus)(ModelGenerator)


class ModelGenerator:
    def __init__(
        self,
        model_path,
        model_name,
        max_memory={0: "80GiB"},
        model_ensemble_weight=1,
        use_cache=True,
        quantization="none",
    ):
        # ================= [第一步：先初始化核心属性，避免提前使用] =================
        # 优先赋值 model_name，防止后续判断时属性不存在
        self.model_name = model_name
        self.model_ensemble_weight = model_ensemble_weight
        self.use_cache = use_cache
        self.model = None  # 先初始化model为None，防止注入逻辑报错
        self.tokenizer = None  # 初始化tokenizer
        
        # ================= [第二步：量化配置（原有逻辑）] =================
        quantization_options = {
            "8bit": BitsAndBytesConfig(load_in_8bit=True),
            "4bit": BitsAndBytesConfig(load_in_4bit=True),
            "none": None,
        }
        quantization_config = quantization_options.get(quantization)
        if quantization_config is None and quantization != "none":
            raise ValueError(
                f"Invalid quantization value '{quantization}'. Allowed values are: 'none', '8bit', '4bit'."
            )

        # ================= [第三步：加载模型（原有逻辑）] =================
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=model._no_split_modules,
            )
            device_map_kwargs = {"device_map": device_map}
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement

            # 赋值self.model，确保后续ZK注入能访问到
            self.model = dispatch_model(model, **device_map_kwargs)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}") from e

        # ================= [修改后的注入逻辑 START] =================
        try:
            # 增加属性存在性检查，防止模型加载失败导致的报错
            if not hasattr(self, 'model') or self.model is None:
                print(f"⚠️ 模型 {self.model_name} 加载失败，跳过ZK探针注入")
            else:
                # 1. 从模型配置中获取名字
                current_model_name = self.model.config._name_or_path if hasattr(self.model.config, '_name_or_path') else self.model_name
                
                # 2. 打印调试信息
                print(f"⚡️ [DEBUG] 当前加载的模型是: {current_model_name}")

                # 3. 定义需要注入ZK探针的模型关键词和对应维度
                target_model_map = {
                    "1.8B": "1.8B",
                    "7B": "7B"
                }
                
                # 匹配模型维度
                model_size = None
                for keyword, size in target_model_map.items():
                    if keyword in current_model_name or keyword in self.model_name:
                        model_size = size
                        break
                
                if model_size:
                    use_model_name_evidence = os.environ.get("ZK_EVIDENCE_USE_MODEL_NAME", "0") in ("1", "true", "True")
                    evidence_model_id = _safe_evidence_model_id(self.model_name) if use_model_name_evidence else model_size
                    print(f"💉 检测到 {model_size} 模型，正在注入 ZK 探针...")
                    ZkQwenWrapperFast.inject(self.model, model_size, evidence_model_id=evidence_model_id)
                else:
                    print(f"ℹ️ 模型 {current_model_name} 不在目标列表中，跳过ZK探针注入")

        except Exception as e:
            print(f"⚠️ ZK 注入逻辑发生意外: {e}")
        # ================= [修改后的注入逻辑 END] =================

        if self.model_name in ["Yi-34B-Chat", "Yi-6B-Chat"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", use_fast=False, trust_remote_code=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", use_fast=False, trust_remote_code=True
            )

        # Make sure use greedy search
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0

        if (
            isinstance(self.model.generation_config.eos_token_id, list)
            and len(self.model.generation_config.eos_token_id) > 1
        ):
            logger.warning(
                f"For model {self.model_name}, the eos_token_id in generation_config more than one, we only take first one."
            )
            self.model.generation_config.eos_token_id = self.model.generation_config.eos_token_id[
                0
            ]

        if self.model.generation_config.eos_token_id and (
            self.model.generation_config.eos_token_id != self.tokenizer.eos_token_id
        ):
            logger.warning(
                f"For model {self.model_name}, the eos_token_id is inconsistent between the generation config and the tokenizer ({self.model.generation_config.eos_token_id} and {self.tokenizer.eos_token_id}). We will forcefully set the tokenizer to be consistent with the generation config ({self.model.generation_config.eos_token_id})."
            )
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        if (
            self.model_name == "Starling-LM-7B-alpha"
            and len(self.tokenizer) > self.model.vocab_size
        ):
            logger.warning(
                f"Model {self.model_name} used! You need remove sep_token from tokenizer_config.json because it cause vocab size +1!"
            )

        # request id (for zk evidence attribution)
        self.request_id = None

    def set_request_id(self, request_id: str):
        """
        Called by API server to tag subsequent forward passes.
        We propagate the request_id onto hooked modules so the forward hook can
        write evidence under a request-scoped directory.
        """
        self.request_id = request_id
        if self.model is None:
            return False

        try:
            for module in self.model.modules():
                if hasattr(module, "layer_idx") and hasattr(module, "proj_type"):
                    module.request_id = request_id
        except Exception as e:
            # Don't fail generation if evidence propagation has issues.
            logger.warning(f"Failed to propagate request_id to modules: {e}")
            return False

        return True

    def get_vocab_size(self):
        # 增加self.model存在性检查
        if self.model is None:
            raise RuntimeError("模型未加载，无法获取词汇表大小")
        if len(self.tokenizer.get_vocab()) != self.model.config.vocab_size:
            logger.warning(
                f"For model {self.model_name}, the vocab_size of the tokenizer and model config are not equal! We will create the mapping matrix base on the model config."
            )
        return self.model.config.vocab_size


    def get_ensemble_weight(self):
        return self.model_ensemble_weight

    def get_input_ids(self):
        return self.state["input_ids"]

    def check_if_stop(self):
        if self.state["unfinished_sequences"].max() == 0:
            self.state["this_peer_finished"] = True

        # stop if we exceed the maximum length
        if torch.all(
            self.state["stopping_criteria"](
                self.state["input_ids"], self.state["scores"]
            )
        ):
            self.state["this_peer_finished"] = True

        return self.state["this_peer_finished"]

    def update_unfinished_sequences(self, unfinished_sequences):
        self.state["unfinished_sequences"] = unfinished_sequences.to(
            self.state["unfinished_sequences"].device
        )

    def get_unfinished_sequences(self):
        return self.state["unfinished_sequences"]

    def update_input_ids_and_model_kwargs(self, next_tokens_list):
        self.state["next_tokens_list"] = next_tokens_list
        (
            self.state["input_ids"],
            self.state["model_kwargs"],
            self.state["unfinished_sequences"],
        ) = update_input_ids_and_model_kwargs(self.model, self.state)

    def get_one_token(self):

        st = time.time()
        self.state["next_tokens_scores"], self.state["outputs"] = get_one_token(
            self.model, self.state
        )
        time_used = time.time() - st

        return self.model_ensemble_weight * self.state["next_tokens_scores"], time_used

    def generate_prepare(self, *args, **kwargs):
        self.state = generate_prepare(model=self.model, **self.inputs, **kwargs)
        self.state["model_kwargs"]["use_cache"] = self.use_cache

    def get_max_position_embeddings(self):
        return self.model.config.max_position_embeddings

    def get_model_name(self):
        return self.model_name

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_inputs_for_model(
        self, chat_list, min_max_position_embeddings=4096, apply_chat_template=False
    ):
        # Calculate the truncation length as 75% of the minimum max_position_embeddings
        truncation_length = int(min_max_position_embeddings * 0.75)
        input_texts = []

        # Apply the chat template and collect the processed text
        for chat in chat_list:
            if apply_chat_template:
                # Assume the tokenizer has an apply_chat_template method
                processed_text = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                # Plain prompt mode must still preserve every message. The API prepends
                # a request_id nonce as a system message; taking only chat[0] would drop
                # the actual user prompt and often makes the first generated token EOS.
                processed_text = "\n".join(
                    str(message.get("content", ""))
                    for message in chat
                    if isinstance(message, dict)
                )
            input_texts.append(processed_text)

        self.inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            max_length=truncation_length,
            truncation=True,
        ).to(next(self.model.parameters()).device)

        return self.inputs.input_ids
