

import os
import sys
import subprocess
import argparse
from typing import Tuple, Optional, List
from dataclasses import dataclass

# ================= 路径配置 =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ZKHOOK_DIR = os.path.join(PROJECT_ROOT, "zkhook")
DEFAULT_TABLE_GEN_PATH = os.path.join(ZKHOOK_DIR, "table_gen")
DEFAULT_TRANSFORMER_PROVE_PATH = os.path.join(ZKHOOK_DIR, "transformer_prove")
DEFAULT_TRANSFORMER_VERIFY_PATH = os.path.join(ZKHOOK_DIR, "transformer_verify")
DEFAULT_TABLES_FILE = os.path.join(ZKHOOK_DIR, "tables.bin")
DEFAULT_PROOF_FILE = os.path.join(ZKHOOK_DIR, "transformer_proof.bin")
DEFAULT_ZK_WEIGHTS_ROOT = os.environ.get("ZK_WEIGHTS_ROOT", os.path.join(PROJECT_ROOT, "zk_weights"))
DEFAULT_WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", DEFAULT_ZK_WEIGHTS_ROOT)
DEFAULT_EVIDENCE_ROOT_DIR = os.environ.get(
    "ZK_EVIDENCE_ROOT_DIR", 
    os.path.join(PROJECT_ROOT, "zk_evidence")
)


@dataclass
class TransformerProveConfig:
    """完整 Transformer 层证明配置"""
    input_file: str          # 输入文件路径 (int32 fixed-point)
    # RMSNorm 输入
    rms_inv_file: str
    input_layernorm_file: str
    post_attn_layernorm_file: str
    # Attention 权重
    q_weight_file: str
    k_weight_file: str
    v_weight_file: str
    o_weight_file: str
    # FFN 权重
    up_weight_file: str
    gate_weight_file: str
    down_weight_file: str
    # 维度
    seq_len: int
    embed_dim: int
    hidden_dim: int
    # 文件
    tables_file: str
    proof_file: str
    nonce: str


def check_executables() -> Tuple[bool, str]:
    """检查必要的可执行文件是否存在"""
    missing = []
    if not os.path.exists(DEFAULT_TABLE_GEN_PATH):
        missing.append(f"table_gen: {DEFAULT_TABLE_GEN_PATH}")
    if not os.path.exists(DEFAULT_TRANSFORMER_PROVE_PATH):
        missing.append(f"transformer_prove: {DEFAULT_TRANSFORMER_PROVE_PATH}")
    if not os.path.exists(DEFAULT_TRANSFORMER_VERIFY_PATH):
        missing.append(f"transformer_verify: {DEFAULT_TRANSFORMER_VERIFY_PATH}")
    
    if missing:
        msg = "缺少可执行文件:\n" + "\n".join(f"  - {m}" for m in missing)
        msg += "\n请先编译: cd zkhook && make all"
        return False, msg
    return True, "OK"


def generate_tables(embed_dim: int, output_file: str) -> Tuple[bool, str]:
    """生成查找表"""
    if not os.path.exists(DEFAULT_TABLE_GEN_PATH):
        return False, f"找不到 table_gen: {DEFAULT_TABLE_GEN_PATH}"
    
    cmd = [DEFAULT_TABLE_GEN_PATH, str(embed_dim), output_file]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    
    if result.returncode != 0:
        return False, f"生成查找表失败:\n{out}"
    
    return True, f"查找表已生成: {output_file}"


def _first_existing_path(*paths: str) -> str:
    """返回第一个存在的路径；如果都不存在，返回第一个候选路径用于报错提示。"""
    for path in paths:
        if path and os.path.exists(path):
            return path
    return next((path for path in paths if path), "")


def _resolve_project_path(path: str) -> str:
    """把项目内相对路径解析为绝对路径；空字符串原样返回。"""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _weight_file(weights_dir: str, short_name: str, full_name: str) -> str:
    """解析导出的 int32 权重文件，兼容扁平命名和 HuggingFace 层名前缀命名。"""
    weights_dir = _resolve_project_path(weights_dir)
    return _first_existing_path(
        os.path.join(weights_dir, short_name),
        os.path.join(weights_dir, full_name),
    )


def _default_weights_dir_for_model(model_size: str) -> str:
    return os.path.join(DEFAULT_ZK_WEIGHTS_ROOT, model_size)


def _table_size_for_model(model_size: str) -> str:
    return "1.8B" if "1.8B" in model_size else "7B"


def _export_command_for_model(model_size: str) -> str:
    if model_size == "qwen-new-7B":
        model_dir = "weights/qwen-new-7B"
    elif model_size == "Qwen1.5-7B-Chat":
        model_dir = "weights/Qwen1.5-7B-Chat"
    else:
        model_dir = "weights/Qwen1.5-1.8B-Chat" if "1.8B" in model_size else "weights/Qwen1.5-7B-Chat"
    return (
        "python export_zkhook_weights.py "
        f"--model-dir {model_dir} --output-dir zk_weights/{model_size}"
    )


def _missing_required_files(config: TransformerProveConfig) -> List[Tuple[str, str]]:
    required_files = [
        ("input_file", config.input_file),
        ("rms_inv_file", config.rms_inv_file),
        ("input_layernorm_file", config.input_layernorm_file),
        ("post_attn_layernorm_file", config.post_attn_layernorm_file),
        ("q_weight_file", config.q_weight_file),
        ("k_weight_file", config.k_weight_file),
        ("v_weight_file", config.v_weight_file),
        ("o_weight_file", config.o_weight_file),
        ("up_weight_file", config.up_weight_file),
        ("gate_weight_file", config.gate_weight_file),
        ("down_weight_file", config.down_weight_file),
        ("tables_file", config.tables_file),
    ]
    return [(name, path) for name, path in required_files if not path or not os.path.exists(path)]


def _format_missing_files_message(missing: List[Tuple[str, str]]) -> str:
    lines = "\n".join(f"  - {name}: {path}" for name, path in missing)
    return (
        "生成证明所需文件缺失:\n"
        f"{lines}\n"
        "请确认 --input-file 是 int32 fixed-point 输入，并确认 "
        "RMSNorm 证据和 zk_weights 下的 int32 权重 .bin 文件存在。"
    )


def transformer_prove(config: TransformerProveConfig, verbose: bool = True) -> Tuple[bool, str]:
    """生成完整 Transformer 层的 ZK 证明"""
    if not os.path.exists(DEFAULT_TRANSFORMER_PROVE_PATH):
        return False, f"找不到 transformer_prove: {DEFAULT_TRANSFORMER_PROVE_PATH}"
    
    missing = _missing_required_files(config)
    if missing:
        return False, _format_missing_files_message(missing)
    
    cmd = [
        DEFAULT_TRANSFORMER_PROVE_PATH,
        config.input_file,
        config.rms_inv_file,
        config.input_layernorm_file,
        config.post_attn_layernorm_file,
        config.q_weight_file,
        config.k_weight_file,
        config.v_weight_file,
        config.o_weight_file,
        config.up_weight_file,
        config.gate_weight_file,
        config.down_weight_file,
        str(config.seq_len),
        str(config.embed_dim),
        str(config.hidden_dim),
        config.tables_file,
        config.proof_file,
        config.nonce,
    ]
    
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    
    if verbose and result.returncode != 0:
        print(out)
    
    if result.returncode == 0 and os.path.exists(config.proof_file):
        return True, f"证明已生成: {config.proof_file}"
    return False, out


def transformer_verify(
    proof_file: str,
    tables_file: str,
    verbose: bool = True
) -> Tuple[bool, str]:
    """验证完整 Transformer 层的 ZK 证明"""
    if not os.path.exists(DEFAULT_TRANSFORMER_VERIFY_PATH):
        return False, f"找不到 transformer_verify: {DEFAULT_TRANSFORMER_VERIFY_PATH}"
    
    if not os.path.exists(proof_file):
        return False, f"找不到证明文件: {proof_file}"
    
    if not os.path.exists(tables_file):
        return False, f"找不到查找表文件: {tables_file}"
    
    cmd = [DEFAULT_TRANSFORMER_VERIFY_PATH, proof_file, "--tables", tables_file]
    
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    
    if verbose:
        print(out)
    
    ok = result.returncode == 0 and (
        "验证通过" in out
        or "Verification passed" in out
        or "All checks passed" in out
    )
    return ok, out


def _resolve_layer_files(weights_dir: str) -> Tuple[str, str]:
    weights_dir = _resolve_project_path(weights_dir)
    input_layernorm_file = _first_existing_path(
        os.path.join(weights_dir, "input_layernorm.weight.bin"),
        os.path.join(weights_dir, "input_layernorm.bin"),
        os.path.join(weights_dir, "model.layers.0.input_layernorm.weight.bin"),
    )
    post_attn_layernorm_file = _first_existing_path(
        os.path.join(weights_dir, "post_attention_layernorm.weight.bin"),
        os.path.join(weights_dir, "post_attn_layernorm.weight.bin"),
        os.path.join(weights_dir, "post_attn_layernorm.bin"),
        os.path.join(weights_dir, "model.layers.0.post_attention_layernorm.weight.bin"),
    )
    return input_layernorm_file, post_attn_layernorm_file


def find_transformer_evidence(
    evidence_root_dir: str,
    request_id: str,
    model_size: str = "1.8B",
    layer_idx: int = 0,
    weights_dir: str = None,
) -> Optional[TransformerProveConfig]:
    """从 evidence 目录中查找完整 Transformer 证据"""
    evidence_root_dir = _resolve_project_path(evidence_root_dir)
    weights_dir = _resolve_project_path(weights_dir or _default_weights_dir_for_model(model_size))
    base_dir = os.path.join(evidence_root_dir, request_id, model_size, f"layer_{layer_idx}")
    attn_dir = os.path.join(base_dir, "attention")
    ffn_dir = os.path.join(base_dir, "ffn")
    rmsnorm_dir = os.path.join(base_dir, "rmsnorm")
    
    if not os.path.exists(attn_dir) or not os.path.exists(ffn_dir):
        return None
    
    # 查找最新的证据文件
    import glob
    
    # transformer_prove expects the original hidden state before input_layernorm.
    # New RMSNorm hooks save that as input_int.bin; keep the old Attention input as fallback.
    rmsnorm_inputs = sorted(glob.glob(os.path.join(rmsnorm_dir, "input_int_*.bin")))
    input_file = ""
    stable_rmsnorm_input = os.path.join(base_dir, "input_int.bin")
    if os.path.exists(stable_rmsnorm_input):
        input_file = stable_rmsnorm_input
    elif rmsnorm_inputs:
        input_file = rmsnorm_inputs[-1]

    # Attention 输入 fallback for older evidence directories.
    attn_inputs = sorted(glob.glob(os.path.join(attn_dir, "attn_input_*.bin")))
    if not input_file and not attn_inputs:
        return None
    if not input_file:
        int_inputs = [f for f in attn_inputs if "_int_" in f]
        float_inputs = [f for f in attn_inputs if "_int_" not in f]
        input_file = (int_inputs or float_inputs)[-1]
    
    input_layernorm_file, post_attn_layernorm_file = _resolve_layer_files(weights_dir)
    rms_inv_file = _first_existing_path(
        os.path.join(base_dir, "rms_inv.bin"),
        os.path.join(base_dir, "rmsnorm", "rms_inv.bin"),
        *(sorted(glob.glob(os.path.join(rmsnorm_dir, "rms_inv_int_*.bin")))[-1:] or [""]),
        os.path.join(weights_dir, "rms_inv.bin"),
    )
    
    # 维度信息
    embed_dim = 2048 if "1.8B" in model_size else 4096
    hidden_dim = 5504 if "1.8B" in model_size else 11008
    
    # 读取 meta 文件获取 seq_len
    meta_files = sorted(glob.glob(os.path.join(attn_dir, "attn_meta_*.txt")))
    seq_len = 1
    if meta_files:
        try:
            with open(meta_files[-1], "r") as f:
                meta = dict(line.strip().split("=", 1) for line in f if "=" in line)
            seq_len = int(meta.get("seq_len", 1))
            embed_dim = int(meta.get("embed_dim", embed_dim))
        except:
            pass

    ffn_meta_files = sorted(glob.glob(os.path.join(ffn_dir, "ffn_meta_*.txt")))
    if ffn_meta_files:
        try:
            with open(ffn_meta_files[-1], "r") as f:
                meta = dict(line.strip().split("=", 1) for line in f if "=" in line)
            embed_dim = int(meta.get("embed_dim", embed_dim))
            hidden_dim = int(meta.get("hidden_dim", hidden_dim))
        except:
            pass
    
    proof_file = os.path.join(evidence_root_dir, request_id, f"transformer_proof_{model_size}.bin")
    tables_file = os.path.join(ZKHOOK_DIR, f"tables_{_table_size_for_model(model_size)}.bin")
    
    return TransformerProveConfig(
        input_file=input_file,
        rms_inv_file=rms_inv_file,
        input_layernorm_file=input_layernorm_file,
        post_attn_layernorm_file=post_attn_layernorm_file,
        q_weight_file=_weight_file(weights_dir, "self_attn.q_proj.weight.bin", "model.layers.0.self_attn.q_proj.weight.bin"),
        k_weight_file=_weight_file(weights_dir, "self_attn.k_proj.weight.bin", "model.layers.0.self_attn.k_proj.weight.bin"),
        v_weight_file=_weight_file(weights_dir, "self_attn.v_proj.weight.bin", "model.layers.0.self_attn.v_proj.weight.bin"),
        o_weight_file=_weight_file(weights_dir, "self_attn.o_proj.weight.bin", "model.layers.0.self_attn.o_proj.weight.bin"),
        up_weight_file=_weight_file(weights_dir, "mlp.up_proj.weight.bin", "model.layers.0.mlp.up_proj.weight.bin"),
        gate_weight_file=_weight_file(weights_dir, "mlp.gate_proj.weight.bin", "model.layers.0.mlp.gate_proj.weight.bin"),
        down_weight_file=_weight_file(weights_dir, "mlp.down_proj.weight.bin", "model.layers.0.mlp.down_proj.weight.bin"),
        seq_len=seq_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        tables_file=tables_file,
        proof_file=proof_file,
        nonce=request_id,
    )


def main():
    parser = argparse.ArgumentParser(
        description="zkhook 完整 Transformer 层零知识证明"
    )
    
    # 操作选择
    parser.add_argument("--prove", action="store_true", help="生成证明")
    parser.add_argument("--verify", action="store_true", help="验证证明")
    parser.add_argument("--gen-tables", action="store_true", help="生成查找表")
    
    # 文件参数
    parser.add_argument("--input-file", help="输入文件路径 (int32 fixed-point)")
    parser.add_argument("--rms-inv-file", help="RMSNorm rms_inv 文件路径 (int32 fixed-point)")
    parser.add_argument("--input-layernorm-file", help="Attention 前 RMSNorm gamma 文件路径 (int32 fixed-point)")
    parser.add_argument("--post-attn-layernorm-file", help="FFN 前 RMSNorm gamma 文件路径 (int32 fixed-point)")
    parser.add_argument("--weights-dir", help="权重目录")
    parser.add_argument("--q-weight-file", help="Q 投影权重文件路径 (int32 fixed-point)")
    parser.add_argument("--k-weight-file", help="K 投影权重文件路径 (int32 fixed-point)")
    parser.add_argument("--v-weight-file", help="V 投影权重文件路径 (int32 fixed-point)")
    parser.add_argument("--o-weight-file", help="O 投影权重文件路径 (int32 fixed-point)")
    parser.add_argument("--up-weight-file", help="FFN up_proj 权重文件路径 (int32 fixed-point)")
    parser.add_argument("--gate-weight-file", help="FFN gate_proj 权重文件路径 (int32 fixed-point)")
    parser.add_argument("--down-weight-file", help="FFN down_proj 权重文件路径 (int32 fixed-point)")
    parser.add_argument("--proof-file", default=DEFAULT_PROOF_FILE, help="证明文件路径")
    parser.add_argument("--tables", default=DEFAULT_TABLES_FILE, help="查找表文件路径")
    
    # 维度参数
    parser.add_argument("--seq-len", type=int, default=1, help="序列长度")
    parser.add_argument("--embed-dim", type=int, default=2048, help="嵌入维度")
    parser.add_argument("--hidden-dim", type=int, default=5504, help="FFN 隐藏维度")
    parser.add_argument("--nonce", default="test_nonce", help="防重放随机数")
    
    # 自动查找参数
    parser.add_argument("--evidence-root", default=DEFAULT_EVIDENCE_ROOT_DIR, help="证据根目录")
    parser.add_argument("--request-id", help="请求ID (用于自动查找证据)")
    parser.add_argument("--model-size", default="1.8B", help="模型大小")
    
    args = parser.parse_args()

    # 检查可执行文件
    exe_ok, exe_msg = check_executables()
    if not exe_ok:
        print(f"❌ {exe_msg}")
        return 1

    # 生成查找表
    if args.gen_tables:
        print(f"生成查找表: embed_dim={args.embed_dim}")
        ok, msg = generate_tables(args.embed_dim, args.tables)
        print(f"{'✅' if ok else '❌'} {msg}")
        return 0 if ok else 1

    do_prove = args.prove
    do_verify = args.verify
    
    # 如果没有指定操作，默认都做
    if not do_prove and not do_verify:
        do_prove = True
        do_verify = True

    # 只验证模式
    if do_verify and not do_prove:
        if not os.path.exists(args.proof_file):
            print(f"❌ 找不到证明文件: {args.proof_file}")
            return 1
        if not os.path.exists(args.tables):
            print(f"❌ 找不到查找表: {args.tables}")
            print(f"   请先运行: ./zkhook/table_gen {args.embed_dim} {args.tables}")
            return 1
        
        ok, msg = transformer_verify(args.proof_file, args.tables, verbose=True)
        return 0 if ok else 1

    # 尝试自动查找证据
    config = None
    if args.request_id and not args.input_file:
        config = find_transformer_evidence(
            args.evidence_root,
            args.request_id,
            args.model_size,
            weights_dir=args.weights_dir or _default_weights_dir_for_model(args.model_size),
        )
        if config:
            print(f"自动发现 Transformer 证据")
            print(f"使用 zk 权重目录: {os.path.dirname(config.q_weight_file)}")

    if not config:
        # 使用命令行参数
        if do_prove:
            if not args.input_file:
                print("❌ 错误: 需要指定 --input-file")
                print("   或者指定 --request-id 从 evidence 目录自动查找")
                return 1
            
            if not args.weights_dir:
                print("❌ 错误: 需要指定 --weights-dir")
                return 1
        
        weights_dir = _resolve_project_path(args.weights_dir or _default_weights_dir_for_model(args.model_size))
        input_layernorm_file, post_attn_layernorm_file = _resolve_layer_files(weights_dir)
        config = TransformerProveConfig(
            input_file=_resolve_project_path(args.input_file or ""),
            rms_inv_file=_resolve_project_path(args.rms_inv_file) if args.rms_inv_file else os.path.join(weights_dir, "rms_inv.bin"),
            input_layernorm_file=_resolve_project_path(args.input_layernorm_file) if args.input_layernorm_file else input_layernorm_file,
            post_attn_layernorm_file=_resolve_project_path(args.post_attn_layernorm_file) if args.post_attn_layernorm_file else post_attn_layernorm_file,
            q_weight_file=_resolve_project_path(args.q_weight_file) if args.q_weight_file else _weight_file(weights_dir, "self_attn.q_proj.weight.bin", "model.layers.0.self_attn.q_proj.weight.bin"),
            k_weight_file=_resolve_project_path(args.k_weight_file) if args.k_weight_file else _weight_file(weights_dir, "self_attn.k_proj.weight.bin", "model.layers.0.self_attn.k_proj.weight.bin"),
            v_weight_file=_resolve_project_path(args.v_weight_file) if args.v_weight_file else _weight_file(weights_dir, "self_attn.v_proj.weight.bin", "model.layers.0.self_attn.v_proj.weight.bin"),
            o_weight_file=_resolve_project_path(args.o_weight_file) if args.o_weight_file else _weight_file(weights_dir, "self_attn.o_proj.weight.bin", "model.layers.0.self_attn.o_proj.weight.bin"),
            up_weight_file=_resolve_project_path(args.up_weight_file) if args.up_weight_file else _weight_file(weights_dir, "mlp.up_proj.weight.bin", "model.layers.0.mlp.up_proj.weight.bin"),
            gate_weight_file=_resolve_project_path(args.gate_weight_file) if args.gate_weight_file else _weight_file(weights_dir, "mlp.gate_proj.weight.bin", "model.layers.0.mlp.gate_proj.weight.bin"),
            down_weight_file=_resolve_project_path(args.down_weight_file) if args.down_weight_file else _weight_file(weights_dir, "mlp.down_proj.weight.bin", "model.layers.0.mlp.down_proj.weight.bin"),
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            tables_file=_resolve_project_path(args.tables),
            proof_file=_resolve_project_path(args.proof_file),
            nonce=args.nonce,
        )

    if do_prove:
        missing_before_tables = [
            item for item in _missing_required_files(config)
            if item[0] != "tables_file"
        ]
        if missing_before_tables:
            print(f"❌ {_format_missing_files_message(missing_before_tables)}")
            print(f"   默认会读取: {_default_weights_dir_for_model(args.model_size)}")
            print(f"   首次或缺失时导出一次: {_export_command_for_model(args.model_size)}")
            return 1

    # 检查/生成查找表
    if not os.path.exists(config.tables_file):
        print(f"查找表不存在，正在生成: {config.tables_file}")
        ok, msg = generate_tables(config.embed_dim, config.tables_file)
        if not ok:
            print(f"❌ {msg}")
            return 1
        print(f"✅ {msg}")

    # 生成证明
    if do_prove:
        print("\n========== 生成证明 ==========")
        prove_ok, prove_msg = transformer_prove(config, verbose=True)
        if not prove_ok:
            print(f"❌ 证明生成失败: {prove_msg}")
            return 1
        print(f"✅ {prove_msg}")

    # 验证证明
    if do_verify:
        print("\n========== 验证证明 ==========")
        verify_ok, verify_msg = transformer_verify(config.proof_file, config.tables_file, verbose=True)
        if not verify_ok:
            print(f"❌ 验证失败")
            return 1
        print(f"✅ 验证通过")

    return 0


if __name__ == "__main__":
    sys.exit(main())
