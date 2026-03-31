"""
verify.py - zkhook 完整 Transformer 层零知识证明验证器

支持完整 Transformer 层零知识证明 (Attention + FFN)

零知识保证:
- 输入 X: 承诺 (验证者不可见)
- 权重 W: 承诺 (验证者不可见)
- 输出 Y: 承诺 (验证者不可见)
- Tables: P 和 V 共用 (公开)

使用方法:
    # 生成查找表 (只需一次)
    ./zkllm/table_gen 2048 tables.bin
    
    # 生成并验证完整 Transformer 证明
    python verify.py --prove --verify \\
        --input-file input.bin \\
        --weights-dir ./weights \\
        --seq-len 1 --embed-dim 2048 --hidden-dim 5504 \\
        --tables tables.bin --nonce test123
    
    # 只验证已有证明
    python verify.py --verify --proof-file proof.bin --tables tables.bin
"""

import os
import sys
import subprocess
import argparse
from typing import Tuple, Optional, List
from dataclasses import dataclass

# ================= 路径配置 =================
ZKLLM_DIR = os.path.join(os.path.dirname(__file__), "zkhook")
DEFAULT_TABLE_GEN_PATH = os.path.join(ZKLLM_DIR, "table_gen")
DEFAULT_TRANSFORMER_PROVE_PATH = os.path.join(ZKLLM_DIR, "transformer_prove")
DEFAULT_TRANSFORMER_VERIFY_PATH = os.path.join(ZKLLM_DIR, "transformer_verify")
DEFAULT_TABLES_FILE = os.path.join(ZKLLM_DIR, "tables.bin")
DEFAULT_PROOF_FILE = os.path.join(ZKLLM_DIR, "transformer_proof.bin")
DEFAULT_EVIDENCE_ROOT_DIR = os.environ.get(
    "ZK_EVIDENCE_ROOT_DIR", 
    os.path.join(os.path.dirname(__file__), "zk_evidence")
)


@dataclass
class TransformerProveConfig:
    """完整 Transformer 层证明配置"""
    input_file: str          # 输入文件路径 (float32)
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


def transformer_prove(config: TransformerProveConfig, verbose: bool = True) -> Tuple[bool, str]:
    """生成完整 Transformer 层的 ZK 证明"""
    if not os.path.exists(DEFAULT_TRANSFORMER_PROVE_PATH):
        return False, f"找不到 transformer_prove: {DEFAULT_TRANSFORMER_PROVE_PATH}"
    
    # 检查查找表
    if not os.path.exists(config.tables_file):
        return False, f"找不到查找表文件: {config.tables_file}\n请先运行: ./zkhook/table_gen {config.embed_dim} {config.tables_file}"
    
    cmd = [
        DEFAULT_TRANSFORMER_PROVE_PATH,
        config.input_file,
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
    
    ok = result.returncode == 0 and "验证通过" in out
    return ok, out


def find_transformer_evidence(
    evidence_root_dir: str,
    request_id: str,
    model_size: str = "1.8B",
    layer_idx: int = 0
) -> Optional[TransformerProveConfig]:
    """从 evidence 目录中查找完整 Transformer 证据"""
    base_dir = os.path.join(evidence_root_dir, request_id, model_size, f"layer_{layer_idx}")
    attn_dir = os.path.join(base_dir, "attention")
    ffn_dir = os.path.join(base_dir, "ffn")
    
    if not os.path.exists(attn_dir) or not os.path.exists(ffn_dir):
        return None
    
    # 查找最新的证据文件
    import glob
    
    # Attention 输入
    attn_inputs = sorted(glob.glob(os.path.join(attn_dir, "attn_input_*.bin")))
    if not attn_inputs:
        return None
    input_file = [f for f in attn_inputs if "_int_" not in f][-1]
    
    # 权重文件 (需要在权重目录中)
    # 这里简化处理，实际应该从配置或环境变量获取
    weights_dir = os.environ.get("WEIGHTS_DIR", "./weights")
    
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
        except:
            pass
    
    proof_file = os.path.join(evidence_root_dir, request_id, f"transformer_proof_{model_size}.bin")
    tables_file = os.path.join(ZKLLM_DIR, f"tables_{model_size}.bin")
    
    return TransformerProveConfig(
        input_file=input_file,
        q_weight_file=os.path.join(weights_dir, "self_attn.q_proj.weight.bin"),
        k_weight_file=os.path.join(weights_dir, "self_attn.k_proj.weight.bin"),
        v_weight_file=os.path.join(weights_dir, "self_attn.v_proj.weight.bin"),
        o_weight_file=os.path.join(weights_dir, "self_attn.o_proj.weight.bin"),
        up_weight_file=os.path.join(weights_dir, "mlp.up_proj.weight.bin"),
        gate_weight_file=os.path.join(weights_dir, "mlp.gate_proj.weight.bin"),
        down_weight_file=os.path.join(weights_dir, "mlp.down_proj.weight.bin"),
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
    parser.add_argument("--input-file", help="输入文件路径 (float32)")
    parser.add_argument("--weights-dir", help="权重目录")
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
            args.model_size
        )
        if config:
            print(f"自动发现 Transformer 证据")

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
        
        weights_dir = args.weights_dir or "./weights"
        config = TransformerProveConfig(
            input_file=args.input_file or "",
            q_weight_file=os.path.join(weights_dir, "self_attn.q_proj.weight.bin"),
            k_weight_file=os.path.join(weights_dir, "self_attn.k_proj.weight.bin"),
            v_weight_file=os.path.join(weights_dir, "self_attn.v_proj.weight.bin"),
            o_weight_file=os.path.join(weights_dir, "self_attn.o_proj.weight.bin"),
            up_weight_file=os.path.join(weights_dir, "mlp.up_proj.weight.bin"),
            gate_weight_file=os.path.join(weights_dir, "mlp.gate_proj.weight.bin"),
            down_weight_file=os.path.join(weights_dir, "mlp.down_proj.weight.bin"),
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            tables_file=args.tables,
            proof_file=args.proof_file,
            nonce=args.nonce,
        )

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
            print(f"❌ 证明生成失败")
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
