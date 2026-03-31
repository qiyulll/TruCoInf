#!/usr/bin/env python3
"""
test.py - zkLLM 完整 Transformer 层测试脚本

完整 Transformer 层零知识证明测试:
1. 调用 GaC API 生成响应并收集证据
2. 使用 transformer_prove 生成完整证明 (Attention + FFN)
3. 使用 transformer_verify 验证证明

使用方法:
    # 完整测试 (API + 证明 + 验证)
    python test.py --request-id <id>
    
    # 跳过 API，直接验证已有证据
    python test.py --skip-api --request-id <request_id>
    
    # 只验证已有证明
    python test.py --verify-only --proof-file <proof_file> --tables <tables_file>
"""

import os
import sys
import time
import uuid
import argparse
import requests
from verify import (
    DEFAULT_EVIDENCE_ROOT_DIR,
    DEFAULT_PROOF_FILE,
    DEFAULT_TABLES_FILE,
    TransformerProveConfig,
    transformer_prove,
    transformer_verify,
    generate_tables,
    check_executables,
    find_transformer_evidence,
)

# ================= 配置 =================
DEFAULT_SERVER_URL = os.environ.get("GAC_SERVER_URL", "http://localhost:8000/api/generate/")
SERVER_URL = DEFAULT_SERVER_URL
PROMPT = "Hello"
MAX_NEW_TOKENS = 10
APPLY_CHAT_TEMPLATE = True
REQUEST_ID = None  # None: 自动生成
SLEEP_SECONDS = 1.0

# 模型配置
MODEL_CONFIGS = {
    "1.8B": {"embed_dim": 2048, "hidden_dim": 5504},
    "7B": {"embed_dim": 4096, "hidden_dim": 11008},
}


def send_generate_request(
    url: str,
    prompt: str,
    request_id: str,
    max_new_tokens: int = 10,
    apply_chat_template: bool = True,
) -> str:
    """发送生成请求到 GaC API"""
    data = {
        "messages_list": [[{"role": "user", "content": prompt}]],
        "max_new_tokens": max_new_tokens,
        "apply_chat_template": apply_chat_template,
        "request_id": request_id,
    }
    res = requests.post(url, json=data, timeout=600)
    res.raise_for_status()
    response_json = res.json()
    return response_json["response"][0]


def test_transformer(request_id: str, model_size: str, weights_dir: str = None) -> dict:
    """
    测试单个模型的完整 Transformer 层证明 (Attention + FFN)
    
    返回: {
        "success": bool,
        "proof_file": str,
        "error": str (如果失败)
    }
    """
    print(f"\n{'=' * 60}")
    print(f"📦 测试模型: {model_size}")
    print(f"   request_id: {request_id}")
    print(f"{'=' * 60}")
    
    result = {
        "success": False,
        "proof_file": "",
        "error": ""
    }
    
    config = MODEL_CONFIGS.get(model_size)
    if not config:
        result["error"] = f"未知模型: {model_size}"
        return result
    
    embed_dim = config["embed_dim"]
    hidden_dim = config["hidden_dim"]
    
    # 查找证据
    evidence = find_transformer_evidence(
        DEFAULT_EVIDENCE_ROOT_DIR,
        request_id,
        model_size
    )
    
    if not evidence:
        result["error"] = f"未找到 {model_size} 模型的证据"
        print(f"   ⚠️ {result['error']}")
        return result
    
    # 如果指定了权重目录，更新配置
    if weights_dir:
        evidence.q_weight_file = os.path.join(weights_dir, "self_attn.q_proj.weight.bin")
        evidence.k_weight_file = os.path.join(weights_dir, "self_attn.k_proj.weight.bin")
        evidence.v_weight_file = os.path.join(weights_dir, "self_attn.v_proj.weight.bin")
        evidence.o_weight_file = os.path.join(weights_dir, "self_attn.o_proj.weight.bin")
        evidence.up_weight_file = os.path.join(weights_dir, "mlp.up_proj.weight.bin")
        evidence.gate_weight_file = os.path.join(weights_dir, "mlp.gate_proj.weight.bin")
        evidence.down_weight_file = os.path.join(weights_dir, "mlp.down_proj.weight.bin")
    
    # 确保查找表存在
    tables_file = os.path.join(os.path.dirname(__file__), "zkllm", f"tables_{model_size}.bin")
    evidence.tables_file = tables_file
    
    if not os.path.exists(tables_file):
        print(f"   生成查找表: {tables_file}")
        ok, msg = generate_tables(embed_dim, tables_file)
        if not ok:
            result["error"] = f"生成查找表失败: {msg}"
            print(f"   ❌ {result['error']}")
            return result
        print(f"   ✅ 查找表已生成")
    
    # 生成证明
    print(f"   生成证明...")
    prove_ok, prove_msg = transformer_prove(evidence, verbose=False)
    
    if not prove_ok:
        result["error"] = f"证明生成失败: {prove_msg}"
        print(f"   ❌ {result['error']}")
        return result
    
    print(f"   ✅ 证明已生成: {evidence.proof_file}")
    
    # 验证证明
    print(f"   验证证明...")
    verify_ok, verify_msg = transformer_verify(evidence.proof_file, evidence.tables_file, verbose=False)
    
    if not verify_ok:
        result["error"] = f"验证失败: {verify_msg}"
        print(f"   ❌ {result['error']}")
        return result
    
    print(f"   ✅ 验证通过!")
    
    result["success"] = True
    result["proof_file"] = evidence.proof_file
    return result


def main():
    parser = argparse.ArgumentParser(description="zkLLM 完整 Transformer 层测试")
    parser.add_argument("--request-id", default=REQUEST_ID, help="Request ID / Nonce")
    parser.add_argument("--skip-api", action="store_true", help="跳过 GaC API 调用")
    parser.add_argument("--verify-only", action="store_true", help="只验证，不生成证明")
    parser.add_argument("--proof-file", help="证明文件 (用于 --verify-only)")
    parser.add_argument("--tables", help="查找表文件 (用于 --verify-only)")
    parser.add_argument("--server-url", default=SERVER_URL, help="GaC API 服务器")
    parser.add_argument("--weights-dir", help="权重目录")
    parser.add_argument("--model-size", default="1.8B", help="模型大小 (1.8B/7B)")
    args = parser.parse_args()
    
    # 检查可执行文件
    exe_ok, exe_msg = check_executables()
    if not exe_ok:
        print(f"❌ {exe_msg}")
        return 1
    
    # request_id 生成
    req_id = args.request_id or uuid.uuid4().hex[:8]
    
    # 只验证模式
    if args.verify_only:
        proof_file = args.proof_file or DEFAULT_PROOF_FILE
        tables_file = args.tables or DEFAULT_TABLES_FILE
        
        if not os.path.exists(proof_file):
            print(f"❌ 找不到证明文件: {proof_file}")
            return 1
        if not os.path.exists(tables_file):
            print(f"❌ 找不到查找表: {tables_file}")
            return 1
        
        print(f"验证证明: {proof_file}")
        ok, msg = transformer_verify(proof_file, tables_file, verbose=True)
        return 0 if ok else 1
    
    # 调用 GaC API
    if not args.skip_api:
        print(f"\n🚀 请求 GaC (id={req_id}, prompt=\"{PROMPT}\")")
        try:
            answer = send_generate_request(
                url=args.server_url,
                prompt=PROMPT,
                request_id=req_id,
                max_new_tokens=MAX_NEW_TOKENS,
                apply_chat_template=APPLY_CHAT_TEMPLATE,
            )
            print(f"📥 回答: {answer}")
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ 请求错误: {e}")
            return 1
    else:
        print(f"⏭️ 跳过 API，使用 id={req_id}")
    
    # 测试完整 Transformer 层
    print("\n" + "=" * 70)
    print("🔐 zkLLM 完整 Transformer 层零知识证明测试")
    print("   包含: Self-Attention + FFN + Skip Connections")
    print("=" * 70)
    
    result = test_transformer(req_id, args.model_size, args.weights_dir)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)
    
    if result["success"]:
        print(f"✅ 测试通过!")
        print(f"   证明文件: {result['proof_file']}")
        print("\n零知识保证:")
        print("   • 验证者未看到输入 X (用户 prompt)")
        print("   • 验证者未看到权重 W (模型参数)")
        print("   • 验证者未看到输出 Y (模型回答)")
        print("   • 验证者只验证了计算的正确性")
        return 0
    else:
        print(f"❌ 测试失败!")
        print(f"   错误: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
