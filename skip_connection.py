#!/usr/bin/env python3
"""
skip_connection.py - Skip Connection 计算

计算 output = input_a + input_b

Skip Connection 不需要单独的零知识证明，因为：
1. 加法在有限域中可以直接验证
2. Verifier 可以通过检查承诺来验证 output = input_a + input_b

用法:
    python skip_connection.py --input_a A.bin --input_b B.bin --output C.bin
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Skip Connection: output = input_a + input_b")
    parser.add_argument("--input_a", required=True, type=str, help="第一个输入文件 (int32 fixed-point)")
    parser.add_argument("--input_b", required=True, type=str, help="第二个输入文件 (int32 fixed-point)")
    parser.add_argument("--output", required=True, type=str, help="输出文件 (int32 fixed-point)")
    args = parser.parse_args()

    # 读取输入
    a = np.fromfile(args.input_a, dtype=np.int32)
    b = np.fromfile(args.input_b, dtype=np.int32)

    if a.shape != b.shape:
        print(f"错误: 输入形状不匹配 {a.shape} vs {b.shape}")
        return 1

    # 计算 skip connection (简单加法)
    # 注意：在有限域中，加法可能会溢出，但这是预期行为
    # 实际的 ZKP 验证会在有限域中进行
    output = a.astype(np.int64) + b.astype(np.int64)
    
    # 裁剪到 int32 范围（模拟有限域行为的简化版本）
    output = output.astype(np.int32)

    # 保存输出
    output.tofile(args.output)

    print(f"Skip Connection 完成")
    print(f"  input_a: {args.input_a} ({a.size} elements)")
    print(f"  input_b: {args.input_b} ({b.size} elements)")
    print(f"  output:  {args.output} ({output.size} elements)")

    return 0


if __name__ == "__main__":
    exit(main())
