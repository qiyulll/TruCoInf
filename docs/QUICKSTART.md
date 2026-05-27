# zkhook 基准测试快速入门

## 📋 前置检查清单

在开始基准测试前，请确保：

- [ ] zkhook 已编译（`zkhook` 目录中有 `table_gen`, `transformer_prove` 等可执行文件）
- [ ] 模型已下载到 `weights/` 目录
- [ ] `config.yaml` 已正确配置
- [ ] Python 3.8+ 已安装
- [ ] 有足够的显存（建议 24GB+）

---

## ⚡ 5分钟快速开始

### 方法 A：使用启动脚本（推荐）

**终端 1：启动 API 服务器**

```bash
cd <repo-root>
chmod +x start_api_server.sh
./start_api_server.sh --config config.yaml --port 8000
```

等待显示：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**终端 2：运行基准测试**

```bash
cd <repo-root>
chmod +x run_benchmark.sh

# 快速测试（各基准10个样本，耗时约5-10分钟）
./run_benchmark.sh --quick --all

# 完整测试（MMLU 12578 个 + GSM8K 8500 个样本，耗时数小时）
./run_benchmark.sh --all
```

### 方法 B：手动命令

**终端 1：启动 API 服务器**

```bash
cd <repo-root>

# 设置环境变量
export ZK_EVIDENCE_ROOT_DIR="$(pwd)/zk_evidence"
export ZK_ATTN_EVIDENCE=1

# 启动服务器
python3 gac_api_server.py --config-path config.yaml --port 8000
```

**终端 2：运行基准测试**

```bash
cd <repo-root>

# 快速测试
python3 benchmark_all.py --all --max-samples 10

# 或分别运行
python3 benchmark_mmlu.py --max-samples 50
python3 benchmark_gsm8k.py --max-samples 50
```

---

## 📊 常见测试场景

### 场景 1：快速验证（5-10分钟）
```bash
./run_benchmark.sh --quick --all
```
✓ 各基准各10个样本  
✓ 快速验证系统是否正常

### 场景 2：中等评估（30-60分钟）
```bash
python3 benchmark_mmlu.py --max-samples 500
python3 benchmark_gsm8k.py --max-samples 500
```
✓ MMLU 500个样本  
✓ GSM8K 500个样本

### 场景 3：完整评估（2-4小时）
```bash
./run_benchmark.sh --all
```
✓ MMLU 完整（12,578个）  
✓ GSM8K 完整（8,500个）

### 场景 4：特定科目测试（30分钟）
```bash
python3 benchmark_mmlu.py --subjects abstract_algebra \
  college_biology college_physics --max-samples 100
```
✓ 仅测试指定科目

---

## 🔧 配置优化

### 显存不足？

编辑 `config.yaml`，启用量化：

```yaml
CONFIG_API_SERVER:
  - weight: './weights/DeepSeek-R1-Distill-Qwen-7B'
    quantization: '4bit'  # 改为 '4bit'
    max_memory:
      0: '12GiB'  # 减少分配
```

### 推理太慢？

调整 `config.yaml` 中的 `THRESHOLD_API_SERVER`：

```yaml
THRESHOLD_API_SERVER: 0.8  # 提高阈值，更多使用单模型
```

---

## 📈 理解结果

### MMLU 结果示例

```
总体准确率: 45.23%
  正确: 5689/12578
平均延迟: 2.34s

各科目准确率 (从高到低):
  abstract_algebra: 62.0% (62/100)
  college_biology: 51.5% (51/100)
  ...
```

**指标说明：**
- **准确率**：答对的比例（0-100%）
- **延迟**：平均每题需要的时间（秒）
- **样本数**：正确数/总数

### GSM8K 结果示例

```
准确率: 35.29%
  正确: 3000/8500
答案提取成功率: 95.1%
平均延迟: 3.12s
```

**指标说明：**
- **准确率**：数学题答对的比例
- **提取成功率**：成功从模型输出中提取数值答案的比例
- **延迟**：平均每题需要的时间

---

## 🐛 常见问题

### Q: 连接失败："Connection refused"
**A:** 确保 API 服务器已启动。检查另一个终端中是否运行了 `start_api_server.sh` 或 `gac_api_server.py`

### Q: 模型加载失败
**A:** 检查：
1. `config.yaml` 中的模型路径是否正确
2. 模型文件是否完整
3. 显存是否充足

### Q: 超时错误
**A:** 模型推理较慢。可以：
1. 减少 `--max-samples`
2. 增加超时时间（修改脚本中的 `timeout` 参数）
3. 减少并发

### Q: 显存溢出
**A:** 尝试：
1. 启用量化（`quantization: '4bit'`）
2. 减少 `max_memory` 的分配
3. 只使用一个模型

### Q: 答案提取错误率高
**A:** 这是正常现象。对于 GSM8K，模型可能输出格式不规范。可以：
1. 调整提示词（修改脚本中的 prompt）
2. 增加 `max_new_tokens` 让模型有更多空间输出

---

## 🎯 下一步

基准测试完成后：

1. **分析结果**
   - 查看生成的 JSON 文件
   - 对比不同模型/配置的性能差异

2. **优化配置**
   - 根据延迟和准确率调整阈值
   - 尝试不同的集成策略

3. **扩展评估**
   - 添加更多基准（如 HellaSwag、ARC 等）
   - 测试不同的 few-shot 设置

---

## 📚 更多信息

- 详细文档：见 `BENCHMARK_README.md`
- 配置说明：见 `config.yaml`
- zkhook 说明：见 `zkhook/README.md`

---

## 💡 提示

1. 第一次运行会下载数据集（MMLU/GSM8K），需要网络连接
2. 数据集会缓存在本地，后续运行会更快
3. 建议先用 `--quick` 进行快速测试，验证系统正常后再进行完整测试
4. 保存结果文件便于后续分析比较

---

祝你测试顺利！有问题欢迎反馈。
