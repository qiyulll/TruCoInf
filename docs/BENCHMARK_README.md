# zkhook 基准测试指南

本目录包含了对 zkhook 系统进行基准测试的脚本，支持 MMLU 和 GSM8K 两个标准数据集。

## 前置要求

### 1. 安装依赖

首先确保安装了必要的 Python 包：

```bash
# 安装基本依赖
pip install -r requirements.txt

# 安装基准测试所需的包
pip install datasets requests
```

### 2. 启动 GaC API 服务器

基准测试脚本通过调用 GaC API 来获取模型的推理结果。

```bash
# 配置 ZK 相关的环境变量
export ZK_EVIDENCE_ROOT_DIR="<repo-root>/zk_evidence"
export ZK_ATTN_EVIDENCE=1

# 启动 API 服务器（需要在另一个终端执行）
python gac_api_server.py --config-path config.yaml --port 8000
```

服务器启动后，应该会显示：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 基准测试脚本

### 1. MMLU 基准测试 (`benchmark_mmlu.py`)

MMLU (Massive Multitask Language Understanding) 是一个包含57个科目的多任务基准，共12,578个多选题。

#### 基本用法

```bash
# 运行完整 MMLU 基准测试
python benchmark_mmlu.py --url http://localhost:8000/api/generate/

# 指定最大样本数（用于快速测试）
python benchmark_mmlu.py --url http://localhost:8000/api/generate/ --max-samples 100

# 只测试特定科目
python benchmark_mmlu.py --url http://localhost:8000/api/generate/ \
  --subjects abstract_algebra college_biology college_physics

# Few-shot 评估
python benchmark_mmlu.py --url http://localhost:8000/api/generate/ --num-shots 5
```

#### 输出

测试完成后，会生成 `mmlu_results_YYYYMMDD_HHMMSS.json` 文件，包含：
- 总体准确率
- 各科目的准确率和平均延迟
- 详细的题目答案记录
- 性能统计

#### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--url` | GaC API URL | `http://localhost:8000/api/generate/` |
| `--num-shots` | Few-shot 数量 | `0` (zero-shot), `5` (5-shot) |
| `--max-samples` | 最大样本数 | `100`, `None` (全部) |
| `--subjects` | 特定科目列表 | `abstract_algebra college_biology` |
| `--no-save` | 不保存结果文件 | - |

---

### 2. GSM8K 基准测试 (`benchmark_gsm8k.py`)

GSM8K (Grade School Math 8K) 是一个包含8,500个小学和中学数学应用题的数据集。

#### 基本用法

```bash
# 运行完整 GSM8K 基准测试
python benchmark_gsm8k.py --url http://localhost:8000/api/generate/

# 只测试 100 个样本
python benchmark_gsm8k.py --url http://localhost:8000/api/generate/ --max-samples 100

# 使用 train 集进行测试
python benchmark_gsm8k.py --url http://localhost:8000/api/generate/ --split train

# 不要求推理过程（只要求最终答案）
python benchmark_gsm8k.py --url http://localhost:8000/api/generate/ --no-reasoning
```

#### 输出

测试完成后，会生成 `gsm8k_results_YYYYMMDD_HHMMSS.json` 文件，包含：
- 总体准确率
- 答案提取成功率
- 平均延迟
- 详细的题目答案记录

#### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--url` | GaC API URL | `http://localhost:8000/api/generate/` |
| `--split` | 数据集分割 | `test` (默认), `train` |
| `--max-samples` | 最大样本数 | `100`, `None` (全部) |
| `--no-save` | 不保存结果文件 | - |
| `--no-reasoning` | 不要求推理过程 | - |

---

### 3. 综合基准测试 (`benchmark_all.py`)

同时运行 MMLU 和 GSM8K，并统计总耗时。

#### 基本用法

```bash
# 运行所有基准测试
python benchmark_all.py --all

# 只运行 MMLU
python benchmark_all.py --mmlu

# 只运行 GSM8K
python benchmark_all.py --gsm8k

# 限制样本数并运行所有基准
python benchmark_all.py --all --max-samples 50

# 自定义每个基准的配置
python benchmark_all.py \
  --mmlu --mmlu-num-shots 5 \
  --gsm8k --gsm8k-split test
```

---

## 完整测试工作流

### 步骤 1: 编译 zkhook

```bash
cd <repo-root>/zkhook

export CUDA_HOME=/usr/local/cuda
make clean
make all
make stepwise
```

### 步骤 2: 下载模型（如果还没下载）

```bash
# 创建权重目录
mkdir -p <repo-root>/weights

# 下载模型（可选，或使用已有的本地模型）
# 参考 复现步骤.txt 中的模型下载部分
```

### 步骤 3: 准备配置文件

编辑 `config.yaml` 确保模型路径正确：

```yaml
NORM_TYPE_API_SERVER: 'average'
THRESHOLD_API_SERVER: 1.0

CONFIG_API_SERVER:
  - weight: './weights/DeepSeek-R1-Distill-Qwen-7B'
    max_memory:
      0: '22GiB'
    num_gpus: 1
    name: 'DeepSeek-R1-Distill-Qwen-7B'
    score: 100
    priority: 'primary'
    quantization: 'none'

  - weight: './weights/Qwen1.5-1.8B-Chat'
    max_memory:
      0: '10GiB'
    num_gpus: 1
    name: 'Qwen1.5-1.8B-Chat'
    score: 100
    priority: 'supportive'
    quantization: 'none'
```

### 步骤 4: 启动 API 服务器

```bash
# 在一个终端中启动服务器
cd <repo-root>

export ZK_EVIDENCE_ROOT_DIR="<repo-root>/zk_evidence"
export ZK_ATTN_EVIDENCE=1

python gac_api_server.py --config-path config.yaml --port 8000
```

### 步骤 5: 运行基准测试

在另一个终端中：

```bash
cd <repo-root>

# 选项 A：快速测试（各基准各10个样本）
python benchmark_all.py --all --max-samples 10

# 选项 B：完整测试
python benchmark_all.py --all

# 选项 C：只测试MMLU
python benchmark_mmlu.py

# 选项 D：只测试GSM8K
python benchmark_gsm8k.py
```

---

## 结果分析

### MMLU 结果

MMLU 的 JSON 结果文件包含：

```json
{
  "config": {
    "num_shots": 0,
    "api_url": "http://localhost:8000/api/generate/",
    "timestamp": "2026-05-14 10:30:00"
  },
  "subjects": {
    "abstract_algebra": {
      "correct": 15,
      "total": 100,
      "accuracy": 0.15,
      "latency_avg": 2.5,
      "results": [...]
    },
    ...
  },
  "overall": {
    "correct": 800,
    "total": 12578,
    "accuracy": 0.0636,
    "latency_avg": 2.3
  }
}
```

### GSM8K 结果

GSM8K 的 JSON 结果文件包含：

```json
{
  "config": {
    "split": "test",
    "api_url": "http://localhost:8000/api/generate/",
    "include_reasoning": true,
    "timestamp": "2026-05-14 10:40:00"
  },
  "results": [
    {
      "question": "...",
      "ground_truth_answer": 42.0,
      "predicted_answer": 42.0,
      "extraction_success": true,
      "is_correct": true,
      "latency": 3.2
    },
    ...
  ],
  "summary": {
    "correct": 3000,
    "total": 8500,
    "accuracy": 0.3529,
    "latency_avg": 3.1,
    "extraction_success_rate": 0.95
  }
}
```

### 性能指标解读

1. **Accuracy（准确率）**：模型给出正确答案的比例
2. **Latency（延迟）**：每个请求的平均处理时间（秒）
3. **Extraction Success Rate（提取成功率）**：仅适用于GSM8K，表示成功提取数值答案的比例

---

## 故障排查

### 问题 1：连接失败

```
✗ 连接失败: Connection refused
```

**解决方案**：确保 GaC API 服务器已启动

```bash
# 检查服务器状态
curl http://localhost:8000/status
```

### 问题 2：模型加载失败

**解决方案**：检查 `config.yaml` 中的模型路径和显存分配

### 问题 3：超时错误

**解决方案**：增加超时时间或减少 `max_new_tokens`

### 问题 4：内存不足

**解决方案**：使用 `--max-samples` 参数限制样本数，或调整 `config.yaml` 中的量化选项

---

## 性能优化建议

1. **并行处理**：修改脚本使用异步请求提高吞吐量
2. **批量推理**：调整 `config.yaml` 中的 batch size
3. **模型量化**：使用 4-bit 量化减少显存占用
4. **缓存**：对重复的请求进行缓存

---

## 数据集信息

### MMLU

- **规模**：12,578 个多选题
- **科目**：57 个，包括数学、科学、历史、法律等
- **评估方式**：多选题（A、B、C、D）
- **难度**：中等到困难
- **来源**：https://github.com/hendrycks/test

### GSM8K

- **规模**：8,500 个数学应用题
- **难度等级**：小学到中学（Grade School）
- **答案类型**：数值型
- **评估方式**：完全匹配（支持浮点数精度容错）
- **来源**：https://github.com/openai/grade-school-math

---

## 许可证

见 LICENSE 文件

---

## 问题反馈

如有问题，请检查：
1. 依赖包是否安装正确
2. API 服务器是否正常运行
3. 模型路径是否正确
4. 显存是否充足
