# 📑 zkhook 基准测试工具索引

> 完整的 MMLU 和 GSM8K 基准测试框架

## 🚀 我想...

### 快速开始
- **3步启动** → 见 [QUICKSTART.md](QUICKSTART.md)
- **查看常用命令** → 见 [USAGE_GUIDE.md](USAGE_GUIDE.md) 中的"常见命令速查表"

### 运行测试
- **快速测试（5分钟）** 
  ```bash
  ./run_benchmark.sh --quick --all
  ```
  
- **测试 MMLU**
  ```bash
  python3 benchmark_mmlu.py --max-samples 100
  ```
  
- **测试 GSM8K**
  ```bash
  python3 benchmark_gsm8k.py --max-samples 100
  ```

### 分析结果
- **查看结果列表**
  ```bash
  python3 analyze_results.py list
  ```

- **分析单个结果**
  ```bash
  python3 analyze_results.py analyze mmlu_results_*.json
  ```

### 配置优化
- **改变 API 服务器地址** → 修改 `config.yaml`
- **启用量化节省显存** → 见 [USAGE_GUIDE.md](USAGE_GUIDE.md) 中的"高级参数配置"
- **调整阈值改变集成策略** → 编辑 `config.yaml` 中的 `THRESHOLD_API_SERVER`

### 理解结果
- **MMLU 准确率怎么算的** → 见 [BENCHMARK_README.md](BENCHMARK_README.md)
- **GSM8K 准确率怎么算的** → 见 [BENCHMARK_README.md](BENCHMARK_README.md)
- **比较两个结果的区别** → 见 [USAGE_GUIDE.md](USAGE_GUIDE.md) 中的"结果分析"

### 遇到问题
- **连接失败** → 见 [BENCHMARK_README.md](BENCHMARK_README.md) 中的"故障排查"
- **显存不足** → 见 [USAGE_GUIDE.md](USAGE_GUIDE.md) 中的"显存优化建议"
- **测试太慢** → 见 [USAGE_GUIDE.md](USAGE_GUIDE.md) 中的"故障排除 - 问题：测试很慢"

---

## 📚 完整文档列表

### 用户指南

| 文档 | 内容 | 读者 |
|------|------|------|
| [SUMMARY.md](SUMMARY.md) | 📋 完成总结，功能清单 | 👀 快速了解 |
| [QUICKSTART.md](QUICKSTART.md) | ⚡ 5分钟快速入门 | 👶 新手用户 |
| [BENCHMARK_README.md](BENCHMARK_README.md) | 📖 详细使用文档 | 📖 深入学习 |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | 🔧 完整使用指南 | 🚀 进阶用户 |

### 脚本文件

#### Python 脚本

| 脚本 | 功能 | 使用方式 |
|------|------|--------|
| `benchmark_mmlu.py` | MMLU基准测试 | `python3 benchmark_mmlu.py [选项]` |
| `benchmark_gsm8k.py` | GSM8K基准测试 | `python3 benchmark_gsm8k.py [选项]` |
| `benchmark_all.py` | 综合基准测试 | `python3 benchmark_all.py --all` |
| `analyze_results.py` | 结果分析工具 | `python3 analyze_results.py analyze file.json` |

#### Shell 脚本

| 脚本 | 功能 | 使用方式 |
|------|------|--------|
| `start_api_server.sh` | 启动API服务器 | `./start_api_server.sh --config config.yaml` |
| `run_benchmark.sh` | 运行基准测试 | `./run_benchmark.sh --quick --all` |

---

## 🎯 常见任务

### 任务 1：验证系统正常（5分钟）

```bash
# 终端1：启动服务器
./start_api_server.sh

# 终端2：快速测试
./run_benchmark.sh --quick --all

# 查看结果
python3 analyze_results.py list
```

### 任务 2：完整评估（2-4小时）

```bash
# 终端1：启动服务器
./start_api_server.sh

# 终端2：运行完整测试
./run_benchmark.sh --all

# 测试完成后分析
python3 analyze_results.py analyze mmlu_results_*.json
python3 analyze_results.py analyze gsm8k_results_*.json
```

### 任务 3：特定科目测试（30分钟）

```bash
# 只测试MMLU中的特定科目
python3 benchmark_mmlu.py \
  --subjects abstract_algebra college_biology \
  --max-samples 100
```

### 任务 4：性能对比（1-2小时）

```bash
# 测试配置A
# 修改 config.yaml，然后：
python3 benchmark_mmlu.py --max-samples 500

# 测试配置B
# 修改 config.yaml，然后：
python3 benchmark_mmlu.py --max-samples 500

# 比较结果
python3 analyze_results.py compare mmlu_results_*.json
```

---

## 💡 快速参考

### MMLU 参数

```bash
python3 benchmark_mmlu.py \
  --url http://localhost:8000/api/generate/    # API地址
  --num-shots 0                                 # Few-shot数
  --max-samples 100                             # 样本限制
  --subjects math science                       # 指定科目
  --no-save                                     # 不保存
```

### GSM8K 参数

```bash
python3 benchmark_gsm8k.py \
  --url http://localhost:8000/api/generate/    # API地址
  --split test                                  # test或train
  --max-samples 100                             # 样本限制
  --no-reasoning                                # 不要推理
  --no-save                                     # 不保存
```

### 启动脚本参数

```bash
./run_benchmark.sh \
  --all                    # 运行所有基准
  --quick                  # 快速测试模式
  --max-samples 50         # 限制样本数
  --url http://...         # API地址
```

---

## 📊 输出文件

### MMLU 结果

文件名：`mmlu_results_YYYYMMDD_HHMMSS.json`

包含：
- 总体准确率和延迟
- 57个科目的逐个结果
- 每个题目的详细答案记录

### GSM8K 结果

文件名：`gsm8k_results_YYYYMMDD_HHMMSS.json`

包含：
- 总体准确率
- 答案提取成功率
- 每个题目的详细答案记录

---

## 🔗 相关文件

- `config.yaml` - API服务器配置
- `gac_api_server.py` - API服务器代码
- `requirements.txt` - Python依赖

---

## ✅ 功能检查清单

- ✅ MMLU 基准测试（57个科目，12,578题）
- ✅ GSM8K 基准测试（8,500题）
- ✅ 快速/中等/完整三档测试模式
- ✅ 灵活的采样选项
- ✅ 详细的结果分析
- ✅ 结果对比工具
- ✅ 启动脚本简化使用
- ✅ 完整的文档

---

## 🆘 获取帮助

### 问题的不同类别

| 类别 | 建议查看 |
|------|--------|
| 如何启动？ | [QUICKSTART.md](QUICKSTART.md) |
| 如何运行测试？ | [BENCHMARK_README.md](BENCHMARK_README.md) - "基准测试脚本" |
| 如何理解结果？ | [BENCHMARK_README.md](BENCHMARK_README.md) - "结果分析" |
| 如何配置优化？ | [USAGE_GUIDE.md](USAGE_GUIDE.md) - "高级参数配置" |
| 遇到错误？ | [BENCHMARK_README.md](BENCHMARK_README.md) - "故障排查" |
| 其他问题？ | [USAGE_GUIDE.md](USAGE_GUIDE.md) - "深度调查" |

---

## 📞 文档导航树

```
README
├── 快速入门
│   ├── QUICKSTART.md (5分钟快速开始)
│   └── run_benchmark.sh (一键启动)
│
├── 详细学习
│   ├── BENCHMARK_README.md
│   │   ├── 前置要求
│   │   ├── 脚本说明
│   │   ├── 完整工作流
│   │   ├── 结果分析
│   │   └── 故障排查
│   │
│   └── USAGE_GUIDE.md
│       ├── 常见命令速查表
│       ├── 高级参数配置
│       ├── 深度调查方法
│       └── 实验设计建议
│
└── 参考文档
    ├── SUMMARY.md (总体总结)
    └── INDEX.md (本文件)
```

---

## 🎓 推荐阅读顺序

**第一次使用**
1. 快速了解 → [SUMMARY.md](SUMMARY.md)
2. 快速启动 → [QUICKSTART.md](QUICKSTART.md)
3. 运行测试 → 使用 `./run_benchmark.sh`

**深入学习**
4. 详细文档 → [BENCHMARK_README.md](BENCHMARK_README.md)
5. 使用指南 → [USAGE_GUIDE.md](USAGE_GUIDE.md)
6. 高级用法 → 修改脚本或配置文件

---

## 🏁 开始测试

最简单的开始方式：

```bash
cd <repo-root>

# 终端1
./start_api_server.sh

# 终端2
./run_benchmark.sh --quick
```

或者查看 [QUICKSTART.md](QUICKSTART.md) 了解详细步骤。

---

**最后更新**: 2026-05-14  
**版本**: 1.0  
**状态**: ✅ 完成
