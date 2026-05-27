# GaC zkhook 本地 Web UI 使用说明

本文档说明如何启动和使用本项目的本地工程控制台 `web_ui.py`，完成两模型协同推理、生成 `zk_evidence`、基于 `zk_weights` 生成 transformer proof、验证 proof，并统计每 token 开销。

## 1. 启动前检查

进入项目目录：

```bash
cd <repo-root>
```

激活运行环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <your-conda-env>
```

确认长期权重目录存在：

```bash
ls zk_weights/qwen-new-7B
ls zk_weights/Qwen1.5-7B-Chat
```

`zk_weights` 是长期保留目录，proof 阶段会反复读取这里的 int32 权重。不要把它当作临时 evidence 删除。

确认 CUDA 可执行文件和查表文件存在：

```bash
ls zkhook/transformer_prove
ls zkhook/transformer_verify
ls zkhook/tables_7B.bin
```

检查 GPU 是否空闲：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

如果已有 Ray 或模型进程残留，先清理：

```bash
ray stop --force
```

## 2. 启动 Web UI

前台启动：

```bash
python web_ui.py
```

浏览器打开：

```text
http://127.0.0.1:7860/
```

后台启动：

```bash
nohup python web_ui.py > log/web_ui.log 2>&1 < /dev/null &
```

检查 Web UI 状态：

```bash
curl -sS http://127.0.0.1:7860/ui/status
```

如果 `7860` 端口被占用，先找到旧进程：

```bash
ps -ef | grep web_ui.py
```

停止旧 Web UI：

```bash
kill <pid>
```

## 3. 页面流程

### 3.1 选择模型

在“模型选择”区域勾选两个模型：

- `qwen-new-7B`
- `Qwen1.5-7B-Chat`

建议配置：

```text
score=100
priority=supportive
quantization=none
max_memory=22GiB
```

点击“生成 YAML”。页面会显示生成的配置文件路径，例如：

```text
generated_configs/ui_config_<timestamp>.yaml
```

页面也会显示实际复现命令，可复制保存。

### 3.2 启动融合 API 服务

在“服务控制”区域点击“启动服务”。

Web UI 后端会调用：

```bash
python gac_api_server.py --config-path <config> --host 127.0.0.1 --port 8001
```

并设置环境变量：

```bash
ZK_EVIDENCE_USE_MODEL_NAME=1
```

等待顶部状态变成：

```text
API: ready :8001
```

注意：

- `7860` 是 Web UI 控制台端口。
- `8001` 是融合生成服务端口。
- 如果生成时报 `Connection refused`，通常说明 `8001` 服务没有启动或尚未 ready。

### 3.3 发送生成请求

在“生成”区域输入：

```text
Hello
```

设置：

```text
max_new_tokens=1
apply_chat_template=false
```

点击“发送生成请求”。

页面会生成或使用当前 `request_id`，例如：

```text
ui_1779700000000
```

`request_id` 支持复制。后续 evidence、proof、metrics 都围绕这个 `request_id` 展开。

## 4. 检查 evidence

生成完成后，页面会扫描：

```text
zk_evidence/<request_id>
```

应看到两个模型目录：

```text
zk_evidence/<request_id>/qwen-new-7B/layer_0
zk_evidence/<request_id>/Qwen1.5-7B-Chat/layer_0
```

每个模型目录应至少包含：

```text
input_int.bin
rms_inv.bin
attention/
ffn/
rmsnorm/
```

检查是否存在重复权重副本：

```bash
find zk_evidence/<request_id> -name 'weight_*.bin' -type f | wc -l
```

结果应为：

```text
0
```

如果页面出现红色警告：

```text
证据中包含重复权重副本，会导致证明开销异常，请检查 hook 逻辑。
```

说明 hook 又把 `weight_*.bin` 写回了 `zk_evidence`，需要优先修复。正常流程应从 `zk_weights/<model-name>` 读取权重，不应把权重复制到 evidence。

## 5. 生成 proof

在“证明”区域确认每个模型的 `zk_weights` 路径：

```text
zk_weights/qwen-new-7B
zk_weights/Qwen1.5-7B-Chat
```

点击：

```text
生成 proof
```

或：

```text
重新生成 proof
```

Web UI 后端会复用 `verify.py`，等价命令类似：

```bash
python verify.py \
  --prove \
  --verify \
  --request-id <request_id> \
  --model-size qwen-new-7B \
  --weights-dir zk_weights/qwen-new-7B
```

另一个模型：

```bash
python verify.py \
  --prove \
  --verify \
  --request-id <request_id> \
  --model-size Qwen1.5-7B-Chat \
  --weights-dir zk_weights/Qwen1.5-7B-Chat
```

proof 文件会生成在：

```text
zk_evidence/<request_id>/transformer_proof_qwen-new-7B.bin
zk_evidence/<request_id>/transformer_proof_Qwen1.5-7B-Chat.bin
```

路径在页面中可复制。

重要注意：

当前真实测试发现，融合服务常驻占用 GPU 时，直接生成 proof 可能触发：

```text
CUDA error at zkSoftmax::zkSoftmax: an illegal memory access was encountered
```

推荐流程是：

1. 先生成回答和 evidence。
2. 点击“停止服务”，释放模型和 Ray。
3. 确认 `nvidia-smi` 没有残留计算进程。
4. 再生成 proof。

## 6. 验证 proof

如果 proof 已经存在，可以点击：

```text
只验证已有 proof
```

Web UI 后端会复用 `verify.py`，等价命令类似：

```bash
python verify.py \
  --verify \
  --proof-file zk_evidence/<request_id>/transformer_proof_qwen-new-7B.bin \
  --tables zkhook/tables_7B.bin
```

验证通过时，页面会显示绿色：

```text
验证通过
```

## 7. 查看开销统计

点击“刷新统计”后，页面会显示：

- evidence MB
- proof MB
- total MB
- tokens
- MB/token

也可以命令行查看：

```bash
curl -sS http://127.0.0.1:7860/ui/metrics/<request_id>
```

## 8. 清理本次 evidence

页面提供：

```text
清理本次 evidence
```

该按钮只删除：

```text
zk_evidence/<request_id>
```

不会删除：

```text
zk_weights/
```

也可以调用接口：

```bash
curl -sS -X POST http://127.0.0.1:7860/ui/cleanup-evidence \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"<request_id>"}'
```

再次提醒：`zk_weights` 是长期权重目录，proof 会反复读取它，不要清理。

## 9. 停止服务

在页面点击：

```text
停止服务
```

Web UI 会：

1. kill `gac_api_server.py` 服务进程。
2. 执行 `ray stop --force`。

命令行也可以调用：

```bash
curl -sS -X POST http://127.0.0.1:7860/ui/stop-server \
  -H 'Content-Type: application/json' \
  -d '{}'
```

停止后检查 GPU：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

如果输出为空，说明没有残留计算进程。

## 10. 常见错误

### 10.1 生成请求 Connection refused

错误示例：

```text
Failed to establish a new connection: [Errno 111] Connection refused
```

原因：

```text
8001 上的融合 API 服务没有启动或尚未 ready。
```

处理：

1. 页面点击“启动服务”。
2. 等待 `API: ready :8001`。
3. 再发送生成请求。

### 10.2 CUDA OOM

错误中通常包含：

```text
CUDA out of memory
```

处理：

- 停止融合服务释放 GPU。
- 执行 `ray stop --force`。
- 降低 `max_new_tokens`。
- 减少同时加载的模型数量。
- 条件允许时，把 proof 放到独立 GPU。

### 10.3 tokenizer.chat_template 缺失

错误中通常包含：

```text
chat_template
```

处理：

- 页面中把 `apply_chat_template` 设为 `false`。
- 或给对应 tokenizer 配置 chat template。

### 10.4 zk_weights 缺失

错误示例：

```text
zk_weights 不存在
```

处理：

确认目录存在：

```bash
ls zk_weights/<model-name>
```

如果确实不存在，需要先从 safetensors 导出 int32 权重。

### 10.5 transformer_prove 找不到

错误中可能包含：

```text
transformer_prove: not found
No such file or directory
```

处理：

检查：

```bash
ls zkhook/transformer_prove
```

如果不存在，需要重新编译 CUDA/zkhook 可执行文件。

### 10.6 tables_7B.bin 缺失

错误示例：

```text
lookup table 文件不存在
tables_7B.bin
```

处理：

检查：

```bash
ls zkhook/tables_7B.bin
```

7B 模型默认使用：

```text
zkhook/tables_7B.bin
```

### 10.7 Ray 残留占用 GPU

表现：

- `nvidia-smi` 仍有 Python/Ray 计算进程。
- proof 报 CUDA 非法访存。
- 服务已经停止但显存没有释放。

处理：

```bash
ray stop --force
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

必要时手动 kill 残留 PID。

## 11. 推荐完整流程

```text
启动 Web UI
  -> 选择 qwen-new-7B + Qwen1.5-7B-Chat
  -> 生成 YAML
  -> 启动融合服务
  -> 等待 API ready
  -> 输入 prompt 并生成
  -> 复制 request_id
  -> 扫描 evidence
  -> 确认 weight_*.bin = 0
  -> 停止融合服务并清理 Ray
  -> 生成/重新生成 proof
  -> 只验证已有 proof
  -> 刷新开销统计
  -> 按需清理本次 evidence
```
