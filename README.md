# LLM Collaborative Reasoning with zkhook Proofs

This repository contains a local research prototype for collaborative LLM inference and zero-knowledge proof generation.

The system can:

- load two or more local HuggingFace causal language models,
- perform GaC-style token fusion during generation,
- capture model-running evidence for layer-0 Transformer computation,
- reuse long-lived `zk_weights/<model-name>` int32 weights,
- generate a CUDA zkhook Transformer proof,
- verify the generated proof,
- inspect evidence/proof size and per-token overhead from a local Web UI.

This repository intentionally does not include model weights, generated zk weights, runtime evidence, proof files, CUDA binaries, or lookup tables.

## Repository Layout

```text
.
├── web_ui.py                      # Local FastAPI engineering console
├── gac_api_server.py              # Collaborative inference API service
├── verify.py                      # transformer_prove / transformer_verify wrapper
├── export_zkhook_weights.py        # Export int32 zk weights from safetensors
├── config.yaml                    # Generic collaborative inference config
├── static/                        # Plain HTML/CSS/JS Web UI
├── utils/                         # Ray actor and generation utilities
├── zkhook/                         # CUDA zk proof source code and Makefile
├── docs/                          # Extra usage notes and historical notes
├── weights/                       # Placeholder for local HuggingFace model dirs
└── zk_weights/                    # Placeholder for exported int32 proof weights
```

## Large Files Are Not Included

You must provide these locally:

```text
weights/<model-name>/              # HuggingFace model directory with safetensors
zk_weights/<model-name>/           # Exported int32 weights for zk proof
zkhook/table_gen                    # Built CUDA executable
zkhook/transformer_prove            # Built CUDA executable
zkhook/transformer_verify           # Built CUDA executable
zkhook/tables_7B.bin                # Generated lookup table
zkhook/tables_1.8B.bin              # Generated lookup table, if needed
```

Do not commit those files to GitHub. They are ignored by `.gitignore`.

## Environment

Use a conda environment that contains PyTorch, CUDA, Ray, FastAPI, Transformers, and safetensors.

```bash
conda activate <your-conda-env>
pip install -r requirements.txt
```

Expected system components:

- NVIDIA GPU and working CUDA toolkit
- `nvcc`
- PyTorch with CUDA
- Ray
- FastAPI / uvicorn
- Transformers / safetensors

## Prepare Model Weights

Put local HuggingFace model directories under `weights/`.

Example:

```text
weights/qwen-new-7B/
weights/Qwen1.5-7B-Chat/
weights/Qwen1.5-1.8B-Chat/
weights/DeepSeek-R1-Distill-Qwen-7B/
```

The Web UI currently exposes these model names:

- `qwen-new-7B`
- `Qwen1.5-7B-Chat`
- `Qwen1.5-1.8B-Chat`
- `DeepSeek-R1-Distill-Qwen-7B`

If your local model path differs, update the generated YAML config or the model path map in `web_ui.py`.

## Build zkhook CUDA Tools

```bash
cd zkhook
make all-targets
```

Generate lookup tables as needed:

```bash
./table_gen 4096 tables_7B.bin
./table_gen 2048 tables_1.8B.bin
cd ..
```

The default `zkhook/Makefile` uses `ARCH := sm_89`. Change it for your GPU if needed, for example:

- A100: `sm_80`
- V100: `sm_70`
- RTX 30 series: `sm_86`
- RTX 40 series: `sm_89`

## Export Long-Lived zk Weights

`zk_weights` should be generated once per model and reused. It should not be regenerated for every request.

```bash
python export_zkhook_weights.py \
  --model-dir weights/qwen-new-7B \
  --output-dir zk_weights/qwen-new-7B

python export_zkhook_weights.py \
  --model-dir weights/Qwen1.5-7B-Chat \
  --output-dir zk_weights/Qwen1.5-7B-Chat
```

Use `--force` only when you intentionally want to regenerate existing exported weights.

## Run the Local Web UI

```bash
conda activate <your-conda-env>
python web_ui.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Recommended test flow:

1. Select two models, for example `qwen-new-7B` and `Qwen1.5-7B-Chat`.
2. Generate a YAML config.
3. Start the fusion API service.
4. Send a short prompt such as `Hello`.
5. Confirm that a `request_id` is generated.
6. Inspect `zk_evidence/<request_id>/`.
7. Confirm evidence does not contain `weight_*.bin`.
8. Select `zk_weights/<model-name>` for each model.
9. Generate proof.
10. Verify proof.
11. Review evidence/proof MB per token in the metrics table.

The Web UI starts `gac_api_server.py` with `ZK_EVIDENCE_USE_MODEL_NAME=1`, so evidence directories are grouped by model name.

## Run the API Directly

```bash
conda activate <your-conda-env>
ZK_EVIDENCE_USE_MODEL_NAME=1 python gac_api_server.py \
  --config-path config.yaml \
  --host 127.0.0.1 \
  --port 8001
```

Generate one token:

```bash
curl -X POST http://127.0.0.1:8001/api/generate/ \
  -H 'Content-Type: application/json' \
  -d '{
    "messages_list": [[{"role": "user", "content": "Hello"}]],
    "max_new_tokens": 1,
    "min_new_tokens": 1,
    "apply_chat_template": false,
    "request_id": "demo_001"
  }'
```

## Proof and Verification Notes

`evidence` and `proof` are different:

- `zk_evidence/<request_id>/...` is runtime data captured from model execution, such as layer input, RMSNorm inverse, attention and FFN intermediate tensors.
- `transformer_proof_<model>.bin` is the cryptographic proof generated by CUDA `transformer_prove` from evidence plus exported `zk_weights`.

The evidence directory must not store model weight copies. If `weight_*.bin` appears under `zk_evidence`, the hook logic is wrong and proof overhead will become abnormally large.

## Common Problems

- `transformer_prove not found`: run `cd zkhook && make all-targets`.
- `tables_7B.bin missing`: run `./table_gen 4096 tables_7B.bin` in `zkhook/`.
- `zk_weights missing`: run `export_zkhook_weights.py` for that model.
- CUDA OOM: reduce selected models, lower `max_memory`, or stop Ray and stale Python GPU processes.
- `tokenizer.chat_template missing`: set `apply_chat_template=false` or add a valid chat template to the tokenizer.
- Ray keeps GPU memory: stop the service from Web UI or run `ray stop --force`.

## Uploading to GitHub

Before committing, check that no large files are included:

```bash
du -sh .
find . -type f -size +50M
git status --ignored
```

The repository should stay small. Model weights and generated proof assets should be distributed separately or generated locally.

## License

See `LICENSE`.
