# Exported zkhook Weights

This directory is for long-lived int32 weights used by `transformer_prove`.

Generate them locally from safetensors:

```bash
python export_zkhook_weights.py \
  --model-dir weights/qwen-new-7B \
  --output-dir zk_weights/qwen-new-7B
```

These files are generated artifacts and can be several GB. Do not commit them to GitHub.
