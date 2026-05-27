# zkhook - Zero-Knowledge Proofs for LLM Inference

## Build

```bash
# Build all targets
make all

# Build stepwise execution targets
make stepwise

# Clean
make clean
```

## Usage

### 1. Generate Lookup Tables (one-time setup)

```bash
./table_gen <embed_dim> <output_tables.bin>

# Example: LLaMA-2-7B (embed_dim=4096)
./table_gen 4096 tables.bin
```

### 2. Stepwise Execution

**RMSNorm:**
```bash
./rmsnorm_prove <input.bin> <rms_inv.bin> <gamma.bin> <seq_len> <embed_dim> <output.bin> <proof.bin>
./rmsnorm_verify <proof.bin>
```

**Attention:**
```bash
./attn_prove <input.bin> <q.bin> <k.bin> <v.bin> <o.bin> <seq_len> <embed_dim> <proof.bin> <nonce>
./attn_verify <proof.bin>
```

**FFN:**
```bash
./ffn_prove <input.bin> <up.bin> <gate.bin> <down.bin> <seq_len> <embed_dim> <hidden_dim> <proof.bin> <nonce>
./ffn_verify <proof.bin>
```

### 3. Full Transformer Layer

**Prove:**
```bash
./transformer_prove <input.bin> <rms_inv.bin> \
    <input_layernorm.bin> <post_attn_layernorm.bin> \
    <q.bin> <k.bin> <v.bin> <o.bin> \
    <up.bin> <gate.bin> <down.bin> \
    <seq_len> <embed_dim> <hidden_dim> \
    <tables.bin> <proof.bin> <nonce>
```

**Verify:**
```bash
./transformer_verify <proof.bin> --tables <tables.bin>
```

## File Formats

- Input/Output tensors: `int32` binary (fixed-point quantized)
- Weights: `int32` binary for FFN/RMSNorm, `float32` for Attention
- Proofs: Binary format with magic header

## LLaMA-2-7B Parameters

| Parameter | Value |
|-----------|-------|
| embed_dim | 4096 |
| hidden_dim | 11008 |
| num_heads | 32 |
| head_dim | 128 |
| num_layers | 32 |
