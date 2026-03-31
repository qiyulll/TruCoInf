#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct SwiGLUProofData {
    std::vector<int32_t> gate_out_ints;
    std::vector<int32_t> swiglu_out_ints;
    std::vector<int32_t> m_ints;
};

struct FFNProof {
    std::string input_file;
    std::string workdir;
    std::string layer_prefix;
    
    uint32_t seq_len = 0;
    uint32_t embed_dim = 0;
    uint32_t hidden_dim = 0;
    
    std::string nonce;
    std::array<uint8_t, 32> seed{};
    
    std::vector<uint8_t> input_commitment;
    std::vector<uint32_t> input_claim;
    std::vector<uint8_t> opening_proof_input;
    
    std::vector<int32_t> output_ints;
    std::vector<uint8_t> output_commitment;
    std::vector<uint32_t> output_claim;
    std::vector<uint8_t> opening_proof_output;
    
    std::vector<int32_t> up_out_ints;
    std::vector<int32_t> gate_out_ints;
    std::vector<int32_t> hidden_ints;
    std::vector<int32_t> down_out_ints;
    
    SwiGLUProofData swiglu_data;
    
    std::vector<uint32_t> weight_claim_up;
    std::vector<uint32_t> weight_claim_gate;
    std::vector<uint32_t> weight_claim_down;
    
    std::vector<uint8_t> opening_proof_up;
    std::vector<uint8_t> opening_proof_gate;
    std::vector<uint8_t> opening_proof_down;
    
    std::vector<uint8_t> proof_messages;
    
    size_t up_proj_proof_offset = 0;
    size_t gate_proj_proof_offset = 0;
    size_t swiglu_proof_offset = 0;
    size_t hadamard_proof_offset = 0;
    size_t down_proj_proof_offset = 0;
};

struct TransformerLayerProof {
    uint32_t layer_idx = 0;
    uint32_t seq_len = 0;
    uint32_t embed_dim = 0;
    uint32_t hidden_dim = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    
    std::string nonce;
    std::array<uint8_t, 32> seed{};
    
    std::vector<uint8_t> input_commitment;
    std::vector<uint8_t> output_commitment;
    std::vector<uint32_t> input_claim;
    std::vector<uint32_t> output_claim;
    std::vector<uint8_t> opening_proof_input;
    std::vector<uint8_t> opening_proof_output;

    std::vector<uint8_t> weight_commitment_q;
    std::vector<uint8_t> weight_commitment_k;
    std::vector<uint8_t> weight_commitment_v;
    std::vector<uint8_t> weight_commitment_o;
    std::vector<uint8_t> weight_commitment_up;
    std::vector<uint8_t> weight_commitment_gate;
    std::vector<uint8_t> weight_commitment_down;
    
    std::vector<int32_t> original_input_ints;
    
    std::vector<int32_t> attn_input_ints;
    std::vector<int32_t> attn_output_ints;
    std::vector<uint8_t> attn_proof_messages;
    
    std::vector<int32_t> ffn_input_ints;
    std::vector<int32_t> ffn_output_ints;
    std::vector<uint8_t> ffn_proof_messages;
    
    std::vector<int32_t> residual_1_ints;
    
    std::vector<uint32_t> weight_claim_q;
    std::vector<uint32_t> weight_claim_k;
    std::vector<uint32_t> weight_claim_v;
    std::vector<uint32_t> weight_claim_o;
    
    std::vector<uint32_t> weight_claim_up;
    std::vector<uint32_t> weight_claim_gate;
    std::vector<uint32_t> weight_claim_down;
    
    std::vector<uint8_t> proof_messages;
};

void write_ffn_proof_bin(const std::string& path, const FFNProof& p);
FFNProof read_ffn_proof_bin(const std::string& path);

void write_transformer_proof_bin(const std::string& path, const TransformerLayerProof& p);
TransformerLayerProof read_transformer_proof_bin(const std::string& path);

void serialize_swiglu_proof(const SwiGLUProofData& data, std::vector<uint8_t>& out);
SwiGLUProofData deserialize_swiglu_proof(const std::vector<uint8_t>& data, size_t& offset);
