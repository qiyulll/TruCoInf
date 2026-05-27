#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct SoftmaxSegmentData {
    std::vector<std::vector<int32_t>> X_segments_ints;
    std::vector<std::vector<int32_t>> Y_segments_ints;
    std::vector<std::vector<int32_t>> m_segments_ints;
};

struct MatmulProofData {
    std::vector<uint32_t> claim;
    std::vector<uint32_t> final_claim;
    std::vector<int32_t> A_partial_ints;
    std::vector<int32_t> B_partial_ints;
};

struct AttnProof {
    std::string input_int_bin;
    std::string workdir;
    std::string layer_prefix;

    uint32_t seq_len = 0;
    uint32_t padded_seq_len = 0;
    uint32_t embed_dim = 0;
    uint32_t head_dim = 0;

    std::string nonce;
    std::array<uint8_t, 32> seed{};

    std::vector<int32_t> output_ints;
    std::vector<uint8_t> output_commitment;
    std::vector<uint32_t> output_claim;
    std::vector<uint8_t> opening_proof_output;

    std::vector<int32_t> Q_ints;
    std::vector<int32_t> K_ints;
    std::vector<int32_t> V_ints;
    
    std::vector<int32_t> X_ints;
    std::vector<int32_t> Y_softmax_ints;
    std::vector<int32_t> shift_ints;
    std::vector<int32_t> X_shifted_ints;
    
    SoftmaxSegmentData softmax_segments;
    
    MatmulProofData matmul_qk;
    MatmulProofData matmul_attnv;
    
    std::vector<uint8_t> input_commitment;
    std::vector<uint32_t> input_claim;
    std::vector<uint8_t> opening_proof_input;
    
    std::vector<uint32_t> weight_claim_q;
    std::vector<uint32_t> weight_claim_k;
    std::vector<uint32_t> weight_claim_v;
    std::vector<uint32_t> weight_claim_o;

    std::vector<uint8_t> opening_proof_q;
    std::vector<uint8_t> opening_proof_k;
    std::vector<uint8_t> opening_proof_v;
    std::vector<uint8_t> opening_proof_o;
    
    std::vector<int32_t> attn_out_ints;
    std::vector<int32_t> o_proj_out_ints;

    std::vector<uint8_t> proof_messages;
    
    size_t softmax_proof_offset = 0;
    size_t matmul_qk_proof_offset = 0;
    size_t matmul_attnv_proof_offset = 0;
};

void write_attn_proof_bin(const std::string& path, const AttnProof& p);
AttnProof read_attn_proof_bin(const std::string& path);

void serialize_softmax_segments(const SoftmaxSegmentData& data, std::vector<uint8_t>& out);
SoftmaxSegmentData deserialize_softmax_segments(const std::vector<uint8_t>& data, size_t& offset);

void serialize_matmul_proof(const MatmulProofData& data, std::vector<uint8_t>& out);
MatmulProofData deserialize_matmul_proof(const std::vector<uint8_t>& data, size_t& offset);
