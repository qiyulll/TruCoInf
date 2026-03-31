#include "ffn_proof.cuh"
#include "proof_stream.cuh"
#include <fstream>
#include <stdexcept>

void serialize_swiglu_proof(const SwiGLUProofData& data, std::vector<uint8_t>& out) {
    ps_write_u32(out, static_cast<uint32_t>(data.gate_out_ints.size()));
    for (auto v : data.gate_out_ints) {
        ps_write_i32(out, v);
    }
    
    ps_write_u32(out, static_cast<uint32_t>(data.swiglu_out_ints.size()));
    for (auto v : data.swiglu_out_ints) {
        ps_write_i32(out, v);
    }
    
    ps_write_u32(out, static_cast<uint32_t>(data.m_ints.size()));
    for (auto v : data.m_ints) {
        ps_write_i32(out, v);
    }
}

SwiGLUProofData deserialize_swiglu_proof(const std::vector<uint8_t>& data, size_t& offset) {
    SwiGLUProofData result;
    
    uint32_t gate_size = ps_read_u32(data, offset);
    result.gate_out_ints.resize(gate_size);
    for (uint32_t i = 0; i < gate_size; ++i) {
        result.gate_out_ints[i] = ps_read_i32(data, offset);
    }
    
    uint32_t swiglu_size = ps_read_u32(data, offset);
    result.swiglu_out_ints.resize(swiglu_size);
    for (uint32_t i = 0; i < swiglu_size; ++i) {
        result.swiglu_out_ints[i] = ps_read_i32(data, offset);
    }
    
    uint32_t m_size = ps_read_u32(data, offset);
    result.m_ints.resize(m_size);
    for (uint32_t i = 0; i < m_size; ++i) {
        result.m_ints[i] = ps_read_i32(data, offset);
    }
    
    return result;
}

void write_ffn_proof_bin(const std::string& path, const FFNProof& p) {
    std::vector<uint8_t> buf;
    
    const char* magic = "ZKFFN001";
    buf.insert(buf.end(), magic, magic + 8);
    
    ps_write_string(buf, p.input_file);
    ps_write_string(buf, p.workdir);
    ps_write_string(buf, p.layer_prefix);
    ps_write_u32(buf, p.seq_len);
    ps_write_u32(buf, p.embed_dim);
    ps_write_u32(buf, p.hidden_dim);
    ps_write_string(buf, p.nonce);
    ps_write_bytes(buf, p.seed.data(), 32);
    
    ps_write_vec_u8(buf, p.input_commitment);
    ps_write_vec_u32(buf, p.input_claim);
    ps_write_vec_u8(buf, p.opening_proof_input);
    
    ps_write_vec_i32(buf, p.output_ints);
    ps_write_vec_u8(buf, p.output_commitment);
    ps_write_vec_u32(buf, p.output_claim);
    ps_write_vec_u8(buf, p.opening_proof_output);
    
    ps_write_vec_i32(buf, p.up_out_ints);
    ps_write_vec_i32(buf, p.gate_out_ints);
    ps_write_vec_i32(buf, p.hidden_ints);
    ps_write_vec_i32(buf, p.down_out_ints);
    
    serialize_swiglu_proof(p.swiglu_data, buf);
    
    ps_write_vec_u32(buf, p.weight_claim_up);
    ps_write_vec_u32(buf, p.weight_claim_gate);
    ps_write_vec_u32(buf, p.weight_claim_down);
    
    ps_write_vec_u8(buf, p.opening_proof_up);
    ps_write_vec_u8(buf, p.opening_proof_gate);
    ps_write_vec_u8(buf, p.opening_proof_down);
    
    ps_write_vec_u8(buf, p.proof_messages);
    
    ps_write_u64(buf, p.up_proj_proof_offset);
    ps_write_u64(buf, p.gate_proj_proof_offset);
    ps_write_u64(buf, p.swiglu_proof_offset);
    ps_write_u64(buf, p.hadamard_proof_offset);
    ps_write_u64(buf, p.down_proj_proof_offset);
    
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot write FFN proof file: " + path);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

FFNProof read_ffn_proof_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot read FFN proof file: " + path);
    
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    
    FFNProof p;
    size_t off = 0;
    
    char magic[9] = {0};
    std::memcpy(magic, buf.data() + off, 8);
    off += 8;
    if (std::string(magic) != "ZKFFN001") {
        throw std::runtime_error("Invalid FFN proof file magic: " + std::string(magic));
    }
    
    p.input_file = ps_read_string(buf, off);
    p.workdir = ps_read_string(buf, off);
    p.layer_prefix = ps_read_string(buf, off);
    p.seq_len = ps_read_u32(buf, off);
    p.embed_dim = ps_read_u32(buf, off);
    p.hidden_dim = ps_read_u32(buf, off);
    p.nonce = ps_read_string(buf, off);
    std::memcpy(p.seed.data(), buf.data() + off, 32);
    off += 32;
    
    p.input_commitment = ps_read_vec_u8(buf, off);
    p.input_claim = ps_read_vec_u32(buf, off);
    p.opening_proof_input = ps_read_vec_u8(buf, off);
    
    p.output_ints = ps_read_vec_i32(buf, off);
    p.output_commitment = ps_read_vec_u8(buf, off);
    p.output_claim = ps_read_vec_u32(buf, off);
    p.opening_proof_output = ps_read_vec_u8(buf, off);
    
    p.up_out_ints = ps_read_vec_i32(buf, off);
    p.gate_out_ints = ps_read_vec_i32(buf, off);
    p.hidden_ints = ps_read_vec_i32(buf, off);
    p.down_out_ints = ps_read_vec_i32(buf, off);
    
    p.swiglu_data = deserialize_swiglu_proof(buf, off);
    
    p.weight_claim_up = ps_read_vec_u32(buf, off);
    p.weight_claim_gate = ps_read_vec_u32(buf, off);
    p.weight_claim_down = ps_read_vec_u32(buf, off);
    
    p.opening_proof_up = ps_read_vec_u8(buf, off);
    p.opening_proof_gate = ps_read_vec_u8(buf, off);
    p.opening_proof_down = ps_read_vec_u8(buf, off);
    
    p.proof_messages = ps_read_vec_u8(buf, off);
    
    p.up_proj_proof_offset = ps_read_u64(buf, off);
    p.gate_proj_proof_offset = ps_read_u64(buf, off);
    p.swiglu_proof_offset = ps_read_u64(buf, off);
    p.hadamard_proof_offset = ps_read_u64(buf, off);
    p.down_proj_proof_offset = ps_read_u64(buf, off);
    
    return p;
}

void write_transformer_proof_bin(const std::string& path, const TransformerLayerProof& p) {
    std::vector<uint8_t> buf;
    
    const char* magic = "ZKTFM001";
    buf.insert(buf.end(), magic, magic + 8);
    
    ps_write_u32(buf, p.layer_idx);
    ps_write_u32(buf, p.seq_len);
    ps_write_u32(buf, p.embed_dim);
    ps_write_u32(buf, p.hidden_dim);
    ps_write_u32(buf, p.num_heads);
    ps_write_u32(buf, p.head_dim);
    ps_write_string(buf, p.nonce);
    ps_write_bytes(buf, p.seed.data(), 32);
    
    ps_write_vec_u8(buf, p.input_commitment);
    ps_write_vec_u8(buf, p.output_commitment);
    ps_write_vec_u32(buf, p.input_claim);
    ps_write_vec_u32(buf, p.output_claim);
    ps_write_vec_u8(buf, p.opening_proof_input);
    ps_write_vec_u8(buf, p.opening_proof_output);

    ps_write_vec_u8(buf, p.weight_commitment_q);
    ps_write_vec_u8(buf, p.weight_commitment_k);
    ps_write_vec_u8(buf, p.weight_commitment_v);
    ps_write_vec_u8(buf, p.weight_commitment_o);
    ps_write_vec_u8(buf, p.weight_commitment_up);
    ps_write_vec_u8(buf, p.weight_commitment_gate);
    ps_write_vec_u8(buf, p.weight_commitment_down);
    
    ps_write_vec_i32(buf, p.original_input_ints);
    
    ps_write_vec_i32(buf, p.attn_input_ints);
    ps_write_vec_i32(buf, p.attn_output_ints);
    ps_write_vec_u8(buf, p.attn_proof_messages);
    
    ps_write_vec_i32(buf, p.ffn_input_ints);
    ps_write_vec_i32(buf, p.ffn_output_ints);
    ps_write_vec_u8(buf, p.ffn_proof_messages);
    
    ps_write_vec_i32(buf, p.residual_1_ints);
    
    ps_write_vec_u32(buf, p.weight_claim_q);
    ps_write_vec_u32(buf, p.weight_claim_k);
    ps_write_vec_u32(buf, p.weight_claim_v);
    ps_write_vec_u32(buf, p.weight_claim_o);
    
    ps_write_vec_u32(buf, p.weight_claim_up);
    ps_write_vec_u32(buf, p.weight_claim_gate);
    ps_write_vec_u32(buf, p.weight_claim_down);
    
    ps_write_vec_u8(buf, p.proof_messages);
    
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot write Transformer proof file: " + path);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

TransformerLayerProof read_transformer_proof_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot read Transformer proof file: " + path);
    
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    
    TransformerLayerProof p;
    size_t off = 0;
    
    char magic[9] = {0};
    std::memcpy(magic, buf.data() + off, 8);
    off += 8;
    if (std::string(magic) != "ZKTFM001") {
        throw std::runtime_error("Invalid Transformer proof file magic: " + std::string(magic));
    }
    
    p.layer_idx = ps_read_u32(buf, off);
    p.seq_len = ps_read_u32(buf, off);
    p.embed_dim = ps_read_u32(buf, off);
    p.hidden_dim = ps_read_u32(buf, off);
    p.num_heads = ps_read_u32(buf, off);
    p.head_dim = ps_read_u32(buf, off);
    p.nonce = ps_read_string(buf, off);
    std::memcpy(p.seed.data(), buf.data() + off, 32);
    off += 32;
    
    p.input_commitment = ps_read_vec_u8(buf, off);
    p.output_commitment = ps_read_vec_u8(buf, off);
    p.input_claim = ps_read_vec_u32(buf, off);
    p.output_claim = ps_read_vec_u32(buf, off);
    p.opening_proof_input = ps_read_vec_u8(buf, off);
    p.opening_proof_output = ps_read_vec_u8(buf, off);

    p.weight_commitment_q = ps_read_vec_u8(buf, off);
    p.weight_commitment_k = ps_read_vec_u8(buf, off);
    p.weight_commitment_v = ps_read_vec_u8(buf, off);
    p.weight_commitment_o = ps_read_vec_u8(buf, off);
    p.weight_commitment_up = ps_read_vec_u8(buf, off);
    p.weight_commitment_gate = ps_read_vec_u8(buf, off);
    p.weight_commitment_down = ps_read_vec_u8(buf, off);
    
    p.original_input_ints = ps_read_vec_i32(buf, off);
    
    p.attn_input_ints = ps_read_vec_i32(buf, off);
    p.attn_output_ints = ps_read_vec_i32(buf, off);
    p.attn_proof_messages = ps_read_vec_u8(buf, off);
    
    p.ffn_input_ints = ps_read_vec_i32(buf, off);
    p.ffn_output_ints = ps_read_vec_i32(buf, off);
    p.ffn_proof_messages = ps_read_vec_u8(buf, off);
    
    p.residual_1_ints = ps_read_vec_i32(buf, off);
    
    p.weight_claim_q = ps_read_vec_u32(buf, off);
    p.weight_claim_k = ps_read_vec_u32(buf, off);
    p.weight_claim_v = ps_read_vec_u32(buf, off);
    p.weight_claim_o = ps_read_vec_u32(buf, off);
    
    p.weight_claim_up = ps_read_vec_u32(buf, off);
    p.weight_claim_gate = ps_read_vec_u32(buf, off);
    p.weight_claim_down = ps_read_vec_u32(buf, off);
    
    p.proof_messages = ps_read_vec_u8(buf, off);
    
    return p;
}
