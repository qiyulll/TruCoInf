#include "attn_proof.cuh"

#include <fstream>
#include <stdexcept>
#include <cstring>

static void write_u32(std::ofstream& out, uint32_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_u64(std::ofstream& out, uint64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static uint32_t read_u32(std::ifstream& in) {
    uint32_t v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (u32)");
    return v;
}
static uint64_t read_u64(std::ifstream& in) {
    uint64_t v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (u64)");
    return v;
}

static void write_str(std::ofstream& out, const std::string& s) {
    if (s.size() > 0xffffffffu) throw std::runtime_error("write_attn_proof_bin: string too large");
    write_u32(out, static_cast<uint32_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

static std::string read_str(std::ifstream& in) {
    uint32_t n = read_u32(in);
    std::string s(n, '\0');
    in.read(s.data(), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (string)");
    return s;
}

static void buf_write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

static void buf_write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
}

static uint32_t buf_read_u32(const std::vector<uint8_t>& buf, size_t& off) {
    if (off + 4 > buf.size()) throw std::runtime_error("buf_read_u32: buffer overflow");
    uint32_t v = static_cast<uint32_t>(buf[off]) |
                 (static_cast<uint32_t>(buf[off + 1]) << 8) |
                 (static_cast<uint32_t>(buf[off + 2]) << 16) |
                 (static_cast<uint32_t>(buf[off + 3]) << 24);
    off += 4;
    return v;
}

static uint64_t buf_read_u64(const std::vector<uint8_t>& buf, size_t& off) {
    if (off + 8 > buf.size()) throw std::runtime_error("buf_read_u64: buffer overflow");
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(buf[off + i]) << (8 * i);
    }
    off += 8;
    return v;
}

static void buf_write_i32_vec(std::vector<uint8_t>& buf, const std::vector<int32_t>& v) {
    buf_write_u64(buf, static_cast<uint64_t>(v.size()));
    for (auto val : v) {
        buf_write_u32(buf, static_cast<uint32_t>(val));
    }
}

static std::vector<int32_t> buf_read_i32_vec(const std::vector<uint8_t>& buf, size_t& off) {
    uint64_t n = buf_read_u64(buf, off);
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<int32_t>(buf_read_u32(buf, off));
    }
    return v;
}

static void buf_write_u32_vec(std::vector<uint8_t>& buf, const std::vector<uint32_t>& v) {
    buf_write_u64(buf, static_cast<uint64_t>(v.size()));
    for (auto val : v) {
        buf_write_u32(buf, val);
    }
}

static std::vector<uint32_t> buf_read_u32_vec(const std::vector<uint8_t>& buf, size_t& off) {
    uint64_t n = buf_read_u64(buf, off);
    std::vector<uint32_t> v(static_cast<size_t>(n));
    for (size_t i = 0; i < n; ++i) {
        v[i] = buf_read_u32(buf, off);
    }
    return v;
}

static void write_i32_vec(std::ofstream& out, const std::vector<int32_t>& v) {
    write_u64(out, static_cast<uint64_t>(v.size()));
    if (!v.empty()) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(int32_t)));
    }
}

static std::vector<int32_t> read_i32_vec(std::ifstream& in, const std::string& name) {
    uint64_t n = read_u64(in);
    if (n > (1ull << 34)) throw std::runtime_error("read_attn_proof_bin: " + name + " too large");
    std::vector<int32_t> v(static_cast<size_t>(n));
    if (n) {
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(int32_t)));
        if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (" + name + ")");
    }
    return v;
}

static void write_u32_vec(std::ofstream& out, const std::vector<uint32_t>& v) {
    write_u64(out, static_cast<uint64_t>(v.size()));
    if (!v.empty()) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(uint32_t)));
    }
}

static std::vector<uint32_t> read_u32_vec(std::ifstream& in, const std::string& name) {
    uint64_t n = read_u64(in);
    if (n > (1ull << 34)) throw std::runtime_error("read_attn_proof_bin: " + name + " too large");
    std::vector<uint32_t> v(static_cast<size_t>(n));
    if (n) {
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(uint32_t)));
        if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (" + name + ")");
    }
    return v;
}

static void write_u8_vec(std::ofstream& out, const std::vector<uint8_t>& v) {
    write_u64(out, static_cast<uint64_t>(v.size()));
    if (!v.empty()) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size()));
    }
}

static std::vector<uint8_t> read_u8_vec(std::ifstream& in, const std::string& name) {
    uint64_t n = read_u64(in);
    if (n > (1ull << 34)) throw std::runtime_error("read_attn_proof_bin: " + name + " too large");
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (n) {
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (" + name + ")");
    }
    return v;
}

void serialize_softmax_segments(const SoftmaxSegmentData& data, std::vector<uint8_t>& out) {
    buf_write_u32(out, static_cast<uint32_t>(data.X_segments_ints.size()));
    for (const auto& seg : data.X_segments_ints) {
        buf_write_i32_vec(out, seg);
    }
    
    buf_write_u32(out, static_cast<uint32_t>(data.Y_segments_ints.size()));
    for (const auto& seg : data.Y_segments_ints) {
        buf_write_i32_vec(out, seg);
    }
    
    buf_write_u32(out, static_cast<uint32_t>(data.m_segments_ints.size()));
    for (const auto& seg : data.m_segments_ints) {
        buf_write_i32_vec(out, seg);
    }
}

SoftmaxSegmentData deserialize_softmax_segments(const std::vector<uint8_t>& buf, size_t& off) {
    SoftmaxSegmentData data;
    
    uint32_t x_count = buf_read_u32(buf, off);
    data.X_segments_ints.resize(x_count);
    for (uint32_t i = 0; i < x_count; ++i) {
        data.X_segments_ints[i] = buf_read_i32_vec(buf, off);
    }
    
    uint32_t y_count = buf_read_u32(buf, off);
    data.Y_segments_ints.resize(y_count);
    for (uint32_t i = 0; i < y_count; ++i) {
        data.Y_segments_ints[i] = buf_read_i32_vec(buf, off);
    }
    
    uint32_t m_count = buf_read_u32(buf, off);
    data.m_segments_ints.resize(m_count);
    for (uint32_t i = 0; i < m_count; ++i) {
        data.m_segments_ints[i] = buf_read_i32_vec(buf, off);
    }
    
    return data;
}

void serialize_matmul_proof(const MatmulProofData& data, std::vector<uint8_t>& out) {
    buf_write_u32_vec(out, data.claim);
    buf_write_u32_vec(out, data.final_claim);
    buf_write_i32_vec(out, data.A_partial_ints);
    buf_write_i32_vec(out, data.B_partial_ints);
}

MatmulProofData deserialize_matmul_proof(const std::vector<uint8_t>& buf, size_t& off) {
    MatmulProofData data;
    data.claim = buf_read_u32_vec(buf, off);
    data.final_claim = buf_read_u32_vec(buf, off);
    data.A_partial_ints = buf_read_i32_vec(buf, off);
    data.B_partial_ints = buf_read_i32_vec(buf, off);
    return data;
}

static void write_softmax_segments(std::ofstream& out, const SoftmaxSegmentData& data) {
    std::vector<uint8_t> buf;
    serialize_softmax_segments(data, buf);
    write_u8_vec(out, buf);
}

static SoftmaxSegmentData read_softmax_segments(std::ifstream& in) {
    auto buf = read_u8_vec(in, "softmax_segments");
    size_t off = 0;
    return deserialize_softmax_segments(buf, off);
}

static void write_matmul_proof(std::ofstream& out, const MatmulProofData& data) {
    std::vector<uint8_t> buf;
    serialize_matmul_proof(data, buf);
    write_u8_vec(out, buf);
}

static MatmulProofData read_matmul_proof(std::ifstream& in) {
    auto buf = read_u8_vec(in, "matmul_proof");
    size_t off = 0;
    return deserialize_matmul_proof(buf, off);
}

void write_attn_proof_bin(const std::string& path, const AttnProof& p) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("write_attn_proof_bin: cannot open " + path);

    const char magic[8] = {'Z','K','A','T','T','N','\0','\x08'};
    out.write(magic, sizeof(magic));

    write_str(out, p.input_int_bin);
    write_str(out, p.workdir);
    write_str(out, p.layer_prefix);
    write_u32(out, p.seq_len);
    write_u32(out, p.padded_seq_len);
    write_u32(out, p.embed_dim);
    write_u32(out, p.head_dim);
    write_str(out, p.nonce);
    out.write(reinterpret_cast<const char*>(p.seed.data()), static_cast<std::streamsize>(p.seed.size()));

    write_i32_vec(out, p.output_ints);
    write_u8_vec(out, p.output_commitment);
    write_u32_vec(out, p.output_claim);
    write_u8_vec(out, p.opening_proof_output);

    write_i32_vec(out, p.Q_ints);
    write_i32_vec(out, p.K_ints);
    write_i32_vec(out, p.V_ints);
    write_i32_vec(out, p.X_ints);
    write_i32_vec(out, p.Y_softmax_ints);
    write_i32_vec(out, p.shift_ints);
    write_i32_vec(out, p.X_shifted_ints);
    
    write_i32_vec(out, p.attn_out_ints);
    write_i32_vec(out, p.o_proj_out_ints);
    
    write_softmax_segments(out, p.softmax_segments);
    
    write_matmul_proof(out, p.matmul_qk);
    write_matmul_proof(out, p.matmul_attnv);
    
    write_u8_vec(out, p.input_commitment);
    write_u32_vec(out, p.input_claim);
    write_u8_vec(out, p.opening_proof_input);
    
    write_u32_vec(out, p.weight_claim_q);
    write_u32_vec(out, p.weight_claim_k);
    write_u32_vec(out, p.weight_claim_v);
    write_u32_vec(out, p.weight_claim_o);
    
    write_u8_vec(out, p.opening_proof_q);
    write_u8_vec(out, p.opening_proof_k);
    write_u8_vec(out, p.opening_proof_v);
    write_u8_vec(out, p.opening_proof_o);

    write_u8_vec(out, p.proof_messages);
    
    write_u64(out, p.softmax_proof_offset);
    write_u64(out, p.matmul_qk_proof_offset);
    write_u64(out, p.matmul_attnv_proof_offset);
    
    if (!out) throw std::runtime_error("write_attn_proof_bin: write failed");
}

AttnProof read_attn_proof_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("read_attn_proof_bin: cannot open " + path);

    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in) throw std::runtime_error("read_attn_proof_bin: short file");
    
    const char expected_v8[8] = {'Z','K','A','T','T','N','\0','\x08'};
    const char expected_v7[8] = {'Z','K','A','T','T','N','\0','\7'};
    const char expected_v6[8] = {'Z','K','A','T','T','N','\0','\6'};
    const char expected_v5[8] = {'Z','K','A','T','T','N','\0','\5'};
    
    int version = 0;
    if (std::memcmp(magic, expected_v8, 8) == 0) version = 8;
    else if (std::memcmp(magic, expected_v7, 8) == 0) version = 7;
    else if (std::memcmp(magic, expected_v6, 8) == 0) version = 6;
    else if (std::memcmp(magic, expected_v5, 8) == 0) version = 5;
    else throw std::runtime_error("read_attn_proof_bin: bad magic/version (need v5, v6, v7 or v8)");

    AttnProof p;
    
    p.input_int_bin = read_str(in);
    p.workdir = read_str(in);
    p.layer_prefix = read_str(in);
    p.seq_len = read_u32(in);
    if (version >= 7) {
        p.padded_seq_len = read_u32(in);
    } else {
        p.padded_seq_len = p.seq_len;
    }
    p.embed_dim = read_u32(in);
    p.head_dim = (version >= 6) ? read_u32(in) : p.embed_dim;
    p.nonce = read_str(in);
    in.read(reinterpret_cast<char*>(p.seed.data()), static_cast<std::streamsize>(p.seed.size()));
    if (!in) throw std::runtime_error("read_attn_proof_bin: unexpected EOF (seed)");

    p.output_ints = read_i32_vec(in, "output_ints");
    
    if (version >= 7) {
        p.output_commitment = read_u8_vec(in, "output_commitment");
        p.output_claim = read_u32_vec(in, "output_claim");
        p.opening_proof_output = read_u8_vec(in, "opening_proof_output");
    }

    p.Q_ints = read_i32_vec(in, "Q_ints");
    p.K_ints = read_i32_vec(in, "K_ints");
    p.V_ints = read_i32_vec(in, "V_ints");
    p.X_ints = read_i32_vec(in, "X_ints");
    p.Y_softmax_ints = read_i32_vec(in, "Y_softmax_ints");
    p.shift_ints = read_i32_vec(in, "shift_ints");
    p.X_shifted_ints = read_i32_vec(in, "X_shifted_ints");
    
    if (version >= 8) {
        p.attn_out_ints = read_i32_vec(in, "attn_out_ints");
        p.o_proj_out_ints = read_i32_vec(in, "o_proj_out_ints");
    }
    
    if (version >= 6) {
        p.softmax_segments = read_softmax_segments(in);
        p.matmul_qk = read_matmul_proof(in);
        p.matmul_attnv = read_matmul_proof(in);
    }
    
    p.input_commitment = read_u8_vec(in, "input_commitment");
    p.input_claim = read_u32_vec(in, "input_claim");
    p.opening_proof_input = read_u8_vec(in, "opening_proof_input");
    
    p.weight_claim_q = read_u32_vec(in, "weight_claim_q");
    p.weight_claim_k = read_u32_vec(in, "weight_claim_k");
    p.weight_claim_v = read_u32_vec(in, "weight_claim_v");
    if (version >= 8) {
        p.weight_claim_o = read_u32_vec(in, "weight_claim_o");
    }
    
    p.opening_proof_q = read_u8_vec(in, "opening_proof_q");
    p.opening_proof_k = read_u8_vec(in, "opening_proof_k");
    p.opening_proof_v = read_u8_vec(in, "opening_proof_v");
    if (version >= 8) {
        p.opening_proof_o = read_u8_vec(in, "opening_proof_o");
    }

    p.proof_messages = read_u8_vec(in, "proof_messages");
    
    if (version >= 6) {
        p.softmax_proof_offset = read_u64(in);
        p.matmul_qk_proof_offset = read_u64(in);
        p.matmul_attnv_proof_offset = read_u64(in);
    }
    
    return p;
}
