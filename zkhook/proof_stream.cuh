#pragma once

#include "fr-tensor.cuh"
#include "polynomial.cuh"

#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// A very small binary stream helper for serializing proof messages into a byte vector.
// We intentionally keep this simple (no compression, little-endian).

inline void ps_write_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

inline uint32_t ps_read_u32(const std::vector<uint8_t>& in, size_t& off) {
    if (off + 4 > in.size()) throw std::runtime_error("proof_stream: EOF (u32)");
    uint32_t v = static_cast<uint32_t>(in[off + 0]) |
                 (static_cast<uint32_t>(in[off + 1]) << 8) |
                 (static_cast<uint32_t>(in[off + 2]) << 16) |
                 (static_cast<uint32_t>(in[off + 3]) << 24);
    off += 4;
    return v;
}

inline void ps_write_i32(std::vector<uint8_t>& out, int32_t v) {
    ps_write_u32(out, static_cast<uint32_t>(v));
}

inline int32_t ps_read_i32(const std::vector<uint8_t>& in, size_t& off) {
    return static_cast<int32_t>(ps_read_u32(in, off));
}

inline void ps_write_u64(std::vector<uint8_t>& out, uint64_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 32) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 40) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 48) & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 56) & 0xff));
}

inline uint64_t ps_read_u64(const std::vector<uint8_t>& in, size_t& off) {
    if (off + 8 > in.size()) throw std::runtime_error("proof_stream: EOF (u64)");
    uint64_t v =
        (static_cast<uint64_t>(in[off + 0]) << 0) |
        (static_cast<uint64_t>(in[off + 1]) << 8) |
        (static_cast<uint64_t>(in[off + 2]) << 16) |
        (static_cast<uint64_t>(in[off + 3]) << 24) |
        (static_cast<uint64_t>(in[off + 4]) << 32) |
        (static_cast<uint64_t>(in[off + 5]) << 40) |
        (static_cast<uint64_t>(in[off + 6]) << 48) |
        (static_cast<uint64_t>(in[off + 7]) << 56);
    off += 8;
    return v;
}

inline void ps_write_bytes(std::vector<uint8_t>& out, const uint8_t* data, size_t len) {
    out.insert(out.end(), data, data + len);
}

inline void ps_read_bytes(const std::vector<uint8_t>& in, size_t& off, uint8_t* dst, size_t len) {
    if (off + len > in.size()) throw std::runtime_error("proof_stream: EOF (bytes)");
    std::memcpy(dst, in.data() + off, len);
    off += len;
}

inline void ps_write_string(std::vector<uint8_t>& out, const std::string& s) {
    ps_write_u32(out, static_cast<uint32_t>(s.size()));
    if (!s.empty()) ps_write_bytes(out, reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

inline std::string ps_read_string(const std::vector<uint8_t>& in, size_t& off) {
    uint32_t n = ps_read_u32(in, off);
    if (off + n > in.size()) throw std::runtime_error("proof_stream: EOF (string)");
    std::string s;
    s.resize(n);
    if (n) std::memcpy(s.data(), in.data() + off, n);
    off += n;
    return s;
}

inline void ps_write_vec_u8(std::vector<uint8_t>& out, const std::vector<uint8_t>& v) {
    ps_write_u32(out, static_cast<uint32_t>(v.size()));
    if (!v.empty()) ps_write_bytes(out, v.data(), v.size());
}

inline std::vector<uint8_t> ps_read_vec_u8(const std::vector<uint8_t>& in, size_t& off) {
    uint32_t n = ps_read_u32(in, off);
    if (off + n > in.size()) throw std::runtime_error("proof_stream: EOF (vec_u8)");
    std::vector<uint8_t> v(n);
    if (n) std::memcpy(v.data(), in.data() + off, n);
    off += n;
    return v;
}

inline void ps_write_vec_u32(std::vector<uint8_t>& out, const std::vector<uint32_t>& v) {
    ps_write_u32(out, static_cast<uint32_t>(v.size()));
    for (auto x : v) ps_write_u32(out, x);
}

inline std::vector<uint32_t> ps_read_vec_u32(const std::vector<uint8_t>& in, size_t& off) {
    uint32_t n = ps_read_u32(in, off);
    std::vector<uint32_t> v;
    v.reserve(n);
    for (uint32_t i = 0; i < n; i++) v.push_back(ps_read_u32(in, off));
    return v;
}

inline void ps_write_vec_i32(std::vector<uint8_t>& out, const std::vector<int32_t>& v) {
    ps_write_u32(out, static_cast<uint32_t>(v.size()));
    for (auto x : v) ps_write_i32(out, x);
}

inline std::vector<int32_t> ps_read_vec_i32(const std::vector<uint8_t>& in, size_t& off) {
    uint32_t n = ps_read_u32(in, off);
    std::vector<int32_t> v;
    v.reserve(n);
    for (uint32_t i = 0; i < n; i++) v.push_back(ps_read_i32(in, off));
    return v;
}

inline void ps_write_fr(std::vector<uint8_t>& out, const Fr_t& x) {
    for (int i = 0; i < 8; i++) ps_write_u32(out, x.val[i]);
}

inline Fr_t ps_read_fr(const std::vector<uint8_t>& in, size_t& off) {
    Fr_t x{};
    for (int i = 0; i < 8; i++) x.val[i] = ps_read_u32(in, off);
    return x;
}

std::vector<uint8_t> ps_serialize_poly(const Polynomial& p);
Polynomial ps_deserialize_poly(const std::vector<uint8_t>& in, size_t& off);

inline void ps_write_fp(std::vector<uint8_t>& out, const blstrs__fp__Fp& fp) {
    for (int i = 0; i < 12; i++) ps_write_u32(out, fp.val[i]);
}

inline blstrs__fp__Fp ps_read_fp(const std::vector<uint8_t>& in, size_t& off) {
    blstrs__fp__Fp fp{};
    for (int i = 0; i < 12; i++) fp.val[i] = ps_read_u32(in, off);
    return fp;
}

inline void ps_write_g1(std::vector<uint8_t>& out, const blstrs__g1__G1Affine_jacobian& g) {
    ps_write_fp(out, g.x);
    ps_write_fp(out, g.y);
    ps_write_fp(out, g.z);
}

inline blstrs__g1__G1Affine_jacobian ps_read_g1(const std::vector<uint8_t>& in, size_t& off) {
    blstrs__g1__G1Affine_jacobian g{};
    g.x = ps_read_fp(in, off);
    g.y = ps_read_fp(in, off);
    g.z = ps_read_fp(in, off);
    return g;
}

inline void ps_write_g1_vec(std::vector<uint8_t>& out, const std::vector<blstrs__g1__G1Affine_jacobian>& gs) {
    ps_write_u32(out, static_cast<uint32_t>(gs.size()));
    for (const auto& g : gs) ps_write_g1(out, g);
}

inline std::vector<blstrs__g1__G1Affine_jacobian> ps_read_g1_vec(const std::vector<uint8_t>& in, size_t& off) {
    uint32_t n = ps_read_u32(in, off);
    std::vector<blstrs__g1__G1Affine_jacobian> gs(n);
    for (uint32_t i = 0; i < n; i++) gs[i] = ps_read_g1(in, off);
    return gs;
}
