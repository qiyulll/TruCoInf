#include "proof_stream.cuh"

std::vector<uint8_t> ps_serialize_poly(const Polynomial& p) {
    // We need coefficients on host.
    auto coeffs = p.to_host_coeffs();
    std::vector<uint8_t> out;
    out.reserve(4 + coeffs.size() * 32);
    ps_write_u32(out, static_cast<uint32_t>(coeffs.size()));
    for (const auto& c : coeffs) ps_write_fr(out, c);
    return out;
}

Polynomial ps_deserialize_poly(const std::vector<uint8_t>& in, size_t& off) {
    uint32_t n = ps_read_u32(in, off);
    if (n == 0) throw std::runtime_error("proof_stream: polynomial with 0 coefficients");
    std::vector<Fr_t> coeffs;
    coeffs.reserve(n);
    for (uint32_t i = 0; i < n; i++) coeffs.push_back(ps_read_fr(in, off));
    return Polynomial(coeffs);
}
