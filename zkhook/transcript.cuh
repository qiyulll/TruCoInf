#pragma once

#include "sha256.cuh"
#include "fr-tensor.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Simple Fiat–Shamir transcript as a hash chain:
//   state_0 = SHA256(domain || 0x00 || statement_bytes)
//   state_{i+1} = SHA256(state_i || msg_i)
//   challenge = H2F(SHA256(state_i || "chal" || label || counter))
//
// This avoids needing an "incremental SHA256 with cloning" API.
class Transcript {
public:
    Transcript() : state_(Sha256::hash("zkhook-transcript-empty")), counter_(0) {}

    Transcript(const std::string& domain_sep, const std::vector<uint8_t>& statement_bytes) : counter_(0) {
        Sha256 h;
        h.update(domain_sep);
        h.update("\0", 1);
        h.update(statement_bytes);
        state_ = h.finalize();
    }

    void absorb(const uint8_t* data, size_t len) {
        Sha256 h;
        h.update(state_.data(), state_.size());
        h.update(data, len);
        state_ = h.finalize();
    }

    void absorb(const std::vector<uint8_t>& data) { absorb(data.data(), data.size()); }

    // Derive one field element challenge.
    Fr_t challenge_fr(const std::string& label) { return challenge_fr(label, counter_++); }

    std::vector<Fr_t> challenge_vec(const std::string& label, uint len) {
        std::vector<Fr_t> out;
        out.reserve(len);
        for (uint i = 0; i < len; i++) out.push_back(challenge_fr(label, counter_++));
        return out;
    }

    std::array<uint8_t, 32> state_digest() const { return state_; }

private:
    static Fr_t hash_to_fr(const std::array<uint8_t, 32>& digest) {
        auto le_u32 = [&](size_t off) -> uint32_t {
            return static_cast<uint32_t>(static_cast<uint8_t>(digest[off + 0])) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(digest[off + 1])) << 8) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(digest[off + 2])) << 16) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(digest[off + 3])) << 24);
        };
        Fr_t out{
            le_u32(0), le_u32(4), le_u32(8), le_u32(12),
            le_u32(16), le_u32(20), le_u32(24), le_u32(28)
        };
        out.val[7] %= 1944954707u;
        return out;
    }

    Fr_t challenge_fr(const std::string& label, uint64_t ctr) const {
        Sha256 h;
        h.update(state_.data(), state_.size());
        h.update("chal");
        h.update("\0", 1);
        h.update(label);
        h.update("\0", 1);
        uint8_t ctr_bytes[8];
        for (int i = 0; i < 8; i++) ctr_bytes[i] = static_cast<uint8_t>((ctr >> (i * 8)) & 0xff);
        h.update(ctr_bytes, 8);
        return hash_to_fr(h.finalize());
    }

    std::array<uint8_t, 32> state_;
    uint64_t counter_;
};
