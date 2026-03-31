#pragma once

#include "fr-tensor.cuh"
#include "sha256.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Fiat–Shamir deterministic "RNG" for producing Fr_t challenges.
//
// NOTE: This is a *seeded* FS generator: challenges are derived from a fixed
// seed (typically H(statement || nonce || params ...)) plus a domain label and
// counter. For a stricter FS transcript, you would also absorb prover messages.
class FsRng {
public:
    explicit FsRng(const std::array<uint8_t, 32>& seed) : seed_(seed), counter_(0) {}

    Fr_t challenge_fr(const std::string& label) { return challenge_fr(label, counter_++); }

    std::vector<Fr_t> challenge_vec(const std::string& label, uint len) {
        std::vector<Fr_t> out;
        out.reserve(len);
        for (uint i = 0; i < len; i++) out.push_back(challenge_fr(label, counter_++));
        return out;
    }

    static std::array<uint8_t, 32> derive_seed(
        const std::string& domain_sep,
        const std::string& nonce,
        const std::vector<std::array<uint8_t, 32>>& digests)
    {
        Sha256 h;
        h.update(domain_sep);
        h.update("\0", 1);
        h.update(nonce);
        h.update("\0", 1);
        for (const auto& d : digests) {
            h.update(d.data(), d.size());
        }
        return h.finalize();
    }

private:
    // Map 32 bytes to an Fr_t in the same "shape" as the repo's random_vec.
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
        // Keep the top limb in a range similar to random_vec() to avoid invalid Scalars.
        out.val[7] %= 1944954707u;
        return out;
    }

    Fr_t challenge_fr(const std::string& label, uint64_t ctr) const {
        Sha256 h;
        h.update(seed_.data(), seed_.size());
        h.update(label);
        h.update("\0", 1);
        uint8_t ctr_bytes[8];
        for (int i = 0; i < 8; i++) ctr_bytes[i] = static_cast<uint8_t>((ctr >> (i * 8)) & 0xff);
        h.update(ctr_bytes, 8);
        return hash_to_fr(h.finalize());
    }

    std::array<uint8_t, 32> seed_;
    uint64_t counter_;
};

