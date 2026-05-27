#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Minimal SHA-256 (byte-oriented). This is a small self-contained implementation
// intended for Fiat–Shamir transcripts in this repo.
//
// References: FIPS 180-4.
class Sha256 {
public:
    Sha256();
    void update(const uint8_t* data, size_t len);
    void update(const char* data, size_t len) { update(reinterpret_cast<const uint8_t*>(data), len); }
    void update(const std::vector<uint8_t>& data);
    void update(const std::string& s);
    std::array<uint8_t, 32> finalize();

    static std::array<uint8_t, 32> hash(const uint8_t* data, size_t len);
    static std::array<uint8_t, 32> hash(const std::vector<uint8_t>& data);
    static std::array<uint8_t, 32> hash(const std::string& s);

private:
    void transform(const uint8_t block[64]);
    void pad_and_finalize();

    uint64_t bit_len_;
    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> buffer_;
    size_t buffer_len_;
    bool finalized_;
};

// Hash a file's raw bytes with SHA-256.
std::array<uint8_t, 32> sha256_file(const std::string& path);
