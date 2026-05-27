/**
 * rmsnorm_prove.cu - RMSNorm Zero-Knowledge Proof Generator (standalone)
 * 
 * Implementation consistent with baseline (zkhook-ccs2024-main)
 * 
 * RMSNorm(X) = X * rms_inv * gamma
 * where:
 *   rms_inv[i] = 1 / sqrt(mean(X[i]^2) + eps)  (precomputed by Python)
 *   gamma is learnable weight (embed_dim)
 */

#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "proof_stream.cuh"
#include "timer.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::cout;
using std::endl;

Timer compute_timer, prove_timer;

void print_usage() {
    std::cerr
        << "RMSNorm Zero-Knowledge Proof Generator\n"
        << "======================================\n"
        << "\n"
        << "Usage:\n"
        << "  rmsnorm_prove <input.bin> <rms_inv.bin> <gamma_weight.bin> \\\n"
        << "    <seq_len> <embed_dim> <output.bin> <proof.bin>\n"
        << "\n"
        << "Arguments:\n"
        << "  input.bin       - Input tensor (int32 fixed-point, seq_len x embed_dim)\n"
        << "  rms_inv.bin     - 1/sqrt(mean(x^2)+eps) (int32 fixed-point, seq_len)\n"
        << "  gamma_weight.bin - RMSNorm gamma weight (int32 fixed-point, embed_dim)\n"
        << "  seq_len         - Sequence length\n"
        << "  embed_dim       - Embedding dimension\n"
        << "  output.bin      - Output tensor (int32 fixed-point)\n"
        << "  proof.bin       - Proof file\n";
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        print_usage();
        return 1;
    }

    // Parse arguments
    const string input_file = argv[1];
    const string rms_inv_file = argv[2];
    const string gamma_file = argv[3];
    const uint seq_len = std::stoul(argv[4]);
    const uint embed_dim = std::stoul(argv[5]);
    const string output_file = argv[6];
    const string proof_file = argv[7];

    cout << "========================================\n";
    cout << "RMSNorm Proof Generator\n";
    cout << "========================================\n";
    cout << "seq_len=" << seq_len << " embed_dim=" << embed_dim << "\n";
    cout << "========================================\n";

    // Check files
    auto check_file = [](const string& path, const string& name) {
        std::ifstream f(path);
        if (!f.good()) {
            std::cerr << "Error: " << name << " does not exist or cannot be read: " << path << "\n";
            return false;
        }
        return true;
    };
    
    if (!check_file(input_file, "input file") ||
        !check_file(rms_inv_file, "rms_inv file") ||
        !check_file(gamma_file, "gamma file")) {
        return 1;
    }

    // Load data
    cout << "\n[1/3] Loading data...\n";
    cout << "   Input file: " << input_file << "\n";
    cout << "   rms_inv: " << rms_inv_file << "\n";
    cout << "   gamma: " << gamma_file << "\n";
    
    FrTensor X = FrTensor::from_int_bin(input_file);
    FrTensor rms_inv = FrTensor::from_int_bin(rms_inv_file);
    FrTensor gamma = FrTensor::from_int_bin(gamma_file);
    
    cout << "   X size: " << X.size << " (expected " << seq_len * embed_dim << ")\n";
    cout << "   rms_inv size: " << rms_inv.size << " (expected " << seq_len << ")\n";
    cout << "   gamma size: " << gamma.size << " (expected " << embed_dim << ")\n";

    // Compute phase
    cout << "\n[2/3] Computing...\n";
    compute_timer.start();
    
    // g_inv_rms = gamma * rms_inv (using zkFC)
    // rms_inv: (seq_len,), gamma: (embed_dim,) 
    // Output: g_inv_rms (seq_len x embed_dim)
    Rescaling rs1(1 << 16), rs2(1 << 16);
    
    zkFC g(1, embed_dim, gamma);  // Input 1 dim, output embed_dim dim
    auto g_inv_rms = g(rms_inv);
    auto g_inv_rms_ = rs1(g_inv_rms);
    
    // Y = g_inv_rms_ * X (element-wise Hadamard product)
    auto Y = g_inv_rms_ * X;
    auto Y_ = rs2(Y);
    
    compute_timer.stop();
    cout << "   Computation complete, time: " << compute_timer.getTotalTime() << " s\n";

    // Save output
    Y_.save_int(output_file);
    cout << "   Output saved to: " << output_file << "\n";

    // Prove phase
    cout << "\n[3/3] Generating proof...\n";
    prove_timer.start();
    
    // Proof data
    vector<uint8_t> proof_data;
    
    // Write metadata
    ps_write_u32(proof_data, seq_len);
    ps_write_u32(proof_data, embed_dim);
    
    // 1. rescale proof for Y
    rs2.prove(Y, Y_);
    
    // 2. Hadamard product sumcheck: g_inv_rms_ * X = Y
    auto u = random_vec(ceilLog2(Y.size));
    auto v = random_vec(ceilLog2(Y.size));
    auto hp_proof = hadamard_product_sumcheck(g_inv_rms_, X, u, v);
    
    // Write Hadamard product proof
    ps_write_u32(proof_data, static_cast<uint32_t>(hp_proof.size()));
    for (const auto& fr : hp_proof) {
        ps_write_fr(proof_data, fr);
    }
    
    // 3. rescale proof for g_inv_rms
    rs1.prove(g_inv_rms, g_inv_rms_);
    
    // 4. zkFC proof for gamma * rms_inv
    auto weight_claim = g.prove(rms_inv, g_inv_rms)[0];
    
    // Write weight claim
    ps_write_fr(proof_data, weight_claim.claim);
    ps_write_u32(proof_data, static_cast<uint32_t>(weight_claim.u.size()));
    for (const auto& u_vec : weight_claim.u) {
        ps_write_u32(proof_data, static_cast<uint32_t>(u_vec.size()));
        for (const auto& fr : u_vec) {
            ps_write_fr(proof_data, fr);
        }
    }
    ps_write_u32(proof_data, static_cast<uint32_t>(weight_claim.dims.size()));
    for (const auto& d : weight_claim.dims) {
        ps_write_u32(proof_data, d);
    }
    
    prove_timer.stop();
    
    // Save proof
    std::ofstream ofs(proof_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Cannot write proof file\n";
        return 1;
    }
    ofs.write(reinterpret_cast<const char*>(proof_data.data()), proof_data.size());
    ofs.close();

    // Output summary
    cout << "\n========================================\n";
    cout << "[TIME] RMSNorm - Compute: " << compute_timer.getTotalTime() << " s\n";
    cout << "[TIME] RMSNorm - Prove: " << prove_timer.getTotalTime() << " s\n";
    cout << "Proof size: " << proof_data.size() << " bytes\n";
    cout << "========================================\n";

    return 0;
}
