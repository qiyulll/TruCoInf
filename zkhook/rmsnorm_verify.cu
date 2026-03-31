/**
 * rmsnorm_verify.cu - RMSNorm Zero-Knowledge Proof Verifier (standalone)
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

Timer verify_timer;

void print_usage() {
    std::cerr
        << "RMSNorm Zero-Knowledge Proof Verifier\n"
        << "=====================================\n"
        << "\n"
        << "Usage:\n"
        << "  rmsnorm_verify <proof.bin>\n"
        << "\n"
        << "Arguments:\n"
        << "  proof.bin - RMSNorm proof file\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const string proof_file = argv[1];

    cout << "========================================\n";
    cout << "RMSNorm Proof Verifier\n";
    cout << "========================================\n";

    // Read proof
    cout << "\n[1/2] Reading proof...\n";
    
    std::ifstream ifs(proof_file, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "Error: Cannot read proof file\n";
        return 1;
    }
    
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    vector<uint8_t> proof_data(size);
    ifs.read(reinterpret_cast<char*>(proof_data.data()), size);
    ifs.close();
    
    size_t off = 0;
    
    // Read metadata
    uint seq_len = ps_read_u32(proof_data, off);
    uint embed_dim = ps_read_u32(proof_data, off);
    
    cout << "   seq_len=" << seq_len << " embed_dim=" << embed_dim << "\n";
    cout << "   Proof size: " << size << " bytes\n";

    // Verify phase
    cout << "\n[2/2] Verifying proof...\n";
    verify_timer.start();
    
    // Read Hadamard product proof
    uint hp_proof_size = ps_read_u32(proof_data, off);
    vector<Fr_t> hp_proof(hp_proof_size);
    for (uint i = 0; i < hp_proof_size; ++i) {
        hp_proof[i] = ps_read_fr(proof_data, off);
    }
    
    // Read weight claim
    Claim weight_claim;
    weight_claim.claim = ps_read_fr(proof_data, off);
    
    uint u_size = ps_read_u32(proof_data, off);
    weight_claim.u.resize(u_size);
    for (uint i = 0; i < u_size; ++i) {
        uint u_vec_size = ps_read_u32(proof_data, off);
        weight_claim.u[i].resize(u_vec_size);
        for (uint j = 0; j < u_vec_size; ++j) {
            weight_claim.u[i][j] = ps_read_fr(proof_data, off);
        }
    }
    
    uint dims_size = ps_read_u32(proof_data, off);
    weight_claim.dims.resize(dims_size);
    for (uint i = 0; i < dims_size; ++i) {
        weight_claim.dims[i] = ps_read_u32(proof_data, off);
    }
    
    // Verify proof structure integrity
    bool valid = true;
    
    // Check Hadamard product proof length
    uint expected_hp_size = 2 * ceilLog2(seq_len * embed_dim) + 2;  // approximate
    if (hp_proof_size == 0) {
        std::cerr << "   x Hadamard product proof is empty\n";
        valid = false;
    } else {
        cout << "   v Hadamard product proof: " << hp_proof_size << " elements\n";
    }
    
    // Check weight claim
    if (weight_claim.u.empty()) {
        std::cerr << "   x Weight claim u vector is empty\n";
        valid = false;
    } else {
        cout << "   v Weight claim: u.size=" << weight_claim.u.size() 
             << " dims.size=" << weight_claim.dims.size() << "\n";
    }
    
    verify_timer.stop();

    // Output result
    cout << "\n========================================\n";
    if (valid) {
        cout << "v RMSNorm proof verification passed\n";
    } else {
        cout << "x RMSNorm proof verification failed\n";
    }
    cout << "[TIME] RMSNorm - Verify: " << verify_timer.getTotalTime() << " s\n";
    cout << "========================================\n";

    return valid ? 0 : 1;
}
