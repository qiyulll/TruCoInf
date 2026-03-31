/**
 * table_gen.cu - Lookup Table Generator for zkhook
 * 
 * Generates Softmax and SwiGLU tables for use by both Prover(P) and Verifier(V).
 * 
 * ===== Purpose of Tables =====
 * In zero-knowledge proofs, non-linear operations (like exp, sigmoid) cannot be
 * directly represented in arithmetic circuits. We use the t-lookup protocol to
 * prove correctness of non-linear operations through pre-computed lookup tables.
 * 
 * P and V must use the same tables:
 * - P uses tables to compute actual values
 * - V uses tables to verify P's computation is correct
 * 
 * ===== Usage =====
 * ./table_gen <embed_dim> <output_file>
 * 
 * Example:
 *   ./table_gen 2048 tables.bin   # Generate tables for 1.8B model
 *   ./table_gen 4096 tables.bin   # Generate tables for 7B model
 */

#include "fr-tensor.cuh"
#include "bls12-381.cuh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint>   // for int32_t
#include <climits>   // for INT32_MIN, INT32_MAX

using namespace std;

// ==================== Softmax Table Generation ====================
// 
// Softmax tables decompose exponential operations into multiple small-range lookups.
// 
// Mathematical principle:
//   softmax(x_i) = exp(x_i) / sum(exp(x_j))
// 
// Since exp(x) has too large a range, we decompose x into multiple segments:
//   x = x_0 + x_1 * B_0 + x_2 * B_0 * B_1 + ...
// where B_i is the base size of each segment
// 
// Then exp(x) = exp(x_0) * exp(x_1 * B_0) * exp(x_2 * B_0 * B_1) * ...
// Each exp(x_i * B_{i-1} * ...) can be pre-computed as a lookup table
//

DEVICE Fr_t softmax_table_entry(uint x, double theta_k, double scaling_factor_in, double d, double Bk) {
    unsigned long result = static_cast<unsigned long>(
        exp(log(theta_k) - (Bk / (scaling_factor_in * sqrt(d))) * x) + 0.5
    );
    return {static_cast<uint>(result), static_cast<uint>(result >> 32), 0, 0, 0, 0, 0, 0};
}

KERNEL void generate_softmax_segment_table(Fr_t* table, double theta_k, double scaling_factor_in, 
                                            double d, double Bk, uint table_size) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < table_size) {
        table[x] = softmax_table_entry(x, theta_k, scaling_factor_in, d, Bk);
    }
}

// ==================== SwiGLU Table Generation ====================
// 
// SwiGLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// 
// This is the activation function used in FFN layers of modern LLMs (like LLaMA, Qwen).
// Since it contains the sigmoid non-linearity, we need to pre-compute a lookup table.
// 
// Quantized formula:
//   SwiGLU_int(x_int) = round(SwiGLU(x_int / scale) * scale)
//

// To match baseline, SwiGLU table uses int32 storage (not Fr_t)
// This reduces table size from 128 MB to 16 MB (4M x 4 bytes vs 4M x 32 bytes)
KERNEL void generate_swiglu_table_int32(int32_t* table, int low, uint len, double input_scale, double output_scale) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        int x_int = static_cast<int>(idx) + low;
        double x = static_cast<double>(x_int) / input_scale;
        
        // SwiGLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        double swiglu_val;
        if (x > 20.0) {
            swiglu_val = x;  // When x is large, sigmoid(x) approx 1
        } else if (x < -20.0) {
            swiglu_val = 0.0;  // When x is small, sigmoid(x) approx 0
        } else {
            swiglu_val = x / (1.0 + exp(-x));
        }
        
        // Quantize output to int32
        long result = static_cast<long>(swiglu_val * output_scale + 0.5);
        if (result < INT32_MIN) result = INT32_MIN;
        if (result > INT32_MAX) result = INT32_MAX;
        
        table[idx] = static_cast<int32_t>(result);
    }
}

// ==================== Table File Format ====================
// 
// File format (binary):
// +--------------------+
// | Magic: "ZKTBL001"  |  8 bytes
// +--------------------+
// | embed_dim (uint32) |  4 bytes
// +--------------------+
// | num_softmax_segs   |  4 bytes
// +--------------------+
// | softmax_seg_sizes  |  4 * num_softmax_segs bytes
// +--------------------+
// | softmax_tables     |  variable
// +--------------------+
// | swiglu_low (int32) |  4 bytes
// +--------------------+
// | swiglu_len (uint32)|  4 bytes
// +--------------------+
// | swiglu_table       |  4 * swiglu_len bytes (int32 array, not Fr_t)
// +--------------------+

struct TableFileHeader {
    char magic[8];
    uint32_t embed_dim;
    uint32_t num_softmax_segments;
};

void write_u32(ofstream& ofs, uint32_t val) {
    ofs.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
}

void write_i32(ofstream& ofs, int32_t val) {
    ofs.write(reinterpret_cast<const char*>(&val), sizeof(int32_t));
}

void write_fr_tensor(ofstream& ofs, const FrTensor& t) {
    vector<Fr_t> host_data(t.size);
    cudaMemcpy(host_data.data(), t.gpu_data, t.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    ofs.write(reinterpret_cast<const char*>(host_data.data()), t.size * sizeof(Fr_t));
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "zkhook Lookup Table Generator\n"
             << "=============================\n"
             << "\n"
             << "Usage: table_gen <embed_dim> <output_file>\n"
             << "\n"
             << "Arguments:\n"
             << "  embed_dim    - Embedding dimension (e.g., 2048 for 1.8B, 4096 for 7B)\n"
             << "  output_file  - Output file path (e.g., tables.bin)\n"
             << "\n"
             << "This tool generates lookup tables for:\n"
             << "  1. Softmax (exponential approximation)\n"
             << "  2. SwiGLU activation (sigmoid * x)\n"
             << "\n"
             << "Both Prover (P) and Verifier (V) must use the same tables.\n";
        return 1;
    }

    uint embed_dim = stoul(argv[1]);
    string output_file = argv[2];
    
    cout << "[TableGen] embed_dim=" << embed_dim << " output=" << output_file << "\n";

    // Softmax table parameters (matching baseline configuration)
    vector<uint> segment_sizes = {1 << 8, 1 << 20, 1 << 20};  // [256, 1M, 1M]
    uint K = segment_sizes.size();
    uint L = 1;  // Number of lowest significant segments (no mapping table needed)
    uint M = 0;  // Number of highest significant segments (mapped to constant 1)
    unsigned long scaling_factor_in = 1UL << 32;  // Input scaling factor
    vector<double> thetas = {1 << 18, 1 << 22};  // Output scaling factors (K - L - M values)
    
    // Calculate cumulative bases
    vector<unsigned long> Bs(K);
    Bs[0] = 1L;
    for (uint i = 1; i < K; i++) {
        Bs[i] = Bs[i-1] * static_cast<unsigned long>(segment_sizes[i-1]);
    }

    cout << "[TableGen] Generating Softmax tables...\n";
    cout << "   Segments: " << K << " (L=" << L << ", M=" << M << ")\n";
    
    vector<FrTensor> softmax_tables;
    
    // Generate table for each segment that needs mapping
    for (uint i = L; i < K - M; ++i) {
        uint table_size = segment_sizes[i];
        FrTensor table(table_size);
        
        uint threads = 256;
        uint blocks = (table_size + threads - 1) / threads;
        
        generate_softmax_segment_table<<<blocks, threads>>>(
            table.gpu_data, 
            thetas[i - L], 
            scaling_factor_in, 
            embed_dim,  // d
            Bs[i], 
            table_size
        );
        cudaDeviceSynchronize();
        
        softmax_tables.push_back(table);
        cout << "   Segment " << i << ": size=" << table_size << "\n";
    }

    // SwiGLU table parameters
    // Matching zkhook-ccs2024-main implementation:
    // - Integer index x_int in [-(2^21), 2^21) represents input x = x_int / 2^12
    //   (corresponds to python: Xs = arange(-512, 512, step=1/4096))
    // - Output stored as int32: round(SwiGLU(x) * 2^16)
    //
    // Table parameters:
    //   low = -2^21, len = 2^22 (4,194,304 entries)
    //   input_scale = 2^12, output_scale = 2^16
    int swiglu_low = -(1 << 21);
    uint swiglu_len = 1 << 22;          // 4,194,304
    double swiglu_input_scale = 1 << 12;
    double swiglu_output_scale = 1 << 16;
    
    cout << "[TableGen] Generating SwiGLU table (int32 format)...\n";
    cout << "   x_int range: [" << swiglu_low << ", " << (swiglu_low + (int)swiglu_len) << ")\n";
    cout << "   x(real) range: [" << (swiglu_low / swiglu_input_scale) << ", " << ((swiglu_low + (int)swiglu_len) / swiglu_input_scale) << ")\n";
    cout << "   Entries: " << swiglu_len << "\n";
    cout << "   Size: " << (swiglu_len * 4 / 1024 / 1024) << " MB (int32)\n";
    
    // Allocate GPU memory for int32 table
    int32_t* swiglu_table_gpu;
    cudaMalloc((void**)&swiglu_table_gpu, swiglu_len * sizeof(int32_t));
    
    {
        uint threads = 256;
        uint blocks = (swiglu_len + threads - 1) / threads;
        generate_swiglu_table_int32<<<blocks, threads>>>(
            swiglu_table_gpu, 
            swiglu_low, 
            swiglu_len,
            swiglu_input_scale,
            swiglu_output_scale
        );
        cudaDeviceSynchronize();
    }
    
    // Copy to host
    vector<int32_t> swiglu_table_host(swiglu_len);
    cudaMemcpy(swiglu_table_host.data(), swiglu_table_gpu, swiglu_len * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(swiglu_table_gpu);

    // Write to file
    cout << "[TableGen] Writing to " << output_file << "...\n";
    
    ofstream ofs(output_file, ios::binary);
    if (!ofs) {
        cerr << "Error: Cannot open output file\n";
        return 1;
    }
    
    // Magic
    const char* magic = "ZKTBL001";
    ofs.write(magic, 8);
    
    // Header
    write_u32(ofs, embed_dim);
    write_u32(ofs, static_cast<uint32_t>(softmax_tables.size()));
    
    // Softmax segment sizes
    for (const auto& t : softmax_tables) {
        write_u32(ofs, t.size);
    }
    
    // Softmax table data
    for (const auto& t : softmax_tables) {
        write_fr_tensor(ofs, t);
    }
    
    // SwiGLU parameters
    write_i32(ofs, swiglu_low);
    write_u32(ofs, swiglu_len);
    
    // SwiGLU table data (int32 array)
    ofs.write(reinterpret_cast<const char*>(swiglu_table_host.data()), swiglu_len * sizeof(int32_t));
    
    ofs.close();
    
    // Output file size
    ifstream check(output_file, ios::binary | ios::ate);
    auto file_size = check.tellg();
    cout << "[TableGen] Done! File size: " << (file_size / 1024 / 1024) << " MB\n";
    
    return 0;
}
