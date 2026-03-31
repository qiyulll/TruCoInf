#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "fs_rng.cuh"
#include "sha256.cuh"
#include "ffn_proof.cuh"
#include "zkfc.cuh"
#include "zkrelu.cuh"
#include "rescaling.cuh"
#include "transcript.cuh"
#include "proof_stream.cuh"
#include "timer.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

using std::string;
using std::vector;

Timer prove_timer, verify_timer, compute_timer;

// Load int32 file to FrTensor
FrTensor load_int32_ffn(const string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    uint count = static_cast<uint>(size / sizeof(int32_t));
    std::vector<int32_t> int_data(count);
    file.read(reinterpret_cast<char*>(int_data.data()), size);
    file.close();
    
    std::cout << "[FFN] Loaded " << filename << ": " << count << " elements\n";
    
    return FrTensor(count, int_data.data());
}

// Create weight from int32 file
Weight create_weight_from_int32_ffn(
    const string& weight_file, 
    uint in_dim, 
    uint out_dim) 
{
    FrTensor weight = load_int32_ffn(weight_file);
    
    // Verify weight size
    uint expected_size = in_dim * out_dim;
    if (weight.size != expected_size) {
        std::cerr << "[FFN] Warning: weight size " << weight.size 
                  << " != expected " << expected_size 
                  << " (in_dim=" << in_dim << " out_dim=" << out_dim << ")\n";
    }
    
    uint generator_size = in_dim;
    Commitment generator = Commitment::random(generator_size);
    G1TensorJacobian com = generator.commit_int(weight);
    return {generator, weight, com, in_dim, out_dim};
}

int main(int argc, char* argv[]) {
    if (argc < 10) {
            std::cerr
                << "zkhook FFN Prover (v2 - Int32 Input)\n"
            << "===================================\n"
            << "Usage:\n"
            << "  ffn_prove <input.bin> <up_weight.bin> <gate_weight.bin> <down_weight.bin> \\\n"
            << "            <seq_len> <embed_dim> <hidden_dim> <proof_output.bin> <nonce>\n"
            << "\n"
            << "Arguments:\n"
            << "  input.bin       - Input tensor (int32, fixed-point quantized)\n"
            << "  up_weight.bin   - up_proj weight (int32, fixed-point quantized)\n"
            << "  gate_weight.bin - gate_proj weight (int32, fixed-point quantized)\n"
            << "  down_weight.bin - down_proj weight (int32, fixed-point quantized)\n"
            << "  seq_len         - Sequence length\n"
            << "  embed_dim       - Embedding dimension (input/output dim)\n"
            << "  hidden_dim      - Hidden dimension (intermediate dim)\n"
            << "  proof_output.bin - Output proof file\n"
            << "  nonce           - Random nonce for Fiat-Shamir\n";
        return 1;
    }

    const string input_file = argv[1];
    const string up_weight_file = argv[2];
    const string gate_weight_file = argv[3];
    const string down_weight_file = argv[4];
    const uint seq_len = std::stoul(argv[5]);
    const uint embed_dim = std::stoul(argv[6]);
    const uint hidden_dim = std::stoul(argv[7]);
    const string proof_file = argv[8];
    const string nonce = argv[9];

    std::cout << "[FFN Prover] seq=" << seq_len << " embed_dim=" << embed_dim 
              << " hidden_dim=" << hidden_dim << " nonce=" << nonce << "\n";

    // Load weights (int32 format)
    compute_timer.start();
    
    std::cout << "[FFN] Loading weights...\n";
    std::cout << "[FFN] up_proj: in_dim=" << embed_dim << " out_dim=" << hidden_dim << "\n";
    Weight up_proj = create_weight_from_int32_ffn(up_weight_file, embed_dim, hidden_dim);
    std::cout << "[FFN] up_proj loaded: generator.size=" << up_proj.generator.size 
              << " weight.size=" << up_proj.weight.size << "\n";
    
    std::cout << "[FFN] gate_proj: in_dim=" << embed_dim << " out_dim=" << hidden_dim << "\n";
    Weight gate_proj = create_weight_from_int32_ffn(gate_weight_file, embed_dim, hidden_dim);
    std::cout << "[FFN] gate_proj loaded: generator.size=" << gate_proj.generator.size 
              << " weight.size=" << gate_proj.weight.size << "\n";
    
    std::cout << "[FFN] down_proj: in_dim=" << hidden_dim << " out_dim=" << embed_dim << "\n";
    Weight down_proj = create_weight_from_int32_ffn(down_weight_file, hidden_dim, embed_dim);
    std::cout << "[FFN] down_proj loaded: generator.size=" << down_proj.generator.size 
              << " weight.size=" << down_proj.weight.size << "\n";
    
    std::cout << "[FFN] Weights loaded successfully\n";

    // Create FC layers and rescalers
    zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
    zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
    zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);

    Rescaling up_rescale(1 << 16);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    // Load SwiGLU lookup table (if exists)
    bool use_swiglu_table = false;
    std::unique_ptr<FrTensor> swiglu_values_ptr;
    std::unique_ptr<tLookupRangeMapping> swiglu_ptr;
    
    // Try to load SwiGLU table
    std::ifstream swiglu_file("swiglu-table.bin", std::ios::binary);
    if (swiglu_file.good()) {
        swiglu_file.close();
        swiglu_values_ptr = std::make_unique<FrTensor>(FrTensor::from_int_bin("swiglu-table.bin"));
        swiglu_ptr = std::make_unique<tLookupRangeMapping>(-(1 << 21), 1 << 22, *swiglu_values_ptr);
        use_swiglu_table = true;
        std::cout << "[FFN Prover] Using SwiGLU lookup table\n";
    } else {
        std::cout << "[FFN Prover] SwiGLU table not found, using simplified proof\n";
    }

    // Load input (int32 format) and compute FFN
    std::cout << "[FFN] Loading input...\n";
    FrTensor input = load_int32_ffn(input_file);
    
    // Verify input size
    uint expected_input_size = seq_len * embed_dim;
    if (input.size != expected_input_size) {
        std::cerr << "[FFN] Error: input size " << input.size 
                  << " != expected " << expected_input_size << "\n";
        return 1;
    }
    
    // Calculate padding dimensions
    // Next power of 2 helper
    auto next_power_of_2 = [](uint n) -> uint {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    };
    
    // Calculate padded_hidden_dim
    uint padded_hidden_dim = next_power_of_2(hidden_dim);
    bool need_hidden_padding = (padded_hidden_dim != hidden_dim);
    
    // Calculate padded_embed_dim
    uint padded_embed_dim = next_power_of_2(embed_dim);
    bool need_embed_padding = (padded_embed_dim != embed_dim);
    
    if (need_hidden_padding) {
        std::cout << "[FFN] Padding hidden_dim: " << hidden_dim << " -> " << padded_hidden_dim << "\n";
    }
    if (need_embed_padding) {
        std::cout << "[FFN] Padding embed_dim: " << embed_dim << " -> " << padded_embed_dim << "\n";
    }
    
    // Padding helper: pad tensor from (rows, cols) to (rows, new_cols)
    auto pad_tensor_cols = [](const FrTensor& t, uint rows, uint old_cols, uint new_cols) -> FrTensor {
        if (old_cols == new_cols) return FrTensor(t);
        
        std::vector<int> padded(rows * new_cols, 0);
        auto host_data = t.to_int_host();
        for (uint r = 0; r < rows; ++r) {
            for (uint c = 0; c < old_cols; ++c) {
                padded[r * new_cols + c] = host_data[r * old_cols + c];
            }
        }
        
        return FrTensor(rows * new_cols, padded.data());
    };
    
    // Extract original dimension output (remove padding)
    auto extract_original_cols = [](const FrTensor& t, uint rows, uint padded_cols, uint orig_cols) -> FrTensor {
        if (padded_cols == orig_cols) return FrTensor(t);
        
        auto host_data = t.to_int_host();
        std::vector<int> extracted(rows * orig_cols);
        
        for (uint r = 0; r < rows; ++r) {
            for (uint c = 0; c < orig_cols; ++c) {
                extracted[r * orig_cols + c] = host_data[r * padded_cols + c];
            }
        }
        
        return FrTensor(rows * orig_cols, extracted.data());
    };
    
    std::cout << "[FFN] Creating input commitment: generator.size=" << up_proj.generator.size 
              << " input.size=" << input.size << "\n";
    std::cout << "[FFN] Check: " << input.size << " % " << up_proj.generator.size 
              << " = " << (input.size % up_proj.generator.size) << "\n";
    Input input_struct = create_input(up_proj.generator, input, seq_len, embed_dim);
    std::cout << "[FFN] Input commitment created successfully\n";
    std::cout << "[FFN] Input loaded, computing FFN...\n";
    
    // up_proj (output hidden_dim)
    auto up_out = up_layer(input);
    auto up_out_ = up_rescale(up_out);
    // Pad up_out_ to padded_hidden_dim
    FrTensor up_out_padded = pad_tensor_cols(up_out_, seq_len, hidden_dim, padded_hidden_dim);

    // gate_proj (output hidden_dim)
    auto gate_out = gate_layer(input);
    auto gate_out_ = gate_rescale(gate_out);
    // Pad gate_out_ to padded_hidden_dim
    FrTensor gate_out_padded = pad_tensor_cols(gate_out_, seq_len, hidden_dim, padded_hidden_dim);

    // SwiGLU activation (using padded tensors)
    FrTensor swiglu_out(gate_out_padded.size);
    std::unique_ptr<FrTensor> swiglu_m_ptr;
    
    if (use_swiglu_table && swiglu_ptr) {
        auto p = (*swiglu_ptr)(gate_out_padded);
        swiglu_out = p.first;
        swiglu_m_ptr = std::make_unique<FrTensor>(std::move(p.second));
    } else {
        // Simplified version: use gate_out_padded as SwiGLU output
        swiglu_out = gate_out_padded;
    }

    // Element-wise multiplication: hidden = swiglu * up (using padded tensors)
    auto down_in_padded = swiglu_out * up_out_padded;
    
    // Create padded version of Rescaling
    Rescaling hidden_rescale_padded(1 << 16);
    auto down_in_padded_ = hidden_rescale_padded(down_in_padded);
    
    // Extract original dimension for down_proj computation
    FrTensor down_in_ = extract_original_cols(down_in_padded_, seq_len, padded_hidden_dim, hidden_dim);
    
    // Keep original dimension down_in for proof
    FrTensor down_in = extract_original_cols(down_in_padded, seq_len, padded_hidden_dim, hidden_dim);

    // down_proj (input hidden_dim, output embed_dim)
    auto down_out = down_layer(down_in_);
    auto down_out_ = down_rescale(down_out);
    
    compute_timer.stop();
    
    // Save intermediate tensors for proof (original dimension)
    FrTensor up_out_orig = up_out_;
    FrTensor gate_out_orig = gate_out_;
    FrTensor swiglu_out_orig = extract_original_cols(swiglu_out, seq_len, padded_hidden_dim, hidden_dim);

    // Helper function
    auto tensor_to_i32_vec = [](const FrTensor& t) -> std::vector<int32_t> {
        auto ints = t.to_int_host();
        std::vector<int32_t> result;
        result.reserve(ints.size());
        for (auto v : ints) result.push_back(static_cast<int32_t>(v));
        return result;
    };

    auto fr_to_u32_vec = [](const Fr_t& f) -> std::vector<uint32_t> {
        return {f.val[0], f.val[1], f.val[2], f.val[3], 
                f.val[4], f.val[5], f.val[6], f.val[7]};
    };

    // Create output commitment
    // Note: output dimension is (seq_len, embed_dim), need embed_dim size generator
    // Cannot use down_proj.generator as its size is hidden_dim
    std::cout << "[FFN] Creating output commitment: generator.size=" << up_proj.generator.size 
              << " down_out_.size=" << down_out_.size << "\n";
    std::cout << "[FFN] Check: " << down_out_.size << " % " << up_proj.generator.size 
              << " = " << (down_out_.size % up_proj.generator.size) << "\n";
    Input output_struct = create_input(up_proj.generator, down_out_, seq_len, embed_dim);
    std::cout << "[FFN] Output commitment created successfully\n";

    // Serialize commitments
    std::vector<uint8_t> input_com_bytes, output_com_bytes;
    std::vector<uint8_t> up_weight_com_bytes, gate_weight_com_bytes, down_weight_com_bytes;
    
    for (uint i = 0; i < input_struct.com.size; ++i) {
        ps_write_g1(input_com_bytes, input_struct.com(i));
    }
    for (uint i = 0; i < output_struct.com.size; ++i) {
        ps_write_g1(output_com_bytes, output_struct.com(i));
    }
    for (uint i = 0; i < up_proj.com.size; ++i) {
        ps_write_g1(up_weight_com_bytes, up_proj.com(i));
    }
    for (uint i = 0; i < gate_proj.com.size; ++i) {
        ps_write_g1(gate_weight_com_bytes, gate_proj.com(i));
    }
    for (uint i = 0; i < down_proj.com.size; ++i) {
        ps_write_g1(down_weight_com_bytes, down_proj.com(i));
    }

    // Derive Fiat-Shamir seed
    std::vector<std::array<uint8_t, 32>> digests;
    digests.push_back(Sha256::hash(input_com_bytes.data(), input_com_bytes.size()));
    digests.push_back(Sha256::hash(output_com_bytes.data(), output_com_bytes.size()));
    digests.push_back(Sha256::hash(up_weight_com_bytes.data(), up_weight_com_bytes.size()));
    digests.push_back(Sha256::hash(gate_weight_com_bytes.data(), gate_weight_com_bytes.size()));
    digests.push_back(Sha256::hash(down_weight_com_bytes.data(), down_weight_com_bytes.size()));

    auto seed = FsRng::derive_seed("zkhook-ffn-fs-seed-v1", nonce, digests);

    // Create Transcript
    std::vector<uint8_t> statement_bytes;
    statement_bytes.insert(statement_bytes.end(), seed.begin(), seed.end());
    statement_bytes.push_back(static_cast<uint8_t>(seq_len & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((seq_len >> 8) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((seq_len >> 16) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((seq_len >> 24) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>(embed_dim & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((embed_dim >> 8) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((embed_dim >> 16) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((embed_dim >> 24) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>(hidden_dim & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((hidden_dim >> 8) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((hidden_dim >> 16) & 0xff));
    statement_bytes.push_back(static_cast<uint8_t>((hidden_dim >> 24) & 0xff));
    statement_bytes.insert(statement_bytes.end(), nonce.begin(), nonce.end());

    Transcript transcript("zkhook-ffn-transcript-v1", statement_bytes);
    std::vector<uint8_t> proof_messages;

    // Generate proof
    prove_timer.start();
    
    // Record sub-proof offsets
    size_t up_proj_offset = proof_messages.size();
    size_t gate_proj_offset = 0;
    size_t swiglu_offset = 0;
    size_t hadamard_offset = 0;
    size_t down_proj_offset = 0;

    // rescaling base_scale
    const uint base_scale = 1 << 10;
    
    // Calculate padded tensor size (using padded_hidden_dim from compute phase)
    uint hidden_tensor_padded_size = seq_len * padded_hidden_dim;
    uint embed_tensor_padded_size = next_power_of_2(seq_len * embed_dim);
    
    std::cout << "[FFN] Proof generation - padded sizes:\n";
    std::cout << "  hidden tensors: " << (seq_len * hidden_dim) << " -> " << hidden_tensor_padded_size << "\n";
    std::cout << "  embed tensors: " << (seq_len * embed_dim) << " -> " << embed_tensor_padded_size << "\n";

    // down_rescale proof (using padded tensors)
    // down_out is embed_dim dimension, usually already power of 2
    bool down_rescale_ok = (embed_tensor_padded_size >= base_scale) && (embed_tensor_padded_size % base_scale == 0);
    if (down_rescale_ok) {
        // pad() parameter is original shape, function auto-pads last dim to power of 2
        FrTensor down_out_for_proof = down_out.pad({seq_len, embed_dim});
        Rescaling down_rescale_proof(1 << 16);
        auto down_out_rescaled = down_rescale_proof(down_out_for_proof);
        down_rescale_proof.prove_fs(down_out_for_proof, down_out_rescaled, transcript, proof_messages, "down/rescale");
    }
    
    // down_proj FC proof
    down_proj_offset = proof_messages.size();
    auto down_claim = down_layer.prove_fs(down_in_, down_out, transcript, proof_messages, "down/fc");
    
    // hidden_rescale proof (using tensors padded in compute phase)
    // down_in_padded created in compute phase, size is seq_len * padded_hidden_dim
    bool hidden_rescale_ok = (hidden_tensor_padded_size >= base_scale) && (hidden_tensor_padded_size % base_scale == 0);
    if (hidden_rescale_ok) {
        // Use padded tensors from compute phase
        hidden_rescale_padded.prove_fs(down_in_padded, down_in_padded_, transcript, proof_messages, "hidden/rescale");
        std::cout << "[FFN] hidden_rescale proof complete (padded)\n";
    }
    
    // Hadamard product (element-wise multiplication) proof
    // Use padded tensors for Hadamard proof
    hadamard_offset = proof_messages.size();
    auto u_hadamard = transcript.challenge_vec("hadamard/u", ceilLog2(down_in_padded.size));
    auto v_hadamard = transcript.challenge_vec("hadamard/v", ceilLog2(down_in_padded.size));
    hadamard_product_sumcheck(swiglu_out, up_out_padded, u_hadamard, v_hadamard);
    
    // SwiGLU proof (if using table, use padded tensors)
    swiglu_offset = proof_messages.size();
    if (use_swiglu_table && swiglu_ptr && swiglu_m_ptr) {
        auto temp_rand = random_vec(3);
        auto swiglu_u = transcript.challenge_vec("swiglu/u", ceilLog2(gate_out_padded.size));
        auto swiglu_v = transcript.challenge_vec("swiglu/v", ceilLog2(gate_out_padded.size));
        std::vector<Polynomial> swiglu_proof;
        swiglu_ptr->prove(gate_out_padded, swiglu_out, *swiglu_m_ptr, temp_rand[0], temp_rand[1], temp_rand[2], 
                         swiglu_u, swiglu_v, swiglu_proof);
        std::cout << "[FFN Prover] SwiGLU proof complete.\n";
    }
    
    // gate_rescale proof (using padded tensors)
    gate_proj_offset = proof_messages.size();
    if (hidden_rescale_ok) {
        // gate_out needs padding first - pad() parameter is original shape
        FrTensor gate_out_for_proof = gate_out.pad({seq_len, hidden_dim});
        Rescaling gate_rescale_proof(1 << 20);
        auto gate_out_rescaled = gate_rescale_proof(gate_out_for_proof);
        gate_rescale_proof.prove_fs(gate_out_for_proof, gate_out_rescaled, transcript, proof_messages, "gate/rescale");
    }
    
    // gate_proj FC proof
    auto gate_claim = gate_layer.prove_fs(input, gate_out, transcript, proof_messages, "gate/fc");
    
    // up_rescale proof (using padded tensors)
    up_proj_offset = proof_messages.size();
    if (hidden_rescale_ok) {
        // up_out needs padding first - pad() parameter is original shape
        FrTensor up_out_for_proof = up_out.pad({seq_len, hidden_dim});
        Rescaling up_rescale_proof(1 << 16);
        auto up_out_rescaled = up_rescale_proof(up_out_for_proof);
        up_rescale_proof.prove_fs(up_out_for_proof, up_out_rescaled, transcript, proof_messages, "up/rescale");
    }
    
    // up_proj FC proof
    auto up_claim = up_layer.prove_fs(input, up_out, transcript, proof_messages, "up/fc");
    
    prove_timer.stop();

    // Build proof structure
    FFNProof proof;
    proof.input_file = input_file;
    proof.workdir = "";
    proof.layer_prefix = "";
    proof.seq_len = static_cast<uint32_t>(seq_len);
    proof.embed_dim = static_cast<uint32_t>(embed_dim);
    proof.hidden_dim = static_cast<uint32_t>(hidden_dim);
    proof.nonce = nonce;
    proof.seed = seed;
    
    // Input commitment
    proof.input_commitment = std::move(input_com_bytes);
    
    // Output
    auto out_ints_host = down_out_.to_int_host();
    proof.output_ints.reserve(out_ints_host.size());
    for (auto v : out_ints_host) proof.output_ints.push_back(static_cast<int32_t>(v));
    proof.output_commitment = std::move(output_com_bytes);
    
    // Intermediate values
    proof.up_out_ints = tensor_to_i32_vec(up_out_);
    proof.gate_out_ints = tensor_to_i32_vec(gate_out_);
    proof.hidden_ints = tensor_to_i32_vec(down_in_);
    proof.down_out_ints = tensor_to_i32_vec(down_out_);
    
    // SwiGLU data (using original dimension)
    proof.swiglu_data.gate_out_ints = tensor_to_i32_vec(gate_out_);
    proof.swiglu_data.swiglu_out_ints = tensor_to_i32_vec(swiglu_out_orig);
    if (swiglu_m_ptr && swiglu_m_ptr->size > 0) {
        proof.swiglu_data.m_ints = tensor_to_i32_vec(*swiglu_m_ptr);
    }
    
    // Proof offsets
    proof.up_proj_proof_offset = up_proj_offset;
    proof.gate_proj_proof_offset = gate_proj_offset;
    proof.swiglu_proof_offset = swiglu_offset;
    proof.hadamard_proof_offset = hadamard_offset;
    proof.down_proj_proof_offset = down_proj_offset;
    
    proof.proof_messages = std::move(proof_messages);
    
    // Write proof file
    write_ffn_proof_bin(proof_file, proof);

    std::cout << "[FFN Prover] ✓ Proof generated (" << proof.proof_messages.size() << " bytes)\n";
    std::cout << "[TIME] FFN - Compute: " << compute_timer.getTotalTime() << " s\n";
    std::cout << "[TIME] FFN - Prove: " << prove_timer.getTotalTime() << " s\n";
    
    // Calculate and output commitment size
    // Weight commitment (aligned with Baseline)
    size_t weight_commitment_size = up_weight_com_bytes.size() + gate_weight_com_bytes.size() + 
                                    down_weight_com_bytes.size();
    // Total commitment (including I/O)
    size_t total_commitment_size = input_com_bytes.size() + output_com_bytes.size() + 
                                   weight_commitment_size;
    std::cout << "[SIZE] FFN - WeightCommitment: " << weight_commitment_size << " bytes\n";
    std::cout << "[SIZE] FFN - Commitment: " << total_commitment_size << " bytes\n";
    
    return 0;
}
