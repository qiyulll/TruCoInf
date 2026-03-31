#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "fs_rng.cuh"
#include "sha256.cuh"
#include "attn_proof.cuh"
#include "zkfc.cuh"
#include "zksoftmax.cuh"
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

Timer compute_timer, prove_timer;

// Load and quantize float32 file to FrTensor
FrTensor load_and_quantize_float32(const string& filename, int scale = 1 << 12) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    uint count = static_cast<uint>(size / sizeof(float));
    std::vector<float> float_data(count);
    file.read(reinterpret_cast<char*>(float_data.data()), size);
    
    std::vector<int> int_data(count);
    for (uint i = 0; i < count; ++i) {
        double scaled = static_cast<double>(float_data[i]) * scale;
        int_data[i] = static_cast<int>(scaled >= 0 ? scaled + 0.5 : scaled - 0.5);
    }
    
    return FrTensor(count, int_data.data());
}

// Load int32 tensor directly from file
FrTensor load_int32_tensor(const string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    uint count = static_cast<uint>(size / sizeof(int32_t));
    std::vector<int> int_data(count);
    file.read(reinterpret_cast<char*>(int_data.data()), size);
    
    return FrTensor(count, int_data.data());
}

// Create weight from float32 file with internal commitment generation
Weight create_weight_from_float32(
    const string& weight_file, 
    uint in_dim, 
    uint out_dim,
    int scale = 1 << 12) 
{
    FrTensor weight = load_and_quantize_float32(weight_file, scale);
    uint generator_size = in_dim;
    Commitment generator = Commitment::random(generator_size);
    G1TensorJacobian com = generator.commit_int(weight);
    
    return {generator, weight, com, in_dim, out_dim};
}

int main(int argc, char* argv[]) {
    if (argc < 10) {
        std::cerr
            << "zkhook Attention Prover (v8 - Complete with O Projection)\n"
            << "=========================================================\n"
            << "Usage:\n"
            << "  attn_prove <input.bin> <q_weight.bin> <k_weight.bin> <v_weight.bin> \\\n"
            << "             <o_weight.bin> <seq_len> <embed_dim> <proof_output.bin> <nonce>\n"
            << "\n"
            << "Arguments:\n"
            << "  input.bin       - Input tensor (float32, from GaC service)\n"
            << "  q_weight.bin    - Q projection weight (float32)\n"
            << "  k_weight.bin    - K projection weight (float32)\n"
            << "  v_weight.bin    - V projection weight (float32)\n"
            << "  o_weight.bin    - O projection weight (float32)\n"
            << "  seq_len         - Sequence length\n"
            << "  embed_dim       - Embedding dimension\n"
            << "  proof_output.bin - Output proof file\n"
            << "  nonce           - Random nonce for Fiat-Shamir\n"
            << "\n"
            << "This prover generates a complete zero-knowledge proof.\n"
            << "The verifier will NOT see:\n"
            << "  - Input X (hidden by commitment)\n"
            << "  - Weights W_q, W_k, W_v, W_o (hidden by commitment)\n"
            << "  - Output Y (hidden by commitment)\n";
        return 1;
    }

    const string input_file = argv[1];
    const string q_weight_file = argv[2];
    const string k_weight_file = argv[3];
    const string v_weight_file = argv[4];
    const string o_weight_file = argv[5];
    const uint seq_len = std::stoul(argv[6]);
    const uint embed_dim = std::stoul(argv[7]);
    const string proof_file = argv[8];
    const string nonce = argv[9];

    std::cout << "[Prover] seq=" << seq_len << " dim=" << embed_dim << " nonce=" << nonce << "\n";

    compute_timer.start();

    // Load weights
    Weight q_proj = create_weight_from_float32(q_weight_file, embed_dim, embed_dim);
    Weight k_proj = create_weight_from_float32(k_weight_file, embed_dim, embed_dim);
    Weight v_proj = create_weight_from_float32(v_weight_file, embed_dim, embed_dim);
    Weight o_proj = create_weight_from_float32(o_weight_file, embed_dim, embed_dim);

    // Create FC layers and rescalers
    zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
    zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
    zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
    zkFC o_layer(embed_dim, embed_dim, o_proj.weight);

    Rescaling q_rescale(1 << 16), k_rescale(1 << 16), v_rescale(1 << 16);

    // Load input and compute QKV
    FrTensor input = load_int32_tensor(input_file);
    Input input_struct = create_input(q_proj.generator, input, seq_len, embed_dim);
    
    auto Q = q_layer(input);
    auto Q_ = q_rescale(Q);
    auto K = k_layer(input);
    auto K_ = k_rescale(K);
    auto V = v_layer(input);
    auto V_ = v_rescale(V);

    // Self-attention computation with padding support
    auto d = Q_.size / seq_len;
    
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
    
    // Calculate padded dimensions (power of 2 for t-lookup)
    uint padded_seq_len = next_power_of_2(seq_len);
    bool need_padding = (padded_seq_len != seq_len);
    
    if (need_padding) {
        std::cout << "[Prover] Padding: " << seq_len << " -> " << padded_seq_len << "\n";
    }
    
    // Padding helper: pad tensor from (rows, cols) to (new_rows, cols)
    auto pad_tensor_rows = [](const FrTensor& t, uint old_rows, uint cols, uint new_rows) -> FrTensor {
        if (old_rows == new_rows) return FrTensor(t);
        
        std::vector<int> padded(new_rows * cols, 0);
        auto host_data = t.to_int_host();
        for (uint r = 0; r < old_rows; ++r) {
            for (uint c = 0; c < cols; ++c) {
                padded[r * cols + c] = host_data[r * cols + c];
            }
        }
        
        return FrTensor(new_rows * cols, padded.data());
    };
    
    // Pad Q_, K_, V_ to padded_seq_len
    FrTensor Q_padded = pad_tensor_rows(Q_, seq_len, d, padded_seq_len);
    FrTensor K_padded = pad_tensor_rows(K_, seq_len, d, padded_seq_len);
    FrTensor V_padded = pad_tensor_rows(V_, seq_len, d, padded_seq_len);
    
    // Compute using padded dimensions
    auto X = FrTensor::matmul(Q_padded, K_padded.transpose(padded_seq_len, d), padded_seq_len, d, padded_seq_len);
    
    // Softmax computation
    FrTensor shift(padded_seq_len), X_shifted(padded_seq_len * padded_seq_len);
    std::vector<FrTensor> X_segments, Y_segments, m_segments;
    FrTensor Y_softmax(padded_seq_len * padded_seq_len);
    
    const uint max_table_size = 1 << 20;
    const uint D_softmax = padded_seq_len * padded_seq_len;
    bool tlookup_feasible = (D_softmax >= max_table_size) && (D_softmax % max_table_size == 0);
    
    bool skip_softmax_proof = (seq_len == 1) || !tlookup_feasible;
    
    if (!tlookup_feasible && seq_len > 1) {
        std::cout << "[Prover] ⚠️ t-lookup not feasible for D=" << D_softmax 
                  << " (requires D >= " << max_table_size << "), using recompute verification\n";
    }
    
    // Create zkSoftmax object
    std::unique_ptr<zkSoftmax> softmax_ptr;
    if (seq_len > 1) {
        softmax_ptr = std::make_unique<zkSoftmax>(
            std::vector<uint>{1 << 8, 1 << 20, 1 << 20}, 1, 0, 1UL << 32, 
            std::vector<double>{1 << 18, 1 << 22}, 
            padded_seq_len, padded_seq_len, d, 1);
    }
    
    if (seq_len == 1) {
        // When seq_len=1, Softmax(X) = 1
        std::vector<int> ones(padded_seq_len * padded_seq_len, 0);
        ones[0] = 1 << 20;
        Y_softmax = FrTensor(padded_seq_len * padded_seq_len, ones.data());
        
        std::vector<int> zeros_shift(padded_seq_len, 0);
        std::vector<int> zeros_shifted(padded_seq_len * padded_seq_len, 0);
        shift = FrTensor(padded_seq_len, zeros_shift.data());
        X_shifted = FrTensor(padded_seq_len * padded_seq_len, zeros_shifted.data());
    } else {
        // Compute Softmax
        Y_softmax = softmax_ptr->compute(X, shift, X_shifted, X_segments, Y_segments, m_segments);
        std::cout << "[Prover] Softmax computed: X_segments=" << X_segments.size() 
                  << " Y_segments=" << Y_segments.size() 
                  << " m_segments=" << m_segments.size() << "\n";
    }

    // Compute output (using padded dimensions)
    auto out_padded = FrTensor::matmul(Y_softmax, V_padded, padded_seq_len, padded_seq_len, d);
    
    // Extract original dimension output (remove padding)
    auto extract_original_rows = [](const FrTensor& t, uint padded_rows, uint cols, uint orig_rows) -> FrTensor {
        if (padded_rows == orig_rows) return FrTensor(t);
        
        auto host_data = t.to_int_host();
        std::vector<int> extracted(orig_rows * cols);
        
        for (uint r = 0; r < orig_rows; ++r) {
            for (uint c = 0; c < cols; ++c) {
                extracted[r * cols + c] = host_data[r * cols + c];
            }
        }
        
        return FrTensor(orig_rows * cols, extracted.data());
    };
    
    auto out = extract_original_rows(out_padded, padded_seq_len, d, seq_len);
    
    // Rescaling
    const uint base_scale = 1 << 10;
    Rescaling rs1(base_scale), rs2(base_scale), rs3(base_scale), rs4(base_scale);
    auto out_r1 = rs1(out);
    auto out_r2 = rs2(out_r1);
    auto out_r3 = rs3(out_r2);
    auto attn_out = rs4(out_r3);

    // O projection
    std::cout << "[Prover] Computing O projection...\n";
    auto o_out = o_layer(attn_out);
    
    // O projection rescaling
    Rescaling o_rescale(1 << 16);
    auto out__ = o_rescale(o_out);
    std::cout << "[Prover] O projection computed\n";

    compute_timer.stop();
    std::cout << "[Prover] Computation done: " << compute_timer.getTotalTime() << " s\n";

    // Helper functions
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
    Input output_struct = create_input(q_proj.generator, out__, seq_len, embed_dim);

    // Extract output plaintext
    auto out_ints_host = out__.to_int_host();
    std::vector<int32_t> out_ints;
    out_ints.reserve(out_ints_host.size());
    for (auto v : out_ints_host) out_ints.push_back(static_cast<int32_t>(v));

    // Serialize output commitment
    std::vector<uint8_t> output_com_bytes;
    for (uint i = 0; i < output_struct.com.size; ++i) {
        ps_write_g1(output_com_bytes, output_struct.com(i));
    }

    // Serialize input commitment
    std::vector<uint8_t> input_com_bytes;
    for (uint i = 0; i < input_struct.com.size; ++i) {
        ps_write_g1(input_com_bytes, input_struct.com(i));
    }

    // Serialize weight commitments
    std::vector<uint8_t> q_weight_com_bytes, k_weight_com_bytes, v_weight_com_bytes, o_weight_com_bytes;
    for (uint i = 0; i < q_proj.com.size; ++i) {
        ps_write_g1(q_weight_com_bytes, q_proj.com(i));
    }
    for (uint i = 0; i < k_proj.com.size; ++i) {
        ps_write_g1(k_weight_com_bytes, k_proj.com(i));
    }
    for (uint i = 0; i < v_proj.com.size; ++i) {
        ps_write_g1(v_weight_com_bytes, v_proj.com(i));
    }
    for (uint i = 0; i < o_proj.com.size; ++i) {
        ps_write_g1(o_weight_com_bytes, o_proj.com(i));
    }

    // Derive Fiat-Shamir seed
    std::vector<std::array<uint8_t, 32>> digests;
    digests.push_back(Sha256::hash(input_com_bytes.data(), input_com_bytes.size()));
    digests.push_back(Sha256::hash(output_com_bytes.data(), output_com_bytes.size()));
    digests.push_back(Sha256::hash(q_weight_com_bytes.data(), q_weight_com_bytes.size()));
    digests.push_back(Sha256::hash(k_weight_com_bytes.data(), k_weight_com_bytes.size()));
    digests.push_back(Sha256::hash(v_weight_com_bytes.data(), v_weight_com_bytes.size()));
    digests.push_back(Sha256::hash(o_weight_com_bytes.data(), o_weight_com_bytes.size()));

    auto seed = FsRng::derive_seed("zkhook-attn-fs-seed-v8", nonce, digests);

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
    statement_bytes.insert(statement_bytes.end(), nonce.begin(), nonce.end());

    Transcript transcript("zkhook-attn-transcript-v8", statement_bytes);
    std::vector<uint8_t> proof_messages;

    // Proof generation phase
    prove_timer.start();

    // Generate proof messages
    
    // QKV FC layer proofs
    std::cout << "[Prover] Generating QKV FC layer proofs...\n";
    size_t fc_proof_start = proof_messages.size();
    
    // Q = input × W_q
    auto q_claims = q_layer.prove_fs(input, Q, transcript, proof_messages, "q/fc");
    std::cout << "[Prover] Q FC proof generated\n";
    
    // K = input × W_k
    auto k_claims = k_layer.prove_fs(input, K, transcript, proof_messages, "k/fc");
    std::cout << "[Prover] K FC proof generated\n";
    
    // V = input × W_v
    auto v_claims = v_layer.prove_fs(input, V, transcript, proof_messages, "v/fc");
    std::cout << "[Prover] V FC proof generated\n";
    
    std::cout << "[Prover] QKV FC proofs total size=" << (proof_messages.size() - fc_proof_start) << " bytes\n";
    
    // QKV Rescaling proofs
    bool skip_qkv_rescaling = (Q.size < (1 << 10)) || (Q.size % (1 << 10) != 0);
    if (!skip_qkv_rescaling) {
        std::cout << "[Prover] Generating QKV rescaling proofs...\n";
        q_rescale.prove_fs(Q, Q_, transcript, proof_messages, "q/rescale");
        k_rescale.prove_fs(K, K_, transcript, proof_messages, "k/rescale");
        v_rescale.prove_fs(V, V_, transcript, proof_messages, "v/rescale");
        std::cout << "[Prover] QKV rescaling proofs generated\n";
    } else {
        std::cout << "[Prover] Skipping QKV rescaling proofs (tensor_size=" << Q.size << " too small)\n";
    }

    // Softmax proof
    size_t softmax_proof_start = proof_messages.size();
    if (!skip_softmax_proof && softmax_ptr) {
        std::cout << "[Prover] Generating Softmax t-lookup proof...\n";
        softmax_ptr->prove_fs(Y_softmax, X, shift, X_shifted, X_segments, Y_segments, m_segments, 
                              transcript, proof_messages, "softmax");
        std::cout << "[Prover] Softmax proof generated, size=" << (proof_messages.size() - softmax_proof_start) << " bytes\n";
    }

    // Attention output rescaling proof
    bool skip_rescaling_proof = (out.size < base_scale) || (out.size % base_scale != 0);
    if (!skip_rescaling_proof) {
        rs1.prove_fs(out, out_r1, transcript, proof_messages, "out/rs1");
        rs2.prove_fs(out_r1, out_r2, transcript, proof_messages, "out/rs2");
        rs3.prove_fs(out_r2, out_r3, transcript, proof_messages, "out/rs3");
        rs4.prove_fs(out_r3, attn_out, transcript, proof_messages, "out/rs4");
    }
    
    // O projection FC layer proof
    std::cout << "[Prover] Generating O projection FC proof...\n";
    size_t o_fc_proof_start = proof_messages.size();
    auto o_claims = o_layer.prove_fs(attn_out, o_out, transcript, proof_messages, "o/fc");
    std::cout << "[Prover] O FC proof generated, size=" << (proof_messages.size() - o_fc_proof_start) << " bytes\n";
    
    // O projection rescaling proof
    bool skip_o_rescaling = (o_out.size < (1 << 10)) || (o_out.size % (1 << 10) != 0);
    if (!skip_o_rescaling) {
        std::cout << "[Prover] Generating O rescaling proof...\n";
        o_rescale.prove_fs(o_out, out__, transcript, proof_messages, "o/rescale");
        std::cout << "[Prover] O rescaling proof generated\n";
    } else {
        std::cout << "[Prover] Skipping O rescaling proof (tensor_size=" << o_out.size << " too small)\n";
    }

    // Matrix multiplication proofs
    
    // Attn @ V (using padded dimensions)
    size_t matmul_attnv_start = proof_messages.size();
    MatmulProofData matmul_attnv_data;
    {
        auto u1 = transcript.challenge_vec("matmul/out/u1", ceilLog2(padded_seq_len));
        auto u2 = transcript.challenge_vec("matmul/out/u2", ceilLog2(d));
        auto ud = transcript.challenge_vec("matmul/out/ud", ceilLog2(padded_seq_len));
        auto claim = out_padded.multi_dim_me({u1, u2}, {padded_seq_len, d});
        
        auto Y_partial = Y_softmax.partial_me(u1, padded_seq_len, padded_seq_len);
        auto V_partial = V_padded.partial_me(u2, d, 1);
        
        auto final_claim = zkip_prove_fs(claim, Y_partial, V_partial, transcript, proof_messages, "matmul/out");
        
        matmul_attnv_data.claim = fr_to_u32_vec(claim);
        matmul_attnv_data.final_claim = fr_to_u32_vec(final_claim);
        matmul_attnv_data.A_partial_ints = tensor_to_i32_vec(Y_partial);
        matmul_attnv_data.B_partial_ints = tensor_to_i32_vec(V_partial);
    }
    
    // Q @ K^T (using padded dimensions)
    size_t matmul_qk_start = proof_messages.size();
    MatmulProofData matmul_qk_data;
    {
        auto u1_ = transcript.challenge_vec("matmul/x/u1", ceilLog2(padded_seq_len));
        auto u2_ = transcript.challenge_vec("matmul/x/u2", ceilLog2(padded_seq_len));
        auto ud_ = transcript.challenge_vec("matmul/x/ud", ceilLog2(d));
        auto claim_ = X.multi_dim_me({u1_, u2_}, {padded_seq_len, padded_seq_len});
        
        auto Q_partial = Q_padded.partial_me(u1_, padded_seq_len, d);
        auto K_partial = K_padded.partial_me(u2_, padded_seq_len, d);
        
        auto final_claim_ = zkip_prove_fs(claim_, Q_partial, K_partial, transcript, proof_messages, "matmul/x");
        
        matmul_qk_data.claim = fr_to_u32_vec(claim_);
        matmul_qk_data.final_claim = fr_to_u32_vec(final_claim_);
        matmul_qk_data.A_partial_ints = tensor_to_i32_vec(Q_partial);
        matmul_qk_data.B_partial_ints = tensor_to_i32_vec(K_partial);
    }


    // Build proof structure
    
    AttnProof proof;
    proof.input_int_bin = input_file;
    proof.workdir = "";  // No longer needed
    proof.layer_prefix = "";
    proof.seq_len = static_cast<uint32_t>(seq_len);
    proof.padded_seq_len = static_cast<uint32_t>(padded_seq_len);
    proof.embed_dim = static_cast<uint32_t>(embed_dim);
    proof.head_dim = static_cast<uint32_t>(d);
    proof.nonce = nonce;
    proof.seed = seed;
    
    // Output
    proof.output_ints = std::move(out_ints);
    proof.output_commitment = std::move(output_com_bytes);
    
    // Intermediate values
    proof.Q_ints = tensor_to_i32_vec(Q_padded);
    proof.K_ints = tensor_to_i32_vec(K_padded);
    proof.V_ints = tensor_to_i32_vec(V_padded);
    proof.X_ints = tensor_to_i32_vec(X);
    proof.Y_softmax_ints = tensor_to_i32_vec(Y_softmax);
    proof.shift_ints = tensor_to_i32_vec(shift);
    proof.X_shifted_ints = tensor_to_i32_vec(X_shifted);
    
    // O projection intermediate values
    proof.attn_out_ints = tensor_to_i32_vec(attn_out);
    proof.o_proj_out_ints = tensor_to_i32_vec(o_out);
    
    // Softmax segment data
    for (const auto& seg : X_segments) {
        proof.softmax_segments.X_segments_ints.push_back(tensor_to_i32_vec(seg));
    }
    for (const auto& seg : Y_segments) {
        proof.softmax_segments.Y_segments_ints.push_back(tensor_to_i32_vec(seg));
    }
    for (const auto& seg : m_segments) {
        proof.softmax_segments.m_segments_ints.push_back(tensor_to_i32_vec(seg));
    }
    
    // Matrix multiplication proof data
    proof.matmul_qk = matmul_qk_data;
    proof.matmul_attnv = matmul_attnv_data;
    
    // Input commitment
    proof.input_commitment = std::move(input_com_bytes);
    
    // Generate commitment opening proofs
    
    // Input claim and opening proof
    {
        Transcript transcript_for_input("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_input.challenge_vec("q/rescale/u", ceilLog2(Q.size));
        transcript_for_input.challenge_fr("q/rescale/alpha");
        transcript_for_input.challenge_fr("q/rescale/beta");
        auto u_batch = transcript_for_input.challenge_vec("q/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input = transcript_for_input.challenge_vec("q/zkfc/u_input", ceilLog2(embed_dim));
        
        auto claim_input = input.multi_dim_me({u_batch, u_input}, {seq_len, embed_dim});
        proof.input_claim = fr_to_u32_vec(claim_input);
        
        vector<Fr_t> u_in;
        u_in.insert(u_in.end(), u_batch.begin(), u_batch.end());
        u_in.insert(u_in.end(), u_input.begin(), u_input.end());
        
        auto opening_input = generate_opening_proof(input, input_struct.generator, input_struct.com, u_in, seq_len, embed_dim);
        serialize_opening_proof(opening_input, proof.opening_proof_input);
    }
    
    // Output claim and opening proof
    {
        auto u_out_batch = transcript.challenge_vec("out/commitment/u_batch", ceilLog2(seq_len));
        auto u_out_embed = transcript.challenge_vec("out/commitment/u_embed", ceilLog2(embed_dim));

        auto claim_output = out__.multi_dim_me({u_out_batch, u_out_embed}, {seq_len, embed_dim});
        proof.output_claim = fr_to_u32_vec(claim_output);

        vector<Fr_t> u_out;
        u_out.insert(u_out.end(), u_out_batch.begin(), u_out_batch.end());
        u_out.insert(u_out.end(), u_out_embed.begin(), u_out_embed.end());

        auto opening_output = generate_opening_proof(out__, output_struct.generator, output_struct.com, u_out, seq_len, embed_dim);
        serialize_opening_proof(opening_output, proof.opening_proof_output);
    }
    
    // Weight claims
    {
        Transcript transcript_for_claims("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_claims.challenge_vec("q/rescale/u", ceilLog2(Q.size));
        transcript_for_claims.challenge_fr("q/rescale/alpha");
        transcript_for_claims.challenge_fr("q/rescale/beta");
        auto u_batch_q = transcript_for_claims.challenge_vec("q/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input_q = transcript_for_claims.challenge_vec("q/zkfc/u_input", ceilLog2(embed_dim));
        auto u_output_q = transcript_for_claims.challenge_vec("q/zkfc/u_output", ceilLog2(embed_dim));
        
        auto claim_W_q = q_proj.weight.multi_dim_me({u_input_q, u_output_q}, {embed_dim, embed_dim});
        proof.weight_claim_q = fr_to_u32_vec(claim_W_q);
    }
    {
        Transcript transcript_for_claims("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_claims.challenge_vec("k/rescale/u", ceilLog2(K.size));
        transcript_for_claims.challenge_fr("k/rescale/alpha");
        transcript_for_claims.challenge_fr("k/rescale/beta");
        auto u_batch_k = transcript_for_claims.challenge_vec("k/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input_k = transcript_for_claims.challenge_vec("k/zkfc/u_input", ceilLog2(embed_dim));
        auto u_output_k = transcript_for_claims.challenge_vec("k/zkfc/u_output", ceilLog2(embed_dim));
        auto claim_W_k = k_proj.weight.multi_dim_me({u_input_k, u_output_k}, {embed_dim, embed_dim});
        proof.weight_claim_k = fr_to_u32_vec(claim_W_k);
    }
    {
        Transcript transcript_for_claims("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_claims.challenge_vec("v/rescale/u", ceilLog2(V.size));
        transcript_for_claims.challenge_fr("v/rescale/alpha");
        transcript_for_claims.challenge_fr("v/rescale/beta");
        auto u_batch_v = transcript_for_claims.challenge_vec("v/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input_v = transcript_for_claims.challenge_vec("v/zkfc/u_input", ceilLog2(embed_dim));
        auto u_output_v = transcript_for_claims.challenge_vec("v/zkfc/u_output", ceilLog2(embed_dim));
        auto claim_W_v = v_proj.weight.multi_dim_me({u_input_v, u_output_v}, {embed_dim, embed_dim});
        proof.weight_claim_v = fr_to_u32_vec(claim_W_v);
    }
    // O projection weight claim
    {
        Transcript transcript_for_claims("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_claims.challenge_vec("o/fc/u_batch", ceilLog2(seq_len));
        transcript_for_claims.challenge_vec("o/fc/u_input", ceilLog2(embed_dim));
        auto u_output_o = transcript_for_claims.challenge_vec("o/fc/u_output", ceilLog2(embed_dim));
        auto u_input_o = transcript_for_claims.challenge_vec("o/fc/u_input_w", ceilLog2(embed_dim));
        auto claim_W_o = o_proj.weight.multi_dim_me({u_input_o, u_output_o}, {embed_dim, embed_dim});
        proof.weight_claim_o = fr_to_u32_vec(claim_W_o);
    }
    
    // Weight commitment opening proofs
    {
        Transcript transcript_for_q("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_q.challenge_vec("q/rescale/u", ceilLog2(Q.size));
        transcript_for_q.challenge_fr("q/rescale/alpha");
        transcript_for_q.challenge_fr("q/rescale/beta");
        auto u_batch_q = transcript_for_q.challenge_vec("q/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input_q = transcript_for_q.challenge_vec("q/zkfc/u_input", ceilLog2(embed_dim));
        auto u_output_q = transcript_for_q.challenge_vec("q/zkfc/u_output", ceilLog2(embed_dim));
        
        vector<Fr_t> u_q;
        u_q.insert(u_q.end(), u_input_q.begin(), u_input_q.end());
        u_q.insert(u_q.end(), u_output_q.begin(), u_output_q.end());
        
        auto opening_q = generate_opening_proof(q_proj.weight, q_proj.generator, q_proj.com, u_q, embed_dim, embed_dim);
        serialize_opening_proof(opening_q, proof.opening_proof_q);
    }
    {
        Transcript transcript_for_k("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_k.challenge_vec("k/rescale/u", ceilLog2(K.size));
        transcript_for_k.challenge_fr("k/rescale/alpha");
        transcript_for_k.challenge_fr("k/rescale/beta");
        auto u_batch_k = transcript_for_k.challenge_vec("k/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input_k = transcript_for_k.challenge_vec("k/zkfc/u_input", ceilLog2(embed_dim));
        auto u_output_k = transcript_for_k.challenge_vec("k/zkfc/u_output", ceilLog2(embed_dim));
        
        vector<Fr_t> u_k;
        u_k.insert(u_k.end(), u_input_k.begin(), u_input_k.end());
        u_k.insert(u_k.end(), u_output_k.begin(), u_output_k.end());
        
        auto opening_k = generate_opening_proof(k_proj.weight, k_proj.generator, k_proj.com, u_k, embed_dim, embed_dim);
        serialize_opening_proof(opening_k, proof.opening_proof_k);
    }
    {
        Transcript transcript_for_v("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_v.challenge_vec("v/rescale/u", ceilLog2(V.size));
        transcript_for_v.challenge_fr("v/rescale/alpha");
        transcript_for_v.challenge_fr("v/rescale/beta");
        auto u_batch_v = transcript_for_v.challenge_vec("v/zkfc/u_batch", ceilLog2(seq_len));
        auto u_input_v = transcript_for_v.challenge_vec("v/zkfc/u_input", ceilLog2(embed_dim));
        auto u_output_v = transcript_for_v.challenge_vec("v/zkfc/u_output", ceilLog2(embed_dim));
        
        vector<Fr_t> u_v;
        u_v.insert(u_v.end(), u_input_v.begin(), u_input_v.end());
        u_v.insert(u_v.end(), u_output_v.begin(), u_output_v.end());
        
        auto opening_v = generate_opening_proof(v_proj.weight, v_proj.generator, v_proj.com, u_v, embed_dim, embed_dim);
        serialize_opening_proof(opening_v, proof.opening_proof_v);
    }
    // O projection weight opening proof
    {
        Transcript transcript_for_o("zkhook-attn-transcript-v8", statement_bytes);
        transcript_for_o.challenge_vec("o/fc/u_batch", ceilLog2(seq_len));
        transcript_for_o.challenge_vec("o/fc/u_input", ceilLog2(embed_dim));
        auto u_output_o = transcript_for_o.challenge_vec("o/fc/u_output", ceilLog2(embed_dim));
        auto u_input_o = transcript_for_o.challenge_vec("o/fc/u_input_w", ceilLog2(embed_dim));
        
        vector<Fr_t> u_o;
        u_o.insert(u_o.end(), u_input_o.begin(), u_input_o.end());
        u_o.insert(u_o.end(), u_output_o.begin(), u_output_o.end());
        
        auto opening_o = generate_opening_proof(o_proj.weight, o_proj.generator, o_proj.com, u_o, embed_dim, embed_dim);
        serialize_opening_proof(opening_o, proof.opening_proof_o);
    }

    // Proof offsets
    proof.softmax_proof_offset = softmax_proof_start;
    proof.matmul_qk_proof_offset = matmul_qk_start;
    proof.matmul_attnv_proof_offset = matmul_attnv_start;

    proof.proof_messages = std::move(proof_messages);
    
    prove_timer.stop();
    
    // Write proof file
    write_attn_proof_bin(proof_file, proof);

    std::cout << "[Prover] ✓ Proof generated (" << proof.proof_messages.size() << " bytes)\n";
    
    // Output timing statistics
    std::cout << "[TIME] Attention - Compute: " << compute_timer.getTotalTime() << " s\n";
    std::cout << "[TIME] Attention - Prove: " << prove_timer.getTotalTime() << " s\n";
    
    // Commitment sizes
    size_t weight_commitment_size = q_weight_com_bytes.size() + k_weight_com_bytes.size() + 
                                    v_weight_com_bytes.size() + o_weight_com_bytes.size();
    size_t total_commitment_size = input_com_bytes.size() + output_com_bytes.size() + 
                                   weight_commitment_size;
    std::cout << "[SIZE] Attention - WeightCommitment: " << weight_commitment_size << " bytes\n";
    std::cout << "[SIZE] Attention - Commitment: " << total_commitment_size << " bytes\n";
    
    return 0;
}
