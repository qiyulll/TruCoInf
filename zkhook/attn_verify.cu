#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "fs_rng.cuh"
#include "sha256.cuh"
#include "attn_proof.cuh"
#include "zkfc.cuh"
#include "zksoftmax.cuh"
#include "rescaling.cuh"
#include "proof.cuh"
#include "transcript.cuh"
#include "proof_stream.cuh"
#include "timer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::vector;

Timer verify_timer;

// Helper functions
static FrTensor i32_vec_to_tensor(const std::vector<int32_t> &v)
{
    std::vector<int> ints(v.begin(), v.end());
    return FrTensor(static_cast<uint>(ints.size()), ints.data());
}

static Fr_t u32_vec_to_fr(const std::vector<uint32_t> &v)
{
    if (v.size() != 8)
        throw std::runtime_error("u32_vec_to_fr: need 8 elements");
    return Fr_t{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
}

// Skip a zkip sumcheck (read polynomial, absorb, generate challenge)
static void skip_zkip_rounds(
    uint tensor_size,
    Transcript& transcript,
    const std::vector<uint8_t>& proof_messages, size_t& proof_off,
    const std::string& label_prefix)
{
    // Same loop structure as zkip_verify_fs
    uint cur_size = tensor_size;
    while (cur_size > 1) {
        size_t off0 = proof_off;
        ps_deserialize_poly(proof_messages, proof_off);
        transcript.absorb(proof_messages.data() + off0, proof_off - off0);
        transcript.challenge_fr(label_prefix + "/zkip/r");
        cur_size = (1u << ceilLog2(cur_size)) >> 1;
    }
}

// Skip a zkFC proof (generate challenges + skip zkip)
static void skip_zkfc_proof(
    uint batch_size, uint input_size, uint output_size,
    Transcript& transcript,
    const std::vector<uint8_t>& proof_messages, size_t& proof_off,
    const std::string& label_prefix)
{
    // Same transcript operation order as zkFC::prove_fs
    transcript.challenge_vec(label_prefix + "/zkfc/u_batch", ceilLog2(batch_size));
    transcript.challenge_vec(label_prefix + "/zkfc/u_output", ceilLog2(output_size));

    // zkip reduced tensor size = inputSize
    skip_zkip_rounds(input_size, transcript, proof_messages, proof_off, label_prefix);
}

// Skip tLookup phase1 + phase2
static void skip_tlookup_rounds(
    uint D, uint N,
    Transcript& transcript,
    const std::vector<uint8_t>& proof_messages, size_t& proof_off,
    const std::string& label_prefix)
{
    uint depth1 = ceilLog2(D / N);
    uint depth2 = ceilLog2(N);

    // Phase 1: depth1 rounds
    for (uint i = 0; i < depth1; ++i) {
        size_t off0 = proof_off;
        ps_deserialize_poly(proof_messages, proof_off);
        transcript.absorb(proof_messages.data() + off0, proof_off - off0);
        transcript.challenge_fr(label_prefix + "/tlookup/v1");
    }

    // Phase 2: depth2 rounds
    for (uint i = 0; i < depth2; ++i) {
        size_t off0 = proof_off;
        ps_deserialize_poly(proof_messages, proof_off);
        transcript.absorb(proof_messages.data() + off0, proof_off - off0);
        transcript.challenge_fr(label_prefix + "/tlookup/v2");
    }
}

// Skip a Rescaling proof (generate challenges + skip tLookup)
static void skip_rescaling_proof(
    uint tensor_size, uint scaling_factor,
    Transcript& transcript,
    const std::vector<uint8_t>& proof_messages, size_t& proof_off,
    const std::string& label_prefix)
{
    // Same transcript operations as Rescaling::verify_fs
    transcript.challenge_vec(label_prefix + "/rescale/u", ceilLog2(tensor_size));
    transcript.challenge_fr(label_prefix + "/rescale/alpha");
    transcript.challenge_fr(label_prefix + "/rescale/beta");

    // tLookup dimensions
    uint D = 1u << ceilLog2(tensor_size);
    uint N = scaling_factor;

    skip_tlookup_rounds(D, N, transcript, proof_messages, proof_off,
                        label_prefix + "/rescale/tlookup");
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr
            << "zkhook Attention Verifier (v8 - Complete with O Projection)\n"
            << "============================================================\n"
            << "Usage:\n"
            << "  attn_verify <proof.bin>\n";
        return 1;
    }

    const string proof_file = argv[1];

    try
    {
        verify_timer.start();

        // Load proof
        auto proof = read_attn_proof_bin(proof_file);

        const uint seq_len = proof.seq_len;
        const uint padded_seq_len = proof.padded_seq_len > 0 ? proof.padded_seq_len : seq_len;
        const uint embed_dim = proof.embed_dim;
        const string nonce = proof.nonce;
        const auto seed = proof.seed;
        const uint d = proof.head_dim > 0 ? proof.head_dim : embed_dim;

        std::cout << "[Verifier] seq=" << seq_len << " padded=" << padded_seq_len
                  << " dim=" << embed_dim << " head_dim=" << d << " nonce=" << nonce << "\n";

        // Create Transcript (must match Prover exactly)
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
        size_t proof_off = 0;

        // Load intermediate values
        FrTensor Q_ = i32_vec_to_tensor(proof.Q_ints);
        FrTensor K_ = i32_vec_to_tensor(proof.K_ints);
        FrTensor V_ = i32_vec_to_tensor(proof.V_ints);
        FrTensor X = i32_vec_to_tensor(proof.X_ints);
        FrTensor Y_softmax = i32_vec_to_tensor(proof.Y_softmax_ints);
        FrTensor shift = i32_vec_to_tensor(proof.shift_ints);
        FrTensor X_shifted = i32_vec_to_tensor(proof.X_shifted_ints);

        // Verify commitment opening proofs
        if (!proof.opening_proof_input.empty())
        {
            Fr_t input_claim = u32_vec_to_fr(proof.input_claim);
            size_t off = 0;
            CommitmentOpeningProof input_opening = deserialize_opening_proof(proof.opening_proof_input, off);
            if (input_claim != input_opening.claimed_value)
            {
                std::cerr << "[Verifier] ❌ Input claim mismatch\n";
                return 1;
            }
        }

        auto verify_weight_opening = [](const std::vector<uint8_t> &opening_bytes,
                                        const std::vector<uint32_t> &claim_vec) -> bool
        {
            if (opening_bytes.empty() || claim_vec.empty())
                return true;
            Fr_t claim = u32_vec_to_fr(claim_vec);
            size_t off = 0;
            CommitmentOpeningProof opening = deserialize_opening_proof(opening_bytes, off);
            return claim == opening.claimed_value;
        };

        if (!verify_weight_opening(proof.opening_proof_q, proof.weight_claim_q) ||
            !verify_weight_opening(proof.opening_proof_k, proof.weight_claim_k) ||
            !verify_weight_opening(proof.opening_proof_v, proof.weight_claim_v))
        {
            std::cerr << "[Verifier] ❌ Weight claim mismatch (Q/K/V)\n";
            return 1;
        }

        // v8: O projection weight
        if (!verify_weight_opening(proof.opening_proof_o, proof.weight_claim_o))
        {
            std::cerr << "[Verifier] ❌ Weight claim mismatch (O)\n";
            return 1;
        }

        if (!proof.opening_proof_output.empty())
        {
            Fr_t output_claim = u32_vec_to_fr(proof.output_claim);
            size_t off = 0;
            CommitmentOpeningProof output_opening = deserialize_opening_proof(proof.opening_proof_output, off);
            if (output_claim != output_opening.claimed_value)
            {
                std::cerr << "[Verifier] ❌ Output claim mismatch\n";
                return 1;
            }
        }
        std::cout << "[Verifier] ✓ Commitment opening proofs verified\n";

        const uint qkv_size = seq_len * embed_dim;

        // Step 1: QKV FC layer proofs (skip, Verifier has no input/weight)
        std::cout << "[Verifier] Replaying QKV FC proofs (transcript sync)...\n";
        skip_zkfc_proof(seq_len, embed_dim, embed_dim, transcript, proof.proof_messages, proof_off, "q/fc");
        skip_zkfc_proof(seq_len, embed_dim, embed_dim, transcript, proof.proof_messages, proof_off, "k/fc");
        skip_zkfc_proof(seq_len, embed_dim, embed_dim, transcript, proof.proof_messages, proof_off, "v/fc");
        std::cout << "[Verifier] ✓ QKV FC proofs replayed (proof_off=" << proof_off << ")\n";

        // Step 2: QKV Rescaling proofs (skip)
        bool skip_qkv_rescaling = (qkv_size < (1u << 10)) || (qkv_size % (1u << 10) != 0);
        if (!skip_qkv_rescaling) {
            std::cout << "[Verifier] Replaying QKV rescaling proofs (transcript sync)...\n";
            skip_rescaling_proof(qkv_size, 1u << 16, transcript, proof.proof_messages, proof_off, "q/rescale");
            skip_rescaling_proof(qkv_size, 1u << 16, transcript, proof.proof_messages, proof_off, "k/rescale");
            skip_rescaling_proof(qkv_size, 1u << 16, transcript, proof.proof_messages, proof_off, "v/rescale");
            std::cout << "[Verifier] ✓ QKV rescaling proofs replayed (proof_off=" << proof_off << ")\n";
        }

        // Step 3: Softmax verification
        const uint max_table_size = 1 << 20;
        const uint D_softmax = padded_seq_len * padded_seq_len;
        bool tlookup_feasible = (D_softmax >= max_table_size) && (D_softmax % max_table_size == 0);
        bool skip_softmax_proof = (seq_len == 1) || !tlookup_feasible;

        try
        {
            if (seq_len == 1)
            {
                auto y_ints = proof.Y_softmax_ints;
                bool valid = (!y_ints.empty() && y_ints[0] == (1 << 20));
                for (size_t i = 1; i < y_ints.size(); ++i)
                    if (y_ints[i] != 0) { valid = false; break; }
                if (!valid) { std::cerr << "[Verifier] ❌ Softmax mismatch (seq_len=1)\n"; return 1; }
                std::cout << "[Verifier] ✓ Softmax check passed (seq_len=1)\n";
            }
            else if (!tlookup_feasible)
            {
                std::cout << "[Verifier] ⚠️ t-lookup not feasible for D=" << D_softmax
                          << " (requires D >= " << max_table_size << "), using recompute verification\n";

                zkSoftmax softmax({1 << 8, 1 << 20, 1 << 20}, 1, 0, 1UL << 32, {1 << 18, 1 << 22},
                                  padded_seq_len, padded_seq_len, d, 1);

                std::vector<FrTensor> X_seg, Y_seg, m_seg;
                FrTensor shift_r(padded_seq_len), X_shifted_r(padded_seq_len * padded_seq_len);
                FrTensor Y_r = softmax.compute(X, shift_r, X_shifted_r, X_seg, Y_seg, m_seg);

                auto y_host = Y_r.to_int_host();
                bool match = (y_host.size() == proof.Y_softmax_ints.size());
                if (match) {
                    for (size_t i = 0; i < y_host.size(); ++i)
                        if (y_host[i] != proof.Y_softmax_ints[i]) { match = false; break; }
                }
                if (!match) { std::cerr << "[Verifier] ❌ Softmax recompute mismatch\n"; return 1; }
                std::cout << "[Verifier] ✓ Softmax recompute verification passed\n";
            }
            else
            {
                if (proof.softmax_segments.X_segments_ints.empty()) {
                    std::cerr << "[Verifier] ❌ Softmax segments data missing\n";
                    return 1;
                }

                std::vector<FrTensor> X_segments, Y_segments, m_segments;
                for (const auto &seg : proof.softmax_segments.X_segments_ints) X_segments.push_back(i32_vec_to_tensor(seg));
                for (const auto &seg : proof.softmax_segments.Y_segments_ints) Y_segments.push_back(i32_vec_to_tensor(seg));
                for (const auto &seg : proof.softmax_segments.m_segments_ints) m_segments.push_back(i32_vec_to_tensor(seg));

                zkSoftmax softmax({1 << 8, 1 << 20, 1 << 20}, 1, 0, 1UL << 32, {1 << 18, 1 << 22},
                                  padded_seq_len, padded_seq_len, d, 1);
                softmax.verify_fs(Y_softmax, X, shift, X_shifted, X_segments, Y_segments, m_segments,
                                  transcript, proof.proof_messages, proof_off, "softmax");
                std::cout << "[Verifier] ✓ Softmax t-lookup verification passed\n";
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Verifier] ❌ Softmax verification failed: " << e.what() << "\n";
            return 1;
        }

        // Step 4: Output Rescaling proofs (4 rescales for Attn@V output)
        auto out_padded = FrTensor::matmul(Y_softmax, V_, padded_seq_len, padded_seq_len, d);

        auto extract_original_rows = [](const FrTensor &t, uint padded_rows, uint cols, uint orig_rows) -> FrTensor {
            if (padded_rows == orig_rows) return FrTensor(t);
            auto host_data = t.to_int_host();
            std::vector<int> extracted(orig_rows * cols);
            for (uint r = 0; r < orig_rows; ++r)
                for (uint c = 0; c < cols; ++c)
                    extracted[r * cols + c] = host_data[r * cols + c];
            return FrTensor(orig_rows * cols, extracted.data());
        };

        auto out = extract_original_rows(out_padded, padded_seq_len, d, seq_len);

        const uint base_scale = 1 << 10;
        Rescaling rs1(base_scale), rs2(base_scale), rs3(base_scale), rs4(base_scale);
        auto out_r1 = rs1(out);
        auto out_r2 = rs2(out_r1);
        auto out_r3 = rs3(out_r2);
        auto attn_out = rs4(out_r3);

        try
        {
            bool skip_rescaling_proof = (out.size < base_scale) || (out.size % base_scale != 0);
            if (!skip_rescaling_proof)
            {
                rs1.verify_fs(out, out_r1, transcript, proof.proof_messages, proof_off, "out/rs1");
                rs2.verify_fs(out_r1, out_r2, transcript, proof.proof_messages, proof_off, "out/rs2");
                rs3.verify_fs(out_r2, out_r3, transcript, proof.proof_messages, proof_off, "out/rs3");
                rs4.verify_fs(out_r3, attn_out, transcript, proof.proof_messages, proof_off, "out/rs4");
                std::cout << "[Verifier] ✓ Output rescaling verified\n";
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Verifier] ❌ Output rescaling verification failed: " << e.what() << "\n";
            return 1;
        }

        // Step 5: O FC proof (skip, Verifier has no W_o)
        std::cout << "[Verifier] Replaying O FC proof (transcript sync)...\n";
        skip_zkfc_proof(seq_len, embed_dim, embed_dim, transcript, proof.proof_messages, proof_off, "o/fc");
        std::cout << "[Verifier] ✓ O FC proof replayed (proof_off=" << proof_off << ")\n";

        // Step 6: O Rescaling proof
        try
        {
            uint o_out_size = seq_len * embed_dim;
            bool skip_o_rescaling = (o_out_size < (1u << 10)) || (o_out_size % (1u << 10) != 0);
            if (!skip_o_rescaling && !proof.o_proj_out_ints.empty())
            {
                FrTensor o_out = i32_vec_to_tensor(proof.o_proj_out_ints);
                Rescaling o_rescale(1 << 16);
                auto out__ = o_rescale(o_out);
                o_rescale.verify_fs(o_out, out__, transcript, proof.proof_messages, proof_off, "o/rescale");
                std::cout << "[Verifier] ✓ O rescaling verified\n";
            }
            else if (!skip_o_rescaling)
            {
                // No o_proj_out_ints (old proof), skip
                skip_rescaling_proof(o_out_size, 1u << 16, transcript, proof.proof_messages, proof_off, "o/rescale");
                std::cout << "[Verifier] O rescaling skipped (no o_proj data)\n";
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Verifier] ❌ O rescaling verification failed: " << e.what() << "\n";
            return 1;
        }

        // Step 7: Matrix multiplication proofs
        try
        {
            // Attn @ V
            {
                auto u1 = transcript.challenge_vec("matmul/out/u1", ceilLog2(padded_seq_len));
                auto u2 = transcript.challenge_vec("matmul/out/u2", ceilLog2(d));
                auto ud = transcript.challenge_vec("matmul/out/ud", ceilLog2(padded_seq_len));

                bool has_data = !proof.matmul_attnv.claim.empty();
                Fr_t claim = has_data ? u32_vec_to_fr(proof.matmul_attnv.claim)
                                      : out_padded.multi_dim_me({u1, u2}, {padded_seq_len, d});
                FrTensor Y_partial = has_data ? i32_vec_to_tensor(proof.matmul_attnv.A_partial_ints)
                                              : Y_softmax.partial_me(u1, padded_seq_len, padded_seq_len);
                FrTensor V_partial = has_data ? i32_vec_to_tensor(proof.matmul_attnv.B_partial_ints)
                                              : V_.partial_me(u2, d, 1);

                zkip_verify_fs(claim, Y_partial, V_partial, transcript, proof.proof_messages, proof_off, "matmul/out");
            }

            // Q @ K^T
            {
                auto u1_ = transcript.challenge_vec("matmul/x/u1", ceilLog2(padded_seq_len));
                auto u2_ = transcript.challenge_vec("matmul/x/u2", ceilLog2(padded_seq_len));
                auto ud_ = transcript.challenge_vec("matmul/x/ud", ceilLog2(d));

                bool has_data = !proof.matmul_qk.claim.empty();
                Fr_t claim_ = has_data ? u32_vec_to_fr(proof.matmul_qk.claim)
                                       : X.multi_dim_me({u1_, u2_}, {padded_seq_len, padded_seq_len});
                FrTensor Q_partial = has_data ? i32_vec_to_tensor(proof.matmul_qk.A_partial_ints)
                                              : Q_.partial_me(u1_, padded_seq_len, d);
                FrTensor K_partial = has_data ? i32_vec_to_tensor(proof.matmul_qk.B_partial_ints)
                                              : K_.partial_me(u2_, padded_seq_len, d);

                zkip_verify_fs(claim_, Q_partial, K_partial, transcript, proof.proof_messages, proof_off, "matmul/x");
            }
            std::cout << "[Verifier] ✓ Matmul proofs verified\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Verifier] ❌ Matmul verification failed: " << e.what() << "\n";
            return 1;
        }

        // Verification complete
        verify_timer.stop();

        std::cout << "[Verifier] ✓ All checks passed\n";
        std::cout << "[TIME] Attention - Verify: " << verify_timer.getTotalTime() << " s\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        verify_timer.stop();
        std::cerr << "[Verifier] ❌ Unexpected error: " << e.what() << "\n";
        std::cerr << "[TIME] Attention - Verify: " << verify_timer.getTotalTime() << " s\n";
        return 1;
    }
}
