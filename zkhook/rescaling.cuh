#ifndef RESCALING_CUH
#define RESCALING_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "tlookup.cuh"
#include "proof.cuh"

class Rescaling {
public:
    uint scaling_factor;
    tLookupRange tl_rem; // table for remainder
    Rescaling decomp(const FrTensor& X, FrTensor& rem);
    FrTensor *rem_tensor_ptr;

    Rescaling(uint scaling_factor);
    FrTensor operator()(const FrTensor& X);
    vector<Claim> prove(const FrTensor& X, const FrTensor& X_);

    // Deterministic / externally-challenged variant (for Fiat–Shamir).
    // Appends lookup/sumcheck polynomials to `proof` (note: current tLookup implementation does not record).
    vector<Claim> prove(
        const FrTensor& X,
        const FrTensor& X_,
        const vector<Fr_t>& u,
        const vector<Fr_t>& v,
        const Fr_t& alpha,
        const Fr_t& beta,
        vector<Polynomial>& proof);

    // Transcript-based proof/verify that emits/consumes tLookup polynomials into/from `proof_messages`.
    vector<Claim> prove_fs(const FrTensor& X, const FrTensor& X_, Transcript& transcript, std::vector<uint8_t>& proof_messages, const std::string& label_prefix);
    vector<Claim> verify_fs(const FrTensor& X, const FrTensor& X_, Transcript& transcript, const std::vector<uint8_t>& proof_messages, size_t& proof_off, const std::string& label_prefix);
    ~Rescaling();
};

#endif // RESCALING_CUH