#ifndef ZKRELU_CUH
#define ZKRELU_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"
#include "fr-tensor.cuh" 
#include "tlookup.cuh"
#include "proof.cuh"
#include "transcript.cuh"

// zkReLU: Zero-knowledge proof ReLU activation
// Supports ReLU and SiLU/SwiGLU activations
class zkReLU {
public:
    uint scaling_factor;
    tLookupRange tl_rem; // table for remainder
    FrTensor decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem);
    FrTensor *sign_tensor_ptr, *abs_tensor_ptr, *rem_tensor_ptr, *m_tensor_ptr;

    zkReLU(uint scaling_factor);
    FrTensor operator()(const FrTensor& X);
    void prove(const FrTensor& Z, const FrTensor& A);
    
    // Fiat-Shamir version
    void prove_fs(const FrTensor& Z, const FrTensor& A, Transcript& transcript, 
                  std::vector<uint8_t>& proof_messages, const std::string& label_prefix);
    void verify_fs(const FrTensor& Z, const FrTensor& A, Transcript& transcript,
                   const std::vector<uint8_t>& proof_messages, size_t& proof_off,
                   const std::string& label_prefix);
    
    ~zkReLU();
};

// SwiGLU activation function (used in LLaMA/Qwen FFN)
// SwiGLU(x) = Swish(gate(x)) * up(x)
// where Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
class zkSwiGLU {
public:
    uint scaling_factor;
    tLookupRangeMapping tl_swiglu;  // SwiGLU lookup table
    FrTensor *m_tensor_ptr;
    
    zkSwiGLU(int min_val, int range, const FrTensor& swiglu_table);
    
    // Compute SwiGLU: out = swiglu(gate_out) * up_out
    std::pair<FrTensor, FrTensor> operator()(const FrTensor& gate_out);
    
    // Prove
    void prove(const FrTensor& gate_out, const FrTensor& swiglu_out, const FrTensor& m,
               const Fr_t& alpha, const Fr_t& beta, const Fr_t& gamma,
               const std::vector<Fr_t>& u, const std::vector<Fr_t>& v,
               std::vector<Polynomial>& proof);
    
    // Fiat-Shamir version
    void prove_fs(const FrTensor& gate_out, const FrTensor& swiglu_out, const FrTensor& m,
                  Transcript& transcript, std::vector<uint8_t>& proof_messages,
                  const std::string& label_prefix);
    
    ~zkSwiGLU();
};

#endif  // ZKRELU_CUH
