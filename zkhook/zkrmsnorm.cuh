/**
 * zkrmsnorm.cuh - RMSNorm Zero-Knowledge Proof
 * 
 * RMSNorm(X) = X * rms_inv * gamma
 * where:
 *   rms_inv[i] = 1 / sqrt(mean(X[i]^2) + eps)
 *   gamma is the learnable weight (embed_dim)
 * 
 * Proof flow:
 *   1. g_inv_rms = gamma * rms_inv  (zkFC, weight is gamma)
 *   2. Y = g_inv_rms * X            (Hadamard product)
 *   3. Y_ = rescale(Y)              (quantization scaling)
 */

#ifndef ZKRMSNORM_CUH
#define ZKRMSNORM_CUH

#include "fr-tensor.cuh"
#include "zkfc.cuh"
#include "rescaling.cuh"
#include "proof.cuh"
#include "commitment.cuh"

class zkRMSNorm {
public:
    uint embed_dim;
    zkFC gamma_layer;
    Rescaling rs1;
    Rescaling rs2;
    
    FrTensor* g_inv_rms_ptr = nullptr;
    FrTensor* g_inv_rms_scaled_ptr = nullptr;
    FrTensor* Y_ptr = nullptr;
    FrTensor* Y_scaled_ptr = nullptr;
    
    zkRMSNorm(uint embed_dim, const FrTensor& gamma_weight)
        : embed_dim(embed_dim),
          gamma_layer(1, embed_dim, gamma_weight),
          rs1(1 << 16),
          rs2(1 << 16)
    {}
    
    ~zkRMSNorm() {
        if (g_inv_rms_ptr) delete g_inv_rms_ptr;
        if (g_inv_rms_scaled_ptr) delete g_inv_rms_scaled_ptr;
        if (Y_ptr) delete Y_ptr;
        if (Y_scaled_ptr) delete Y_scaled_ptr;
    }
    
    FrTensor compute(const FrTensor& X, const FrTensor& rms_inv) {
        uint seq_len = X.size / embed_dim;
        
        if (g_inv_rms_ptr) { delete g_inv_rms_ptr; g_inv_rms_ptr = nullptr; }
        if (g_inv_rms_scaled_ptr) { delete g_inv_rms_scaled_ptr; g_inv_rms_scaled_ptr = nullptr; }
        if (Y_ptr) { delete Y_ptr; Y_ptr = nullptr; }
        if (Y_scaled_ptr) { delete Y_scaled_ptr; Y_scaled_ptr = nullptr; }
        
        g_inv_rms_ptr = new FrTensor(gamma_layer(rms_inv));
        g_inv_rms_scaled_ptr = new FrTensor(rs1(*g_inv_rms_ptr));
        
        Y_ptr = new FrTensor(*g_inv_rms_scaled_ptr * X);
        Y_scaled_ptr = new FrTensor(rs2(*Y_ptr));
        
        return *Y_scaled_ptr;
    }
    
    Claim prove(const FrTensor& X, const FrTensor& rms_inv) {
        if (!g_inv_rms_ptr || !g_inv_rms_scaled_ptr || !Y_ptr || !Y_scaled_ptr) {
            throw std::runtime_error("zkRMSNorm::prove - must call compute() first");
        }
        
        rs2.prove(*Y_ptr, *Y_scaled_ptr);
        
        auto u = random_vec(ceilLog2(Y_ptr->size));
        auto v = random_vec(ceilLog2(Y_ptr->size));
        hadamard_product_sumcheck(*g_inv_rms_scaled_ptr, X, u, v);
        
        rs1.prove(*g_inv_rms_ptr, *g_inv_rms_scaled_ptr);
        
        auto weight_claim = gamma_layer.prove(rms_inv, *g_inv_rms_ptr)[0];
        
        return weight_claim;
    }
};

#endif // ZKRMSNORM_CUH
