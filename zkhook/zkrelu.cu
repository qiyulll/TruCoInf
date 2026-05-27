#include "zkrelu.cuh"
#include "proof_stream.cuh"

// ==================== zkReLU Implementation ====================

zkReLU::zkReLU(uint scaling_factor): 
    scaling_factor(scaling_factor), 
    tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), 
    sign_tensor_ptr(nullptr), 
    abs_tensor_ptr(nullptr), 
    rem_tensor_ptr(nullptr), 
    m_tensor_ptr(nullptr)
{
}

KERNEL void zkrelu_decomp_kernel(Fr_t* X_ptr, Fr_t* sign_ptr, Fr_t* abs_ptr, Fr_t* rem_ptr, Fr_t* res_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {   
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(X_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = temp < 0 ? temp + scaling_factor : temp;
        x_rem -= hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;

        bool pos = x_rescaled >= 0;

        sign_ptr[tid] = {static_cast<uint>(pos), 0, 0, 0, 0, 0, 0, 0};
        abs_ptr[tid] = pos? long_to_scalar(x_rescaled) : long_to_scalar(-x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
        res_ptr[tid] = pos? long_to_scalar(x_rescaled) : blstrs__scalar__Scalar_ZERO;
    }
}

FrTensor zkReLU::decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem)
{
    uint N = X.size;
    FrTensor res(N);
    uint block_size = 256;
    uint grid_size = (N + block_size - 1) / block_size;
    zkrelu_decomp_kernel<<<grid_size, block_size>>>(X.gpu_data, sign.gpu_data, abs.gpu_data, rem.gpu_data, res.gpu_data, static_cast<long>(scaling_factor), N);
    cudaDeviceSynchronize();
    return res;
}

FrTensor zkReLU::operator()(const FrTensor& X)
{
    if (sign_tensor_ptr) delete sign_tensor_ptr;
    sign_tensor_ptr = new FrTensor(X.size);
    if (abs_tensor_ptr) delete abs_tensor_ptr;
    abs_tensor_ptr = new FrTensor(X.size);
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);

    FrTensor res = decomp(X, *sign_tensor_ptr, *abs_tensor_ptr, *rem_tensor_ptr);
    m_tensor_ptr = new FrTensor(tl_rem.prep(*rem_tensor_ptr));
    return res;
}

void zkReLU::prove(const FrTensor& Z, const FrTensor& A)
{
    // Basic proof logic (using random challenges)
    auto u = random_vec(ceilLog2(Z.size));
    auto v = random_vec(ceilLog2(Z.size));
    auto temp_rand = random_vec(2);
    std::vector<Polynomial> proof;
    
    if (rem_tensor_ptr) {
        auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
        auto m = tl_rem.prep(rem);
        tl_rem.prove(rem, m, temp_rand[0], temp_rand[1], u, v, proof);
    }
    
    std::cout << "zkReLU proof complete." << std::endl;
}

void zkReLU::prove_fs(const FrTensor& Z, const FrTensor& A, Transcript& transcript, 
                      std::vector<uint8_t>& proof_messages, const std::string& label_prefix)
{
    auto u = transcript.challenge_vec(label_prefix + "/relu/u", ceilLog2(Z.size));
    auto alpha = transcript.challenge_fr(label_prefix + "/relu/alpha");
    auto beta = transcript.challenge_fr(label_prefix + "/relu/beta");
    
    if (rem_tensor_ptr) {
        auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
        auto m = tl_rem.prep(rem);
        tl_rem.prove_fs(rem, m, alpha, beta, u, transcript, proof_messages, 
                        label_prefix + "/relu/tlookup");
    }
    
    std::cout << "zkReLU FS proof complete." << std::endl;
}

void zkReLU::verify_fs(const FrTensor& Z, const FrTensor& A, Transcript& transcript,
                       const std::vector<uint8_t>& proof_messages, size_t& proof_off,
                       const std::string& label_prefix)
{
    auto u = transcript.challenge_vec(label_prefix + "/relu/u", ceilLog2(Z.size));
    auto alpha = transcript.challenge_fr(label_prefix + "/relu/alpha");
    auto beta = transcript.challenge_fr(label_prefix + "/relu/beta");
    
    if (rem_tensor_ptr) {
        auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
        auto m = tl_rem.prep(rem);
        tl_rem.verify_fs(rem, m, alpha, beta, u, transcript, proof_messages, proof_off,
                         label_prefix + "/relu/tlookup");
    }
    
    std::cout << "zkReLU FS verify complete." << std::endl;
}

zkReLU::~zkReLU()
{
    if (sign_tensor_ptr) delete sign_tensor_ptr;
    if (abs_tensor_ptr) delete abs_tensor_ptr;
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    if (m_tensor_ptr) delete m_tensor_ptr;
}

// ==================== zkSwiGLU Implementation ====================

zkSwiGLU::zkSwiGLU(int min_val, int range, const FrTensor& swiglu_table):
    scaling_factor(range),
    tl_swiglu(min_val, range, swiglu_table),
    m_tensor_ptr(nullptr)
{
}

std::pair<FrTensor, FrTensor> zkSwiGLU::operator()(const FrTensor& gate_out)
{
    // Compute SwiGLU lookup
    auto result = tl_swiglu(gate_out);
    
    // Save m tensor for proof
    if (m_tensor_ptr) delete m_tensor_ptr;
    m_tensor_ptr = new FrTensor(result.second);
    
    return result;
}

void zkSwiGLU::prove(const FrTensor& gate_out, const FrTensor& swiglu_out, const FrTensor& m,
                     const Fr_t& alpha, const Fr_t& beta, const Fr_t& gamma,
                     const std::vector<Fr_t>& u, const std::vector<Fr_t>& v,
                     std::vector<Polynomial>& proof)
{
    tl_swiglu.prove(gate_out, swiglu_out, m, alpha, beta, gamma, u, v, proof);
    std::cout << "SwiGLU proof complete." << std::endl;
}

void zkSwiGLU::prove_fs(const FrTensor& gate_out, const FrTensor& swiglu_out, const FrTensor& m,
                        Transcript& transcript, std::vector<uint8_t>& proof_messages,
                        const std::string& label_prefix)
{
    auto u = transcript.challenge_vec(label_prefix + "/swiglu/u", ceilLog2(gate_out.size));
    auto v = transcript.challenge_vec(label_prefix + "/swiglu/v", ceilLog2(gate_out.size));
    auto alpha = transcript.challenge_fr(label_prefix + "/swiglu/alpha");
    auto beta = transcript.challenge_fr(label_prefix + "/swiglu/beta");
    auto gamma = transcript.challenge_fr(label_prefix + "/swiglu/gamma");
    
    tl_swiglu.prove_fs(gate_out, swiglu_out, m, alpha, beta, gamma, u, transcript, 
                       proof_messages, label_prefix + "/swiglu/tlookup");
    
    std::cout << "SwiGLU FS proof complete." << std::endl;
}

zkSwiGLU::~zkSwiGLU()
{
    if (m_tensor_ptr) delete m_tensor_ptr;
}
