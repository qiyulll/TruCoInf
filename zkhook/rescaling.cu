#include "rescaling.cuh"

Rescaling::Rescaling(uint scaling_factor): scaling_factor(scaling_factor), tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), rem_tensor_ptr(nullptr)
{
}

// void decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem, FrTensor& rem_ind);
KERNEL void rescaling_kernel(Fr_t* in_ptr, Fr_t* out_ptr, Fr_t* rem_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {   
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(in_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = (temp < 0 ? temp + scaling_factor : temp) - hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;
        out_ptr[tid] = long_to_scalar(x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
    }
}

FrTensor Rescaling::operator()(const FrTensor& X)
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);

    FrTensor out(X.size);
    uint block_size = 256;
    rescaling_kernel<<<(X.size + block_size - 1) / block_size, block_size>>>(X.gpu_data, out.gpu_data, rem_tensor_ptr->gpu_data, scaling_factor, X.size);
    cudaDeviceSynchronize();
    
    return out;
}

Rescaling::~Rescaling()
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
}

vector<Claim> Rescaling::prove(const FrTensor& X, const FrTensor& X_)
{
    if (X.size != X_.size)
    {
        throw std::runtime_error("Error: the size of X and X_ should be the same.");
    }

    auto u = random_vec(ceilLog2(X.size));
    auto v = random_vec(ceilLog2(X.size));
    auto rand_temp = random_vec(2);
    vector<Polynomial> proof;
    return prove(X, X_, u, v, rand_temp[0], rand_temp[1], proof);
}

vector<Claim> Rescaling::prove(
    const FrTensor& X,
    const FrTensor& X_,
    const vector<Fr_t>& u,
    const vector<Fr_t>& v,
    const Fr_t& alpha,
    const Fr_t& beta,
    vector<Polynomial>& proof)
{
    if (X.size != X_.size) throw std::runtime_error("Rescaling::prove: size mismatch");
    if (!rem_tensor_ptr) throw std::runtime_error("Rescaling::prove: rem_tensor_ptr is null (call operator() first)");
    if (u.size() != ceilLog2(X.size)) throw std::runtime_error("Rescaling::prove: u size mismatch");
    if (v.size() != ceilLog2(X.size)) throw std::runtime_error("Rescaling::prove: v size mismatch");

    auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
    auto m = tl_rem.prep(rem);

    if (X(u) != X_(u) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + rem(u)) {
        throw std::runtime_error("Rescaling::prove: rem relation check failed");
    }

    tl_rem.prove(rem, m, alpha, beta, u, v, proof);
    cout << "Rescaling proof complete." << endl;
    return {};
}

vector<Claim> Rescaling::prove_fs(const FrTensor& X, const FrTensor& X_, Transcript& transcript, std::vector<uint8_t>& proof_messages, const std::string& label_prefix)
{
    if (X.size != X_.size) throw std::runtime_error("Rescaling::prove_fs: size mismatch");
    if (!rem_tensor_ptr) throw std::runtime_error("Rescaling::prove_fs: rem_tensor_ptr is null (call operator() first)");

    auto u = transcript.challenge_vec(label_prefix + "/rescale/u", ceilLog2(X.size));
    auto alpha = transcript.challenge_fr(label_prefix + "/rescale/alpha");
    auto beta = transcript.challenge_fr(label_prefix + "/rescale/beta");

    auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
    auto m = tl_rem.prep(rem);

    if (X(u) != X_(u) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + rem(u)) {
        throw std::runtime_error("Rescaling::prove_fs: rem relation check failed");
    }

    tl_rem.prove_fs(rem, m, alpha, beta, u, transcript, proof_messages, label_prefix + "/rescale/tlookup");
    cout << "Rescaling proof complete." << endl;
    return {};
}

vector<Claim> Rescaling::verify_fs(const FrTensor& X, const FrTensor& X_, Transcript& transcript, const std::vector<uint8_t>& proof_messages, size_t& proof_off, const std::string& label_prefix)
{
    if (X.size != X_.size) throw std::runtime_error("Rescaling::verify_fs: size mismatch");
    if (!rem_tensor_ptr) throw std::runtime_error("Rescaling::verify_fs: rem_tensor_ptr is null (call operator() first)");

    auto u = transcript.challenge_vec(label_prefix + "/rescale/u", ceilLog2(X.size));
    auto alpha = transcript.challenge_fr(label_prefix + "/rescale/alpha");
    auto beta = transcript.challenge_fr(label_prefix + "/rescale/beta");

    auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
    auto m = tl_rem.prep(rem);

    if (X(u) != X_(u) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + rem(u)) {
        throw std::runtime_error("Rescaling::verify_fs: rem relation check failed");
    }

    tl_rem.verify_fs(rem, m, alpha, beta, u, transcript, proof_messages, proof_off, label_prefix + "/rescale/tlookup");
    cout << "Rescaling proof complete." << endl;
    return {};
}