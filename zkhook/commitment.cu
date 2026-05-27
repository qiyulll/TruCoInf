#include "commitment.cuh"
#include "proof_stream.cuh"

Commitment Commitment::random(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out;
}

G1TensorJacobian Commitment::commit(const FrTensor &t) const
{
    if (t.size % size != 0)
        throw std::runtime_error("Commitment::commit - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp = (*this) * t;
    return temp.rowwise_sum(m, size);
}

DEVICE G1Jacobian_t commit_int_dev_func(G1Jacobian_t a, Fr_t s)
{
    const int x = scalar_to_int(s);
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
#pragma unroll
    for (uint i = 0; i < 31; ++i)
    {
        if ((x >> i) & 1)
            out = blstrs__g1__G1Affine_add(out, a);
        a = blstrs__g1__G1Affine_double(a);
    }

    if (x < 0)
        out = blstrs__g1__G1Affine_add(out, G1Jacobian_minus(a));
    return out;
}

KERNEL void commit_int_kernel(const G1Jacobian_t *generators, const Fr_t *scalars, G1Jacobian_t *out, uint n, uint m)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n)
        return;
    out[gid] = commit_int_dev_func(generators[gid % n], scalars[gid]);
}

G1TensorJacobian Commitment::commit_int(const FrTensor &t) const
{
    if (t.size % size != 0)
        throw std::runtime_error("Commitment::commit_int - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp(t.size);
    commit_int_kernel<<<(m * size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(gpu_data, t.gpu_data, temp.gpu_data, size, m);
    cudaDeviceSynchronize();
    return temp.rowwise_sum(m, size);
}

G1TensorJacobian Commitment::commit_int_multi(const vector<FrTensor> &ts) const
{
    uint num_row = 0;
    for (auto &t : ts)
    {
        if (t.size % size != 0)
            throw std::runtime_error("Commitment::commit_int_multi - Incompatible dimensions");
        num_row += t.size / size;
    }

    G1TensorJacobian temp(num_row * size);
    auto temp_start = temp.gpu_data;
    for (auto &t : ts)
    {
        uint m = t.size / size;
        commit_int_kernel<<<(m * size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(gpu_data, t.gpu_data, temp_start, size, m);
        cudaDeviceSynchronize();
        temp_start += m * size;
    }
    return temp.rowwise_sum(temp.size / size, size);
}

KERNEL void me_open_step(GLOBAL Fr_t *scalars, GLOBAL G1Jacobian_t *generators, Fr_t u,
                         GLOBAL Fr_t *new_scalars, GLOBAL G1Jacobian_t *new_generators,
                         GLOBAL G1Jacobian_t *temp_out, GLOBAL G1Jacobian_t *temp_out0, GLOBAL G1Jacobian_t *temp_out1,
                         uint old_size, uint new_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size)
        return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    if (gid1 >= old_size)
    {
        new_scalars[gid] = blstrs__scalar__Scalar_sub(scalars[gid0],
                                                      blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, scalars[gid0])));
        new_generators[gid] = G1Jacobian_mul(generators[gid0], u);
        temp_out[gid] = G1Jacobian_mul(generators[gid0], scalars[gid0]);
        temp_out0[gid] = blstrs__g1__G1Affine_ZERO;
        temp_out1[gid] = blstrs__g1__G1Affine_ZERO;
        return;
    }

    new_scalars[gid] = blstrs__scalar__Scalar_add(scalars[gid0], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0]))));
    new_generators[gid] = blstrs__g1__G1Affine_add(generators[gid1], G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalars[gid0]), G1Jacobian_mul(generators[gid1], scalars[gid1]));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalars[gid0]);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalars[gid1]);
}

Fr_t Commitment::me_open(const FrTensor &t, const Commitment &generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t> &proof)
{
    if (t.size != generators.size)
        throw std::runtime_error("Commitment::me_open - Incompatible dimensions " + std::to_string(t.size) + " " + std::to_string(generators.size));
    if (begin >= end)
    {
        proof.push_back(generators(0));
        return t(0);
    }
    uint new_size = (t.size + 1) / 2;
    FrTensor new_scalars(new_size);
    Commitment new_generators(new_size);
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(t.gpu_data, generators.gpu_data, *begin,
                                                                              new_scalars.gpu_data, new_generators.gpu_data, temp.gpu_data, temp0.gpu_data, temp1.gpu_data,
                                                                              t.size, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
    return me_open(new_scalars, new_generators, begin + 1, end, proof);
}

Fr_t Commitment::open(const FrTensor &t, const G1TensorJacobian &com, const vector<Fr_t> &u) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1) ? com(0) : com(u_out);
    vector<G1Jacobian_t> proof;
    return me_open(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), proof);
}

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim, uint out_dim)
{
    Commitment generator(generator_filename);
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com(com_filename);
    return {generator, weight, com, in_dim, out_dim};
}

WeightPublic create_weight_public(string generator_filename, string com_filename, uint in_dim, uint out_dim)
{
    Commitment generator(generator_filename);
    G1TensorJacobian com(com_filename);
    return {generator, com, in_dim, out_dim};
}

Input create_input(const Commitment &generator, const FrTensor &data, uint seq_len, uint embed_dim)
{
    G1TensorJacobian com = generator.commit_int(data);
    return {generator, data, com, seq_len, embed_dim};
}

Input create_input_from_file(string generator_filename, string input_filename, uint seq_len, uint embed_dim)
{
    Commitment generator(generator_filename);
    FrTensor data = FrTensor::from_int_bin(input_filename);
    G1TensorJacobian com = generator.commit_int(data);
    return {generator, data, com, seq_len, embed_dim};
}

InputPublic create_input_public(string generator_filename, string com_filename, uint seq_len, uint embed_dim)
{
    Commitment generator(generator_filename);
    G1TensorJacobian com(com_filename);
    return {generator, com, seq_len, embed_dim};
}

InputPublic create_input_public_from_com(const Commitment &generator, const G1TensorJacobian &com, uint seq_len, uint embed_dim)
{
    return {generator, com, seq_len, embed_dim};
}

CommitmentOpeningProof generate_opening_proof(
    const FrTensor &f,
    const Commitment &generators,
    const G1TensorJacobian &com,
    const vector<Fr_t> &u,
    uint in_dim,
    uint out_dim)
{
    CommitmentOpeningProof proof;
    proof.u = u;

    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));

    auto f_partial = f.partial_me(u_out, f.size / com.size);

    Commitment::me_open(f_partial, generators, u_in.begin(), u_in.end(), proof.proof_points);

    proof.claimed_value = f.multi_dim_me({vector<Fr_t>(u.begin(), u.begin() + ceilLog2(in_dim)),
                                          vector<Fr_t>(u.begin() + ceilLog2(in_dim), u.end())},
                                         {in_dim, out_dim});

    return proof;
}

KERNEL void verify_fold_generators_kernel(
    GLOBAL G1Jacobian_t *generators,
    GLOBAL G1Jacobian_t *new_generators,
    Fr_t challenge,
    uint old_size, uint new_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size)
        return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    if (gid1 >= old_size)
    {
        new_generators[gid] = G1Jacobian_mul(generators[gid0], challenge);
    }
    else
    {
        new_generators[gid] = blstrs__g1__G1Affine_add(
            generators[gid1],
            G1Jacobian_mul(
                blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])),
                challenge));
    }
}

bool verify_opening_proof(
    const CommitmentOpeningProof &proof,
    const Commitment &generators,
    const G1TensorJacobian &com,
    uint in_dim,
    uint out_dim)
{
    const vector<Fr_t> &u = proof.u;
    if (u.size() < ceilLog2(com.size)) {
        std::cerr << "verify_opening_proof: u vector too small" << std::endl;
        return false;
    }
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));

    G1Jacobian_t com_folded = (com.size == 1) ? com(0) : com(u_out);

    uint num_rounds = ceilLog2(generators.size);
    if (proof.proof_points.size() != num_rounds * 3 + 1)
    {
        std::cerr << "verify_opening_proof: wrong number of proof points" << std::endl;
        return false;
    }

    G1Jacobian_t current_com = com_folded;

    for (uint round = 0; round < num_rounds; ++round)
    {
        uint idx = round * 3;
        G1Jacobian_t temp = proof.proof_points[idx];
        G1Jacobian_t temp0 = proof.proof_points[idx + 1];
        G1Jacobian_t temp1 = proof.proof_points[idx + 2];

        Fr_t challenge = u_in[round];

        FrTensor challenge_tensor(1, &challenge);
        FrTensor challenge_sq_tensor = challenge_tensor * challenge_tensor;
        Fr_t challenge_sq = challenge_sq_tensor(0);

        G1TensorJacobian temp_tensor(1, &temp);
        G1TensorJacobian temp0_tensor(1, &temp0);
        G1TensorJacobian temp1_tensor(1, &temp1);
        FrTensor challenge_fr(1, &challenge);
        FrTensor challenge_sq_fr(1, &challenge_sq);

        G1TensorJacobian term1 = temp0_tensor * challenge_fr;
        G1TensorJacobian term2 = temp1_tensor * challenge_sq_fr;
        G1TensorJacobian sum1 = temp_tensor + term1;
        G1TensorJacobian expected_next_tensor = sum1 + term2;
        current_com = expected_next_tensor(0);
    }

    G1Jacobian_t final_generator = proof.proof_points.back();
    G1TensorJacobian final_gen_tensor(1, &final_generator);
    FrTensor claimed_tensor(1, &proof.claimed_value);
    G1TensorJacobian expected_com_tensor = final_gen_tensor * claimed_tensor;

    return true;
}

void serialize_opening_proof(const CommitmentOpeningProof &proof, std::vector<uint8_t> &out)
{
    ps_write_fr(out, proof.claimed_value);

    ps_write_u32(out, static_cast<uint32_t>(proof.u.size()));
    for (const auto &ui : proof.u)
    {
        ps_write_fr(out, ui);
    }

    ps_write_g1_vec(out, proof.proof_points);
}

CommitmentOpeningProof deserialize_opening_proof(const std::vector<uint8_t> &in, size_t &off)
{
    CommitmentOpeningProof proof;

    proof.claimed_value = ps_read_fr(in, off);

    uint32_t u_size = ps_read_u32(in, off);
    proof.u.resize(u_size);
    for (uint32_t i = 0; i < u_size; ++i)
    {
        proof.u[i] = ps_read_fr(in, off);
    }

    proof.proof_points = ps_read_g1_vec(in, off);

    return proof;
}
