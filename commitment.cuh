#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"

class Commitment: public G1TensorJacobian
{   
    public:
    using G1TensorJacobian::G1TensorJacobian;

    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;
    using G1TensorJacobian::operator*=;

    G1TensorJacobian commit(const FrTensor& t) const;
    G1TensorJacobian commit_int (const FrTensor& t) const;
    G1TensorJacobian commit_int_multi(const vector<FrTensor>& t) const;

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u) const;

    static Commitment random(uint size);
    static Fr_t me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);
};

struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
    uint in_dim;
    uint out_dim;
};

struct WeightPublic {
    Commitment generator;
    G1TensorJacobian com;
    uint in_dim;
    uint out_dim;
};

struct Input {
    Commitment generator;
    FrTensor data;
    G1TensorJacobian com;
    uint seq_len;
    uint embed_dim;
};

struct InputPublic {
    Commitment generator;
    G1TensorJacobian com;
    uint seq_len;
    uint embed_dim;
};

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim, uint out_dim);

WeightPublic create_weight_public(string generator_filename, string com_filename, uint in_dim, uint out_dim);

Input create_input(const Commitment& generator, const FrTensor& data, uint seq_len, uint embed_dim);

Input create_input_from_file(string generator_filename, string input_filename, uint seq_len, uint embed_dim);

InputPublic create_input_public(string generator_filename, string com_filename, uint seq_len, uint embed_dim);

InputPublic create_input_public_from_com(const Commitment& generator, const G1TensorJacobian& com, uint seq_len, uint embed_dim);

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u,
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size);

struct CommitmentOpeningProof {
    Fr_t claimed_value;
    vector<G1Jacobian_t> proof_points;
    vector<Fr_t> u;
};

CommitmentOpeningProof generate_opening_proof(
    const FrTensor& f,
    const Commitment& generators,
    const G1TensorJacobian& com,
    const vector<Fr_t>& u,
    uint in_dim,
    uint out_dim
);

bool verify_opening_proof(
    const CommitmentOpeningProof& proof,
    const Commitment& generators,
    const G1TensorJacobian& com,
    uint in_dim,
    uint out_dim
);

void serialize_opening_proof(const CommitmentOpeningProof& proof, std::vector<uint8_t>& out);
CommitmentOpeningProof deserialize_opening_proof(const std::vector<uint8_t>& in, size_t& off);

#endif
