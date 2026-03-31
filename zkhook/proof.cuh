#ifndef PROOF_CUH
#define PROOF_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "bls12-381.cuh"
#include "polynomial.cuh"
#include "transcript.cuh"
#include "proof_stream.cuh"

#include <vector>
#include <random>

struct Claim {
    Fr_t claim;
    std::vector<std::vector<Fr_t>> u;
    std::vector<uint> dims;
};

struct Weight;
struct WeightPublic;

void verifyWeightClaim(const Weight& w, const Claim& c);

void verifyWeightClaimPublic(const WeightPublic& w, const Claim& c, 
    Transcript& transcript, const std::vector<uint8_t>& proof_messages, 
    size_t& proof_off, const std::string& label_prefix);

KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size);

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof);

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u);

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v);

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size);

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v);


bool operator==(const Fr_t& a, const Fr_t& b);
bool operator!=(const Fr_t& a, const Fr_t& b);


Fr_t multi_hadamard_sumchecks(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);

Fr_t multi_hadamard_sumchecks_prove_fs(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, Transcript& transcript, std::vector<uint8_t>& proof_messages, const std::string& label_prefix);
Fr_t multi_hadamard_sumchecks_verify_fs(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, Transcript& transcript, const std::vector<uint8_t>& proof_messages, size_t& proof_off, const std::string& label_prefix);

#endif
