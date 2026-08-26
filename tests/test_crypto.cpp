#include "crypto.h"
#include <iostream>
#include "check.h"

using namespace inspire;

void test_rlwe_enc_dec() {
    std::cout << "test_rlwe_enc_dec... " << std::flush;
    std::mt19937_64 rng(42);
    uint8_t seed[SEED_BYTES] = {};

    RnsPoly s = sk_gen(rng);
    RnsPoly s_ntt = s; s_ntt.to_ntt();

    // Encrypt m(X) = 5 + 3X
    RnsPoly a = expand_seed(seed, "test", 0);
    RnsPoly e = sample_error(rng);
    RnsPoly m = RnsPoly::zero();
    m.set_coeff(0, ((__uint128_t)5 * DELTA_MOD_Q0) % Q0,
                   ((__uint128_t)5 * DELTA_MOD_Q1) % Q1);
    m.set_coeff(1, ((__uint128_t)3 * DELTA_MOD_Q0) % Q0,
                   ((__uint128_t)3 * DELTA_MOD_Q1) % Q1);

    RnsPoly a_ntt = a; a_ntt.to_ntt();
    RnsPoly b_ntt = a_ntt; b_ntt.mul_inplace(s_ntt);
    b_ntt.to_coeff();
    b_ntt.add_inplace(e);
    b_ntt.add_inplace(m);

    RlweCt ct;
    ct.a = a; ct.a.to_ntt();
    ct.b = b_ntt; ct.b.to_ntt();

    auto result = rlwe_dec(s_ntt, ct);
    INSPIRE_CHECK(result[0] == 5);
    INSPIRE_CHECK(result[1] == 3);
    INSPIRE_CHECK(result[2] == 0);
    std::cout << "PASS (m[0]=" << result[0] << ", m[1]=" << result[1] << ")" << std::endl;
}

void test_lwe_enc_b() {
    std::cout << "test_lwe_enc_b... " << std::flush;
    std::mt19937_64 rng(55);
    uint8_t seed[SEED_BYTES] = {};

    RnsPoly s = sk_gen(rng);
    size_t ell = 2 * N;
    auto lwe = lwe_enc_b(s, 42, ell, seed, rng);
    INSPIRE_CHECK(lwe.b_limb0.size() == ell);
    INSPIRE_CHECK(lwe.b_limb1.size() == ell);
    std::cout << "PASS (generated " << ell << " b-values)" << std::endl;
}

void test_ext_prod() {
    std::cout << "test_ext_prod... " << std::flush;
    std::mt19937_64 rng(77);
    uint8_t seed[SEED_BYTES] = {};

    RnsPoly s = sk_gen(rng);
    RnsPoly s_ntt = s; s_ntt.to_ntt();

    // Encrypt mu = 1 (constant polynomial) via RGSW
    RnsPoly mu = RnsPoly::zero();
    mu.set_coeff(0, 1, 1);
    auto rgsw = rgsw_enc_b(s, mu, seed, rng);

    // Encrypt m = 42 via RLWE
    RnsPoly a = expand_seed(seed, "test_ext", 0);
    RnsPoly e = sample_error(rng);
    RnsPoly m_enc = RnsPoly::zero();
    m_enc.set_coeff(0, ((__uint128_t)42 * DELTA_MOD_Q0) % Q0,
                       ((__uint128_t)42 * DELTA_MOD_Q1) % Q1);
    RnsPoly a_ntt = a; a_ntt.to_ntt();
    RnsPoly b_ntt = a_ntt; b_ntt.mul_inplace(s_ntt);
    b_ntt.to_coeff();
    b_ntt.add_inplace(e);
    b_ntt.add_inplace(m_enc);

    RlweCt ct;
    ct.a = a; ct.a.to_ntt();
    ct.b = b_ntt; ct.b.to_ntt();

    // ExtProd(RGSW(1), RLWE(42)) should give RLWE(1*42) = RLWE(42)
    auto result_ct = ext_prod(rgsw, ct);
    auto dec = rlwe_dec(s_ntt, result_ct);
    std::cout << "dec[0]=" << dec[0] << " (expect 42)" << std::endl;
    INSPIRE_CHECK(dec[0] == 42);
    std::cout << "PASS" << std::endl;
}

int main() {
    test_rlwe_enc_dec();
    test_lwe_enc_b();
    test_ext_prod();
    std::cout << "All crypto tests passed." << std::endl;
    return inspire_test_status();
}
