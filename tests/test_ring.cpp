#include "ring.h"
#include "check.h"
#include <iostream>

using namespace inspire;

void test_ntt_roundtrip() {
    std::cout << "test_ntt_roundtrip... " << std::flush;
    std::mt19937_64 rng(42);
    RnsPoly p = sample_ternary(rng);
    auto orig0 = p.limbs[0], orig1 = p.limbs[1];
    p.to_ntt();
    INSPIRE_CHECK(p.is_ntt);
    p.to_coeff();
    INSPIRE_CHECK(!p.is_ntt);
    for (size_t i = 0; i < N; i++) {
        INSPIRE_CHECK(p.limbs[0][i] == orig0[i]);
        INSPIRE_CHECK(p.limbs[1][i] == orig1[i]);
    }
    std::cout << "PASS" << std::endl;
}

void test_poly_mul() {
    std::cout << "test_poly_mul... " << std::flush;
    // (1+X) * (1+X) = 1 + 2X + X^2
    RnsPoly a = RnsPoly::zero(), b = RnsPoly::zero();
    a.set_coeff(0, 1, 1); a.set_coeff(1, 1, 1);
    b.set_coeff(0, 1, 1); b.set_coeff(1, 1, 1);
    a.to_ntt(); b.to_ntt();
    a.mul_inplace(b);
    a.to_coeff();
    INSPIRE_CHECK(a.limbs[0][0] == 1 && a.limbs[0][1] == 2 && a.limbs[0][2] == 1);
    std::cout << "PASS" << std::endl;
}

void test_automorphism() {
    std::cout << "test_automorphism... " << std::flush;
    // f(X) = X, tau_5(f) = X^5
    RnsPoly f = RnsPoly::zero();
    f.set_coeff(1, 1, 1);
    RnsPoly g = automorphism(f, 5);
    INSPIRE_CHECK(g.limbs[0][5] == 1);
    for (size_t i = 0; i < N; i++)
        if (i != 5) INSPIRE_CHECK(g.limbs[0][i] == 0);
    std::cout << "PASS" << std::endl;
}

void test_decomp_reconstruction() {
    std::cout << "test_decomp_reconstruction... " << std::flush;
    std::mt19937_64 rng(123);
    RnsPoly f = expand_seed(reinterpret_cast<const uint8_t*>("testseed01234567890123456789012345678901234567890123456789012345"), "test", 0);

    auto digits = decomp(f);
    INSPIRE_CHECK(digits.size() == (size_t)D_EFF);

    // Reconstruct: sum digits[j] * B^{j+1} should approximate f (up to d0)
    // Check per-coefficient: |recon - f| <= B/2 (mod each prime)
    for (size_t i = 0; i < N; i++) {
        // Reconstruct in mod Q0
        uint64_t recon0 = 0;
        uint64_t Bj = BASE % Q0;
        for (int j = 0; j < D_EFF; j++) {
            recon0 = (recon0 + ((__uint128_t)digits[j].limbs[0][i] * Bj) % Q0) % Q0;
            Bj = ((__uint128_t)Bj * (BASE % Q0)) % Q0;
        }
        int64_t diff = (int64_t)recon0 - (int64_t)f.limbs[0][i];
        if (diff < 0) diff = -diff;
        if (diff > (int64_t)(Q0 / 2)) diff = Q0 - diff;
        INSPIRE_CHECK(diff <= (int64_t)(BASE / 2 + 1)); // allow small rounding
    }
    std::cout << "PASS" << std::endl;
}

void test_seed_deterministic() {
    std::cout << "test_seed_deterministic... " << std::flush;
    uint8_t seed[SEED_BYTES] = {};
    auto a = expand_seed(seed, "rlwe", 0);
    auto b = expand_seed(seed, "rlwe", 0);
    for (size_t i = 0; i < N; i++) {
        INSPIRE_CHECK(a.limbs[0][i] == b.limbs[0][i]);
        INSPIRE_CHECK(a.limbs[1][i] == b.limbs[1][i]);
    }
    // Different index gives different result
    auto c = expand_seed(seed, "rlwe", 1);
    bool diff = false;
    for (size_t i = 0; i < N; i++)
        if (a.limbs[0][i] != c.limbs[0][i]) { diff = true; break; }
    INSPIRE_CHECK(diff);
    std::cout << "PASS" << std::endl;
}

int main() {
    test_ntt_roundtrip();
    test_poly_mul();
    test_automorphism();
    test_decomp_reconstruction();
    test_seed_deterministic();
    std::cout << "All ring tests passed." << std::endl;
    return inspire_test_status();
}
