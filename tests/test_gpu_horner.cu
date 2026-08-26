// Test GPU horner_eval against CPU horner_eval.
#include "check.h"
#include "ring.h"
#include "crypto.h"
#include "gpu.cuh"
#include <iostream>
#include <random>
#include <vector>

using namespace inspire;

static std::vector<uint32_t> rnspoly_to_u32_coeff(const RnsPoly& p_in) {
    RnsPoly p = p_in;
    if (p.is_ntt) p.to_coeff();
    std::vector<uint32_t> out(2 * N);
    for (size_t i = 0; i < N; i++) {
        out[i]     = (uint32_t)p.limbs[0][i];
        out[N + i] = (uint32_t)p.limbs[1][i];
    }
    return out;
}

int main() {
    std::cout << "=== GPU horner_eval vs CPU horner_eval ===" << std::endl;

    constexpr size_t D_TEST = 8;  // small D for quick test

    std::mt19937_64 rng(11);
    uint8_t seed[SEED_BYTES] = {};
    for (auto& b : seed) b = rng() & 0xFF;

    // Generate sk + RGSW
    RnsPoly s = sk_gen(rng);
    RnsPoly s_ntt = s; s_ntt.to_ntt();
    RnsPoly mu = RnsPoly::zero();
    mu.set_coeff(0, 1, 1);  // mu = X^0 = 1
    RgswCt rgsw = rgsw_enc_b(s, mu, seed, rng);

    // Generate D random RLWE ciphertexts in NTT form
    std::vector<RlweCt> cts(D_TEST);
    for (size_t k = 0; k < D_TEST; k++) {
        RnsPoly a_poly = expand_seed(seed, "horner", k);
        RnsPoly e_poly = sample_error(rng);
        RnsPoly m_poly = RnsPoly::zero();
        for (size_t i = 0; i < N; i++) {
            uint64_t v = rng() % P;
            m_poly.limbs[0][i] = (v * DELTA_MOD_Q0) % Q0;
            m_poly.limbs[1][i] = (v * DELTA_MOD_Q1) % Q1;
        }
        RnsPoly a_ntt = a_poly; a_ntt.to_ntt();
        RnsPoly b_ntt = a_ntt; b_ntt.mul_inplace(s_ntt);
        b_ntt.to_coeff();
        b_ntt.add_inplace(e_poly);
        b_ntt.add_inplace(m_poly);
        cts[k].a = a_poly;
        cts[k].a.to_ntt();
        cts[k].b = b_ntt;
        cts[k].b.to_ntt();
    }

    // CPU reference
    RlweCt ref = horner_eval(cts, rgsw, D_TEST);
    if (ref.a.is_ntt) ref.a.to_coeff();
    if (ref.b.is_ntt) ref.b.to_coeff();

    // ===== GPU =====
    std::vector<uint32_t> fwd_q0, inv_q0, fwd_q1, inv_q1;
    uint32_t inv_n_q0, inv_n_q1;
    get_ntt_twiddles(0, fwd_q0, inv_q0, inv_n_q0);
    get_ntt_twiddles(1, fwd_q1, inv_q1, inv_n_q1);

    uint32_t *d_fwd_q0, *d_inv_q0, *d_fwd_q1, *d_inv_q1;
    cudaMalloc(&d_fwd_q0, N * sizeof(uint32_t));
    cudaMalloc(&d_inv_q0, N * sizeof(uint32_t));
    cudaMalloc(&d_fwd_q1, N * sizeof(uint32_t));
    cudaMalloc(&d_inv_q1, N * sizeof(uint32_t));
    cudaMemcpy(d_fwd_q0, fwd_q0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_q0, inv_q0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fwd_q1, fwd_q1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_q1, inv_q1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Upload all D packed ciphertexts (each in CPU NTT form, must convert)
    uint32_t *d_packed_a, *d_packed_b;
    cudaMalloc(&d_packed_a, D_TEST * 2 * N * sizeof(uint32_t));
    cudaMalloc(&d_packed_b, D_TEST * 2 * N * sizeof(uint32_t));
    for (size_t k = 0; k < D_TEST; k++) {
        auto a_coeff = rnspoly_to_u32_coeff(cts[k].a);
        auto b_coeff = rnspoly_to_u32_coeff(cts[k].b);
        // Upload, then NTT each limb on GPU
        cudaMemcpy(d_packed_a + k * 2 * N, a_coeff.data(), 2 * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_packed_b + k * 2 * N, b_coeff.data(), 2 * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        inspire::gpu::gpu_ntt_forward(d_packed_a + k * 2 * N,         N, Q0, d_fwd_q0);
        inspire::gpu::gpu_ntt_forward(d_packed_a + k * 2 * N + N,     N, Q1, d_fwd_q1);
        inspire::gpu::gpu_ntt_forward(d_packed_b + k * 2 * N,         N, Q0, d_fwd_q0);
        inspire::gpu::gpu_ntt_forward(d_packed_b + k * 2 * N + N,     N, Q1, d_fwd_q1);
    }

    // Upload RGSW (same helper as test_gpu_extprod)
    auto upload_rgsw_part = [&](const std::vector<RnsPoly>& parts, uint32_t** d_out) {
        cudaMalloc(d_out, 2 * D_EFF * N * sizeof(uint32_t));
        for (int j = 0; j < D_EFF; j++) {
            auto coeff = rnspoly_to_u32_coeff(parts[j]);
            cudaMemcpy(*d_out + j * N,                 coeff.data(),     N * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(*d_out + D_EFF * N + j * N,     coeff.data() + N, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        }
        for (int j = 0; j < D_EFF; j++)
            inspire::gpu::gpu_ntt_forward(*d_out + j * N, N, Q0, d_fwd_q0);
        for (int j = 0; j < D_EFF; j++)
            inspire::gpu::gpu_ntt_forward(*d_out + D_EFF * N + j * N, N, Q1, d_fwd_q1);
    };
    uint32_t *d_top_a, *d_top_b, *d_bot_a, *d_bot_b;
    upload_rgsw_part(rgsw.top.a_parts, &d_top_a);
    upload_rgsw_part(rgsw.top.b_parts, &d_top_b);
    upload_rgsw_part(rgsw.bottom.a_parts, &d_bot_a);
    upload_rgsw_part(rgsw.bottom.b_parts, &d_bot_b);
    cudaDeviceSynchronize();

    // Allocate output and scratch
    uint32_t *d_acc_a, *d_acc_b, *d_scratch_coeff, *d_dig, *d_res_a, *d_res_b;
    cudaMalloc(&d_acc_a,         2 * N * sizeof(uint32_t));
    cudaMalloc(&d_acc_b,         2 * N * sizeof(uint32_t));
    cudaMalloc(&d_res_a,         2 * N * sizeof(uint32_t));
    cudaMalloc(&d_res_b,         2 * N * sizeof(uint32_t));
    cudaMalloc(&d_scratch_coeff, 4 * N * sizeof(uint32_t));
    cudaMalloc(&d_dig,           4 * D_EFF * N * sizeof(uint32_t));

    // Run GPU horner
    inspire::gpu::gpu_horner_eval(
        d_acc_a, d_acc_b, d_packed_a, d_packed_b, D_TEST,
        d_top_a, d_top_b, d_bot_a, d_bot_b,
        d_fwd_q0, d_inv_q0, inv_n_q0,
        d_fwd_q1, d_inv_q1, inv_n_q1,
        d_scratch_coeff, d_dig, d_res_a, d_res_b,
        D_GSW, D_EFF, BASE_LOG, N);

    // INTT result
    inspire::gpu::gpu_ntt_inverse(d_acc_a,     N, Q0, d_inv_q0, inv_n_q0);
    inspire::gpu::gpu_ntt_inverse(d_acc_a + N, N, Q1, d_inv_q1, inv_n_q1);
    inspire::gpu::gpu_ntt_inverse(d_acc_b,     N, Q0, d_inv_q0, inv_n_q0);
    inspire::gpu::gpu_ntt_inverse(d_acc_b + N, N, Q1, d_inv_q1, inv_n_q1);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_acc_a(2 * N), h_acc_b(2 * N);
    cudaMemcpy(h_acc_a.data(), d_acc_a, 2 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_acc_b.data(), d_acc_b, 2 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int mm_a = 0, mm_b = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_acc_a[i] != (uint32_t)ref.a.limbs[0][i]) mm_a++;
        if (h_acc_a[N + i] != (uint32_t)ref.a.limbs[1][i]) mm_a++;
        if (h_acc_b[i] != (uint32_t)ref.b.limbs[0][i]) mm_b++;
        if (h_acc_b[N + i] != (uint32_t)ref.b.limbs[1][i]) mm_b++;
    }
    std::cout << "  D = " << D_TEST << std::endl;
    std::cout << "  res.a mismatches: " << mm_a << " / " << 2*N << std::endl;
    std::cout << "  res.b mismatches: " << mm_b << " / " << 2*N << std::endl;
    std::cout << "  Test: " << (mm_a == 0 && mm_b == 0 ? "PASS" : "FAIL") << std::endl;

    return (mm_a == 0 && mm_b == 0) ? 0 : 1;
}
