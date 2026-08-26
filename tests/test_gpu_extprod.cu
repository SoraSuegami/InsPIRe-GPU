// Test GPU ext_prod against CPU ext_prod.
// Strategy:
//   1. Generate CPU sk, RGSW(mu), RLWE(m).
//   2. Run CPU ext_prod -> reference output.
//   3. Convert all NTT-domain inputs to coefficient form (CPU INTT),
//      then upload to GPU and re-NTT with our GPU NTT.
//   4. Run GPU ext_prod, INTT result on GPU, download.
//   5. INTT the CPU reference (so both are coefficient form).
//   6. Compare per-coefficient.
//
// The GPU and CPU NTT representations differ, so we compare only in
// coefficient form (unambiguous).
#include "check.h"
#include "ring.h"
#include "crypto.h"
#include "gpu.cuh"
#include <iostream>
#include <random>
#include <vector>
#include <cstring>

using namespace inspire;

// Convert a 64-bit RnsPoly (CPU NTT form) to a contiguous 2-limb uint32
// array and INTT it on the CPU so we can re-NTT on GPU.
// Returns coefficient-form data: [limb0 N values][limb1 N values]
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

// Upload coefficient-form data and NTT each limb on GPU.
static void upload_and_gpu_ntt(uint32_t* d_dst, const std::vector<uint32_t>& h_coeff,
                                const uint32_t* d_fwd_q0, const uint32_t* d_fwd_q1) {
    cudaMemcpy(d_dst, h_coeff.data(), 2 * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    inspire::gpu::gpu_ntt_forward(d_dst,     N, Q0, d_fwd_q0);
    inspire::gpu::gpu_ntt_forward(d_dst + N, N, Q1, d_fwd_q1);
}

int main() {
    std::cout << "=== GPU ext_prod vs CPU ext_prod ===" << std::endl;

    std::mt19937_64 rng(7);
    uint8_t seed[SEED_BYTES] = {};
    for (auto& b : seed) b = rng() & 0xFF;

    // Generate sk, RGSW, RLWE on CPU
    RnsPoly s = sk_gen(rng);

    // mu = X^k for some random k (the j*-style evaluation point)
    RnsPoly mu = RnsPoly::zero();
    mu.set_coeff(123, 1, 1);  // X^123

    RgswCt rgsw = rgsw_enc_b(s, mu, seed, rng);

    // Generate an RLWE(plaintext) for testing
    RnsPoly s_ntt = s; s_ntt.to_ntt();
    RnsPoly a_poly = expand_seed(seed, "test_rlwe", 0);
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
    // Now (a, b) = (-a_poly, b_ntt) is an RLWE(m).
    // For the protocol's convention, b = a*s + e + m. Here we follow that.
    RlweCt ct;
    ct.a = a_poly;
    ct.a.to_ntt();
    ct.b = b_ntt;
    ct.b.to_ntt();

    // CPU ext_prod -> reference
    RlweCt ref = ext_prod(rgsw, ct);
    // Bring reference to coefficient form
    if (ref.a.is_ntt) ref.a.to_coeff();
    if (ref.b.is_ntt) ref.b.to_coeff();

    // ====== GPU side ======
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

    // Upload ct
    auto ct_a_coeff = rnspoly_to_u32_coeff(ct.a);
    auto ct_b_coeff = rnspoly_to_u32_coeff(ct.b);
    uint32_t *d_ct_a, *d_ct_b;
    cudaMalloc(&d_ct_a, 2 * N * sizeof(uint32_t));
    cudaMalloc(&d_ct_b, 2 * N * sizeof(uint32_t));
    upload_and_gpu_ntt(d_ct_a, ct_a_coeff, d_fwd_q0, d_fwd_q1);
    upload_and_gpu_ntt(d_ct_b, ct_b_coeff, d_fwd_q0, d_fwd_q1);

    // Upload RGSW. Layout per array: [limb0: d_eff polys][limb1: d_eff polys] = 2*d_eff*N.
    auto upload_rgsw_part = [&](const std::vector<RnsPoly>& parts, uint32_t** d_out) {
        cudaMalloc(d_out, 2 * D_EFF * N * sizeof(uint32_t));
        for (int j = 0; j < D_EFF; j++) {
            auto coeff = rnspoly_to_u32_coeff(parts[j]);
            // Limb 0 piece
            cudaMemcpy(*d_out + j * N,                   coeff.data(),     N * sizeof(uint32_t), cudaMemcpyHostToDevice);
            // Limb 1 piece (offset by d_eff*N)
            cudaMemcpy(*d_out + D_EFF * N + j * N,       coeff.data() + N, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        }
        // NTT each polynomial in place.
        // Limb 0: d_eff polys
        for (int j = 0; j < D_EFF; j++)
            inspire::gpu::gpu_ntt_forward(*d_out + j * N, N, Q0, d_fwd_q0);
        // Limb 1: d_eff polys
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
    uint32_t *d_res_a, *d_res_b, *d_scratch_coeff, *d_dig;
    cudaMalloc(&d_res_a,         2 * N * sizeof(uint32_t));
    cudaMalloc(&d_res_b,         2 * N * sizeof(uint32_t));
    cudaMalloc(&d_scratch_coeff, 4 * N * sizeof(uint32_t));
    cudaMalloc(&d_dig,           4 * D_EFF * N * sizeof(uint32_t));

    // Run GPU ext_prod
    inspire::gpu::gpu_ext_prod(
        d_res_a, d_res_b, d_ct_a, d_ct_b,
        d_top_a, d_top_b, d_bot_a, d_bot_b,
        d_fwd_q0, d_inv_q0, inv_n_q0,
        d_fwd_q1, d_inv_q1, inv_n_q1,
        d_scratch_coeff, d_dig,
        D_GSW, D_EFF, BASE_LOG, N);

    // INTT result on GPU
    inspire::gpu::gpu_ntt_inverse(d_res_a,     N, Q0, d_inv_q0, inv_n_q0);
    inspire::gpu::gpu_ntt_inverse(d_res_a + N, N, Q1, d_inv_q1, inv_n_q1);
    inspire::gpu::gpu_ntt_inverse(d_res_b,     N, Q0, d_inv_q0, inv_n_q0);
    inspire::gpu::gpu_ntt_inverse(d_res_b + N, N, Q1, d_inv_q1, inv_n_q1);
    cudaDeviceSynchronize();

    // Download
    std::vector<uint32_t> h_res_a(2 * N), h_res_b(2 * N);
    cudaMemcpy(h_res_a.data(), d_res_a, 2 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_b.data(), d_res_b, 2 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Compare with CPU reference (in coefficient form)
    int mm_a = 0, mm_b = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_res_a[i] != (uint32_t)ref.a.limbs[0][i]) mm_a++;
        if (h_res_a[N + i] != (uint32_t)ref.a.limbs[1][i]) mm_a++;
        if (h_res_b[i] != (uint32_t)ref.b.limbs[0][i]) mm_b++;
        if (h_res_b[N + i] != (uint32_t)ref.b.limbs[1][i]) mm_b++;
    }
    std::cout << "  res.a mismatches: " << mm_a << " / " << 2*N << std::endl;
    std::cout << "  res.b mismatches: " << mm_b << " / " << 2*N << std::endl;
    std::cout << "  Test: " << (mm_a == 0 && mm_b == 0 ? "PASS" : "FAIL") << std::endl;

    return (mm_a == 0 && mm_b == 0) ? 0 : 1;
}
