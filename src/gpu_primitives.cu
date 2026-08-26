// gpu_primitives.cu: phase-agnostic GPU building blocks — NTT (fwd/inv/batch),
// gadget decomposition, polynomial pointwise ops, device memory + context init.
// Used by both the preprocess and the online phases.

#include "gpu_common.cuh"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

namespace inspire {
namespace gpu {


// ============================================================
// NTT kernel: radix-2 Cooley-Tukey, shared memory, matching the CPU NTT
//
// Forward NTT convention (shared with the CPU NTT):
//   for m = 1, 2, 4, ..., n/2:
//     t = n / (2*m)                   // stride
//     for group i = 0..m-1:
//       w = twiddles[m + i]           // ONE twiddle per group
//       for j in group's range:
//         butterfly(a[j], a[j+t], w)
//
// CT butterfly: u = a[j], v = a[j+t]*w;  a[j] = u+v, a[j+t] = u-v
// ============================================================

__global__
void ntt_fwd_kernel(uint32_t* data, size_t n, uint32_t q,
                    const uint32_t* twiddles) {
    extern __shared__ uint32_t shared[];

    size_t poly_offset = blockIdx.x * n;
    uint32_t* poly = data + poly_offset;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        shared[i] = poly[i];
    __syncthreads();

    for (size_t m = 1; m < n; m <<= 1) {
        size_t t = n / (2 * m); // stride between butterfly pairs
        // Total n/2 butterflies per stage
        for (size_t idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            // Which group (0..m-1) and which pair within the group (0..t-1)?
            size_t group_idx = idx / t;
            size_t pair_idx = idx % t;
            size_t j = group_idx * 2 * t + pair_idx;
            size_t k = j + t;

            uint32_t w = twiddles[m + group_idx]; // same w for all pairs in group
            uint32_t u = shared[j];
            uint32_t v = mod_mul(shared[k], w, q);
            shared[j] = mod_add(u, v, q);
            shared[k] = mod_sub(u, v, q);
        }
        __syncthreads();
    }

    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        poly[i] = shared[i];
}


// ============================================================
// INTT kernel: Gentleman-Sande butterfly, undoes forward CT
//
// Inverts the forward CT in reverse stage order. Each forward CT
// butterfly was: (u, v) → (u + v·W, u − v·W) using W = twiddles[m+i].
// To undo: given (a, b) = (u+v·W, u−v·W),
//   u = (a + b) / 2
//   v = (a − b) · W^{-1} / 2
// We omit the /2 here and absorb all of them into the final
// multiplication by n^{-1} at the end.
//
// GS butterfly (per stage, with stride t):
//   u = a[j], v = a[j+t]
//   a[j]   = u + v
//   a[j+t] = (u − v) · inv_twiddles[m + group_idx]
// where inv_twiddles[i] = mod_inv(fwd_twiddles[i], q).
//
// Stages run in reverse order: m = n/2, n/4, ..., 1.
// ============================================================

__global__
void ntt_inv_kernel(uint32_t* data, size_t n, uint32_t q,
                    const uint32_t* inv_twiddles, uint32_t inv_n) {
    extern __shared__ uint32_t shared[];

    size_t poly_offset = blockIdx.x * n;
    uint32_t* poly = data + poly_offset;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        shared[i] = poly[i];
    __syncthreads();

    // Reverse-stage GS butterflies.
    // m starts at n/2 and halves each iteration; t = n/(2m) doubles.
    for (size_t m = n / 2; m >= 1; m >>= 1) {
        size_t t = n / (2 * m);
        for (size_t idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            size_t group_idx = idx / t;
            size_t pair_idx = idx % t;
            size_t j = group_idx * 2 * t + pair_idx;
            size_t k = j + t;

            uint32_t w = inv_twiddles[m + group_idx];
            uint32_t u = shared[j];
            uint32_t v = shared[k];
            shared[j] = mod_add(u, v, q);
            shared[k] = mod_mul(mod_sub(u, v, q), w, q);
        }
        __syncthreads();
        if (m == 1) break; // unsigned m: can't go lower without underflow
    }

    // Scale by n^{-1} (absorbs all the /2 factors)
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        shared[i] = mod_mul(shared[i], inv_n, q);
    __syncthreads();

    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        poly[i] = shared[i];
}


// ============================================================
// Pointwise polynomial ops (NTT domain or coefficient domain — they're
// pointwise either way for these three operations)
// ============================================================

__global__
void poly_add_kernel(uint32_t* c, const uint32_t* a, const uint32_t* b,
                     size_t n, uint32_t q) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = mod_add(a[i], b[i], q);
}


__global__
void poly_sub_kernel(uint32_t* c, const uint32_t* a, const uint32_t* b,
                     size_t n, uint32_t q) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = mod_sub(a[i], b[i], q);
}


// NTT-domain pointwise multiply
__global__
void poly_mul_kernel(uint32_t* c, const uint32_t* a, const uint32_t* b,
                     size_t n, uint32_t q, uint32_t mu) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = barrett_mod((uint64_t)a[i] * b[i], q, mu);
}


// c[i] -= a[i] * b[i]   (NTT-domain mul-sub)
__global__
void poly_mul_sub_kernel(uint32_t* c, const uint32_t* a, const uint32_t* b,
                         size_t n, uint32_t q, uint32_t mu) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t s = barrett_mod((uint64_t)a[i] * b[i], q, mu);
        c[i] = mod_sub(c[i], s, q);
    }
}


// ============================================================
// Gadget decomposition (matches CPU decomp() in ring.cpp).
// Inputs: a0[i] mod Q0, a1[i] mod Q1 in coefficient form.
// Output: d_eff polynomials, each (out0[j][i], out1[j][i]) per RNS limb.
//
// Algorithm:
//   1. CRT-lift to val ∈ [0, Q) where Q = Q0 * Q1
//   2. Center to r ∈ [-Q/2, Q/2)
//   3. Extract digits using arithmetic shift (sign-extended)
//   4. Drop digit 0; emit digits 1..d-1 as (mod Q0, mod Q1) per limb
// ============================================================

__global__
void gadget_decomp_kernel(const uint32_t* a0, const uint32_t* a1,
                          uint32_t* out0, uint32_t* out1,
                          size_t n, int d_gsw, int d_eff, int base_log,
                          uint64_t q0, uint64_t q1,
                          uint64_t q1_inv_q0,    // Q1^{-1} mod Q0
                          uint64_t q0_inv_q1,    // Q0^{-1} mod Q1
                          uint64_t q_full_lo,    // low 64 bits of Q (= Q0*Q1)
                          uint64_t q_full_hi) {  // high 64 bits of Q
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t v0 = a0[i], v1 = a1[i];

    // CRT lift: val = (v0 * Q1_inv_Q0 mod Q0) * Q1 + (v1 * Q0_inv_Q1 mod Q1) * Q0
    uint64_t t0 = ((__uint128_t)v0 * q1_inv_q0) % q0;
    uint64_t t1 = ((__uint128_t)v1 * q0_inv_q1) % q1;
    __uint128_t Q = ((__uint128_t)q_full_hi << 64) | q_full_lo;
    __uint128_t val = (__uint128_t)t0 * q1 + (__uint128_t)t1 * q0;
    val = val % Q;

    // Center to [-Q/2, Q/2)
    int64_t r;
    if (val <= Q / 2) r = (int64_t)val;
    else r = (int64_t)(val) - (int64_t)Q;  // val - Q is negative; cast to int64

    // Actually since Q < 2^53, and we cast to int64, the value fits.
    // Just need to handle the sign carefully:
    // val ∈ [0, Q), Q < 2^55. val ≤ Q/2 ⇒ r = val (positive int64).
    // val > Q/2 ⇒ r = val - Q (negative int64).
    // The above casts work since |Q/2| < 2^54 < 2^63.

    // Extract digits using arithmetic shift
    // K = 64 - base_log; (r << K) >> K extracts low base_log bits sign-extended
    int K = 64 - base_log;
    int64_t digs[16]; // d_gsw <= 16
    for (int j = 0; j < d_gsw; j++) {
        int64_t d = ((int64_t)((uint64_t)r << K)) >> K;
        digs[j] = d;
        r = (r - d) >> base_log;
    }

    // Store digits 1..d-1 (skip digit 0)
    for (int j = 0; j < d_eff; j++) {
        int64_t d = digs[j + 1];
        // Map negative to (q + d) mod q
        uint64_t u0 = (d >= 0) ? (uint64_t)d : (uint64_t)(q0 + d);
        uint64_t u1 = (d >= 0) ? (uint64_t)d : (uint64_t)(q1 + d);
        out0[j * n + i] = (uint32_t)(u0 % q0);
        out1[j * n + i] = (uint32_t)(u1 % q1);
    }
}


// Strided variant for the lockstep batched Horner (Phase B4): poly y of
// `count` lives at a{0,1}_base + y*n; digits go to out{0,1}_base + y*d_eff*n.
// Same arithmetic as gadget_decomp_kernel, one grid.y lane per poly.
__global__
void gadget_decomp_strided_kernel(const uint32_t* a0_base, const uint32_t* a1_base,
                                  uint32_t* out0_base, uint32_t* out1_base,
                                  size_t n, int d_gsw, int d_eff, int base_log,
                                  uint64_t q0, uint64_t q1,
                                  uint64_t q1_inv_q0, uint64_t q0_inv_q1,
                                  uint64_t q_full_lo, uint64_t q_full_hi) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t y = blockIdx.y;
    const uint32_t* a0 = a0_base + y * n;
    const uint32_t* a1 = a1_base + y * n;
    uint32_t* out0 = out0_base + y * (size_t)d_eff * n;
    uint32_t* out1 = out1_base + y * (size_t)d_eff * n;

    uint64_t v0 = a0[i], v1 = a1[i];
    uint64_t t0 = ((__uint128_t)v0 * q1_inv_q0) % q0;
    uint64_t t1 = ((__uint128_t)v1 * q0_inv_q1) % q1;
    __uint128_t Q = ((__uint128_t)q_full_hi << 64) | q_full_lo;
    __uint128_t val = (__uint128_t)t0 * q1 + (__uint128_t)t1 * q0;
    val = val % Q;
    int64_t r;
    if (val <= Q / 2) r = (int64_t)val;
    else r = (int64_t)(val) - (int64_t)Q;

    int K = 64 - base_log;
    int64_t digs[16];
    for (int j = 0; j < d_gsw; j++) {
        int64_t d = ((int64_t)((uint64_t)r << K)) >> K;
        digs[j] = d;
        r = (r - d) >> base_log;
    }
    for (int j = 0; j < d_eff; j++) {
        int64_t d = digs[j + 1];
        uint64_t u0 = (d >= 0) ? (uint64_t)d : (uint64_t)(q0 + d);
        uint64_t u1 = (d >= 0) ? (uint64_t)d : (uint64_t)(q1 + d);
        out0[j * n + i] = (uint32_t)(u0 % q0);
        out1[j * n + i] = (uint32_t)(u1 % q1);
    }
}

// ============================================================
// Host-side wrappers
// ============================================================

// compute_q_inv definition lives in gpu_common.cuh.

void gpu_init_with_twiddles(GpuContext& ctx,
                           const uint32_t* fwd0, const uint32_t* inv0, uint32_t inv_n0,
                           const uint32_t* fwd1, const uint32_t* inv1, uint32_t inv_n1) {
    ctx.q_inv[0] = compute_q_inv(Q0);
    ctx.q_inv[1] = compute_q_inv(Q1);
    ctx.mont_one[0] = (uint32_t)(((uint64_t)1 << 32) % Q0);
    ctx.mont_one[1] = (uint32_t)(((uint64_t)1 << 32) % Q1);
    ctx.inv_n[0] = inv_n0;
    ctx.inv_n[1] = inv_n1;

    const uint32_t* fwd_ptrs[2] = {fwd0, fwd1};
    const uint32_t* inv_ptrs[2] = {inv0, inv1};
    for (int c = 0; c < 2; c++) {
        CUDA_CHECK(cudaMalloc(&ctx.d_twiddles[c], N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx.d_inv_twiddles[c], N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(ctx.d_twiddles[c], fwd_ptrs[c], N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_inv_twiddles[c], inv_ptrs[c], N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    ctx.initialized = true;
}


void gpu_init(GpuContext& ctx) {
    // Legacy: allocate but don't upload twiddles (use gpu_init_with_twiddles instead)
    ctx.q_inv[0] = compute_q_inv(Q0);
    ctx.q_inv[1] = compute_q_inv(Q1);
    ctx.mont_one[0] = (uint32_t)(((uint64_t)1 << 32) % Q0);
    ctx.mont_one[1] = (uint32_t)(((uint64_t)1 << 32) % Q1);
    ctx.inv_n[0] = 0;
    ctx.inv_n[1] = 0;
    for (int c = 0; c < 2; c++) {
        CUDA_CHECK(cudaMalloc(&ctx.d_twiddles[c], N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx.d_inv_twiddles[c], N * sizeof(uint32_t)));
    }
    ctx.initialized = true;
}


void gpu_free(GpuContext& ctx) {
    if (!ctx.initialized) return;
    for (int c = 0; c < 2; c++) {
        cudaFree(ctx.d_twiddles[c]);
        cudaFree(ctx.d_inv_twiddles[c]);
    }
    ctx.initialized = false;
}


void gpu_ntt_forward(uint32_t* d_poly, size_t n, uint32_t q, const uint32_t* d_twiddles) {
    size_t shared_size = n * sizeof(uint32_t);
    ntt_fwd_kernel<<<1, 256, shared_size>>>(d_poly, n, q, d_twiddles);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_ntt_forward_stream(uint32_t* d_poly, size_t n, uint32_t q,
                             const uint32_t* d_twiddles, cudaStream_t stream) {
    size_t shared_size = n * sizeof(uint32_t);
    ntt_fwd_kernel<<<1, 256, shared_size, stream>>>(d_poly, n, q, d_twiddles);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_ntt_inverse(uint32_t* d_poly, size_t n, uint32_t q,
                     const uint32_t* d_inv_twiddles, uint32_t inv_n) {
    size_t shared_size = n * sizeof(uint32_t);
    ntt_inv_kernel<<<1, 256, shared_size>>>(d_poly, n, q, d_inv_twiddles, inv_n);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_ntt_batch(uint32_t* d_polys, size_t n, size_t batch_size, uint32_t q,
                   const uint32_t* d_twiddles, size_t threads) {
    size_t shared_size = n * sizeof(uint32_t);
    ntt_fwd_kernel<<<batch_size, threads, shared_size>>>(d_polys, n, q, d_twiddles);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_ntt_inverse_batch(uint32_t* d_polys, size_t n, size_t batch_size, uint32_t q,
                           const uint32_t* d_inv_twiddles, uint32_t inv_n, size_t threads) {
    size_t shared_size = n * sizeof(uint32_t);
    ntt_inv_kernel<<<batch_size, threads, shared_size>>>(d_polys, n, q, d_inv_twiddles, inv_n);
    CUDA_CHECK(cudaGetLastError());
}


// gpu_fused_ip_sub_stream and gpu_lazy_collapse_stream live in gpu_collapse.cu.

// ============================================================
// Polynomial primitives (host wrappers)
// ============================================================

void gpu_poly_add(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                  size_t n, uint32_t q) {
    size_t blocks = (n + 255) / 256;
    poly_add_kernel<<<blocks, 256>>>(d_c, d_a, d_b, n, q);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_poly_sub(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                  size_t n, uint32_t q) {
    size_t blocks = (n + 255) / 256;
    poly_sub_kernel<<<blocks, 256>>>(d_c, d_a, d_b, n, q);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_poly_mul(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                  size_t n, uint32_t q) {
    uint32_t mu = (q == Q0) ? Q0_BARRETT_MU : Q1_BARRETT_MU;
    size_t blocks = (n + 255) / 256;
    poly_mul_kernel<<<blocks, 256>>>(d_c, d_a, d_b, n, q, mu);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_poly_mul_sub(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                      size_t n, uint32_t q) {
    uint32_t mu = (q == Q0) ? Q0_BARRETT_MU : Q1_BARRETT_MU;
    size_t blocks = (n + 255) / 256;
    poly_mul_sub_kernel<<<blocks, 256>>>(d_c, d_a, d_b, n, q, mu);
    CUDA_CHECK(cudaGetLastError());
}


// Gadget decomposition. Inputs in coefficient form; outputs digits[0..d_eff-1]
// in coefficient form, layout: out0[d_eff * n], out1[d_eff * n].
void gpu_gadget_decomp(const uint32_t* d_a0, const uint32_t* d_a1,
                       uint32_t* d_out0, uint32_t* d_out1,
                       size_t n, int d_gsw, int d_eff, int base_log) {
    // Precompute CRT constants on host
    auto modinv = [](uint64_t a, uint64_t m) {
        int64_t old_r = (int64_t)a, r = (int64_t)m;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            int64_t qq = old_r / r;
            int64_t tmp = r; r = old_r - qq * r; old_r = tmp;
            tmp = s; s = old_s - qq * s; old_s = tmp;
        }
        return (uint64_t)((old_s % (int64_t)m + (int64_t)m) % (int64_t)m);
    };
    uint64_t q1_inv_q0 = modinv(Q1, Q0);
    uint64_t q0_inv_q1 = modinv(Q0, Q1);
    __uint128_t Q_full = (__uint128_t)Q0 * Q1;
    uint64_t q_full_lo = (uint64_t)Q_full;
    uint64_t q_full_hi = (uint64_t)(Q_full >> 64);

    size_t blocks = (n + 255) / 256;
    gadget_decomp_kernel<<<blocks, 256>>>(d_a0, d_a1, d_out0, d_out1,
                                          n, d_gsw, d_eff, base_log,
                                          Q0, Q1, q1_inv_q0, q0_inv_q1,
                                          q_full_lo, q_full_hi);
    CUDA_CHECK(cudaGetLastError());
}

// Strided decomposition of `count` contiguous polys (see the strided kernel).
void gpu_gadget_decomp_strided(const uint32_t* d_a0, const uint32_t* d_a1,
                               uint32_t* d_out0, uint32_t* d_out1,
                               size_t n, int count,
                               int d_gsw, int d_eff, int base_log) {
    auto modinv = [](uint64_t a, uint64_t m) {
        int64_t old_r = (int64_t)a, r = (int64_t)m;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            int64_t qq = old_r / r;
            int64_t tmp = r; r = old_r - qq * r; old_r = tmp;
            tmp = s; s = old_s - qq * s; old_s = tmp;
        }
        return (uint64_t)((old_s % (int64_t)m + (int64_t)m) % (int64_t)m);
    };
    uint64_t q1_inv_q0 = modinv(Q1, Q0);
    uint64_t q0_inv_q1 = modinv(Q0, Q1);
    __uint128_t Q_full = (__uint128_t)Q0 * Q1;
    dim3 grid((n + 255) / 256, count);
    gadget_decomp_strided_kernel<<<grid, 256>>>(d_a0, d_a1, d_out0, d_out1,
                                                n, d_gsw, d_eff, base_log,
                                                Q0, Q1, q1_inv_q0, q0_inv_q1,
                                                (uint64_t)Q_full,
                                                (uint64_t)(Q_full >> 64));
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================
// Memory management
// ============================================================

uint32_t* gpu_alloc(size_t count) {
    uint32_t* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(uint32_t)));
    return ptr;
}


void gpu_free_mem(uint32_t* ptr) {
    cudaFree(ptr);
}


void gpu_upload(uint32_t* d_dst, const uint32_t* h_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(uint32_t), cudaMemcpyHostToDevice));
}


void gpu_download(uint32_t* h_dst, const uint32_t* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

} // namespace gpu
} // namespace inspire
