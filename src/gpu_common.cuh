// Shared GPU helpers — included from every src/gpu_*.cu TU.
//
// Contents:
//   - CUDA_CHECK macro
//   - Scalar modular arithmetic device helpers (mod_add, mod_sub, mod_mul,
//     mont_mul) for 27-bit primes Q0 / Q1.
//   - Barrett reduction: barrett_mod (x < 2^55) and barrett_u64_dev (full
//     u64; uses __umul64hi).
//   - Mersenne reduction: mod_p_mersenne (P=65535).
//   - Bit-reverse helper br_bits.
//   - NTT shared-memory primitives:
//       ntt_forward_shared      — plain (uses % q via modmul_u32)
//       ntt_forward_shared_shoup / ntt_inverse_shared_shoup — Shoup-Harvey
//       lazy variant (Phase 7) operating in [0, 2q) / [0, 4q).
//   - compute_q_inv host helper for Montgomery setup.
//
// All __device__ helpers are __forceinline__ or static inline so multiple
// TUs can include this header without ODR conflicts.

#pragma once

#include "gpu.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// DB element address for column `col`, row `row`. Production uses RM=true
// (row-major db[row*db_cols+col]) so the online matvec reads the DB directly
// with no transpose; the RM=false (column-major) branch is retained only as a
// layout-agnostic helper and is not instantiated. `if constexpr` => zero overhead.
template<bool RM> __device__ __forceinline__
size_t db_index(size_t col, size_t row, size_t db_rows, size_t db_cols) {
    if constexpr (RM) return row * db_cols + col;
    else              return col * db_rows + row;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

namespace inspire {
namespace gpu {

// ============================================================
// Scalar modular arithmetic (per-element, 32-bit modulus)
// ============================================================

__device__ static __forceinline__
uint32_t mont_mul(uint32_t a, uint32_t b, uint32_t q, uint32_t q_inv) {
    uint64_t ab = (uint64_t)a * b;
    uint32_t lo = (uint32_t)ab;
    uint32_t m  = lo * q_inv;
    uint64_t t  = ab + (uint64_t)m * q;
    uint32_t r  = (uint32_t)(t >> 32);
    return r >= q ? r - q : r;
}

__device__ static __forceinline__
uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    return s >= q ? s - q : s;
}

__device__ static __forceinline__
uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    return a >= b ? a - b : a + q - b;
}

__device__ static __forceinline__
uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)((uint64_t)a * b % q);
}

// Older-named aliases used by NTT/matmul code.
__device__ static __forceinline__ uint32_t modmul_u32(uint32_t a, uint32_t b, uint32_t q) { return mod_mul(a, b, q); }
__device__ static __forceinline__ uint32_t modadd_u32(uint32_t a, uint32_t b, uint32_t q) { return mod_add(a, b, q); }
__device__ static __forceinline__ uint32_t modsub_u32(uint32_t a, uint32_t b, uint32_t q) { return mod_sub(a, b, q); }

// ============================================================
// Barrett reductions
// ============================================================

// Barrett reduction (mu = floor(2^55 / q)). Input x < 2^55, q < 2^28.
__device__ static __forceinline__
uint32_t barrett_mod(uint64_t x, uint32_t q, uint32_t mu) {
    uint64_t t = ((x >> 27) * mu) >> 28;
    uint32_t r = (uint32_t)(x - t * q);
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

// Full 64-bit Barrett reduction.
// cr = floor(2^64 / modulus); quotient estimate via __umul64hi, off by ≤1.
__device__ static __forceinline__
uint64_t barrett_u64_dev(uint64_t val, uint64_t cr, uint64_t modulus) {
    uint64_t tmp = __umul64hi(val, cr);
    uint64_t res = val - tmp * modulus;
    if (res >= modulus) res -= modulus;
    return res;
}

// Mersenne reduction for P = 65535 = 2^16 - 1 (Phase 14 plaintext modulus).
// Two folds of (lo + hi); valid for any 32-bit input.
__device__ static __forceinline__ uint32_t mod_p_mersenne(uint32_t x) {
    uint32_t r = (x & 0xFFFFu) + (x >> 16);
    r = (r & 0xFFFFu) + (r >> 16);
    if (r >= 65535u) r -= 65535u;
    return r;
}

// ============================================================
// Bit reverse (Phase 6 closed-form perm)
// ============================================================

// Reverse `bits` lowest bits of x. For N=2048, bits=11.
__device__ static __forceinline__ uint32_t br_bits(uint32_t x, int bits) {
    return __brev(x) >> (32 - bits);
}

// ============================================================
// Shared-memory NTT primitives
// ============================================================

// Plain forward NTT (uses modmul_u32 / mod_add / mod_sub).
// Used by a_side_matmul_a32_kernel.
__device__ static inline void ntt_forward_shared(uint32_t* poly, size_t n,
                                                  const uint32_t* twiddles, uint32_t q) {
    for (size_t m = 1; m < n; m <<= 1) {
        size_t stride = n / (2 * m);
        for (size_t idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            size_t group_idx = idx / stride;
            size_t pair_idx  = idx % stride;
            size_t jj = group_idx * 2 * stride + pair_idx;
            size_t kk = jj + stride;
            uint32_t w = twiddles[m + group_idx];
            uint32_t u = poly[jj];
            uint32_t v = modmul_u32(poly[kk], w, q);
            poly[jj]   = modadd_u32(u, v, q);
            poly[kk]   = modsub_u32(u, v, q);
        }
        __syncthreads();
    }
}

// ============================================================
// Shoup-Harvey lazy NTT (Phase 7). Returns value in [0, 2q).
//   w_prime = floor(w * 2^32 / q)
//   q_tmp   = (x * w_prime) >> 32       ≈ x*w/q
//   q_new   = x * w - q_tmp * q         ∈ [0, 2q)
// Butterflies operate in [0, 2q) → [0, 4q); a final reduction brings the
// result back to [0, q).
// ============================================================

__device__ static inline uint32_t shoup_mul_2q(uint32_t x, uint32_t w, uint32_t w_prime, uint32_t q) {
    uint64_t q_tmp = ((uint64_t)x * (uint64_t)w_prime) >> 32;
    return (uint32_t)((uint64_t)x * (uint64_t)w - q_tmp * (uint64_t)q);
}

// Cooley-Tukey DIT forward NTT, Shoup-Harvey lazy variant.
// Input [0, q). Output [0, q) (final reduction included).
__device__ static inline void ntt_forward_shared_shoup(
    uint32_t* sh_poly, size_t n, uint32_t q,
    const uint32_t* fwd_twiddles, const uint32_t* fwd_primes)
{
    const uint32_t two_q = q + q;
    for (size_t m = 1; m < n; m <<= 1) {
        size_t stride = n / (2 * m);
        for (size_t idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            size_t group_idx = idx / stride;
            size_t pair_idx  = idx % stride;
            size_t j = group_idx * 2 * stride + pair_idx;
            size_t k = j + stride;
            uint32_t w  = fwd_twiddles[m + group_idx];
            uint32_t wp = fwd_primes  [m + group_idx];
            uint32_t x  = sh_poly[j];
            uint32_t y  = sh_poly[k];
            uint32_t curr_x = (x >= two_q) ? x - two_q : x;
            uint32_t prod   = shoup_mul_2q(y, w, wp, q);
            sh_poly[j] = curr_x + prod;
            sh_poly[k] = curr_x + (two_q - prod);
        }
        __syncthreads();
    }
    // Final reduction: [0, 4q) -> [0, q)
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        uint32_t v = sh_poly[i];
        if (v >= two_q) v -= two_q;
        if (v >= q)     v -= q;
        sh_poly[i] = v;
    }
    __syncthreads();
}

// Gentleman-Sande DIF inverse NTT, Shoup-Harvey lazy variant + N^{-1} scaling.
__device__ static inline void ntt_inverse_shared_shoup(
    uint32_t* sh_poly, size_t n, uint32_t q,
    const uint32_t* inv_twiddles, const uint32_t* inv_primes,
    uint32_t inv_n, uint32_t inv_n_prime)
{
    const uint32_t two_q = q + q;
    for (size_t m = n / 2; m >= 1; m >>= 1) {
        size_t stride = n / (2 * m);
        for (size_t idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            size_t group_idx = idx / stride;
            size_t pair_idx  = idx % stride;
            size_t j = group_idx * 2 * stride + pair_idx;
            size_t k = j + stride;
            uint32_t w  = inv_twiddles[m + group_idx];
            uint32_t wp = inv_primes  [m + group_idx];
            uint32_t a  = sh_poly[j];
            uint32_t b  = sh_poly[k];
            uint32_t curr_a = (a >= two_q) ? a - two_q : a;
            uint32_t curr_b = (b >= two_q) ? b - two_q : b;
            uint32_t diff = curr_a + (two_q - curr_b);
            if (diff >= two_q) diff -= two_q;
            sh_poly[j] = curr_a + curr_b;
            sh_poly[k] = shoup_mul_2q(diff, w, wp, q);
        }
        __syncthreads();
        if (m == 1) break;
    }
    // N^{-1} scale + final reduction to [0, q)
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        uint32_t v = sh_poly[i];
        if (v >= two_q) v -= two_q;
        if (v >= q)     v -= q;
        v = shoup_mul_2q(v, inv_n, inv_n_prime, q);
        if (v >= q) v -= q;
        sh_poly[i] = v;
    }
    __syncthreads();
}

// ============================================================
// Host helpers
// ============================================================

// Compute Montgomery constant: -q^{-1} mod 2^32.
inline uint32_t compute_q_inv(uint32_t q) {
    uint64_t r0 = (uint64_t)1 << 32, r1 = q;
    int64_t  s0 = 0, s1 = 1;
    while (r1 != 0) {
        uint64_t quotient = r0 / r1;
        uint64_t tmp = r1; r1 = r0 - quotient * r1; r0 = tmp;
        int64_t  st  = s1; s1 = s0 - (int64_t)quotient * s1; s0 = st;
    }
    return (uint32_t)(-(int32_t)s0);
}

} // namespace gpu
} // namespace inspire
