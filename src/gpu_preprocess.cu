// gpu_preprocess.cu: one-time server preprocess kernels — in-place row-major
// inverse-DFT encode, a-side matmul, ring_embed, automorphism/permutation, and the
// fused offline collapse. Run once per DB by gpu_preprocess (see gpu_protocol.cu).

#include "gpu_common.cuh"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

namespace inspire {
namespace gpu {


// fused_ip_sub_kernel + the lazy-collapse partial/reduce kernels and their
// stream wrappers live in gpu_collapse.cu, alongside collapse_fused_kernel.

// ============================================================
// Ring embed (GPU port of packing.cpp:ring_embed)
//
// For each k = 0..half-1, c = 0..1, i = 0..n-1:
//   acc_fwd[k][c][i]  = sum_{j=0..γ-1} a32[j][c][i] * mono32[idx_f][c][i]  mod q_c
//   acc_conj[k][c][i] = sum_{j=0..γ-1} a32[j][c][i] * mono32[idx_c][c][i]  mod q_c
// where:
//   gen_pow_nk = gp[(n-k) mod n] = 5^{n-k} mod 2n
//   idx_f = (j * gen_pow_nk) mod (2n)
//   idx_c = (2n - idx_f) mod (2n)
//
// All inputs/outputs are in NTT form (CPU and GPU NTT conventions agree).
//
// Layout convention:
//   a32[j][c][i]      → a32[j*2*n + c*n + i]
//   mono32[idx][c][i] → mono32[idx*2*n + c*n + i]
//   acc_fwd[k][c][i]  → acc_fwd[k*2*n + c*n + i]
// ============================================================

__global__
void ring_embed_kernel(
    uint32_t* acc_fwd,         // [half * 2 * n]
    uint32_t* acc_conj,        // [half * 2 * n]
    const uint32_t* a32,       // [γ * 2 * n]
    const uint32_t* mono32,    // [2n * 2 * n]
    const uint32_t* gp,        // [n], gp[k] = 5^k mod 2n
    int gamma, int half, int n,
    uint32_t q0, uint64_t cr0,
    uint32_t q1, uint64_t cr1) {

    int kbase = blockIdx.x;
    int k = blockIdx.y;
    int c = blockIdx.z;
    int i = kbase * blockDim.x + threadIdx.x;
    if (i >= n || k >= half) return;

    const uint32_t q  = (c == 0) ? q0 : q1;
    const uint64_t cr = (c == 0) ? cr0 : cr1;
    const uint32_t two_n_mask = (uint32_t)(2 * n - 1);   // 2n is pow2

    uint32_t gen_pow_nk = gp[(n - k) % n];

    // Phase 11: increment idx_f by gen_pow_nk each iter instead of multiplying
    // (j * gen_pow_nk) from scratch. `% (2n)` is a bitmask since 2n is pow2.
    uint32_t idx_f = 0;

    uint64_t sf = 0, sc = 0;
    for (int j = 0; j < gamma; j++) {
        uint32_t aj = a32[(size_t)j * 2 * n + (size_t)c * n + i];

        uint32_t idx_c_idx = ((uint32_t)(2 * n) - idx_f) & two_n_mask;

        uint32_t mf = mono32[(size_t)idx_f      * 2 * n + (size_t)c * n + i];
        uint32_t mc = mono32[(size_t)idx_c_idx  * 2 * n + (size_t)c * n + i];

        sf += (uint64_t)aj * mf;
        sc += (uint64_t)aj * mc;

        // Reduce every 256 iterations: 256 * (q-1)^2 < 2^62, fits uint64.
        // Phase 11: Barrett (no software 64-bit divide).
        if ((j & 0xFF) == 0xFF) {
            sf = barrett_u64_dev(sf, cr, (uint64_t)q);
            sc = barrett_u64_dev(sc, cr, (uint64_t)q);
        }

        idx_f = (idx_f + gen_pow_nk) & two_n_mask;
    }

    acc_fwd[(size_t)k * 2 * n + (size_t)c * n + i]  =
        (uint32_t)barrett_u64_dev(sf, cr, (uint64_t)q);
    acc_conj[(size_t)k * 2 * n + (size_t)c * n + i] =
        (uint32_t)barrett_u64_dev(sc, cr, (uint64_t)q);
}


// Apply NTT-domain automorphism in place: dst[i] = src[perm[i]] for both limbs.
// d_dst layout: [limb0: n][limb1: n]. d_src same. d_perm: [n] uint32 indices.
__global__
void apply_auto_perm_kernel(uint32_t* dst, const uint32_t* src,
                             const uint32_t* perm, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t p = perm[i];
    dst[i]     = src[p];
    dst[n + i] = src[n + p];
}


// Same as apply_auto_perm_kernel but computes the permutation index inline
// from the automorphism factor t (no precomputed table).
//
// Our NTT (negacyclic, Harvey-style layout) stores
// evaluations at psi^{2*BR(i)+1} in slot i — the slot indices are themselves
// bit-reversed. Applying τ_t in coefficient domain then converting to NTT
// induces this slot permutation:
//
//     perm[i] = BR_inv( ( t * (2*BR(i)+1) mod 2n  −  1 ) / 2 mod n )
//
// (Verified bit-exact against the brute-force NTT-based table in
//  tools/verify_perm.cpp for t=5, N=2048: 0 mismatches.)
//
// For N=2048 = 2^11, BR is a 11-bit reverse; we use __brev shifted right.
// (Definition lives in gpu_common.cuh.)

__global__
void apply_auto_perm_inline_kernel(uint32_t* dst, const uint32_t* src,
                                    uint32_t t, size_t n, int log_n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // 2n is a power of two (n is a power of two here), so `mod 2n` is a mask.
    const uint32_t mask = (uint32_t)(2 * n - 1);
    uint32_t br_i  = br_bits((uint32_t)i, log_n);
    uint32_t i_exp = (t * (2 * br_i + 1)) & mask;
    uint32_t br_j  = (i_exp - 1) >> 1;
    uint32_t p     = br_bits(br_j, log_n);
    dst[i]     = src[p];
    dst[n + i] = src[n + p];
}


void gpu_apply_auto_perm_inline(uint32_t* d_dst, const uint32_t* d_src,
                                 uint32_t t, size_t n) {
    int log_n = 0;
    while (((size_t)1 << log_n) < n) log_n++;
    size_t blocks = (n + 255) / 256;
    apply_auto_perm_inline_kernel<<<blocks, 256>>>(d_dst, d_src, t, n, log_n);
    CUDA_CHECK(cudaGetLastError());
}


// Phase 11: batched per-k automorphism, both fwd and conj directions in one
// kernel. Replaces the 2*half tiny apply_auto_perm_inline launches that
// run_after_a32 used to do in a host loop.
//
// Grid: <<<half, 256>>> — one block per k slot. Each block computes both
// τ_{5^k} (forward) and τ_{2n - 5^k} (conjugate) on its (limb0|limb1) pair.
// d_gp[k] = 5^k mod 2n is the per-k automorphism exponent (uploaded once at
// GpuRingEmbedHelper construction time, reused across all groups).
__global__
void apply_auto_perm_batched_kernel(
    uint32_t* __restrict__ d_dst_fwd,     // [half * 2 * n]
    uint32_t* __restrict__ d_dst_conj,    // [half * 2 * n]
    const uint32_t* __restrict__ d_src_fwd,
    const uint32_t* __restrict__ d_src_conj,
    const uint32_t* __restrict__ d_gp,    // [n], gp[k] = 5^k mod 2n
    int half, int n, int log_n, uint32_t two_n_mask)
{
    const int k   = blockIdx.x;
    const int tid = threadIdx.x;
    if (k >= half) return;

    const uint32_t t_fwd  = d_gp[k];
    const uint32_t t_conj = ((uint32_t)(2 * n) - t_fwd) & two_n_mask;

    const size_t koff = (size_t)k * 2 * n;
    uint32_t*       dfwd = d_dst_fwd  + koff;
    uint32_t*       dcnj = d_dst_conj + koff;
    const uint32_t* sfwd = d_src_fwd  + koff;
    const uint32_t* scnj = d_src_conj + koff;

    for (int i = tid; i < n; i += blockDim.x) {
        const uint32_t br_i = __brev((uint32_t)i) >> (32 - log_n);
        const uint32_t two_br_i_p1 = 2 * br_i + 1;

        const uint32_t ix_f = (t_fwd  * two_br_i_p1) & two_n_mask;
        const uint32_t ix_c = (t_conj * two_br_i_p1) & two_n_mask;
        const uint32_t p_f  = __brev((ix_f - 1) >> 1) >> (32 - log_n);
        const uint32_t p_c  = __brev((ix_c - 1) >> 1) >> (32 - log_n);

        dfwd[i]     = sfwd[p_f];
        dfwd[n + i] = sfwd[n + p_f];
        dcnj[i]     = scnj[p_c];
        dcnj[n + i] = scnj[n + p_c];
    }
}


void gpu_apply_auto_perm_batched(
    uint32_t* d_dst_fwd, uint32_t* d_dst_conj,
    const uint32_t* d_src_fwd, const uint32_t* d_src_conj,
    const uint32_t* d_gp, int half, size_t n)
{
    int log_n = 0;
    while (((size_t)1 << log_n) < n) log_n++;
    const uint32_t two_n_mask = (uint32_t)(2 * n - 1);
    apply_auto_perm_batched_kernel<<<half, 256>>>(
        d_dst_fwd, d_dst_conj, d_src_fwd, d_src_conj, d_gp,
        half, (int)n, log_n, two_n_mask);
    CUDA_CHECK(cudaGetLastError());
}


// Phase 12: GPU-side ExpandKSK_a permutation step.
//
// CPU expand_ksk_a expands a seed via SHAKE256 into 4 ksk5 polynomials (in
// coefficient form), NTTs them, then permutes each by τ_{5^k} and τ_{2n-5^k}
// for k = 0..n_steps-1 to produce K.plus[k][j] / K.minus[k][j]. On CPU the
// 4 * 2 * n_steps permutations dominate the 64-ms runtime.
//
// Here we keep SHAKE256+sampling+NTT on CPU (each is fast) and move only the
// per-(k, j, dir) permutation to GPU. The kernel writes directly into the
// final d_ksk_plus / d_ksk_minus layout consumed by collapse_fused_kernel:
//   d_ksk_plus[c * n_steps * d_eff * n + s * d_eff * n + j * n + i].
//
// Source layout: d_ksk5_ntt is [d_eff][limb][coeff] flat = d_eff * 2 * n u32.
//
// Grid: <<<(n_steps, d_eff), 256>>>. Each block does both fwd and conj
// directions and both limbs for its (s, j) pair, 4 writes per coeff.
__global__
void ksk_perm_batched_kernel(
    uint32_t* __restrict__ d_ksk_plus,
    uint32_t* __restrict__ d_ksk_minus,
    const uint32_t* __restrict__ d_ksk5_ntt,    // [d_eff * 2 * n]
    const uint32_t* __restrict__ d_gp,          // [n]
    int n_steps, int d_eff, int n, int log_n, uint32_t two_n_mask)
{
    const int s = blockIdx.x;       // 0..n_steps-1
    const int j = blockIdx.y;       // 0..d_eff-1
    const int tid = threadIdx.x;
    if (s >= n_steps || j >= d_eff) return;

    // CPU uses atables.fwd[s] = τ_{5^s}; d_gp[s] = 5^s mod 2n
    const uint32_t t_fwd  = d_gp[s];
    const uint32_t t_conj = ((uint32_t)(2 * n) - t_fwd) & two_n_mask;

    // Source: digit j's two limbs (contiguous, j*2*n then j*2*n + n)
    const uint32_t* src_q0 = d_ksk5_ntt + (size_t)j * 2 * n;
    const uint32_t* src_q1 = src_q0 + n;

    // Destination offsets in [limb][step][digit][coeff] layout
    const size_t per_limb_dst = (size_t)n_steps * d_eff * n;
    const size_t out_off = (size_t)s * d_eff * n + (size_t)j * n;
    uint32_t* dst_p_q0 = d_ksk_plus  + out_off;
    uint32_t* dst_p_q1 = d_ksk_plus  + per_limb_dst + out_off;
    uint32_t* dst_m_q0 = d_ksk_minus + out_off;
    uint32_t* dst_m_q1 = d_ksk_minus + per_limb_dst + out_off;

    for (int i = tid; i < n; i += blockDim.x) {
        const uint32_t br_i = __brev((uint32_t)i) >> (32 - log_n);
        const uint32_t v    = 2 * br_i + 1;
        const uint32_t ixf  = (t_fwd  * v) & two_n_mask;
        const uint32_t ixc  = (t_conj * v) & two_n_mask;
        const uint32_t pf   = __brev((ixf - 1) >> 1) >> (32 - log_n);
        const uint32_t pc   = __brev((ixc - 1) >> 1) >> (32 - log_n);

        dst_p_q0[i] = src_q0[pf];
        dst_p_q1[i] = src_q1[pf];
        dst_m_q0[i] = src_q0[pc];
        dst_m_q1[i] = src_q1[pc];
    }
}


void gpu_ksk_perm_batched(
    uint32_t* d_ksk_plus, uint32_t* d_ksk_minus,
    const uint32_t* d_ksk5_ntt, const uint32_t* d_gp,
    int n_steps, int d_eff, size_t n)
{
    int log_n = 0;
    while (((size_t)1 << log_n) < n) log_n++;
    const uint32_t two_n_mask = (uint32_t)(2 * n - 1);
    dim3 grid((unsigned)n_steps, (unsigned)d_eff);
    ksk_perm_batched_kernel<<<grid, 256>>>(
        d_ksk_plus, d_ksk_minus, d_ksk5_ntt, d_gp,
        n_steps, d_eff, (int)n, log_n, two_n_mask);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_apply_auto_perm(uint32_t* d_dst, const uint32_t* d_src,
                          const uint32_t* d_perm, size_t n) {
    size_t blocks = (n + 255) / 256;
    apply_auto_perm_kernel<<<blocks, 256>>>(d_dst, d_src, d_perm, n);
    CUDA_CHECK(cudaGetLastError());
}


void gpu_ring_embed(uint32_t* d_acc_fwd, uint32_t* d_acc_conj,
                    const uint32_t* d_a32, const uint32_t* d_mono32,
                    const uint32_t* d_gp,
                    int gamma, int half, size_t n) {
    int threads = 256;
    int kbase_blocks = (n + threads - 1) / threads;
    dim3 grid(kbase_blocks, half, 2);
    // Phase 11: pass full-u64 Barrett constants (cr = floor(2^64 / q)) instead
    // of the legacy 2^55-shift Barrett mu — the inner accumulator can exceed
    // 2^55 between reductions.
    const uint64_t cr0 = (uint64_t)(((__uint128_t)1 << 64) / (uint64_t)Q0);
    const uint64_t cr1 = (uint64_t)(((__uint128_t)1 << 64) / (uint64_t)Q1);
    ring_embed_kernel<<<grid, threads>>>(
        d_acc_fwd, d_acc_conj, d_a32, d_mono32, d_gp,
        gamma, half, (int)n,
        Q0, cr0,
        Q1, cr1);
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================
// Fused a-side matmul + build_a32, for ring_embed input.
//
// Replaces a CPU loop over (j, r, i) that does, per output column j of
// a packing group g:
//   acc = 0  (NTT form)
//   for r in 0..n_r-1:
//     db_col[i] = d_db[ db_index<RM>(g*N+j, r*N+i) ]   (row=r*N+i, col=g*N+j)
//     acc += NTT(db_col) * a_crs_rev_ntt[r]
//   lwe_a[j] = INTT(acc)                              -- coefficient form
//   d_a32[j] = NTT(neg_rev(lwe_a[j]) * (1/N))         -- ring_embed input
//
// Algebraic identity (verified): NTT(neg_rev(INTT(g)))[k] = g[N-1-k].
// So d_a32[j][k] = (1/N) * acc[j][N-1-k]; the INTT + neg_rev + NTT
// round-trip collapses to a single reverse-and-scale write.
//
// One block per (j); each block loads db_col into shared memory, NTTs in
// place, multiplies + accumulates over r, then writes d_a32 with reversal.
//
// Layout:
//   d_db:                row-major u16, shape [db_rows × db_cols] (db[row*db_cols+col]);
//                        addressed via db_index<RM> (RM=true in production)
//   d_a_crs_q*_ntt:      n_r × n, NTT-form, one per RNS prime
//   d_a32:               [gamma=N][limb][coeff], 2n uint32 per j (NTT form)
// ============================================================

// modmul_u32 / modadd_u32 / modsub_u32 / ntt_forward_shared definitions
// are in gpu_common.cuh.

template<bool RM>
__global__
void a_side_matmul_a32_kernel(
    uint32_t* d_a32,                   // [gamma][limb][coeff], 2n uint32/j, NTT form
    const uint16_t* d_db,              // u16 DB; layout per RM (col-major or row-major)
    size_t db_rows, size_t db_cols,
    size_t g, size_t n_r, size_t n,
    const uint32_t* d_a_crs_q0_ntt,    // n_r × n
    const uint32_t* d_a_crs_q1_ntt,    // n_r × n
    uint32_t nu_inv_q0, uint32_t nu_inv_q1,
    const uint32_t* d_fwd_q0, const uint32_t* d_fwd_q1,
    uint32_t Q0c, uint32_t Q1c)
{
    extern __shared__ uint32_t shared[];
    // Shared layout: [limb0_work: n][limb1_work: n][limb0_acc: n][limb1_acc: n]
    // = 4n uint32. For n=2048 that's 32 KB.
    uint32_t* l0 = shared;
    uint32_t* l1 = shared + n;
    uint32_t* a0 = shared + 2 * n;
    uint32_t* a1 = shared + 3 * n;

    const size_t j = blockIdx.x;
    const size_t col = g * n + j;

    // acc = 0
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        a0[i] = 0;
        a1[i] = 0;
    }
    __syncthreads();

    for (size_t r = 0; r < n_r; r++) {
        // Load db_col -> l0 and l1 (same values; different moduli for NTT).
        // Phase 14: d_db is uint16_t; widen to u32 for NTT arithmetic.
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
            uint32_t v = (uint32_t)d_db[db_index<RM>(col, r * n + i, db_rows, db_cols)];
            l0[i] = v;
            l1[i] = v;
        }
        __syncthreads();

        // NTT both limbs (decimation-in-time, mirrors ntt_fwd_kernel)
        ntt_forward_shared(l0, n, d_fwd_q0, Q0c);
        ntt_forward_shared(l1, n, d_fwd_q1, Q1c);

        // l_c *= a_crs_rev_ntt[r] elementwise; acc_c += l_c
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
            uint32_t p0 = modmul_u32(l0[i], d_a_crs_q0_ntt[r * n + i], Q0c);
            uint32_t p1 = modmul_u32(l1[i], d_a_crs_q1_ntt[r * n + i], Q1c);
            a0[i] = modadd_u32(a0[i], p0, Q0c);
            a1[i] = modadd_u32(a1[i], p1, Q1c);
        }
        __syncthreads();
    }

    // Write d_a32 with index reversal k -> n-1-k and scale by 1/N.
    // d_a32 layout per j: [limb0: n][limb1: n]
    uint32_t* out = d_a32 + j * 2 * n;
    for (size_t k = threadIdx.x; k < n; k += blockDim.x) {
        size_t src = n - 1 - k;
        out[k]     = modmul_u32(a0[src], nu_inv_q0, Q0c);
        out[n + k] = modmul_u32(a1[src], nu_inv_q1, Q1c);
    }
}


void gpu_a_side_matmul_a32(
    uint32_t* d_a32,
    const uint16_t* d_db, size_t db_rows, size_t db_cols,
    size_t g, size_t n_r, size_t n,
    const uint32_t* d_a_crs_q0_ntt, const uint32_t* d_a_crs_q1_ntt,
    uint32_t nu_inv_q0, uint32_t nu_inv_q1,
    const uint32_t* d_fwd_q0, const uint32_t* d_fwd_q1,
    uint32_t Q0c, uint32_t Q1c)
{
    size_t shared_bytes = 4 * n * sizeof(uint32_t);
    // DB is row-major (db[row*db_cols+col]); matvec reads it directly online.
    a_side_matmul_a32_kernel<true><<<n, 256, shared_bytes>>>(
        d_a32, d_db, db_rows, db_cols, g, n_r, n,
        d_a_crs_q0_ntt, d_a_crs_q1_ntt,
        nu_inv_q0, nu_inv_q1,
        d_fwd_q0, d_fwd_q1,
        Q0c, Q1c);
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================
// Phase 8/10: register-based fused collapse_a kernel.
//
// Per block (one packing group, blockIdx.x = col):
//   Forward direction: for s = 0..n_steps-1 (k = N/2-1..1):
//     INTT state → coeff (per limb), CRT compose + sign-center, gadget
//     extract (skip digit 0), NTT each piece, store to D_plus[s], lazy
//     multiply-accumulate ksk_plus[s]; reload a[k-1] − acc.
//   Conjugate direction: same on a_post_conj, writes D_minus.
//   Final: decompose state, write D_final, out_a = saved_fwd0 − D_final·ksk_neg1.
//
// Intermediate per-coefficient state (state_qX[8], coeff_qX[8], signed_v[8],
// acc_qX[8], dig_ntt_qX[8], saved_fwd0_qX[8]) lives in per-thread registers.
// Shared mem holds only one ntt_buf of N u32 (8 KB), reused per NTT/INTT.
//
// Layout of d_out_D_plus / d_out_D_minus: [limb][step][digit][coeff].
// ============================================================

__global__ void collapse_fused_kernel(
    // Phase 10: per-group input/output pointer arrays; blockIdx.x selects the
    // group. KSK and twiddles remain shared across groups (single pointers).
    const uint32_t* const* __restrict__ d_a_post_fwd_ptrs,
    const uint32_t* const* __restrict__ d_a_post_conj_ptrs,
    const uint32_t* d_ksk_plus,
    const uint32_t* d_ksk_minus,
    const uint32_t* d_ksk_neg1,
    const uint32_t* d_fwd_q0,        const uint32_t* d_fwd_q0_prime,
    const uint32_t* d_inv_q0,        const uint32_t* d_inv_q0_prime,
    const uint32_t* d_fwd_q1,        const uint32_t* d_fwd_q1_prime,
    const uint32_t* d_inv_q1,        const uint32_t* d_inv_q1_prime,
    uint32_t inv_n_q0, uint32_t inv_n_q0_prime,
    uint32_t inv_n_q1, uint32_t inv_n_q1_prime,
    uint32_t* const* __restrict__ d_out_a_ptrs,
    uint32_t* const* __restrict__ d_out_D_plus_ptrs,
    uint32_t* const* __restrict__ d_out_D_minus_ptrs,
    uint32_t* const* __restrict__ d_out_D_final_ptrs,
    size_t n, size_t half, size_t n_steps,
    uint32_t Q0c, uint32_t Q1c,
    uint64_t Q0c_cr, uint64_t Q1c_cr,    // Phase 9: Barrett constants = floor(2^64 / q)
    uint64_t q1_inv_q0, uint64_t q0_inv_q1,
    uint64_t q_full_lo, uint64_t /*q_full_hi*/,
    int d_gsw, int /*d_eff*/, int base_log)
{
    extern __shared__ uint32_t ntt_buf[];   // N u32 only (one NTT in flight)

    // Phase 10: pick this block's per-group buffers
    const int col = blockIdx.x;
    const uint32_t* d_a_post_fwd  = d_a_post_fwd_ptrs[col];
    const uint32_t* d_a_post_conj = d_a_post_conj_ptrs[col];
    uint32_t* d_out_a       = d_out_a_ptrs[col];
    uint32_t* d_out_D_plus  = d_out_D_plus_ptrs[col];
    uint32_t* d_out_D_minus = d_out_D_minus_ptrs[col];
    uint32_t* d_out_D_final = d_out_D_final_ptrs[col];

    constexpr int ELEMS_PER_THREAD = 8;     // N=2048 / 256 threads
    const int tid = threadIdx.x;
    const int THREADS = 256;

    const int d_eff_local      = d_gsw - 1;  // signed-decomp skips digit 0
    const size_t per_limb_dp   = n_steps * d_eff_local * n;
    const size_t per_limb_ksk  = n_steps * d_eff_local * n;
    const size_t neg1_per_limb = d_eff_local * n;
    const int K_SHIFT = 64 - base_log;
    const uint64_t Q_FULL = q_full_lo;       // 27-bit primes ⇒ Q_full < 2^53, q_full_hi == 0
    const uint64_t Q_HALF = Q_FULL >> 1;

    // ---- Long-lived register state across iters (current a[k] in NTT form) ----
    uint32_t state_q0[ELEMS_PER_THREAD];
    uint32_t state_q1[ELEMS_PER_THREAD];

    // ============== FORWARD DIRECTION ==============
    {
        const size_t off = (half - 1) * 2 * n;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int z = tid + e * THREADS;
            state_q0[e] = d_a_post_fwd[off + z];
            state_q1[e] = d_a_post_fwd[off + n + z];
        }
    }

    for (size_t s = 0; s < n_steps; s++) {
        const size_t k = half - 1 - s;
        const size_t kskstep = n_steps - 1 - s;

        // INTT state_q0 (state → coeff_q0[8])
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++)
            ntt_buf[tid + e * THREADS] = state_q0[e];
        __syncthreads();
        ntt_inverse_shared_shoup(ntt_buf, n, Q0c, d_inv_q0, d_inv_q0_prime, inv_n_q0, inv_n_q0_prime);
        uint32_t coeff_q0[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++)
            coeff_q0[e] = ntt_buf[tid + e * THREADS];
        __syncthreads();

        // INTT state_q1 → coeff_q1[8]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++)
            ntt_buf[tid + e * THREADS] = state_q1[e];
        __syncthreads();
        ntt_inverse_shared_shoup(ntt_buf, n, Q1c, d_inv_q1, d_inv_q1_prime, inv_n_q1, inv_n_q1_prime);
        uint32_t coeff_q1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++)
            coeff_q1[e] = ntt_buf[tid + e * THREADS];
        __syncthreads();

        // CRT compose + sign-center
        int64_t signed_v[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const uint64_t v0 = coeff_q0[e];
            const uint64_t v1 = coeff_q1[e];
            const uint64_t t0 = barrett_u64_dev(v0 * q1_inv_q0, Q0c_cr, Q0c);
            const uint64_t t1 = barrett_u64_dev(v1 * q0_inv_q1, Q1c_cr, Q1c);
            uint64_t comp = t0 * (uint64_t)Q1c + t1 * (uint64_t)Q0c;
            if (comp >= Q_FULL) comp -= Q_FULL;
            signed_v[e] = (comp > Q_HALF) ? (int64_t)comp - (int64_t)Q_FULL : (int64_t)comp;
        }

        // Accumulator (u64 lazy)
        uint64_t acc_q0[ELEMS_PER_THREAD];
        uint64_t acc_q1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) { acc_q0[e] = 0; acc_q1[e] = 0; }

        // Gadget loop: D_GSW extracts; skip j_extract==0 (signed-decomp).
        for (int j_extract = 0; j_extract < d_gsw; j_extract++) {
            int64_t pieces[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int64_t r = signed_v[e];
                const int64_t d = ((int64_t)((uint64_t)r << K_SHIFT)) >> K_SHIFT;
                pieces[e] = d;
                signed_v[e] = (r - d) >> base_log;
            }
            if (j_extract == 0) continue;
            const int j_use = j_extract - 1;

            // Limb 0: piece mod Q0 → NTT → dig_ntt_q0
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int64_t d = pieces[e];
                const uint32_t u = (d >= 0) ? (uint32_t)d : (uint32_t)((int64_t)Q0c + d);
                ntt_buf[tid + e * THREADS] = u;
            }
            __syncthreads();
            ntt_forward_shared_shoup(ntt_buf, n, Q0c, d_fwd_q0, d_fwd_q0_prime);
            uint32_t dig_ntt_q0[ELEMS_PER_THREAD];
            const size_t dp_base_q0 = s * d_eff_local * n + (size_t)j_use * n;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                const uint32_t v = ntt_buf[z];
                dig_ntt_q0[e] = v;
                d_out_D_plus[dp_base_q0 + z] = v;
            }
            __syncthreads();

            // Limb 1
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int64_t d = pieces[e];
                const uint32_t u = (d >= 0) ? (uint32_t)d : (uint32_t)((int64_t)Q1c + d);
                ntt_buf[tid + e * THREADS] = u;
            }
            __syncthreads();
            ntt_forward_shared_shoup(ntt_buf, n, Q1c, d_fwd_q1, d_fwd_q1_prime);
            uint32_t dig_ntt_q1[ELEMS_PER_THREAD];
            const size_t dp_base_q1 = per_limb_dp + dp_base_q0;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                const uint32_t v = ntt_buf[z];
                dig_ntt_q1[e] = v;
                d_out_D_plus[dp_base_q1 + z] = v;
            }
            __syncthreads();

            // Multiply-accumulate against ksk_plus
            const uint32_t* ksk_q0 = d_ksk_plus + kskstep * d_eff_local * n + (size_t)j_use * n;
            const uint32_t* ksk_q1 = d_ksk_plus + per_limb_ksk + kskstep * d_eff_local * n + (size_t)j_use * n;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                acc_q0[e] += (uint64_t)dig_ntt_q0[e] * (uint64_t)ksk_q0[z];
                acc_q1[e] += (uint64_t)dig_ntt_q1[e] * (uint64_t)ksk_q1[z];
            }
        }

        // state = a[k-1] - acc mod q
        {
            const size_t off = (k - 1) * 2 * n;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                const uint32_t a0 = d_a_post_fwd[off + z];
                const uint32_t a1 = d_a_post_fwd[off + n + z];
                const uint32_t r0 = (uint32_t)barrett_u64_dev(acc_q0[e], Q0c_cr, Q0c);
                const uint32_t r1 = (uint32_t)barrett_u64_dev(acc_q1[e], Q1c_cr, Q1c);
                state_q0[e] = (a0 >= r0) ? a0 - r0 : a0 + Q0c - r0;
                state_q1[e] = (a1 >= r1) ? a1 - r1 : a1 + Q1c - r1;
            }
        }
    }

    // Save a_agg_fwd[0] for the final step.
    uint32_t saved_fwd0_q0[ELEMS_PER_THREAD];
    uint32_t saved_fwd0_q1[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        saved_fwd0_q0[e] = state_q0[e];
        saved_fwd0_q1[e] = state_q1[e];
    }

    // ============== CONJUGATE DIRECTION ==============
    {
        const size_t off = (half - 1) * 2 * n;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int z = tid + e * THREADS;
            state_q0[e] = d_a_post_conj[off + z];
            state_q1[e] = d_a_post_conj[off + n + z];
        }
    }

    for (size_t s = 0; s < n_steps; s++) {
        const size_t k = half - 1 - s;
        const size_t kskstep = n_steps - 1 - s;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) ntt_buf[tid + e * THREADS] = state_q0[e];
        __syncthreads();
        ntt_inverse_shared_shoup(ntt_buf, n, Q0c, d_inv_q0, d_inv_q0_prime, inv_n_q0, inv_n_q0_prime);
        uint32_t coeff_q0[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) coeff_q0[e] = ntt_buf[tid + e * THREADS];
        __syncthreads();

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) ntt_buf[tid + e * THREADS] = state_q1[e];
        __syncthreads();
        ntt_inverse_shared_shoup(ntt_buf, n, Q1c, d_inv_q1, d_inv_q1_prime, inv_n_q1, inv_n_q1_prime);
        uint32_t coeff_q1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) coeff_q1[e] = ntt_buf[tid + e * THREADS];
        __syncthreads();

        int64_t signed_v[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const uint64_t v0 = coeff_q0[e];
            const uint64_t v1 = coeff_q1[e];
            const uint64_t t0 = barrett_u64_dev(v0 * q1_inv_q0, Q0c_cr, Q0c);
            const uint64_t t1 = barrett_u64_dev(v1 * q0_inv_q1, Q1c_cr, Q1c);
            uint64_t comp = t0 * (uint64_t)Q1c + t1 * (uint64_t)Q0c;
            if (comp >= Q_FULL) comp -= Q_FULL;
            signed_v[e] = (comp > Q_HALF) ? (int64_t)comp - (int64_t)Q_FULL : (int64_t)comp;
        }

        uint64_t acc_q0[ELEMS_PER_THREAD];
        uint64_t acc_q1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) { acc_q0[e] = 0; acc_q1[e] = 0; }

        for (int j_extract = 0; j_extract < d_gsw; j_extract++) {
            int64_t pieces[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int64_t r = signed_v[e];
                const int64_t d = ((int64_t)((uint64_t)r << K_SHIFT)) >> K_SHIFT;
                pieces[e] = d;
                signed_v[e] = (r - d) >> base_log;
            }
            if (j_extract == 0) continue;
            const int j_use = j_extract - 1;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int64_t d = pieces[e];
                const uint32_t u = (d >= 0) ? (uint32_t)d : (uint32_t)((int64_t)Q0c + d);
                ntt_buf[tid + e * THREADS] = u;
            }
            __syncthreads();
            ntt_forward_shared_shoup(ntt_buf, n, Q0c, d_fwd_q0, d_fwd_q0_prime);
            uint32_t dig_ntt_q0[ELEMS_PER_THREAD];
            const size_t dm_base_q0 = s * d_eff_local * n + (size_t)j_use * n;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                const uint32_t v = ntt_buf[z];
                dig_ntt_q0[e] = v;
                d_out_D_minus[dm_base_q0 + z] = v;
            }
            __syncthreads();

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int64_t d = pieces[e];
                const uint32_t u = (d >= 0) ? (uint32_t)d : (uint32_t)((int64_t)Q1c + d);
                ntt_buf[tid + e * THREADS] = u;
            }
            __syncthreads();
            ntt_forward_shared_shoup(ntt_buf, n, Q1c, d_fwd_q1, d_fwd_q1_prime);
            uint32_t dig_ntt_q1[ELEMS_PER_THREAD];
            const size_t dm_base_q1 = per_limb_dp + dm_base_q0;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                const uint32_t v = ntt_buf[z];
                dig_ntt_q1[e] = v;
                d_out_D_minus[dm_base_q1 + z] = v;
            }
            __syncthreads();

            const uint32_t* ksk_q0 = d_ksk_minus + kskstep * d_eff_local * n + (size_t)j_use * n;
            const uint32_t* ksk_q1 = d_ksk_minus + per_limb_ksk + kskstep * d_eff_local * n + (size_t)j_use * n;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                acc_q0[e] += (uint64_t)dig_ntt_q0[e] * (uint64_t)ksk_q0[z];
                acc_q1[e] += (uint64_t)dig_ntt_q1[e] * (uint64_t)ksk_q1[z];
            }
        }

        {
            const size_t off = (k - 1) * 2 * n;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                const int z = tid + e * THREADS;
                const uint32_t a0 = d_a_post_conj[off + z];
                const uint32_t a1 = d_a_post_conj[off + n + z];
                const uint32_t r0 = (uint32_t)barrett_u64_dev(acc_q0[e], Q0c_cr, Q0c);
                const uint32_t r1 = (uint32_t)barrett_u64_dev(acc_q1[e], Q1c_cr, Q1c);
                state_q0[e] = (a0 >= r0) ? a0 - r0 : a0 + Q0c - r0;
                state_q1[e] = (a1 >= r1) ? a1 - r1 : a1 + Q1c - r1;
            }
        }
    }

    // ============== FINAL STEP ==============
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) ntt_buf[tid + e * THREADS] = state_q0[e];
    __syncthreads();
    ntt_inverse_shared_shoup(ntt_buf, n, Q0c, d_inv_q0, d_inv_q0_prime, inv_n_q0, inv_n_q0_prime);
    uint32_t coeff_q0[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) coeff_q0[e] = ntt_buf[tid + e * THREADS];
    __syncthreads();

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) ntt_buf[tid + e * THREADS] = state_q1[e];
    __syncthreads();
    ntt_inverse_shared_shoup(ntt_buf, n, Q1c, d_inv_q1, d_inv_q1_prime, inv_n_q1, inv_n_q1_prime);
    uint32_t coeff_q1[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) coeff_q1[e] = ntt_buf[tid + e * THREADS];
    __syncthreads();

    int64_t signed_v[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        const uint64_t v0 = coeff_q0[e];
        const uint64_t v1 = coeff_q1[e];
        const uint64_t t0 = barrett_u64_dev(v0 * q1_inv_q0, Q0c_cr, Q0c);
        const uint64_t t1 = barrett_u64_dev(v1 * q0_inv_q1, Q1c_cr, Q1c);
        uint64_t comp = t0 * (uint64_t)Q1c + t1 * (uint64_t)Q0c;
        if (comp >= Q_FULL) comp -= Q_FULL;
        signed_v[e] = (comp > Q_HALF) ? (int64_t)comp - (int64_t)Q_FULL : (int64_t)comp;
    }

    uint64_t acc_q0[ELEMS_PER_THREAD];
    uint64_t acc_q1[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) { acc_q0[e] = 0; acc_q1[e] = 0; }

    for (int j_extract = 0; j_extract < d_gsw; j_extract++) {
        int64_t pieces[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int64_t r = signed_v[e];
            const int64_t d = ((int64_t)((uint64_t)r << K_SHIFT)) >> K_SHIFT;
            pieces[e] = d;
            signed_v[e] = (r - d) >> base_log;
        }
        if (j_extract == 0) continue;
        const int j_use = j_extract - 1;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int64_t d = pieces[e];
            const uint32_t u = (d >= 0) ? (uint32_t)d : (uint32_t)((int64_t)Q0c + d);
            ntt_buf[tid + e * THREADS] = u;
        }
        __syncthreads();
        ntt_forward_shared_shoup(ntt_buf, n, Q0c, d_fwd_q0, d_fwd_q0_prime);
        uint32_t dig_ntt_q0[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int z = tid + e * THREADS;
            const uint32_t v = ntt_buf[z];
            dig_ntt_q0[e] = v;
            d_out_D_final[(size_t)j_use * n + z] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int64_t d = pieces[e];
            const uint32_t u = (d >= 0) ? (uint32_t)d : (uint32_t)((int64_t)Q1c + d);
            ntt_buf[tid + e * THREADS] = u;
        }
        __syncthreads();
        ntt_forward_shared_shoup(ntt_buf, n, Q1c, d_fwd_q1, d_fwd_q1_prime);
        uint32_t dig_ntt_q1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int z = tid + e * THREADS;
            const uint32_t v = ntt_buf[z];
            dig_ntt_q1[e] = v;
            d_out_D_final[neg1_per_limb + (size_t)j_use * n + z] = v;
        }
        __syncthreads();

        const uint32_t* knx0 = d_ksk_neg1 + (size_t)j_use * n;
        const uint32_t* knx1 = d_ksk_neg1 + neg1_per_limb + (size_t)j_use * n;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            const int z = tid + e * THREADS;
            acc_q0[e] += (uint64_t)dig_ntt_q0[e] * (uint64_t)knx0[z];
            acc_q1[e] += (uint64_t)dig_ntt_q1[e] * (uint64_t)knx1[z];
        }
    }

    // out_a = saved_fwd0 - acc mod q
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        const int z = tid + e * THREADS;
        const uint32_t r0 = (uint32_t)barrett_u64_dev(acc_q0[e], Q0c_cr, Q0c);
        const uint32_t r1 = (uint32_t)barrett_u64_dev(acc_q1[e], Q1c_cr, Q1c);
        const uint32_t a0 = saved_fwd0_q0[e];
        const uint32_t a1 = saved_fwd0_q1[e];
        d_out_a[z]     = (a0 >= r0) ? a0 - r0 : a0 + Q0c - r0;
        d_out_a[n + z] = (a1 >= r1) ? a1 - r1 : a1 + Q1c - r1;
    }
}


void gpu_collapse_fused(
    const uint32_t* const* d_a_post_fwd_ptrs,
    const uint32_t* const* d_a_post_conj_ptrs,
    const uint32_t* d_ksk_plus, const uint32_t* d_ksk_minus,
    const uint32_t* d_ksk_neg1,
    const uint32_t* d_fwd_q0, const uint32_t* d_fwd_q0_prime,
    const uint32_t* d_inv_q0, const uint32_t* d_inv_q0_prime,
    const uint32_t* d_fwd_q1, const uint32_t* d_fwd_q1_prime,
    const uint32_t* d_inv_q1, const uint32_t* d_inv_q1_prime,
    uint32_t inv_n_q0, uint32_t inv_n_q0_prime,
    uint32_t inv_n_q1, uint32_t inv_n_q1_prime,
    uint32_t* const* d_out_a_ptrs,
    uint32_t* const* d_out_D_plus_ptrs,
    uint32_t* const* d_out_D_minus_ptrs,
    uint32_t* const* d_out_D_final_ptrs,
    size_t n, size_t half, size_t n_steps,
    uint32_t Q0c, uint32_t Q1c,
    uint64_t q1_inv_q0, uint64_t q0_inv_q1,
    uint64_t q_full_lo, uint64_t q_full_hi,
    int d_gsw, int d_eff, int base_log,
    int num_groups)
{
    assert(n == 2048);
    const size_t shared_bytes = n * sizeof(uint32_t);

    // Phase 9: Barrett constant cr = floor(2^64 / q) for each limb.
    const uint64_t Q0c_cr = (uint64_t)(((__uint128_t)1 << 64) / Q0c);
    const uint64_t Q1c_cr = (uint64_t)(((__uint128_t)1 << 64) / Q1c);

    collapse_fused_kernel<<<num_groups, 256, shared_bytes>>>(
        d_a_post_fwd_ptrs, d_a_post_conj_ptrs,
        d_ksk_plus, d_ksk_minus, d_ksk_neg1,
        d_fwd_q0, d_fwd_q0_prime,
        d_inv_q0, d_inv_q0_prime,
        d_fwd_q1, d_fwd_q1_prime,
        d_inv_q1, d_inv_q1_prime,
        inv_n_q0, inv_n_q0_prime, inv_n_q1, inv_n_q1_prime,
        d_out_a_ptrs, d_out_D_plus_ptrs, d_out_D_minus_ptrs, d_out_D_final_ptrs,
        n, half, n_steps,
        Q0c, Q1c,
        Q0c_cr, Q1c_cr,
        q1_inv_q0, q0_inv_q1, q_full_lo, q_full_hi,
        d_gsw, d_eff, base_log);
    CUDA_CHECK(cudaGetLastError());
}


template<bool RM>
__global__ void encode_inverse_dft_kernel(
    uint16_t* db,
    size_t db_rows, size_t db_cols,
    size_t N_poly,
    size_t D_actual,
    int log2_D,
    uint16_t D_inv)
{
    extern __shared__ uint32_t shared[];
    // [sh_even: N][sh_odd: N][sh_temp: N] — 3N uint32, 24 KB at N=2048
    uint32_t* sh_even = shared;
    uint32_t* sh_odd  = shared + N_poly;
    uint32_t* sh_temp = shared + 2 * N_poly;

    const size_t row = blockIdx.x;
    const size_t ct  = blockIdx.y;
    if (row >= db_rows) return;

    const size_t base_offset = ct * D_actual * N_poly;
    const size_t two_N = 2 * N_poly;
    constexpr uint32_t P_mod = 65535u;

    // Step 1: bit-reverse permutation across d-axis (in-place swap).
    for (size_t d = 0; d < D_actual; d++) {
        size_t rev = 0;
        for (int b = 0; b < log2_D; b++) {
            if (d & ((size_t)1 << b)) rev |= (size_t)1 << (log2_D - 1 - b);
        }
        if (d < rev) {
            size_t col_d   = base_offset + d   * N_poly;
            size_t col_rev = base_offset + rev * N_poly;
            for (size_t k = threadIdx.x; k < N_poly; k += blockDim.x) {
                size_t ad = db_index<RM>(col_d   + k, row, db_rows, db_cols);
                size_t ar = db_index<RM>(col_rev + k, row, db_rows, db_cols);
                uint16_t tmp = db[ad]; db[ad] = db[ar]; db[ar] = tmp;
            }
            __syncthreads();
        }
    }

    // Step 2: log2(D) levels of butterflies (monomial twiddles, no NTT).
    for (int level = 0; level < log2_D; level++) {
        const size_t size_grp = (size_t)2 << level; // 2, 4, ..., D
        const size_t half     = size_grp / 2;

        for (size_t base = 0; base < D_actual; base += size_grp) {
            for (size_t i = 0; i < half; i++) {
                // Monomial twiddle exponent: X^{exp_val} in Z_p[X]/(X^N+1).
                const size_t exp_val =
                    (two_N - ((two_N * i) / size_grp) % two_N) % two_N;

                const size_t col_even = base_offset + (base + i)        * N_poly;
                const size_t col_odd  = base_offset + (base + i + half) * N_poly;

                // Load even/odd into shared (widen u16 -> u32 for arithmetic)
                for (size_t k = threadIdx.x; k < N_poly; k += blockDim.x) {
                    sh_even[k] = (uint32_t)db[db_index<RM>(col_even + k, row, db_rows, db_cols)];
                    sh_odd[k]  = (uint32_t)db[db_index<RM>(col_odd  + k, row, db_rows, db_cols)];
                }
                __syncthreads();

                // Butterfly with monomial-twiddle multiplication = coefficient
                // rotation with negacyclic wrap (sign flip on wrap).
                for (size_t k = threadIdx.x; k < N_poly; k += blockDim.x) {
                    size_t src = k + two_N - exp_val;
                    if (src >= two_N) src -= two_N;

                    uint32_t temp_k;
                    if (src < N_poly) {
                        temp_k = sh_odd[src];
                    } else {
                        uint32_t v = sh_odd[src - N_poly];
                        temp_k = (v == 0) ? 0 : (P_mod - v);
                    }

                    uint32_t even_val = sh_even[k];
                    uint32_t sum  = even_val + temp_k;       if (sum  >= P_mod) sum  -= P_mod;
                    uint32_t diff = even_val + P_mod - temp_k; if (diff >= P_mod) diff -= P_mod;
                    sh_even[k] = sum;
                    sh_temp[k] = diff;
                }
                __syncthreads();

                for (size_t k = threadIdx.x; k < N_poly; k += blockDim.x) {
                    db[db_index<RM>(col_even + k, row, db_rows, db_cols)] = (uint16_t)sh_even[k];
                    db[db_index<RM>(col_odd  + k, row, db_rows, db_cols)] = (uint16_t)sh_temp[k];
                }
                __syncthreads();
            }
        }
    }

    // Step 3: scale by D^{-1} mod P. Multiply max ≈ 65535 × 65535 < 2^32 so
    // mod_p_mersenne handles it.
    for (size_t d = 0; d < D_actual; d++) {
        const size_t col_d = base_offset + d * N_poly;
        for (size_t k = threadIdx.x; k < N_poly; k += blockDim.x) {
            const size_t idx = db_index<RM>(col_d + k, row, db_rows, db_cols);
            uint32_t v = (uint32_t)db[idx] * (uint32_t)D_inv;
            db[idx] = (uint16_t)mod_p_mersenne(v);
        }
    }
}


// Slot-native encode: d_db holds plaintext slots in [0,P), ROW-MAJOR
// (db[row*db_cols+col]); applies only the in-place inverse-DFT. The online
// matvec reads this row-major DB directly — no transpose, single resident copy.
void gpu_encode_inverse_dft(uint16_t* d_db, size_t db_rows, size_t db_cols,
                            size_t D, size_t poly_len) {
    // Inverse DFT (only if D > 1 after clamping to n_packed).
    const size_t n_packed = db_cols / poly_len;
    const size_t D_actual = D < n_packed ? D : n_packed;
    if (D_actual <= 1) return;

    const size_t num_cts = n_packed / D_actual;

    // D^{-1} mod 65535 via extended Euclidean
    constexpr int64_t P_int = 65535;
    int64_t old_r = (int64_t)D_actual, r = P_int;
    int64_t old_s = 1, s = 0;
    while (r != 0) {
        int64_t qq = old_r / r;
        int64_t t1 = r; r = old_r - qq * r; old_r = t1;
        int64_t t2 = s; s = old_s - qq * s; old_s = t2;
    }
    const uint16_t D_inv =
        (uint16_t)((old_s % P_int + P_int) % P_int);

    int log2_D = 0;
    while (((size_t)1 << log2_D) < D_actual) log2_D++;

    const int threads = 256;
    dim3 grid((unsigned)db_rows, (unsigned)num_cts);
    size_t shared_bytes = 3 * poly_len * sizeof(uint32_t);
    encode_inverse_dft_kernel<true><<<grid, threads, shared_bytes>>>(
        d_db, db_rows, db_cols, poly_len, D_actual, log2_D, D_inv);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace inspire
