// gpu_online.cu: per-query online kernels — mat-vec, lazy-collapse
// (partial/reduce), external product, Horner, and the final fused inner-product.
// Run on every query by gpu_answer (see gpu_protocol.cu).

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
// Mat-vec kernel: result[j] = sum_i db_rm[i*db_cols + j] * query[i] mod q
// (DB is row-major, so consecutive output columns j coalesce at a fixed row i)
// ============================================================

// Dual-limb matvec: decodes the in-place centered-byte representation and
// writes directly to interleaved packed-polynomial storage.
__device__ __forceinline__ uint16_t decode_centered_u16(uint16_t packed) {
    int lo = (int)(int8_t)(packed & 0xffu) + 128;
    int hi = (int)(int8_t)(packed >> 8) + 128;
    return (uint16_t)(lo | (hi << 8));
}

__global__
void matvec_kernel_dual(uint32_t* result0, uint32_t* result1,
                        const uint16_t* db_rm,
                        const uint32_t* query_mod0, const uint32_t* query_mod1,
                        size_t db_rows, size_t db_cols,
                        uint32_t q0, uint32_t q1) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= db_cols) return;

    uint64_t acc0 = 0, acc1 = 0;
    for (size_t i = 0; i < db_rows; i++) {
        uint64_t db_val = decode_centered_u16(db_rm[i * db_cols + j]);
        acc0 += db_val * query_mod0[i];
        acc1 += db_val * query_mod1[i];
        if ((i & 0xFFF) == 0xFFF) { acc0 %= q0; acc1 %= q1; }
    }
    size_t out = (j / N) * 2 * N + (j % N);
    result0[out] = (uint32_t)(acc0 % q0);
    result1[out + N] = (uint32_t)(acc1 % q1);
}


// Dual-limb matvec with parallel reduction across row blocks (tall matrices).
__global__
void matvec_kernel_dual_par(uint32_t* partials0, uint32_t* partials1,
                            const uint16_t* db_rm,
                            const uint32_t* query_mod0, const uint32_t* query_mod1,
                            size_t db_rows, size_t db_cols,
                            uint32_t q0, uint32_t q1,
                            size_t rows_per_block) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= db_cols) return;

    size_t row_start = blockIdx.y * rows_per_block;
    size_t row_end = row_start + rows_per_block;
    if (row_end > db_rows) row_end = db_rows;

    uint64_t acc0 = 0, acc1 = 0;
    for (size_t i = row_start; i < row_end; i++) {
        uint64_t db_val = decode_centered_u16(db_rm[i * db_cols + j]);
        acc0 += db_val * query_mod0[i];
        acc1 += db_val * query_mod1[i];
        if ((i & 0xFFF) == 0xFFF) { acc0 %= q0; acc1 %= q1; }
    }
    partials0[blockIdx.y * db_cols + j] = (uint32_t)(acc0 % q0);
    partials1[blockIdx.y * db_cols + j] = (uint32_t)(acc1 % q1);
}


__global__
void matvec_reduce2_kernel(uint32_t* result0, uint32_t* result1,
                           const uint32_t* partials0, const uint32_t* partials1,
                           size_t db_cols, int n_row_blocks,
                           uint32_t q0, uint32_t q1) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= db_cols) return;
    uint64_t s0 = 0, s1 = 0;
    for (int b = 0; b < n_row_blocks; b++) {
        s0 += partials0[b * db_cols + j];
        s1 += partials1[b * db_cols + j];
    }
    size_t out = (j / N) * 2 * N + (j % N);
    result0[out] = (uint32_t)(s0 % q0);
    result1[out + N] = (uint32_t)(s1 % q1);
}


// ============================================================
// Batched dual-limb matvec (Phase B2): BT queries share one DB stream.
// One thread per output column; per row the u16 DB value is loaded once and
// applied to BT register accumulators, so arithmetic intensity scales with
// the batch while memory traffic stays one DB read per launch. Pointer
// arrays live on device (built once at setup from the slot pool). Lanes
// b >= batch compute against lane 0's query as padding and are not written
// back. Accumulation order per query is identical to matvec_kernel_dual,
// so batched results are bit-identical to the single-query kernel's.
// ============================================================

template <int BT>
__global__
void matvec_kernel_dual_batched(uint32_t* const* results0, uint32_t* const* results1,
                                const uint16_t* db_rm,
                                const uint32_t* const* queries0,
                                const uint32_t* const* queries1,
                                size_t db_rows, size_t db_cols,
                                uint32_t q0, uint32_t q1, int batch) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= db_cols) return;

    const uint32_t* qp0[BT];
    const uint32_t* qp1[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        int src = (b < batch) ? b : 0;
        qp0[b] = queries0[src];
        qp1[b] = queries1[src];
    }

    uint64_t acc0[BT], acc1[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) { acc0[b] = 0; acc1[b] = 0; }

    for (size_t i = 0; i < db_rows; i++) {
        uint64_t db_val = decode_centered_u16(db_rm[i * db_cols + j]);
        #pragma unroll
        for (int b = 0; b < BT; b++) {
            acc0[b] += db_val * qp0[b][i];
            acc1[b] += db_val * qp1[b][i];
        }
        if ((i & 0xFFF) == 0xFFF) {
            #pragma unroll
            for (int b = 0; b < BT; b++) { acc0[b] %= q0; acc1[b] %= q1; }
        }
    }
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        if (b < batch) {
            size_t out = (j / N) * 2 * N + (j % N);
            results0[b][out] = (uint32_t)(acc0[b] % q0);
            results1[b][out + N] = (uint32_t)(acc1[b] % q1);
        }
    }
}

void gpu_matvec_dual_batched(uint32_t* const* d_results0, uint32_t* const* d_results1,
                             const uint16_t* d_db_rm,
                             const uint32_t* const* d_queries0,
                             const uint32_t* const* d_queries1,
                             size_t db_rows, size_t db_cols,
                             uint32_t q0, uint32_t q1, int batch) {
    size_t threads = 256;
    size_t blocks = (db_cols + threads - 1) / threads;

    // Tile of 4 queries per DB stream: measured on RTX 5090 (16 GB DB) the
    // 8-wide tile's register pressure costs more than its extra reuse buys;
    // 4 is the sweet spot. Each chunk streams the DB once.
    for (int off = 0; off < batch; off += 4) {
        int chunk = batch - off > 4 ? 4 : batch - off;
        auto r0 = d_results0 + off, r1 = d_results1 + off;
        auto s0 = d_queries0 + off, s1 = d_queries1 + off;
        if (chunk > 2)
            matvec_kernel_dual_batched<4><<<blocks, threads>>>(
                r0, r1, d_db_rm, s0, s1, db_rows, db_cols, q0, q1, chunk);
        else
            matvec_kernel_dual_batched<2><<<blocks, threads>>>(
                r0, r1, d_db_rm, s0, s1, db_rows, db_cols, q0, q1, chunk);
    }
}

// The batched kernel is only profitable when one-thread-per-column fills the
// GPU on its own (same condition as gpu_matvec_dual's simple path). Tall
// narrow geometries need the row-split kernel, which the batched path does
// not implement; the caller falls back to per-query gpu_matvec_dual there.
bool gpu_matvec_batched_profitable(size_t db_cols) {
    return (db_cols + 255) / 256 >= 150;
}

// ============================================================
// External product accumulator (single RNS limb).
// Computes:
//   res_a[i] = sum_{j=0..d_eff-1} ( a_dig[j][i]*rgsw_top_a[j][i]
//                                 + b_dig[j][i]*rgsw_bot_a[j][i] ) mod q
//   res_b[i] = sum_{j=0..d_eff-1} ( a_dig[j][i]*rgsw_top_b[j][i]
//                                 + b_dig[j][i]*rgsw_bot_b[j][i] ) mod q
// All arguments are in NTT form. Reduces inside the loop to keep
// intermediate sums in Barrett's safe range (< 2^55).
// ============================================================

__global__
void ext_prod_acc_kernel(
    uint32_t* res_a, uint32_t* res_b,
    const uint32_t* a_dig, const uint32_t* b_dig,
    const uint32_t* rgsw_top_a, const uint32_t* rgsw_top_b,
    const uint32_t* rgsw_bot_a, const uint32_t* rgsw_bot_b,
    int d_eff, size_t n, uint32_t q, uint32_t mu,
    const uint32_t* add_a, const uint32_t* add_b) {  // optional: res += add (fused Horner step)

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t sa = 0, sb = 0;
    for (int j = 0; j < d_eff; j++) {
        // Each iteration's sub-sum is two products: < 2 * q^2 < 2^55, Barrett-safe.
        uint64_t pa = (uint64_t)a_dig[j*n+i] * rgsw_top_a[j*n+i]
                    + (uint64_t)b_dig[j*n+i] * rgsw_bot_a[j*n+i];
        uint64_t pb = (uint64_t)a_dig[j*n+i] * rgsw_top_b[j*n+i]
                    + (uint64_t)b_dig[j*n+i] * rgsw_bot_b[j*n+i];
        uint32_t ra = barrett_mod(pa, q, mu);
        uint32_t rb = barrett_mod(pb, q, mu);
        sa = mod_add(sa, ra, q);
        sb = mod_add(sb, rb, q);
    }
    if (add_a) sa = mod_add(sa, add_a[i], q);
    if (add_b) sb = mod_add(sb, add_b[i], q);
    res_a[i] = sa;
    res_b[i] = sb;
}


void gpu_matvec_dual(uint32_t* d_result0, uint32_t* d_result1,
                     const uint16_t* d_db_rm,
                     const uint32_t* d_query_mod0, const uint32_t* d_query_mod1,
                     size_t db_rows, size_t db_cols,
                     uint32_t q0, uint32_t q1) {
    size_t threads = 256;

    // Use the simple one-thread-per-column kernel whenever there are enough
    // columns to fill the GPU on their own (db_cols/256 blocks). It streams the
    // DB once with no reduction pass — strictly less memory traffic than the
    // row-split path (which also reads/writes ~n_row_blocks×db_cols partials).
    // Only fall back to row-splitting for narrow matrices that can't otherwise
    // occupy the SMs. (~150 blocks ≈ saturates the 170-SM RTX 5090.)
    size_t col_blocks_simple = (db_cols + threads - 1) / threads;
    bool enough_columns = col_blocks_simple >= 150;

    if (enough_columns || (db_cols >= 8192 && db_rows <= 8192)) {
        // Wide matrix (or enough columns for occupancy): simple kernel is optimal
        size_t blocks = (db_cols + threads - 1) / threads;
        matvec_kernel_dual<<<blocks, threads>>>(d_result0, d_result1, d_db_rm,
                                                d_query_mod0, d_query_mod1,
                                                db_rows, db_cols, q0, q1);
    } else {
        // Tall matrix: split rows for parallelism
        size_t rows_per_block = (db_rows + 255) / 256;
        if (rows_per_block < 64) rows_per_block = 64;
        size_t n_row_blocks = (db_rows + rows_per_block - 1) / rows_per_block;

        uint32_t *d_p0, *d_p1;
        CUDA_CHECK(cudaMalloc(&d_p0, n_row_blocks * db_cols * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_p1, n_row_blocks * db_cols * sizeof(uint32_t)));

        size_t col_blocks = (db_cols + threads - 1) / threads;
        dim3 grid(col_blocks, n_row_blocks);
        matvec_kernel_dual_par<<<grid, threads>>>(d_p0, d_p1, d_db_rm,
                                                   d_query_mod0, d_query_mod1,
                                                   db_rows, db_cols, q0, q1,
                                                   rows_per_block);
        matvec_reduce2_kernel<<<col_blocks, threads>>>(d_result0, d_result1,
                                                        d_p0, d_p1, db_cols,
                                                        n_row_blocks, q0, q1);
        cudaFree(d_p0); cudaFree(d_p1);
    }
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================
// GPU RNS-poly bundle: contiguous storage for two limbs of N uint32_t
//   layout: limbs[c][i] is at &storage[c * n + i]
// Lets us pass a single pointer to "an RNS poly".
// ============================================================

// External product (both RNS limbs).
//
// Inputs (NTT form, layout: limb0 contiguous, then limb1):
//   d_ct_a:        2*n  — ciphertext a-component
//   d_ct_b:        2*n  — ciphertext b-component
//   d_rgsw_top_a:  d_eff * 2*n — RGSW top row a-parts (d_eff polys × 2 limbs)
//   d_rgsw_top_b, d_rgsw_bot_a, d_rgsw_bot_b: same shape
//
// Output (NTT form):
//   d_res_a, d_res_b: 2*n each
//
// Scratch (caller-provided to avoid per-call cudaMalloc):
//   d_scratch_coeff: 4*n  — a/b in coeff form, both limbs, laid out [a0,b0,a1,b1]
//   d_scratch_dig:   4*d_eff*n  — merged digit buffer, laid out by modulus so the
//                    forward NTTs batch: [a_q0 | b_q0 | a_q1 | b_q1]
// Optional d_add_a/d_add_b: if set, res += add (the fused Horner +packed step).
// ============================================================

void gpu_ext_prod(
    uint32_t* d_res_a, uint32_t* d_res_b,
    const uint32_t* d_ct_a, const uint32_t* d_ct_b,
    const uint32_t* d_rgsw_top_a, const uint32_t* d_rgsw_top_b,
    const uint32_t* d_rgsw_bot_a, const uint32_t* d_rgsw_bot_b,
    const uint32_t* d_fwd_twiddles_q0, const uint32_t* d_inv_twiddles_q0, uint32_t inv_n_q0,
    const uint32_t* d_fwd_twiddles_q1, const uint32_t* d_inv_twiddles_q1, uint32_t inv_n_q1,
    uint32_t* d_scratch_coeff,    // 4*n
    uint32_t* d_scratch_dig,      // 4*d_eff*n, layout [a_q0][b_q0][a_q1][b_q1]
    int d_gsw, int d_eff, int base_log, size_t n,
    const uint32_t* d_add_a, const uint32_t* d_add_b) {  // optional fused +add (Horner)

    // Scratch layout groups the two same-modulus polynomials contiguously so the
    // inverse NTTs batch: [a0,b0] (Q0) then [a1,b1] (Q1). Each batched launch runs
    // its 2 polys as 2 concurrent blocks instead of 2 serial single-block kernels.
    uint32_t* a0_coeff = d_scratch_coeff;          // Q0 pair
    uint32_t* b0_coeff = d_scratch_coeff + n;
    uint32_t* a1_coeff = d_scratch_coeff + 2*n;    // Q1 pair
    uint32_t* b1_coeff = d_scratch_coeff + 3*n;

    // Step 1: INTT each limb of ct_a, ct_b → coeff form (batched per modulus).
    CUDA_CHECK(cudaMemcpyAsync(a0_coeff, d_ct_a,     n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(b0_coeff, d_ct_b,     n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(a1_coeff, d_ct_a + n, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(b1_coeff, d_ct_b + n, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    // Batched INTT via the primitives wrapper (1024 threads): [a0,b0] / [a1,b1].
    gpu_ntt_inverse_batch(a0_coeff, n, 2, Q0, d_inv_twiddles_q0, inv_n_q0, 1024);
    gpu_ntt_inverse_batch(a1_coeff, n, 2, Q1, d_inv_twiddles_q1, inv_n_q1, 1024);

    // Step 2: gadget decompose. Merged digit buffer laid out by modulus so the
    // forward NTTs batch: [a_q0 | b_q0 | a_q1 | b_q1], each block d_eff*n.
    uint32_t* a_dig_q0 = d_scratch_dig;
    uint32_t* b_dig_q0 = d_scratch_dig + d_eff * n;
    uint32_t* a_dig_q1 = d_scratch_dig + 2 * d_eff * n;
    uint32_t* b_dig_q1 = d_scratch_dig + 3 * d_eff * n;
    gpu_gadget_decomp(a0_coeff, a1_coeff, a_dig_q0, a_dig_q1, n, d_gsw, d_eff, base_log);
    gpu_gadget_decomp(b0_coeff, b1_coeff, b_dig_q0, b_dig_q1, n, d_gsw, d_eff, base_log);

    // Step 3: NTT each digit polynomial. The Q0 polys [a_q0,b_q0] and Q1 polys
    // [a_q1,b_q1] are each contiguous, so one batched launch of 2*d_eff blocks
    // per modulus replaces 4 separate d_eff-block launches (more concurrent blocks).
    // Batched forward NTT via the primitives wrapper (1024 threads).
    gpu_ntt_batch(a_dig_q0, n, 2 * d_eff, Q0, d_fwd_twiddles_q0, 1024);
    gpu_ntt_batch(a_dig_q1, n, 2 * d_eff, Q1, d_fwd_twiddles_q1, 1024);

    // Step 4: multiply-accumulate, per limb. If d_add_a/b are given (Horner),
    // the per-limb +add is fused in here (skips a separate poly_add pass).
    {
        size_t blocks = (n + 255) / 256;
        // Limb 0 (Q0)
        ext_prod_acc_kernel<<<blocks, 256>>>(
            d_res_a, d_res_b,
            a_dig_q0, b_dig_q0,
            d_rgsw_top_a,           d_rgsw_top_b,
            d_rgsw_bot_a,           d_rgsw_bot_b,
            d_eff, n, Q0, Q0_BARRETT_MU,
            d_add_a,     d_add_b);
        // Limb 1 (Q1) — RGSW arrays for limb1 are at offset d_eff*n
        ext_prod_acc_kernel<<<blocks, 256>>>(
            d_res_a + n, d_res_b + n,
            a_dig_q1, b_dig_q1,
            d_rgsw_top_a + d_eff*n, d_rgsw_top_b + d_eff*n,
            d_rgsw_bot_a + d_eff*n, d_rgsw_bot_b + d_eff*n,
            d_eff, n, Q1, Q1_BARRETT_MU,
            d_add_a ? d_add_a + n : nullptr,
            d_add_b ? d_add_b + n : nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}


// ============================================================
// Lockstep batched Horner (Phase B4). All B chains advance the same step
// index together, so each Horner step is a fixed 8-launch sequence whose
// kernels cover every query at once — launch count stays that of a single
// chain while the per-launch work scales with B.
//
// Contiguous scratch layout per modulus (y = 2b for query b's a-component,
// 2b+1 for its b-component):
//   coeff_qX: [y][n]           — INTT input/output, 2*batch polys
//   dig_qX:   [y][d_eff][n]    — decomposed digits, NTT'd in place
// ============================================================

// acc (per-slot, NTT form) → contiguous coeff blocks, both limbs.
__global__
void horner_gather_batched_kernel(const uint32_t* const* __restrict__ acc_a_ptrs,
                                  const uint32_t* const* __restrict__ acc_b_ptrs,
                                  uint32_t* __restrict__ coeff_q0,
                                  uint32_t* __restrict__ coeff_q1, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t b = blockIdx.y;
    const uint32_t* aa = acc_a_ptrs[b];
    const uint32_t* ab = acc_b_ptrs[b];
    coeff_q0[(2 * b)     * n + i] = aa[i];
    coeff_q0[(2 * b + 1) * n + i] = ab[i];
    coeff_q1[(2 * b)     * n + i] = aa[n + i];
    coeff_q1[(2 * b + 1) * n + i] = ab[n + i];
}

// Batched multiply-accumulate (one RNS limb): grid.y = query. Reads that
// query's digits from the contiguous dig block, its RGSW parts / packed add
// / acc output via slot-base pointer arrays plus limb offsets. Semantics
// match ext_prod_acc_kernel with the fused +add, so per-query results stay
// bit-identical to the sequential Horner.
__global__
void ext_prod_acc_batched_kernel(
        uint32_t* const* __restrict__ acc_a_ptrs,
        uint32_t* const* __restrict__ acc_b_ptrs, size_t acc_off,
        const uint32_t* __restrict__ dig_base,
        const uint32_t* const* __restrict__ top_a_ptrs,
        const uint32_t* const* __restrict__ top_b_ptrs,
        const uint32_t* const* __restrict__ bot_a_ptrs,
        const uint32_t* const* __restrict__ bot_b_ptrs, size_t rgsw_off,
        const uint32_t* const* __restrict__ padd_a_ptrs,
        const uint32_t* const* __restrict__ padd_b_ptrs, size_t add_off,
        int d_eff, size_t n, uint32_t q, uint32_t mu) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t b = blockIdx.y;
    const uint32_t* a_dig = dig_base + (size_t)(2 * b)     * d_eff * n;
    const uint32_t* b_dig = dig_base + (size_t)(2 * b + 1) * d_eff * n;
    const uint32_t* ta = top_a_ptrs[b] + rgsw_off;
    const uint32_t* tb = top_b_ptrs[b] + rgsw_off;
    const uint32_t* ba = bot_a_ptrs[b] + rgsw_off;
    const uint32_t* bb = bot_b_ptrs[b] + rgsw_off;

    uint32_t sa = 0, sb = 0;
    for (int j = 0; j < d_eff; j++) {
        uint64_t pa = (uint64_t)a_dig[j*n+i] * ta[j*n+i]
                    + (uint64_t)b_dig[j*n+i] * ba[j*n+i];
        uint64_t pb = (uint64_t)a_dig[j*n+i] * tb[j*n+i]
                    + (uint64_t)b_dig[j*n+i] * bb[j*n+i];
        sa = mod_add(sa, barrett_mod(pa, q, mu), q);
        sb = mod_add(sb, barrett_mod(pb, q, mu), q);
    }
    sa = mod_add(sa, padd_a_ptrs[b][add_off + i], q);
    sb = mod_add(sb, padd_b_ptrs[b][add_off + i], q);
    acc_a_ptrs[b][acc_off + i] = sa;
    acc_b_ptrs[b][acc_off + i] = sb;
}

void gpu_horner_step_batched(
        uint32_t* const* d_acc_a_ptrs, uint32_t* const* d_acc_b_ptrs,
        const uint32_t* const* d_packed_a_ptrs,
        const uint32_t* const* d_packed_b_ptrs, size_t add_off,
        const uint32_t* const* d_top_a_ptrs, const uint32_t* const* d_top_b_ptrs,
        const uint32_t* const* d_bot_a_ptrs, const uint32_t* const* d_bot_b_ptrs,
        const uint32_t* d_fwd_twiddles_q0, const uint32_t* d_inv_twiddles_q0, uint32_t inv_n_q0,
        const uint32_t* d_fwd_twiddles_q1, const uint32_t* d_inv_twiddles_q1, uint32_t inv_n_q1,
        uint32_t* d_coeff_q0, uint32_t* d_coeff_q1,
        uint32_t* d_dig_q0, uint32_t* d_dig_q1,
        int d_gsw, int d_eff, int base_log, size_t n, int batch) {
    size_t blocks = (n + 255) / 256;
    dim3 gridB(blocks, batch);
    horner_gather_batched_kernel<<<gridB, 256>>>(
        d_acc_a_ptrs, d_acc_b_ptrs, d_coeff_q0, d_coeff_q1, n);
    gpu_ntt_inverse_batch(d_coeff_q0, n, 2 * batch, Q0, d_inv_twiddles_q0, inv_n_q0, 1024);
    gpu_ntt_inverse_batch(d_coeff_q1, n, 2 * batch, Q1, d_inv_twiddles_q1, inv_n_q1, 1024);
    gpu_gadget_decomp_strided(d_coeff_q0, d_coeff_q1, d_dig_q0, d_dig_q1,
                              n, 2 * batch, d_gsw, d_eff, base_log);
    gpu_ntt_batch(d_dig_q0, n, (size_t)2 * batch * d_eff, Q0, d_fwd_twiddles_q0, 1024);
    gpu_ntt_batch(d_dig_q1, n, (size_t)2 * batch * d_eff, Q1, d_fwd_twiddles_q1, 1024);
    ext_prod_acc_batched_kernel<<<gridB, 256>>>(
        d_acc_a_ptrs, d_acc_b_ptrs, 0,
        d_dig_q0, d_top_a_ptrs, d_top_b_ptrs, d_bot_a_ptrs, d_bot_b_ptrs, 0,
        d_packed_a_ptrs, d_packed_b_ptrs, add_off,
        d_eff, n, Q0, Q0_BARRETT_MU);
    ext_prod_acc_batched_kernel<<<gridB, 256>>>(
        d_acc_a_ptrs, d_acc_b_ptrs, n,
        d_dig_q1, d_top_a_ptrs, d_top_b_ptrs, d_bot_a_ptrs, d_bot_b_ptrs, (size_t)d_eff * n,
        d_packed_a_ptrs, d_packed_b_ptrs, add_off + n,
        d_eff, n, Q1, Q1_BARRETT_MU);
    CUDA_CHECK(cudaGetLastError());
}

// HornerEval: acc = packed[D-1]; for i = D-2 .. 0: acc = ext_prod(rgsw, acc) + packed[i]
//
// Layouts (all NTT form):
//   d_packed_a, d_packed_b: D × 2n  (D ciphertexts × 2 limbs)
//   d_rgsw_*: 2*d_eff*n each (4 arrays of d_eff polys × 2 limbs)
//   d_acc_a, d_acc_b: 2n (output, in NTT form)
//
// Scratch: same as gpu_ext_prod, plus d_scratch_res for swap target (4n total).
void gpu_horner_eval(
    uint32_t* d_acc_a, uint32_t* d_acc_b,                 // output: 2*n each
    const uint32_t* d_packed_a, const uint32_t* d_packed_b,
    int D,
    const uint32_t* d_rgsw_top_a, const uint32_t* d_rgsw_top_b,
    const uint32_t* d_rgsw_bot_a, const uint32_t* d_rgsw_bot_b,
    const uint32_t* d_fwd_twiddles_q0, const uint32_t* d_inv_twiddles_q0, uint32_t inv_n_q0,
    const uint32_t* d_fwd_twiddles_q1, const uint32_t* d_inv_twiddles_q1, uint32_t inv_n_q1,
    uint32_t* d_scratch_coeff, uint32_t* d_scratch_dig,
    uint32_t* d_scratch_res_a, uint32_t* d_scratch_res_b,
    int d_gsw, int d_eff, int base_log, size_t n) {

    // Optional fine timing: separate ext_prod cost from poly_add cost, and
    // expose GPU-busy time vs wall time to reveal launch/latency boundedness.
    bool fine = std::getenv("GPU_ANSWER_FINE") != nullptr;
    auto wall0 = std::chrono::high_resolution_clock::now();
    std::vector<cudaEvent_t> ev0, ev1, ev2;   // ext_prod start / poly_add start / iter end
    auto new_ev = [&]() { cudaEvent_t e; cudaEventCreate(&e); return e; };

    // Initialize: acc = packed[D-1]
    CUDA_CHECK(cudaMemcpyAsync(d_acc_a, d_packed_a + (D-1) * 2 * n,
                               2 * n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_acc_b, d_packed_b + (D-1) * 2 * n,
                               2 * n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));

    for (int i = D - 2; i >= 0; i--) {
        if (fine) { ev0.push_back(new_ev()); cudaEventRecord(ev0.back(), 0); }
        // acc = ext_prod(rgsw, acc) + packed[i], computed in-place: ext_prod
        // copies acc to scratch before its MAC overwrites acc, and the +packed[i]
        // is fused into the MAC (no separate poly_add pass / res scratch).
        const uint32_t* pa = d_packed_a + i * 2 * n;
        const uint32_t* pb = d_packed_b + i * 2 * n;
        gpu_ext_prod(
            d_acc_a, d_acc_b,
            d_acc_a, d_acc_b,
            d_rgsw_top_a, d_rgsw_top_b, d_rgsw_bot_a, d_rgsw_bot_b,
            d_fwd_twiddles_q0, d_inv_twiddles_q0, inv_n_q0,
            d_fwd_twiddles_q1, d_inv_twiddles_q1, inv_n_q1,
            d_scratch_coeff, d_scratch_dig,
            d_gsw, d_eff, base_log, n,
            pa, pb);
        CUDA_CHECK(cudaGetLastError());
        if (fine) { ev1.push_back(new_ev()); cudaEventRecord(ev1.back(), 0);
                    ev2.push_back(new_ev()); cudaEventRecord(ev2.back(), 0); }
    }

    if (fine) {
        cudaDeviceSynchronize();
        auto wall1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
        double ext_ms = 0.0, add_ms = 0.0;
        int iters = (int)ev0.size();
        for (int k = 0; k < iters; k++) {
            float m1 = 0.f, m2 = 0.f;
            cudaEventElapsedTime(&m1, ev0[k], ev1[k]);   // ext_prod
            cudaEventElapsedTime(&m2, ev1[k], ev2[k]);   // poly_add
            ext_ms += m1; add_ms += m2;
            cudaEventDestroy(ev0[k]); cudaEventDestroy(ev1[k]); cudaEventDestroy(ev2[k]);
        }
        std::fprintf(stderr,
            "  [fine/horner] %d iters: ext_prod GPU=%.2f ms (%.4f/iter, 12 klaunch),"
            " poly_add GPU=%.2f ms (%.4f/iter, 4 klaunch); wall=%.2f ms\n"
            "  [fine/horner] launch/latency overhead = wall - GPU-busy = %.2f ms (%.1f%% of horner)\n",
            iters, ext_ms, ext_ms / std::max(1, iters),
            add_ms, add_ms / std::max(1, iters),
            wall_ms, wall_ms - (ext_ms + add_ms),
            100.0 * (wall_ms - (ext_ms + add_ms)) / std::max(1e-9, wall_ms));
    }
}


// ============================================================
// Fused inner-product-subtract: b[i] -= sum_j D[j*n+i] * K[j*n+i]
// ============================================================

__global__
void fused_ip_sub_kernel(uint32_t* b, const uint32_t* D, const uint32_t* K,
                         int d_eff, size_t n, uint32_t q, uint32_t /*q_inv*/) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t sum = 0;
    for (int j = 0; j < d_eff; j++) {
        sum += (uint64_t)D[j * n + i] * K[j * n + i];
    }
    uint32_t s = (uint32_t)(sum % q);
    b[i] = mod_sub(b[i], s, q);
}


void gpu_fused_ip_sub_stream(uint32_t* d_b, const uint32_t* d_D, const uint32_t* d_K,
                              int d_eff, size_t n, uint32_t q, cudaStream_t stream) {
    uint32_t q_inv = compute_q_inv(q);
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    fused_ip_sub_kernel<<<blocks, threads, 0, stream>>>(d_b, d_D, d_K, d_eff, n, q, q_inv);
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================
// Lazy collapse (online): on-the-fly permutation, no ExpandKSK
// b[i] -= sum_j D[step][j][i] * ksk_ntt[j][ perm[step][i] ]
// ============================================================

// The original kernel ran the whole n_steps loop on one thread per coefficient
// (n=2048 threads = 8 blocks), updating b[i] each step. Each thread touches only
// b[i], so there is no cross-thread dependency — but 2048 threads can never fill
// the GPU (170 SMs), and the kernel sat at ~0.6% of HBM bandwidth, memory-latency
// bound. We split the work over BOTH coefficients and step-chunks:
//
//   b[i] = b[i] - ( sum_{step} s_step[i] ) mod q,   s_step[i] = sum_j D[step][j][i]·ksk[perm]
//
// Kernel 1 (partial): grid (n/threads, n_chunks). Block (x,y) sums s_step[i] over
//   step-chunk y into partials[y*n + i] (mod q). n_chunks×(n/threads) blocks fill
//   the GPU and give the memory subsystem enough in-flight warps to hide latency.
// Kernel 2 (reduce): b[i] -= ( sum_y partials[y*n + i] ) mod q.
//
// Each s_step ∈ [0,q), q<2^27; a chunk sum (≤~32 terms) and the reduce sum
// (n_chunks terms) both stay well under 2^64, so a single final mod is exact.

__global__
void lazy_collapse_partial_kernel(const uint32_t* __restrict__ D_all,   // [n_steps*d_eff*n]
                                  const uint32_t* __restrict__ ksk_ntt, // [d_eff*n]
                                  const uint32_t* __restrict__ perm_all,// [n_steps*n]
                                  uint32_t* __restrict__ partials,      // [n_chunks*n]
                                  int n_steps, int d_eff, size_t n,
                                  uint32_t q, uint32_t mu, int n_chunks) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int chunk = (n_steps + n_chunks - 1) / n_chunks;
    int step0 = blockIdx.y * chunk;
    int step1 = min(step0 + chunk, n_steps);

    uint64_t acc = 0;
    for (int step = step0; step < step1; step++) {
        uint32_t perm_idx = perm_all[(size_t)step * n + i];
        const uint32_t* D_step = D_all + (size_t)step * d_eff * n;
        uint64_t sum = 0;
        for (int j = 0; j < d_eff; j++) {
            uint32_t K_val = ksk_ntt[j * n + perm_idx];
            sum += (uint64_t)D_step[j * n + i] * K_val;
        }
        acc += barrett_mod(sum, q, mu);   // each in [0,q)
    }
    partials[(size_t)blockIdx.y * n + i] = (uint32_t)(acc % q);
}


__global__
void lazy_collapse_reduce_kernel(uint32_t* __restrict__ b,
                                 const uint32_t* __restrict__ partials,
                                 int n_chunks, size_t n, uint32_t q) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t total = 0;
    for (int y = 0; y < n_chunks; y++)
        total += partials[(size_t)y * n + i];   // each in [0,q)
    b[i] = mod_sub(b[i], (uint32_t)(total % q), q);
}


// ============================================================
// Batched lazy collapse (Phase B3): B queries share one tensor stream.
// The precomp tensor D_all is identical for every query of a batch (it is
// the database-side offline data); only the client key ksk and the b
// polynomial differ. Each tensor element is loaded once and multiplied
// against BT per-query key values in registers — the per-query 2×(n_steps×
// d_eff×n) tensor traffic, which dominates the packing phase, is paid once
// per tile instead of once per query. Per-query accumulation order matches
// lazy_collapse_partial_kernel exactly, so results stay bit-identical.
//
// Pointer plumbing: slot-base device pointer arrays (built once at setup)
// plus a scalar offset, so no per-launch pointer uploads are needed.
// ============================================================

template <int BT>
__global__
void lazy_collapse_partial_batched_kernel(
        const uint32_t* __restrict__ D_all,        // shared: [n_steps*d_eff*n]
        const uint32_t* const* __restrict__ ksk_bases, size_t ksk_off,
        const uint32_t* __restrict__ perm_all,     // [n_steps*n]
        uint32_t* const* __restrict__ part_ptrs,   // [batch] partial buffers
        int n_steps, int d_eff, size_t n,
        uint32_t q, uint32_t mu, int n_chunks, int batch) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const uint32_t* kp[BT];
    uint32_t* pp[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        int src = (b < batch) ? b : 0;
        kp[b] = ksk_bases[src] + ksk_off;
        pp[b] = part_ptrs[src];
    }

    int chunk = (n_steps + n_chunks - 1) / n_chunks;
    int step0 = blockIdx.y * chunk;
    int step1 = min(step0 + chunk, n_steps);

    uint64_t acc[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0;

    for (int step = step0; step < step1; step++) {
        uint32_t perm_idx = perm_all[(size_t)step * n + i];
        const uint32_t* D_step = D_all + (size_t)step * d_eff * n;
        uint64_t sum[BT];
        #pragma unroll
        for (int b = 0; b < BT; b++) sum[b] = 0;
        for (int j = 0; j < d_eff; j++) {
            uint64_t Dv = D_step[j * n + i];
            size_t kidx = (size_t)j * n + perm_idx;
            #pragma unroll
            for (int b = 0; b < BT; b++)
                sum[b] += Dv * kp[b][kidx];
        }
        #pragma unroll
        for (int b = 0; b < BT; b++)
            acc[b] += barrett_mod(sum[b], q, mu);
    }
    #pragma unroll
    for (int b = 0; b < BT; b++)
        if (b < batch)
            pp[b][(size_t)blockIdx.y * n + i] = (uint32_t)(acc[b] % q);
}

__global__
void lazy_collapse_reduce_batched_kernel(
        uint32_t* const* __restrict__ b_bases, size_t b_off,
        uint32_t* const* __restrict__ part_ptrs,
        int n_chunks, size_t n, uint32_t q) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t* b = b_bases[blockIdx.y] + b_off;
    const uint32_t* partials = part_ptrs[blockIdx.y];
    uint64_t total = 0;
    for (int y = 0; y < n_chunks; y++)
        total += partials[(size_t)y * n + i];
    b[i] = mod_sub(b[i], (uint32_t)(total % q), q);
}

void gpu_lazy_collapse_batched_stream(
        uint32_t* const* d_b_bases, size_t b_off,
        const uint32_t* d_D_all,
        const uint32_t* const* d_ksk_bases, size_t ksk_off,
        const uint32_t* d_perm_all,
        uint32_t* const* d_part_ptrs,
        int n_steps, int d_eff, size_t n, uint32_t q,
        int batch, cudaStream_t stream) {
    uint32_t mu = (q == Q0) ? Q0_BARRETT_MU : Q1_BARRETT_MU;
    int n_chunks = std::min(LAZY_COLLAPSE_STEP_CHUNKS, n_steps);
    size_t threads = 256;
    size_t blocks_x = (n + threads - 1) / threads;

    // Register tiles of up to 8 queries per tensor stream.
    for (int off = 0; off < batch; off += 8) {
        int cnt = batch - off > 8 ? 8 : batch - off;
        auto kb = d_ksk_bases + off;
        auto pb = d_part_ptrs + off;
        dim3 grid(blocks_x, n_chunks);
        if (cnt > 4)
            lazy_collapse_partial_batched_kernel<8><<<grid, threads, 0, stream>>>(
                d_D_all, kb, ksk_off, d_perm_all, pb, n_steps, d_eff, n, q, mu, n_chunks, cnt);
        else if (cnt > 2)
            lazy_collapse_partial_batched_kernel<4><<<grid, threads, 0, stream>>>(
                d_D_all, kb, ksk_off, d_perm_all, pb, n_steps, d_eff, n, q, mu, n_chunks, cnt);
        else
            lazy_collapse_partial_batched_kernel<2><<<grid, threads, 0, stream>>>(
                d_D_all, kb, ksk_off, d_perm_all, pb, n_steps, d_eff, n, q, mu, n_chunks, cnt);
        dim3 rgrid(blocks_x, cnt);
        lazy_collapse_reduce_batched_kernel<<<rgrid, threads, 0, stream>>>(
            d_b_bases + off, b_off, pb, n_chunks, n, q);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Batched fused inner-product-subtract: D shared (24 KB, cache-resident),
// key and b per query via base arrays + offset. grid.y = query index.
__global__
void fused_ip_sub_batched_kernel(
        uint32_t* const* __restrict__ b_bases, size_t b_off,
        const uint32_t* __restrict__ D,
        const uint32_t* const* __restrict__ k_bases, size_t k_off,
        int d_eff, size_t n, uint32_t q) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t* b = b_bases[blockIdx.y] + b_off;
    const uint32_t* K = k_bases[blockIdx.y] + k_off;
    uint64_t sum = 0;
    for (int j = 0; j < d_eff; j++)
        sum += (uint64_t)D[j * n + i] * K[j * n + i];
    b[i] = mod_sub(b[i], (uint32_t)(sum % q), q);
}

void gpu_fused_ip_sub_batched_stream(
        uint32_t* const* d_b_bases, size_t b_off,
        const uint32_t* d_D,
        const uint32_t* const* d_k_bases, size_t k_off,
        int d_eff, size_t n, uint32_t q, int batch, cudaStream_t stream) {
    size_t threads = 256;
    dim3 grid((n + threads - 1) / threads, batch);
    fused_ip_sub_batched_kernel<<<grid, threads, 0, stream>>>(
        d_b_bases, b_off, d_D, d_k_bases, k_off, d_eff, n, q);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_lazy_collapse_stream(uint32_t* d_b, const uint32_t* d_D_all,
                              const uint32_t* d_ksk_ntt, const uint32_t* d_perm_all,
                              uint32_t* d_partials,
                              int n_steps, int d_eff, size_t n, uint32_t q,
                              cudaStream_t stream) {
    uint32_t mu = (q == Q0) ? Q0_BARRETT_MU : Q1_BARRETT_MU;
    int n_chunks = std::min(LAZY_COLLAPSE_STEP_CHUNKS, n_steps);
    size_t threads = 256;
    size_t blocks_x = (n + threads - 1) / threads;
    dim3 grid(blocks_x, n_chunks);
    lazy_collapse_partial_kernel<<<grid, threads, 0, stream>>>(
        d_D_all, d_ksk_ntt, d_perm_all, d_partials, n_steps, d_eff, n, q, mu, n_chunks);
    lazy_collapse_reduce_kernel<<<blocks_x, threads, 0, stream>>>(
        d_b, d_partials, n_chunks, n, q);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace inspire
