#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace inspire {
namespace gpu {

// ============================================================
// GPU parameters (matching CPU params.h)
// ============================================================

constexpr size_t N = 2048;
constexpr uint32_t Q0 = 94519297;
constexpr uint32_t Q1 = 95293441;
constexpr int NUM_LIMBS = 2;
constexpr int D_EFF = 2;
constexpr int BASE_LOG = 18;

// Barrett reduction constants: mu = floor(2^55 / q)
constexpr uint32_t Q0_BARRETT_MU = 381179274;
constexpr uint32_t Q1_BARRETT_MU = 378082653;

// Montgomery form: R = 2^32
constexpr uint32_t MONT_R_LOG = 32;

// ============================================================
// This header is the public API for the inspire_gpu library. Declarations are
// grouped by the .cu translation unit that defines them, mirroring the
// pipeline phases:
//   PRIMITIVES (gpu_primitives.cu)  — context, NTT, poly ops, decomp, memory
//   PREPROCESS (gpu_preprocess.cu)  — encode, a-side matmul, ring_embed, autos, collapse
//   ONLINE     (gpu_online.cu)      — matvec, lazy_collapse, ext_prod, horner
// High-level orchestration (GpuServerCtx, gpu_preprocess/setup_server/answer)
// lives in gpu_protocol.{h,cu}, not here.
// ============================================================

// ============================================================
// PRIMITIVES (gpu_primitives.cu) — phase-agnostic building blocks
// ============================================================

// --- GPU context: holds precomputed NTT roots, Montgomery constants ---

struct GpuContext {
    // NTT twiddle factors per prime (in Montgomery form)
    uint32_t* d_twiddles[2];      // forward NTT
    uint32_t* d_inv_twiddles[2];  // inverse NTT
    uint32_t inv_n[2];            // n^{-1} mod q_i (Montgomery form)

    // Montgomery constants
    uint32_t q_inv[2];  // -q^{-1} mod R (for Montgomery reduction)
    uint32_t mont_one[2]; // R mod q (Montgomery form of 1)

    bool initialized = false;
};

// Initialize GPU context with the CPU NTT's twiddle factors
void gpu_init_with_twiddles(GpuContext& ctx,
                           const uint32_t* fwd0, const uint32_t* inv0, uint32_t inv_n0,
                           const uint32_t* fwd1, const uint32_t* inv1, uint32_t inv_n1);

// Legacy init (no twiddles uploaded)
void gpu_init(GpuContext& ctx);
void gpu_free(GpuContext& ctx);

// --- NTT on device memory ---

// NTT forward/inverse on device memory (in-place, per limb)
void gpu_ntt_forward(uint32_t* d_poly, size_t n, uint32_t q, const uint32_t* d_twiddles);
void gpu_ntt_inverse(uint32_t* d_poly, size_t n, uint32_t q, const uint32_t* d_inv_twiddles, uint32_t inv_n);

// Batch NTT: process `batch_size` polynomials at once (one block each, `threads`
// threads/block). Forward and inverse variants.
void gpu_ntt_batch(uint32_t* d_polys, size_t n, size_t batch_size, uint32_t q,
                   const uint32_t* d_twiddles, size_t threads = 256);
void gpu_ntt_inverse_batch(uint32_t* d_polys, size_t n, size_t batch_size, uint32_t q,
                           const uint32_t* d_inv_twiddles, uint32_t inv_n, size_t threads = 256);

// Stream-aware NTT (forward only — INTT isn't on the per-group hot path)
void gpu_ntt_forward_stream(uint32_t* d_poly, size_t n, uint32_t q,
                             const uint32_t* d_twiddles, cudaStream_t stream);

// --- Polynomial primitives (single RNS limb each) ---

// c[i] = (a[i] + b[i]) mod q
void gpu_poly_add(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                  size_t n, uint32_t q);

// c[i] = (a[i] - b[i]) mod q
void gpu_poly_sub(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                  size_t n, uint32_t q);

// c[i] = a[i] * b[i] mod q (NTT-domain multiply)
void gpu_poly_mul(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                  size_t n, uint32_t q);

// c[i] -= a[i] * b[i] mod q (NTT-domain mul-sub)
void gpu_poly_mul_sub(uint32_t* d_c, const uint32_t* d_a, const uint32_t* d_b,
                      size_t n, uint32_t q);

// Gadget decomposition (matches CPU decomp() in ring.cpp)
// Inputs: a0, a1 (coefficient form, one per RNS limb)
// Outputs: out0[d_eff*n], out1[d_eff*n] — d_eff digit polynomials per limb
void gpu_gadget_decomp(const uint32_t* d_a0, const uint32_t* d_a1,
                       uint32_t* d_out0, uint32_t* d_out1,
                       size_t n, int d_gsw, int d_eff, int base_log);

// --- Memory management helpers ---

uint32_t* gpu_alloc(size_t count);
void gpu_free_mem(uint32_t* ptr);
void gpu_upload(uint32_t* d_dst, const uint32_t* h_src, size_t count);
void gpu_download(uint32_t* h_dst, const uint32_t* d_src, size_t count);

// ============================================================
// PREPROCESS (gpu_preprocess.cu) — server one-time setup
// ============================================================

// Slot-native encode: d_db holds plaintext slots in [0,P), ROW-MAJOR
// (db[row*db_cols+col]); applies only the in-place inverse-DFT. Row-major lets
// the online matvec read the DB directly — no transpose, single resident copy.
void gpu_encode_inverse_dft(uint16_t* d_db, size_t db_rows, size_t db_cols,
                            size_t D, size_t poly_len);

// Fused a-side matmul producing ring_embed input directly (NTT form, with
// neg_rev + 1/N scale absorbed into the writeback). One block per output j.
// See gpu_preprocess.cu for the full algorithm and shared-memory layout.
void gpu_a_side_matmul_a32(
    uint32_t* d_a32,                   // [gamma=n][limb][coeff], NTT form, output
    const uint16_t* d_db,              // u16 DB (P=65535), ROW-MAJOR
    size_t db_rows, size_t db_cols,
    size_t g, size_t n_r, size_t n,
    const uint32_t* d_a_crs_q0_ntt,    // n_r * n, NTT form
    const uint32_t* d_a_crs_q1_ntt,    // n_r * n, NTT form
    uint32_t nu_inv_q0, uint32_t nu_inv_q1,
    const uint32_t* d_fwd_q0, const uint32_t* d_fwd_q1,
    uint32_t Q0c, uint32_t Q1c);

// Ring embed: gamma LWE a-vectors (in compact NTT form) → 2*half output
// polynomials (in NTT form) via a fused inner-product with monomial NTTs.
//   acc_fwd[k][c][i]  = sum_j a32[j][c][i] * mono32[(j * 5^{n-k}) mod 2n][c][i]
//   acc_conj[k][c][i] = sum_j a32[j][c][i] * mono32[(2n - j*5^{n-k}) mod 2n][c][i]
// Both for k = 0..half-1.
//
// Caller still needs to apply the τ_{5^k} (forward) and τ_{2n - 5^k} (conjugate)
// automorphism to acc_fwd[k] / acc_conj[k] respectively before use.
void gpu_ring_embed(uint32_t* d_acc_fwd, uint32_t* d_acc_conj,
                    const uint32_t* d_a32, const uint32_t* d_mono32,
                    const uint32_t* d_gp,
                    int gamma, int half, size_t n);

// Apply NTT-domain automorphism via permutation table (both limbs at once)
// dst layout: [limb0: n][limb1: n]; same for src; perm is a uint32 index table of length n.
void gpu_apply_auto_perm(uint32_t* d_dst, const uint32_t* d_src,
                          const uint32_t* d_perm, size_t n);

// As above, but compute the permutation index inline from the automorphism
// factor t — no precomputed perm table needed. Eliminates the ~1.5 s brute-
// force perm-table build in GpuRingEmbedHelper. Formula:
//   perm[i] = (t * (2i+1) mod 2n − 1) / 2,   t odd in [1, 2n).
void gpu_apply_auto_perm_inline(uint32_t* d_dst, const uint32_t* d_src,
                                 uint32_t t, size_t n);

// Phase 11: batched per-k automorphism. Computes τ_{5^k} on d_src_fwd and
// τ_{2n - 5^k} on d_src_conj, writing to d_dst_fwd / d_dst_conj for all
// k = 0..half-1 in a single kernel grid (<<<half, 256>>>). The d_gp table
// is reused (same one built by GpuRingEmbedHelper).
void gpu_apply_auto_perm_batched(
    uint32_t* d_dst_fwd, uint32_t* d_dst_conj,
    const uint32_t* d_src_fwd, const uint32_t* d_src_conj,
    const uint32_t* d_gp, int half, size_t n);

// Phase 12: GPU-side ExpandKSK_a permutation. Given d_eff base ksk5 polys
// (in NTT form, [d_eff * 2 * n]) and d_gp[s] = 5^s mod 2n, produces:
//   d_ksk_plus [c=limb, s=step, j=digit, i] = τ_{5^s}(ksk5[j])[c, i]
//   d_ksk_minus[c=limb, s=step, j=digit, i] = τ_{2n - 5^s}(ksk5[j])[c, i]
// for all s = 0..n_steps-1, j = 0..d_eff-1. Single-launch batched.
void gpu_ksk_perm_batched(
    uint32_t* d_ksk_plus, uint32_t* d_ksk_minus,
    const uint32_t* d_ksk5_ntt, const uint32_t* d_gp,
    int n_steps, int d_eff, size_t n);

// Fused collapse_a kernel — replaces the host-orchestrated collapse loop
// (~16K kernel launches per group) with a single CUDA kernel that runs all
// 2*(N/2-1) iterations + the final step in shared memory.
//
// Phase 10: takes per-group pointer arrays (device-resident, length num_groups).
// Each block uses blockIdx.x to pick its group's input/output buffer, so all
// num_groups collapses run concurrently in a single kernel launch.
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
    int num_groups);

// ============================================================
// ONLINE (gpu_online.cu) — per-query answer hot path
// ============================================================

// Convert the encoded row-major u16 DB in place to centered signed bytes.
// The allocation size is unchanged: each u16 becomes interleaved low/high
// int8 digits.  The resulting memory is a column-major (2*db_cols) x db_rows
// INT8 matrix without a transpose or a second resident DB copy.  byte_sums
// contains the per-matrix-row signed sums needed to undo centering exactly.
void gpu_matvec_tensor_prepare_db(uint16_t* d_db_rm, int32_t* d_byte_sums,
                                  size_t db_rows, size_t db_cols);

// Tensor-core batched mat-vec.  A single INT8 GEMM handles every query and
// both RNS limbs.  Each uint32 query residue is decomposed into four centered
// byte planes; the correction/recomposition kernel writes ordinary q0/q1
// results, bit-exact with the scalar modular dot product.
void gpu_matvec_tensor_batched(
    uint32_t* const* d_results0, uint32_t* const* d_results1,
    const uint16_t* d_centered_db, const int32_t* d_db_byte_sums,
    const uint32_t* const* d_queries0, const uint32_t* const* d_queries1,
    int8_t* d_query_planes, int32_t* d_query_sums, int32_t* d_gemm_out,
    size_t db_rows, size_t db_cols, uint32_t q0, uint32_t q1,
    int batch, cublasHandle_t handle);

// Dual-limb mat-vec: reads u16 DB once, computes both RNS limbs in one pass.
// Dispatches between a "wide" simple kernel (db_cols >> db_rows) and a tall
// parallel-reduction kernel that splits rows into row blocks.
void gpu_matvec_dual(uint32_t* d_result0, uint32_t* d_result1,
                     const uint16_t* d_db_rm,
                     const uint32_t* d_query_mod0, const uint32_t* d_query_mod1,
                     size_t db_rows, size_t db_cols,
                     uint32_t q0, uint32_t q1);

// Batched dual-limb mat-vec: `batch` queries share the DB stream (register
// tiles of up to 8 per launch; larger batches loop in chunks). All pointer
// arrays are DEVICE-resident arrays of device pointers, entry b = query b's
// vector / result buffer. Bit-identical per query to gpu_matvec_dual.
void gpu_matvec_dual_batched(uint32_t* const* d_results0, uint32_t* const* d_results1,
                             const uint16_t* d_db_rm,
                             const uint32_t* const* d_queries0,
                             const uint32_t* const* d_queries1,
                             size_t db_rows, size_t db_cols,
                             uint32_t q0, uint32_t q1, int batch);

// Whether the batched matvec beats per-query row-split dispatch for this
// geometry (wide enough for one-thread-per-column occupancy).
bool gpu_matvec_batched_profitable(size_t db_cols);

// Batched lazy collapse: `batch` queries share one stream over the group's
// precomp tensor D_all; keys / b polys / partials are per query via
// device-resident base-pointer arrays plus a scalar offset (no per-launch
// pointer uploads). Bit-identical per query to gpu_lazy_collapse_stream.
void gpu_lazy_collapse_batched_stream(
        uint32_t* const* d_b_bases, size_t b_off,
        const uint32_t* d_D_all,
        const uint32_t* const* d_ksk_bases, size_t ksk_off,
        const uint32_t* d_perm_all,
        uint32_t* const* d_part_ptrs,
        int n_steps, int d_eff, size_t n, uint32_t q,
        int batch, cudaStream_t stream);

// Batched final CollapseOne (inner-product-subtract), same plumbing.
void gpu_fused_ip_sub_batched_stream(
        uint32_t* const* d_b_bases, size_t b_off,
        const uint32_t* d_D,
        const uint32_t* const* d_k_bases, size_t k_off,
        int d_eff, size_t n, uint32_t q, int batch, cudaStream_t stream);

// Strided gadget decomposition: `count` contiguous polys at a{0,1} + y*n,
// digits to out{0,1} + y*d_eff*n (Phase B4 lockstep Horner).
void gpu_gadget_decomp_strided(const uint32_t* d_a0, const uint32_t* d_a1,
                               uint32_t* d_out0, uint32_t* d_out1,
                               size_t n, int count,
                               int d_gsw, int d_eff, int base_log);

// One lockstep batched Horner step for `batch` chains: acc[b] =
// ext_prod(rgsw[b], acc[b]) + packed[b][add_off..]. Eight launches total,
// independent of batch. Per-query semantics match gpu_ext_prod's fused-add
// path bit-for-bit. Scratch: coeff_qX = 2*batch*n u32, dig_qX =
// 2*batch*d_eff*n u32, contiguous, caller-allocated.
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
        int d_gsw, int d_eff, int base_log, size_t n, int batch);

// Online-phase lazy collapse: per-step inner-product + on-the-fly permutation.
// Used by `gpu_answer` to fold the query's b vector against the stored
// gadget-decomposed precomp. Stream-aware so multiple groups can overlap.
//   ksk_ntt:    [d_eff * N] raw NTT-domain key
//   perm_all:   [n_steps * N] uint32 permutation tables
//   d_partials: scratch of LAZY_COLLAPSE_STEP_CHUNKS * N uint32 (one slice per
//               concurrent stream) for the step-parallel partial sums.
//
// The n_steps×n work is parallelized over BOTH coefficients (grid.x) and
// step-chunks (grid.y = LAZY_COLLAPSE_STEP_CHUNKS) so it fills the GPU — the
// 2048-coefficient dimension alone is far too small to occupy all SMs.
constexpr int LAZY_COLLAPSE_STEP_CHUNKS = 32;
void gpu_lazy_collapse_stream(uint32_t* d_b, const uint32_t* d_D_all,
                              const uint32_t* d_ksk_ntt, const uint32_t* d_perm_all,
                              uint32_t* d_partials,
                              int n_steps, int d_eff, size_t n, uint32_t q,
                              cudaStream_t stream);

// Fused inner-product-subtract (stream-aware): b -= sum_j D[j] * K[j].
// Used by online gpu_answer for the final-step inner products.
void gpu_fused_ip_sub_stream(uint32_t* d_b, const uint32_t* d_D, const uint32_t* d_K,
                              int d_eff, size_t n, uint32_t q, cudaStream_t stream);

// External product (RGSW × RLWE → RLWE), both limbs.
// All NTT-form arrays use the layout: limb0 contiguous, then limb1 (so 2*n total).
// RGSW arrays: limb0's d_eff polys then limb1's d_eff polys (2*d_eff*n total).
// Caller-provided scratch buffers avoid per-call cudaMalloc in Horner.
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
    const uint32_t* d_add_a = nullptr, const uint32_t* d_add_b = nullptr);

// Horner evaluation: acc = packed[D-1]; for i=D-2..0: acc = ext_prod(rgsw, acc) + packed[i]
void gpu_horner_eval(
    uint32_t* d_acc_a, uint32_t* d_acc_b,
    const uint32_t* d_packed_a, const uint32_t* d_packed_b,
    int D,
    const uint32_t* d_rgsw_top_a, const uint32_t* d_rgsw_top_b,
    const uint32_t* d_rgsw_bot_a, const uint32_t* d_rgsw_bot_b,
    const uint32_t* d_fwd_twiddles_q0, const uint32_t* d_inv_twiddles_q0, uint32_t inv_n_q0,
    const uint32_t* d_fwd_twiddles_q1, const uint32_t* d_inv_twiddles_q1, uint32_t inv_n_q1,
    uint32_t* d_scratch_coeff, uint32_t* d_scratch_dig,
    uint32_t* d_scratch_res_a, uint32_t* d_scratch_res_b,
    int d_gsw, int d_eff, int base_log, size_t n);

} // namespace gpu
} // namespace inspire
