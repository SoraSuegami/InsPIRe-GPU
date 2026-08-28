#include "gpu_protocol.h"
#include "ring.h"
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", \
                __FILE__, __LINE__, (int)status_); \
        std::exit(1); \
    } \
} while(0)

namespace inspire {

// ============================================================
// Layout conventions for GPU data
// ============================================================
//
// An "RLWE component" (a or b polynomial) is laid out as 2*N uint32:
//   [limb0: N values][limb1: N values]
//
// An "RGSW part" (top.a_parts or similar — a vector of d_eff polynomials,
// each with 2 limbs) is laid out as 2*d_eff*N uint32:
//   [limb0: d_eff polys × N][limb1: d_eff polys × N]
//
// Per-group precomp:
//   d_a:        2*N         (a polynomial, NTT form)
//   d_D_plus:   (n/2-1) × d_eff × 2*N
//   d_D_minus:  (n/2-1) × d_eff × 2*N
//   d_D_final:  d_eff × 2*N
// All in NTT form.
//
// Permutation tables (shared across groups, derived from CRS):
//   d_perm_fwd, d_perm_conj: each (n/2-1) × N uint32

// Upload d_eff polynomials in the layout [limb0: d_eff × N][limb1: d_eff × N].
static void upload_rgsw_part(uint32_t* d_dst, const std::vector<RnsPoly>& parts,
                             uint32_t* h_buf) {
    assert((int)parts.size() == D_EFF);
    for (int j = 0; j < D_EFF; j++) {
        RnsPoly p = parts[j];
        if (!p.is_ntt) p.to_ntt();
        for (size_t i = 0; i < N; i++) {
            h_buf[j * N + i]                      = (uint32_t)p.limbs[0][i];
            h_buf[D_EFF * N + j * N + i]          = (uint32_t)p.limbs[1][i];
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_buf,
                               2 * D_EFF * N * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, 0));
}

// ============================================================
// GpuServerCtx
// ============================================================

// Per-query scratch: one slot per concurrently-resident query. All slots are
// allocated at setup time (max_batch of them) — nothing is allocated on the
// request path. gpu_answer_batch runs the B2/B3/B4 batched kernels across
// all slots of a batch (single-query gpu_answer uses slot 0 alone).
struct QuerySlot {
    uint32_t* d_query_b_q0;       // db_rows uint32
    uint32_t* d_query_b_q1;       // db_rows uint32
    uint32_t* d_ksk5_b;           // 2*d_eff*N
    uint32_t* d_kskneg1_b;        // 2*d_eff*N (in coeff form, NTT applied during use)
    uint32_t* d_rgsw_top_a;       // 2*d_eff*N
    uint32_t* d_rgsw_top_b;
    uint32_t* d_rgsw_bot_a;
    uint32_t* d_rgsw_bot_b;
    uint32_t* d_packed_b;         // n_packed × 2*N
    // Persistent pinned staging makes request H2D copies genuinely async.
    uint32_t* h_query_b_q0;
    uint32_t* h_query_b_q1;
    uint32_t* h_ksk5_b;
    uint32_t* h_kskneg1_b;
    uint32_t* h_rgsw_top_a;
    uint32_t* h_rgsw_top_b;
    uint32_t* h_rgsw_bot_a;
    uint32_t* h_rgsw_bot_b;
    // Horner scratch
    uint32_t* d_acc_a;
    uint32_t* d_acc_b;
    uint32_t* d_res_a;
    uint32_t* d_res_b;
    uint32_t* d_scratch_coeff;    // 4*N
    uint32_t* d_dig;              // 4*d_eff*N, [a_q0][b_q0][a_q1][b_q1] (merged for batched NTT)
};

struct GpuServerCtx {
    PublicParams pp;
    GpuServerConfig cfg;

    // Twiddles (per RNS limb)
    uint32_t *d_fwd_q0, *d_inv_q0;
    uint32_t *d_fwd_q1, *d_inv_q1;
    uint32_t inv_n_q0, inv_n_q1;

    // DB (row-major u16): gpu_preprocess produces it row-major; adopted directly. Phase 14: P=65535 / 15-bit
    // packing, each slot fits in uint16_t.
    // Encoded DB, row-major (db[row*db_cols+col]) — the layout gpu_encode produces
    // and the online matvec reads directly (no transpose, single resident copy).
    uint16_t* d_db_rm;
    size_t db_rows, db_cols, n_packed;

    // Tensor-core mat-vec state.  d_db_rm is centered in place as interleaved
    // signed low/high bytes after preprocessing; no second DB copy is kept.
    cublasHandle_t cublas;
    int32_t* d_db_byte_sums = nullptr;  // 2*db_cols
    int8_t* d_mv_query_planes = nullptr; // max_batch*8*db_rows
    int32_t* d_mv_query_sums = nullptr;  // max_batch*8
    int32_t* d_mv_gemm_out = nullptr;    // max_batch*8*2*db_cols
    uint32_t* d_mv_tall_partials0 = nullptr;
    uint32_t* d_mv_tall_partials1 = nullptr;
    size_t mv_tall_partial_count = 0;

    // Permutation tables (shared, derived from CRS at setup time)
    uint32_t* d_perm_fwd;   // (n/2-1) × N uint32
    uint32_t* d_perm_conj;  // (n/2-1) × N uint32

    // Per-group precomp (one device buffer per group)
    std::vector<uint32_t*> d_precomp_a;        // [n_packed], each 2*N
    std::vector<uint32_t*> d_precomp_D_plus;   // [n_packed], each (n/2-1)*d_eff*2*N
    std::vector<uint32_t*> d_precomp_D_minus;  // same shape
    std::vector<uint32_t*> d_precomp_D_final;  // [n_packed], each d_eff*2*N
    uint32_t* d_packed_a = nullptr;            // shared contiguous [n_packed][2*N]

    // Per-query scratch slots (cfg.max_batch of them, allocated at setup)
    std::vector<QuerySlot> slots;

    // Device-resident pointer arrays over the slot pool (built at setup),
    // consumed by the batched matvec: entry i = slots[i]'s buffers.
    const uint32_t** d_mv_q0_ptrs = nullptr;
    const uint32_t** d_mv_q1_ptrs = nullptr;
    uint32_t** d_mv_r0_ptrs = nullptr;
    uint32_t** d_mv_r1_ptrs = nullptr;

    // B3: slot-base pointer arrays + per-(stream, slot) partials pool for
    // the batched collapse (tensor stream shared across the batch).
    uint32_t** d_slot_packedb_ptrs = nullptr;        // [max_batch] = slot.d_packed_b
    const uint32_t** d_slot_ksk5_ptrs = nullptr;     // [max_batch] = slot.d_ksk5_b
    const uint32_t** d_slot_kskneg1_ptrs = nullptr;  // [max_batch] = slot.d_kskneg1_b
    uint32_t* d_batch_partials = nullptr;            // n_streams × max_batch × CHUNKS × N
    uint32_t** d_batch_part_ptrs = nullptr;          // [n_streams × max_batch]

    // B4: lockstep batched Horner — slot-base pointer arrays and contiguous
    // per-modulus scratch (coeff: 2*max_batch polys, dig: 2*max_batch*d_eff).
    uint32_t** d_slot_acc_a_ptrs = nullptr;          // [max_batch]
    uint32_t** d_slot_acc_b_ptrs = nullptr;
    const uint32_t** d_slot_packeda_ptrs = nullptr;  // packed_a as Horner add input
    const uint32_t** d_slot_rgsw_ta_ptrs = nullptr;
    const uint32_t** d_slot_rgsw_tb_ptrs = nullptr;
    const uint32_t** d_slot_rgsw_ba_ptrs = nullptr;
    const uint32_t** d_slot_rgsw_bb_ptrs = nullptr;
    uint32_t* d_h_coeff_q0 = nullptr;
    uint32_t* d_h_coeff_q1 = nullptr;
    uint32_t* d_h_dig_q0 = nullptr;
    uint32_t* d_h_dig_q1 = nullptr;

    // Lazy-collapse step-parallel partials: one slice per (stream, slot),
    // each LAZY_COLLAPSE_STEP_CHUNKS * N uint32.
    uint32_t* d_lazy_partials;

    // Per-group streams for collapse parallelism
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> stream_done_events;

    // Bookkeeping for gpu_server_caps
    size_t resident_bytes = 0;
};

// ============================================================
// Permutation-table generation for the online lazy-collapse.
//
// Each collapse step applies an NTT-domain automorphism τ_{5^e} (and its
// conjugate τ_{2n-5^e}); the kernel realises it as a coefficient permutation.
// We recover that permutation empirically: NTT a polynomial with distinct
// coefficients (v[i] = i+1), apply the automorphism via automorphism_ntt(),
// then match each output value back to its source index through a reverse
// lookup. Runs once at setup. fwd_flat/conj_flat are [n_steps × N]; entry
// perm[i] is the source index that lands at position i.
// ============================================================
static void build_perm_tables(std::vector<uint32_t>& fwd_flat,
                              std::vector<uint32_t>& conj_flat) {
    size_t half = N / 2;
    size_t n_steps = half - 1;
    fwd_flat.resize(n_steps * N);
    conj_flat.resize(n_steps * N);

    // Build base NTT polynomial with distinct coefficients: v[i] = i+1
    RnsPoly base;
    base.is_ntt = false;
    for (size_t i = 0; i < N; i++) {
        base.limbs[0][i] = (uint64_t)(i + 1);
        base.limbs[1][i] = (uint64_t)(i + 1);
    }
    base.to_ntt();

    // Build a reverse lookup: ntt_value (from limb 0) -> index
    // (Unique since v[i] = i+1 distinct, and NTT preserves bijection;
    // but check for collisions just in case.)
    std::vector<size_t> reverse_lookup(Q0, (size_t)-1);
    for (size_t i = 0; i < N; i++) {
        reverse_lookup[base.limbs[0][i]] = i;
    }

    // Collapse step s (= 0..n_steps-1) folds at 5-power exponent half-2-s:
    // the online lazy-collapse walks steps in reverse (k = half-1..1), so step
    // s applies τ_{5^(half-2-s)} forward and τ_{2n-5^(half-2-s)} conjugate.
    for (size_t s = 0; s < n_steps; s++) {
        size_t exp = half - 2 - s;  // 5-power exponent
        uint64_t fwd_power = 1;
        for (size_t e = 0; e < exp; e++) fwd_power = (fwd_power * 5) % (2 * N);
        RnsPoly fwd_applied = automorphism_ntt(base, fwd_power);
        for (size_t i = 0; i < N; i++) {
            uint64_t v = fwd_applied.limbs[0][i];
            size_t src = reverse_lookup[v];
            fwd_flat[s * N + i] = (uint32_t)src;
        }

        uint64_t conj_power = (2 * N - fwd_power) % (2 * N);
        RnsPoly conj_applied = automorphism_ntt(base, conj_power);
        for (size_t i = 0; i < N; i++) {
            uint64_t v = conj_applied.limbs[0][i];
            size_t src = reverse_lookup[v];
            conj_flat[s * N + i] = (uint32_t)src;
        }
    }
}

// ============================================================
// GPU-accelerated preprocess.
//
// Runs the full InsPIRe preprocess on device: in-place row-major encode,
// a-side matmul, ring_embed, and the fused collapse — producing the
// device-resident PreprocessData (per-group precomp + encoded DB) that
// gpu_setup_server adopts.
// ============================================================

namespace {

// Stateful helper: holds GPU buffers shared across groups.
struct GpuRingEmbedHelper {
    size_t n;
    int half;
    int gamma;

    // Static (one-time) buffers
    uint32_t* d_mono32;       // [2n × 2 × n] uint32
    uint32_t* d_gp;           // [n] uint32 (5^k mod 2n)
    std::vector<uint32_t> h_gp;  // host-side copy of 5^k mod 2n for inline τ_t kernel
    // (perm tables removed — gpu_apply_auto_perm_inline computes the
    //  permutation from the closed-form formula instead of looking it up)

    // Per-group reusable buffers
    uint32_t* d_a32;          // [γ × 2 × n]
    uint32_t* d_acc_fwd;      // [half × 2 × n]  pre-automorphism
    uint32_t* d_acc_conj;     // [half × 2 × n]  pre-automorphism
    uint32_t* d_a_post_fwd;   // [half × 2 × n]  post-automorphism (= a_agg[0..half-1])
    uint32_t* d_a_post_conj;  // [half × 2 × n]  post-automorphism (= a_agg[half..n-1])

    GpuRingEmbedHelper(size_t n_, int gamma_) : n(n_), half((int)(n_ / 2)), gamma(gamma_) {
        // Allocate static buffers and upload mono32, gp, perm tables.

        // mono32
        size_t mono_bytes = (size_t)2 * n * 2 * n * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_mono32, mono_bytes));
        std::vector<uint32_t> h_mono(2 * n * 2 * n);
        for (size_t k = 0; k < 2 * n; k++) {
            RnsPoly p = RnsPoly::zero();
            if (k < n) p.set_coeff(k, 1, 1);
            else       p.set_coeff(k - n, Q0 - 1, Q1 - 1);
            p.to_ntt();
            for (size_t i = 0; i < n; i++) {
                h_mono[k * 2 * n + 0 * n + i] = (uint32_t)p.limbs[0][i];
                h_mono[k * 2 * n + 1 * n + i] = (uint32_t)p.limbs[1][i];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_mono32, h_mono.data(), mono_bytes, cudaMemcpyHostToDevice));

        // gp[k] = 5^k mod 2n. Keep host copy too so the inline-τ kernel can
        // be parametrised without a device-side table read.
        h_gp.assign(n, 0);
        uint64_t v = 1;
        for (size_t k = 0; k < n; k++) {
            h_gp[k] = (uint32_t)v;
            v = (v * 5) % (2 * n);
        }
        CUDA_CHECK(cudaMalloc(&d_gp, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_gp, h_gp.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Per-group reusable
        CUDA_CHECK(cudaMalloc(&d_a32,         (size_t)gamma * 2 * n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_acc_fwd,     (size_t)half  * 2 * n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_acc_conj,    (size_t)half  * 2 * n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_a_post_fwd,  (size_t)half  * 2 * n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_a_post_conj, (size_t)half  * 2 * n * sizeof(uint32_t)));
    }

    ~GpuRingEmbedHelper() {
        cudaFree(d_mono32); cudaFree(d_gp);
        cudaFree(d_a32);
        cudaFree(d_acc_fwd); cudaFree(d_acc_conj);
        cudaFree(d_a_post_fwd); cudaFree(d_a_post_conj);
    }

    // Run ring_embed + automorphism on device, leaving the output in
    // d_a_post_fwd / d_a_post_conj. The caller (gpu_preprocess) populates
    // d_a32 first via the fused GPU matmul kernel, then invokes this.
    void compute_device_with_external_a32() {
        run_after_a32();
    }

    // Common postlude: ring_embed + per-k automorphism. Reads d_a32, writes
    // d_a_post_fwd / d_a_post_conj. The per-k automorphism uses the inline
    // closed-form permutation kernel — no precomputed perm tables.
    //
    // Phase 11: replaced the host loop (2*half tiny kernel launches per group)
    // with a single batched kernel grid covering all k slots and both fwd /
    // conj directions. At N=2048 that's 1024-block one-wave saturation of the
    // GPU vs the previous 5-us-launch-overhead-dominated 2048 micro-launches.
    void run_after_a32() {
        inspire::gpu::gpu_ring_embed(d_acc_fwd, d_acc_conj, d_a32, d_mono32, d_gp,
                                     gamma, half, n);
        inspire::gpu::gpu_apply_auto_perm_batched(
            d_a_post_fwd, d_a_post_conj,
            d_acc_fwd, d_acc_conj,
            d_gp, half, n);
    }
};

// =====================================================================
// GpuCollapseHelper — runs the InspiRING CollapseA algorithm on device.
// Uploads NTT twiddles + the expanded KSK once at construction (shared across
// all groups in a `gpu_preprocess` call). Per-group `compute()` runs the
// full collapse on device-resident input/output buffers; no NTTs on host.
//
// Algorithm (CollapseA, full: +half / -half / final):
//   forward dir (k = half-1..1):   INTT a_fwd[k] → decomp → NTT digits →
//                                  a_fwd[k-1] -= Σ_j D_plus[s][j] * ksk_plus[k-1][j]
//   conjugate dir (k = half-1..1): same on a_conj using ksk_minus
//   final:                         INTT a_conj[0] → decomp → NTT digits →
//                                  out_a = a_fwd[0] - Σ_j D_final[j] * ksk_neg1[j]
// Collapse step `s` folds at `k = half-1-s` in the reference CollapseA loop;
// its expanded key K_+^(a)[k-1] is therefore at kskstep = n_steps-1-s.
//
// Output layout (the device-resident form gpu_setup_server adopts directly, see
// d_precomp_D_plus etc.): per-limb contiguous, [limb][step][digit][coeff].
// =====================================================================
struct GpuCollapseHelper {
    static constexpr size_t n_steps = N / 2 - 1;

    // Twiddle tables (per limb) — both standard twiddles and Shoup-Harvey
    // "primes" (precomputed `floor(w * 2^32 / q)`) for fast modmul in the
    // fused kernel.
    uint32_t *d_fwd_q0, *d_fwd_q0_prime, *d_inv_q0, *d_inv_q0_prime;
    uint32_t *d_fwd_q1, *d_fwd_q1_prime, *d_inv_q1, *d_inv_q1_prime;
    uint32_t inv_n_q0, inv_n_q0_prime;
    uint32_t inv_n_q1, inv_n_q1_prime;

    // ExpandedKSK on device
    uint32_t *d_ksk_plus;     // 2 × n_steps × D_EFF × N uint32, layout [limb][step][digit][coeff]
    uint32_t *d_ksk_minus;    // same shape
    uint32_t *d_ksk_neg1;     // 2 × D_EFF × N uint32 (NTT'd here)

    // Scratch: one polynomial in coefficient form (2 limbs contiguous)
    uint32_t *d_scratch_coeff;

    // Phase 12: GPU-native KSK setup. Replaces the CPU `expand_ksk_a` +
    // host-side ExpandedKSK marshalling with:
    //   1. CPU expand_seed × (2 * D_EFF) — SHAKE256 + uniform sample to
    //      coefficient form. Tiny (~ms), keeps OpenSSL SHAKE on CPU.
    //   2. Host-flat upload of the 4 polys (~32 KB ksk5 + 32 KB ksk_neg1).
    //   3. GPU CT-NTT of each (4 polys × 2 limbs = 8 launches; cheap).
    //   4. GPU batched permutation kernel populates d_ksk_plus / d_ksk_minus
    //      directly in the final layout.
    //   5. ksk_neg1's NTT'd polys are reshuffled into d_ksk_neg1 layout.
    GpuCollapseHelper(const uint8_t* seed) {
        // ---- Twiddles + Shoup primes (same as before) ----
        std::vector<uint32_t> fwd0, fwd0p, inv0, inv0p, fwd1, fwd1p, inv1, inv1p;
        uint32_t in0, in0p, in1, in1p;
        get_ntt_twiddles_and_primes(0, fwd0, fwd0p, inv0, inv0p, in0, in0p);
        get_ntt_twiddles_and_primes(1, fwd1, fwd1p, inv1, inv1p, in1, in1p);
        inv_n_q0 = in0; inv_n_q0_prime = in0p;
        inv_n_q1 = in1; inv_n_q1_prime = in1p;

        const size_t tw_bytes = N * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_fwd_q0,        tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_fwd_q0_prime,  tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_inv_q0,        tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_inv_q0_prime,  tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_fwd_q1,        tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_fwd_q1_prime,  tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_inv_q1,        tw_bytes));
        CUDA_CHECK(cudaMalloc(&d_inv_q1_prime,  tw_bytes));
        CUDA_CHECK(cudaMemcpy(d_fwd_q0,        fwd0 .data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fwd_q0_prime,  fwd0p.data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv_q0,        inv0 .data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv_q0_prime,  inv0p.data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fwd_q1,        fwd1 .data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fwd_q1_prime,  fwd1p.data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv_q1,        inv1 .data(), tw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv_q1_prime,  inv1p.data(), tw_bytes, cudaMemcpyHostToDevice));

        // ---- d_gp[k] = 5^k mod 2N (for the perm kernel) ----
        uint32_t* d_gp = nullptr;
        {
            std::vector<uint32_t> h_gp(N);
            uint64_t v = 1;
            for (size_t k = 0; k < N; k++) {
                h_gp[k] = (uint32_t)v;
                v = (v * 5) % (2 * N);
            }
            CUDA_CHECK(cudaMalloc(&d_gp, N * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_gp, h_gp.data(), N * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
        }

        // ---- CPU expand_seed × 4: 2 ksk5 + 2 ksk_neg1, coeff form. Each is
        // ~32 KB of uniform-mod-q samples from SHAKE256(seed || "ksk" || idx).
        std::vector<uint32_t> h_ksk5_coeff((size_t)D_EFF * 2 * N);
        std::vector<uint32_t> h_neg1_coeff((size_t)D_EFF * 2 * N);
        for (int j = 0; j < D_EFF; j++) {
            int digit_idx = j + 1;
            RnsPoly r5 = expand_seed(seed, "ksk", AUTO_GEN * 100 + digit_idx);
            RnsPoly rn = expand_seed(seed, "ksk", (uint64_t)(2 * N - 1) * 100 + digit_idx);
            // Pack [j][limb][coeff] flat.
            for (size_t i = 0; i < N; i++) {
                h_ksk5_coeff[(size_t)j * 2 * N + 0 * N + i] = (uint32_t)r5.limbs[0][i];
                h_ksk5_coeff[(size_t)j * 2 * N + 1 * N + i] = (uint32_t)r5.limbs[1][i];
                h_neg1_coeff[(size_t)j * 2 * N + 0 * N + i] = (uint32_t)rn.limbs[0][i];
                h_neg1_coeff[(size_t)j * 2 * N + 1 * N + i] = (uint32_t)rn.limbs[1][i];
            }
        }

        // ---- Upload coeff-form polys + GPU NTT in place ----
        uint32_t* d_ksk5_ntt = nullptr;
        const size_t ksk5_bytes = (size_t)D_EFF * 2 * N * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_ksk5_ntt, ksk5_bytes));
        CUDA_CHECK(cudaMemcpy(d_ksk5_ntt, h_ksk5_coeff.data(), ksk5_bytes,
                              cudaMemcpyHostToDevice));

        const size_t neg1_per_limb = (size_t)D_EFF * N;
        const size_t neg1_size = 2 * neg1_per_limb;
        CUDA_CHECK(cudaMalloc(&d_ksk_neg1, neg1_size * sizeof(uint32_t)));

        // NTT ksk5_ntt: 4 polys × 2 limbs in place.
        for (int j = 0; j < D_EFF; j++) {
            uint32_t* p = d_ksk5_ntt + (size_t)j * 2 * N;
            inspire::gpu::gpu_ntt_forward(p,     N, Q0, d_fwd_q0);
            inspire::gpu::gpu_ntt_forward(p + N, N, Q1, d_fwd_q1);
        }

        // NTT ksk_neg1: 4 polys × 2 limbs, but final layout splits limbs.
        // Stage in d_ksk_neg1 in the same [j][limb][coeff] form first, NTT,
        // then transpose-on-write into the [limb][digit][coeff] layout.
        uint32_t* d_neg1_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_neg1_tmp, ksk5_bytes));
        CUDA_CHECK(cudaMemcpy(d_neg1_tmp, h_neg1_coeff.data(), ksk5_bytes,
                              cudaMemcpyHostToDevice));
        for (int j = 0; j < D_EFF; j++) {
            uint32_t* p = d_neg1_tmp + (size_t)j * 2 * N;
            inspire::gpu::gpu_ntt_forward(p,     N, Q0, d_fwd_q0);
            inspire::gpu::gpu_ntt_forward(p + N, N, Q1, d_fwd_q1);
            // Transpose into d_ksk_neg1[limb][j][coeff]
            CUDA_CHECK(cudaMemcpy(d_ksk_neg1 + (size_t)j * N, p,
                                  N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ksk_neg1 + neg1_per_limb + (size_t)j * N, p + N,
                                  N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }
        cudaFree(d_neg1_tmp);

        // ---- GPU batched permute: d_ksk5_ntt × τ_{5^s}, τ_{2n-5^s} →
        //      d_ksk_plus, d_ksk_minus.
        const size_t per_limb_ksk = (size_t)n_steps * D_EFF * N;
        const size_t ksk_size = 2 * per_limb_ksk;
        CUDA_CHECK(cudaMalloc(&d_ksk_plus,  ksk_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ksk_minus, ksk_size * sizeof(uint32_t)));
        inspire::gpu::gpu_ksk_perm_batched(
            d_ksk_plus, d_ksk_minus, d_ksk5_ntt, d_gp,
            (int)n_steps, D_EFF, N);

        cudaFree(d_ksk5_ntt);
        cudaFree(d_gp);

        // Scratch (legacy: used by orchestrated `compute()`)
        CUDA_CHECK(cudaMalloc(&d_scratch_coeff, 2 * N * sizeof(uint32_t)));
    }

    ~GpuCollapseHelper() {
        cudaFree(d_fwd_q0); cudaFree(d_fwd_q0_prime);
        cudaFree(d_inv_q0); cudaFree(d_inv_q0_prime);
        cudaFree(d_fwd_q1); cudaFree(d_fwd_q1_prime);
        cudaFree(d_inv_q1); cudaFree(d_inv_q1_prime);
        cudaFree(d_ksk_plus); cudaFree(d_ksk_minus); cudaFree(d_ksk_neg1);
        cudaFree(d_scratch_coeff);
    }

    // Phase 10: batched collapse — runs all num_groups groups in one kernel
    // launch using per-group pointer arrays (device-resident). The caller
    // builds the pointer arrays and passes them as device pointers.
    void compute_fused_batched(
        const uint32_t* const* d_a_post_fwd_ptrs,
        const uint32_t* const* d_a_post_conj_ptrs,
        uint32_t* const* d_out_a_ptrs,
        uint32_t* const* d_out_D_plus_ptrs,
        uint32_t* const* d_out_D_minus_ptrs,
        uint32_t* const* d_out_D_final_ptrs,
        int num_groups)
    {
        auto modinv = [](uint64_t a, uint64_t m) -> uint64_t {
            int64_t old_r = (int64_t)a, r = (int64_t)m;
            int64_t old_s = 1, s = 0;
            while (r != 0) {
                int64_t qq = old_r / r;
                int64_t t = r; r = old_r - qq * r; old_r = t;
                t = s; s = old_s - qq * s; old_s = t;
            }
            return (uint64_t)((old_s % (int64_t)m + (int64_t)m) % (int64_t)m);
        };
        const uint64_t q1_inv_q0 = modinv(Q1, Q0);
        const uint64_t q0_inv_q1 = modinv(Q0, Q1);
        const __uint128_t Q_full = (__uint128_t)Q0 * Q1;
        const uint64_t q_full_lo = (uint64_t)Q_full;
        const uint64_t q_full_hi = (uint64_t)(Q_full >> 64);

        inspire::gpu::gpu_collapse_fused(
            d_a_post_fwd_ptrs, d_a_post_conj_ptrs,
            d_ksk_plus, d_ksk_minus, d_ksk_neg1,
            d_fwd_q0, d_fwd_q0_prime,
            d_inv_q0, d_inv_q0_prime,
            d_fwd_q1, d_fwd_q1_prime,
            d_inv_q1, d_inv_q1_prime,
            inv_n_q0, inv_n_q0_prime,
            inv_n_q1, inv_n_q1_prime,
            d_out_a_ptrs, d_out_D_plus_ptrs,
            d_out_D_minus_ptrs, d_out_D_final_ptrs,
            N, N / 2, n_steps,
            (uint32_t)Q0, (uint32_t)Q1,
            q1_inv_q0, q0_inv_q1, q_full_lo, q_full_hi,
            D_GSW, D_EFF, BASE_LOG,
            num_groups);
    }

};

}  // anonymous namespace

PreprocessData gpu_preprocess(const PublicParams& pp, const uint16_t* db) {
    PreprocessData precomp;

    const bool profile = (std::getenv("INSPIRE_PROFILE_PREPROCESS") != nullptr);
    auto tic = []() { return std::chrono::high_resolution_clock::now(); };
    auto ms_since = [](std::chrono::high_resolution_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };
    double t_encode = 0, t_crs = 0, t_ringhelper = 0, t_expand_ksk = 0,
           t_collhelper = 0, t_db_upload = 0, t_a_crs_upload = 0;

    // Phase: encode_database (GPU). Phase 14: u16 device DB with 15-bit
    // packing per slot. CPU precomp.db is left empty (gpu_setup_server adopts
    // d_db_col).
    auto T = tic();
    uint16_t* d_db = nullptr;
    {
        const size_t db_count = (size_t)pp.db_rows * pp.db_cols;
        CUDA_CHECK(cudaMalloc(&d_db, db_count * sizeof(uint16_t)));
        // Upload the slot DB (one u16 copy — no raw byte DB, no byte-pack) and
        // apply the in-place inverse-DFT. Peak = encoded DB only.
        CUDA_CHECK(cudaMemcpy(d_db, db, db_count * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        inspire::gpu::gpu_encode_inverse_dft(d_db, pp.db_rows, pp.db_cols, pp.D, N);
        cudaDeviceSynchronize();
    }
    if (profile) t_encode = ms_since(T);

    size_t n_groups = pp.n_packed;
    size_t n_r = pp.db_rows / N;

    // Phase: CRS polynomial derivation (CPU NTT, n_r polynomials)
    T = tic();
    std::vector<RnsPoly> a_crs_rev(n_r);
    for (size_t r = 0; r < n_r; r++) {
        RnsPoly a = expand_seed(pp.seed.data(), "rlwe", r);
        RnsPoly rev = RnsPoly::zero();
        rev.limbs[0][0] = a.limbs[0][0];
        rev.limbs[1][0] = a.limbs[1][0];
        for (size_t i = 1; i < N; i++) {
            rev.limbs[0][i] = (a.limbs[0][N - i] == 0) ? 0 : Q0 - a.limbs[0][N - i];
            rev.limbs[1][i] = (a.limbs[1][N - i] == 0) ? 0 : Q1 - a.limbs[1][N - i];
        }
        rev.is_ntt = false;
        rev.to_ntt();
        a_crs_rev[r] = rev;
    }
    if (profile) t_crs = ms_since(T);

    // Phase: GpuRingEmbedHelper construction (uploads mono32, gp, perms)
    T = tic();
    GpuRingEmbedHelper helper(N, (int)N);
    cudaDeviceSynchronize();
    if (profile) t_ringhelper = ms_since(T);

    // Phase 12: GPU-native ExpandKSK + GpuCollapseHelper construction.
    // Replaces the CPU expand_ksk_a (~64 ms, dominated by 8184 permutations)
    // and the host-side ExpandedKSK marshalling. SHAKE256+sampling stay on
    // CPU; NTT + permutation happen on device.
    if (profile) t_expand_ksk = 0;          // folded into t_collhelper now
    T = tic();
    GpuCollapseHelper collapse_helper(pp.seed.data());
    cudaDeviceSynchronize();
    if (profile) t_collhelper = ms_since(T);

    // (No separate d_db upload phase — already on device from gpu_encode_inverse_dft.)
    t_db_upload = 0;
    // Phase: a_crs_rev upload (small, n_r*2N u32)
    T = tic();
    uint32_t* d_a_crs_q0 = nullptr;
    uint32_t* d_a_crs_q1 = nullptr;
    {
        std::vector<uint32_t> h_q0(n_r * N), h_q1(n_r * N);
        for (size_t r = 0; r < n_r; r++) {
            assert(a_crs_rev[r].is_ntt);
            for (size_t i = 0; i < N; i++) {
                h_q0[r * N + i] = (uint32_t)a_crs_rev[r].limbs[0][i];
                h_q1[r * N + i] = (uint32_t)a_crs_rev[r].limbs[1][i];
            }
        }
        CUDA_CHECK(cudaMalloc(&d_a_crs_q0, n_r * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_a_crs_q1, n_r * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_a_crs_q0, h_q0.data(), n_r * N * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_crs_q1, h_q1.data(), n_r * N * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }
    cudaDeviceSynchronize();
    if (profile) t_a_crs_upload = ms_since(T);

    // nu_inv (1/N mod q) from collapse_helper's tables — same as build_a32 uses
    const uint32_t nu_inv_q0 = collapse_helper.inv_n_q0;
    const uint32_t nu_inv_q1 = collapse_helper.inv_n_q1;

    // Phase 3: allocate per-group output buffers ON DEVICE in the final layout
    // (matches what gpu_setup_server expects). The buffers are NOT downloaded;
    // gpu_setup_server later adopts these pointers directly.
    const size_t n_steps = N / 2 - 1;
    precomp.d_precomp_a.assign(n_groups, nullptr);
    precomp.d_precomp_D_plus.assign(n_groups, nullptr);
    precomp.d_precomp_D_minus.assign(n_groups, nullptr);
    precomp.d_precomp_D_final.assign(n_groups, nullptr);

    // --- INSTRUMENTATION (temporary): per-phase CUDA event timing ---
    // `profile` already declared at the top of this function.
    cudaEvent_t ev_a, ev_b, ev_c, ev_d, ev_e;
    float t_alloc = 0, t_matmul = 0, t_ring = 0, t_collapse = 0;
    if (profile) {
        cudaEventCreate(&ev_a); cudaEventCreate(&ev_b); cudaEventCreate(&ev_c);
        cudaEventCreate(&ev_d); cudaEventCreate(&ev_e);
    }

    // Phase 10: stage each group's post-ring-embed input in its own device
    // buffer so the batched collapse kernel (one launch covering all groups)
    // can read them concurrently via pointer arrays.
    std::vector<uint32_t*> per_group_apf(n_groups, nullptr);
    std::vector<uint32_t*> per_group_apc(n_groups, nullptr);
    {
        const size_t apf_bytes = (size_t)(N / 2) * 2 * N * sizeof(uint32_t);
        for (size_t g = 0; g < n_groups; g++) {
            CUDA_CHECK(cudaMalloc(&per_group_apf[g], apf_bytes));
            CUDA_CHECK(cudaMalloc(&per_group_apc[g], apf_bytes));
        }
    }

    for (size_t g = 0; g < n_groups; g++) {
        if (profile) cudaEventRecord(ev_a);

        CUDA_CHECK(cudaMalloc(&precomp.d_precomp_a[g],       2 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&precomp.d_precomp_D_plus[g],  2 * n_steps * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&precomp.d_precomp_D_minus[g], 2 * n_steps * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&precomp.d_precomp_D_final[g], 2 * D_EFF * N * sizeof(uint32_t)));

        if (profile) cudaEventRecord(ev_b);

        // Phase 2 GPU matmul writes ring_embed's d_a32 directly (no CPU NTT).
        inspire::gpu::gpu_a_side_matmul_a32(
            helper.d_a32,
            d_db, pp.db_rows, pp.db_cols,
            g, n_r, N,
            d_a_crs_q0, d_a_crs_q1,
            nu_inv_q0, nu_inv_q1,
            collapse_helper.d_fwd_q0, collapse_helper.d_fwd_q1,
            (uint32_t)Q0, (uint32_t)Q1);

        if (profile) cudaEventRecord(ev_c);

        // GPU ring_embed + automorphisms (d_a32 already populated).
        helper.compute_device_with_external_a32();

        if (profile) cudaEventRecord(ev_d);

        // Stage this group's post-ring-embed into its dedicated buffer; the
        // batched collapse kernel reads from all n_groups buffers in one
        // launch after the loop.
        {
            const size_t apf_bytes = (size_t)(N / 2) * 2 * N * sizeof(uint32_t);
            CUDA_CHECK(cudaMemcpyAsync(per_group_apf[g], helper.d_a_post_fwd,
                                       apf_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyAsync(per_group_apc[g], helper.d_a_post_conj,
                                       apf_bytes, cudaMemcpyDeviceToDevice));
        }

        if (profile) {
            cudaEventRecord(ev_e);
            cudaEventSynchronize(ev_e);
            float ms;
            cudaEventElapsedTime(&ms, ev_a, ev_b); t_alloc    += ms;
            cudaEventElapsedTime(&ms, ev_b, ev_c); t_matmul   += ms;
            cudaEventElapsedTime(&ms, ev_c, ev_d); t_ring     += ms;
            cudaEventElapsedTime(&ms, ev_d, ev_e); t_collapse += ms;
        }
    }

    // Phase 10: batched collapse launch — all n_groups groups in one kernel.
    {
        cudaEvent_t bc_start, bc_stop;
        if (profile) {
            cudaEventCreate(&bc_start);
            cudaEventCreate(&bc_stop);
            cudaEventRecord(bc_start);
        }

        // Upload pointer arrays to device.
        const uint32_t** d_apf_ptrs = nullptr;
        const uint32_t** d_apc_ptrs = nullptr;
        uint32_t** d_dpa_ptrs = nullptr;
        uint32_t** d_dpp_ptrs = nullptr;
        uint32_t** d_dpm_ptrs = nullptr;
        uint32_t** d_dpf_ptrs = nullptr;
        const size_t ptr_bytes = n_groups * sizeof(uint32_t*);
        CUDA_CHECK(cudaMalloc(&d_apf_ptrs, ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_apc_ptrs, ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_dpa_ptrs, ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_dpp_ptrs, ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_dpm_ptrs, ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_dpf_ptrs, ptr_bytes));
        CUDA_CHECK(cudaMemcpy(d_apf_ptrs, per_group_apf.data(), ptr_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_apc_ptrs, per_group_apc.data(), ptr_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dpa_ptrs, precomp.d_precomp_a.data(),       ptr_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dpp_ptrs, precomp.d_precomp_D_plus.data(),  ptr_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dpm_ptrs, precomp.d_precomp_D_minus.data(), ptr_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dpf_ptrs, precomp.d_precomp_D_final.data(), ptr_bytes, cudaMemcpyHostToDevice));

        collapse_helper.compute_fused_batched(
            d_apf_ptrs, d_apc_ptrs,
            d_dpa_ptrs, d_dpp_ptrs, d_dpm_ptrs, d_dpf_ptrs,
            (int)n_groups);

        if (profile) {
            cudaEventRecord(bc_stop);
            cudaEventSynchronize(bc_stop);
            float ms;
            cudaEventElapsedTime(&ms, bc_start, bc_stop);
            t_collapse += ms;
            cudaEventDestroy(bc_start);
            cudaEventDestroy(bc_stop);
        }

        // Free per-group staging + pointer arrays. The output buffers
        // (d_precomp_*) are kept; ctx adopts them in gpu_setup_server.
        for (size_t g = 0; g < n_groups; g++) {
            cudaFree(per_group_apf[g]);
            cudaFree(per_group_apc[g]);
        }
        cudaFree(d_apf_ptrs); cudaFree(d_apc_ptrs);
        cudaFree(d_dpa_ptrs); cudaFree(d_dpp_ptrs);
        cudaFree(d_dpm_ptrs); cudaFree(d_dpf_ptrs);
    }

    if (profile) {
        float loop_total = t_alloc + t_matmul + t_ring + t_collapse;
        double pre_total = t_encode + t_crs + t_ringhelper + t_expand_ksk
                         + t_collhelper + t_db_upload + t_a_crs_upload;
        double grand_total = pre_total + (double)loop_total;
        std::fprintf(stderr,
            "[gpu_preprocess profile, 1GB-ish=%zu groups, n_r=%zu]\n"
            "  PRE-LOOP:\n"
            "    encode_database (CPU)     = %7.1f ms (%5.1f%%)\n"
            "    CRS derivation (CPU NTT)  = %7.1f ms (%5.1f%%)\n"
            "    GpuRingEmbedHelper init   = %7.1f ms (%5.1f%%)\n"
            "    expand_ksk_a (CPU)        = %7.1f ms (%5.1f%%)\n"
            "    GpuCollapseHelper init    = %7.1f ms (%5.1f%%)\n"
            "    d_db H2D upload           = %7.1f ms (%5.1f%%)\n"
            "    a_crs upload              = %7.1f ms (%5.1f%%)\n"
            "    pre-loop subtotal         = %7.1f ms (%5.1f%%)\n"
            "  PER-GROUP LOOP:\n"
            "    alloc                     = %7.1f ms (%5.1f%%)\n"
            "    matmul                    = %7.1f ms (%5.1f%%)\n"
            "    ring_embed                = %7.1f ms (%5.1f%%)\n"
            "    collapse                  = %7.1f ms (%5.1f%%)\n"
            "    loop subtotal             = %7.1f ms (%5.1f%%)\n"
            "  grand total                 = %7.1f ms\n",
            n_groups, n_r,
            t_encode,      100.0 * t_encode      / grand_total,
            t_crs,         100.0 * t_crs         / grand_total,
            t_ringhelper,  100.0 * t_ringhelper  / grand_total,
            t_expand_ksk,  100.0 * t_expand_ksk  / grand_total,
            t_collhelper,  100.0 * t_collhelper  / grand_total,
            t_db_upload,   100.0 * t_db_upload   / grand_total,
            t_a_crs_upload,100.0 * t_a_crs_upload/ grand_total,
            pre_total,     100.0 * pre_total     / grand_total,
            (double)t_alloc,    100.0 * t_alloc    / grand_total,
            (double)t_matmul,   100.0 * t_matmul   / grand_total,
            (double)t_ring,     100.0 * t_ring     / grand_total,
            (double)t_collapse, 100.0 * t_collapse / grand_total,
            (double)loop_total, 100.0 * loop_total / grand_total,
            grand_total);
        cudaEventDestroy(ev_a); cudaEventDestroy(ev_b); cudaEventDestroy(ev_c);
        cudaEventDestroy(ev_d); cudaEventDestroy(ev_e);
    }

    cudaFree(d_a_crs_q0); cudaFree(d_a_crs_q1);

    // Phase 4: transfer ownership of d_db to PreprocessData so that
    // gpu_setup_server can adopt it (no CPU detour).
    precomp.d_db_col = d_db;

    return precomp;
}

// ============================================================
// Setup
// ============================================================

GpuServerCtx* gpu_setup_server(const PublicParams& pp, const PreprocessData& precomp) {
    return gpu_setup_server(pp, precomp, GpuServerConfig{});
}

GpuServerCtx* gpu_setup_server(const PublicParams& pp, const PreprocessData& precomp,
                               const GpuServerConfig& cfg) {
    if (cfg.max_batch == 0) {
        fprintf(stderr, "gpu_setup_server: cfg.max_batch must be >= 1\n");
        std::exit(1);
    }
    auto* ctx = new GpuServerCtx;
    ctx->pp = pp;
    ctx->cfg = cfg;
    ctx->db_rows = pp.db_rows;
    ctx->db_cols = pp.db_cols;
    ctx->n_packed = pp.n_packed;

    // Twiddles
    std::vector<uint32_t> fwd_q0, inv_q0, fwd_q1, inv_q1;
    get_ntt_twiddles(0, fwd_q0, inv_q0, ctx->inv_n_q0);
    get_ntt_twiddles(1, fwd_q1, inv_q1, ctx->inv_n_q1);
    CUDA_CHECK(cudaMalloc(&ctx->d_fwd_q0, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_inv_q0, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_fwd_q1, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_inv_q1, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(ctx->d_fwd_q0, fwd_q0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_inv_q0, inv_q0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_fwd_q1, fwd_q1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_inv_q1, inv_q1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // DB (u16, P=65535, 15-bit packing) is ROW-MAJOR: gpu_preprocess produces it
    // row-major in precomp.d_db_col, adopted directly as d_db_rm (no transpose,
    // single resident copy). Ownership transfers to ctx.
    ctx->d_db_rm = precomp.d_db_col;
    const_cast<PreprocessData&>(precomp).d_db_col = nullptr;

    // Turn the encoded u16 DB into the tensor-core matrix in place.  In byte
    // view the row-major DB is already the column-major transpose needed by
    // cuBLAS: (2*db_cols) rows x db_rows columns.
    CUBLAS_CHECK(cublasCreate(&ctx->cublas));
    CUBLAS_CHECK(cublasSetMathMode(ctx->cublas, CUBLAS_TENSOR_OP_MATH));
    CUDA_CHECK(cudaMalloc(&ctx->d_db_byte_sums, 2 * pp.db_cols * sizeof(int32_t)));
    inspire::gpu::gpu_matvec_tensor_prepare_db(
        ctx->d_db_rm, ctx->d_db_byte_sums, pp.db_rows, pp.db_cols);

    // Upload permutation tables
    std::vector<uint32_t> fwd_flat, conj_flat;
    build_perm_tables(fwd_flat, conj_flat);
    CUDA_CHECK(cudaMalloc(&ctx->d_perm_fwd, fwd_flat.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_perm_conj, conj_flat.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(ctx->d_perm_fwd, fwd_flat.data(),
                          fwd_flat.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_perm_conj, conj_flat.data(),
                          conj_flat.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Per-group precomp: gpu_preprocess filled the per-group device buffers in
    // the exact layout we need; transfer ownership to ctx.
    ctx->d_precomp_a.resize(pp.n_packed);
    ctx->d_precomp_D_plus.resize(pp.n_packed);
    ctx->d_precomp_D_minus.resize(pp.n_packed);
    ctx->d_precomp_D_final.resize(pp.n_packed);
    assert(precomp.d_precomp_a.size() == pp.n_packed);
    for (size_t g = 0; g < pp.n_packed; g++) {
        ctx->d_precomp_a[g]       = precomp.d_precomp_a[g];
        ctx->d_precomp_D_plus[g]  = precomp.d_precomp_D_plus[g];
        ctx->d_precomp_D_minus[g] = precomp.d_precomp_D_minus[g];
        ctx->d_precomp_D_final[g] = precomp.d_precomp_D_final[g];
    }
    const_cast<PreprocessData&>(precomp).d_precomp_a.clear();
    const_cast<PreprocessData&>(precomp).d_precomp_D_plus.clear();
    const_cast<PreprocessData&>(precomp).d_precomp_D_minus.clear();
    const_cast<PreprocessData&>(precomp).d_precomp_D_final.clear();

    // The a-side of every packed ciphertext is query-independent. Keep one
    // contiguous copy shared by every slot, instead of copying/replicating it
    // for every request in the batch.
    CUDA_CHECK(cudaMalloc(&ctx->d_packed_a,
                          pp.n_packed * 2 * N * sizeof(uint32_t)));
    for (size_t g = 0; g < pp.n_packed; g++)
        CUDA_CHECK(cudaMemcpyAsync(ctx->d_packed_a + g * 2 * N,
                                   ctx->d_precomp_a[g],
                                   2 * N * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, 0));

    // Per-query scratch: cfg.max_batch slots, all allocated here so the
    // request path never allocates.
    ctx->slots.resize(cfg.max_batch);
    for (auto& sl : ctx->slots) {
        CUDA_CHECK(cudaMalloc(&sl.d_query_b_q0, pp.db_rows * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_query_b_q1, pp.db_rows * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_ksk5_b,     2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_kskneg1_b,  2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_rgsw_top_a, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_rgsw_top_b, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_rgsw_bot_a, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_rgsw_bot_b, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_packed_b,   pp.n_packed * 2 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_acc_a,      2 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_acc_b,      2 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_res_a,      2 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_res_b,      2 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_scratch_coeff, 4 * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_dig,        4 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_query_b_q0, pp.db_rows * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_query_b_q1, pp.db_rows * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_ksk5_b,    2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_kskneg1_b, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_rgsw_top_a, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_rgsw_top_b, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_rgsw_bot_a, 2 * D_EFF * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&sl.h_rgsw_bot_b, 2 * D_EFF * N * sizeof(uint32_t)));
    }

    // Device pointer arrays over the slot pool for the batched matvec.
    {
        std::vector<const uint32_t*> q0(cfg.max_batch), q1(cfg.max_batch);
        std::vector<uint32_t*> r0(cfg.max_batch), r1(cfg.max_batch);
        for (size_t i = 0; i < cfg.max_batch; i++) {
            q0[i] = ctx->slots[i].d_query_b_q0;
            q1[i] = ctx->slots[i].d_query_b_q1;
            r0[i] = ctx->slots[i].d_packed_b;
            r1[i] = ctx->slots[i].d_packed_b;
        }
        const size_t pb = cfg.max_batch * sizeof(uint32_t*);
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_q0_ptrs, pb));
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_q1_ptrs, pb));
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_r0_ptrs, pb));
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_r1_ptrs, pb));
        CUDA_CHECK(cudaMemcpy((void*)ctx->d_mv_q0_ptrs, q0.data(), pb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)ctx->d_mv_q1_ptrs, q1.data(), pb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)ctx->d_mv_r0_ptrs, r0.data(), pb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)ctx->d_mv_r1_ptrs, r1.data(), pb, cudaMemcpyHostToDevice));

        // One query contributes four byte planes for each of two RNS limbs.
        // GEMM output is [plane-column][interleaved DB byte/output column].
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_query_planes,
                              cfg.max_batch * 8 * pp.db_rows * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_query_sums,
                              cfg.max_batch * 8 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_mv_gemm_out,
                              cfg.max_batch * 8 * 2 * pp.db_cols * sizeof(int32_t)));

        const size_t col_blocks = (pp.db_cols + 255) / 256;
        if (col_blocks < 150 && !(pp.db_cols >= 8192 && pp.db_rows <= 8192)) {
            size_t rows_per_block = (pp.db_rows + 255) / 256;
            if (rows_per_block < 64) rows_per_block = 64;
            const size_t row_blocks = (pp.db_rows + rows_per_block - 1) / rows_per_block;
            ctx->mv_tall_partial_count = row_blocks * pp.db_cols;
            CUDA_CHECK(cudaMalloc(&ctx->d_mv_tall_partials0,
                                  ctx->mv_tall_partial_count * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&ctx->d_mv_tall_partials1,
                                  ctx->mv_tall_partial_count * sizeof(uint32_t)));
        }
    }

    // Resident-memory bookkeeping for gpu_server_caps.
    {
        const size_t n_steps_bk = N / 2 - 1;
        const size_t per_group = (2 * N)                              // a
                               + 2 * (2 * n_steps_bk * D_EFF * N)     // D_plus + D_minus
                               + (2 * D_EFF * N);                     // D_final
        const size_t per_slot = 2 * pp.db_rows
                              + 6 * (2 * D_EFF * N)
                              + (pp.n_packed * 2 * N)
                              + 4 * (2 * N) + 4 * N + 4 * D_EFF * N;
        ctx->resident_bytes =
              (size_t)pp.db_rows * pp.db_cols * sizeof(uint16_t)      // encoded DB
            + pp.n_packed * per_group * sizeof(uint32_t)              // precomp
            + cfg.max_batch * per_slot * sizeof(uint32_t)             // device slot pools
            + pp.n_packed * 2 * N * sizeof(uint32_t)                  // shared packed a
            + 2 * (n_steps_bk * N) * sizeof(uint32_t)                 // perm tables
            + 4 * N * sizeof(uint32_t)                                // twiddles
            + 2 * pp.db_cols * sizeof(int32_t)                        // centered DB sums
            + cfg.max_batch * 8 * pp.db_rows * sizeof(int8_t)         // query byte planes
            + cfg.max_batch * 8 * sizeof(int32_t)                     // query-plane sums
            + cfg.max_batch * 8 * 2 * pp.db_cols * sizeof(int32_t)    // tensor GEMM output
            + 2 * ctx->mv_tall_partial_count * sizeof(uint32_t);      // reusable tall scratch
    }

    // Per-group streams (capped at 32 to avoid GPU oversubscription).
    // Multiple groups share streams round-robin; ops within a stream are
    // serialized (so fwd→conj dependency is preserved when both run on the
    // same stream).
    size_t n_streams = std::min(pp.n_packed, (size_t)32);
    if (n_streams == 0) n_streams = 1;
    ctx->streams.resize(n_streams);
    ctx->stream_done_events.resize(n_streams);
    for (size_t i = 0; i < n_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&ctx->streams[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx->stream_done_events[i],
                                            cudaEventDisableTiming));
    }

    // One lazy-collapse partials slice per concurrent stream (reused across the
    // fwd/conj × q0/q1 calls, which serialize within a group's stream).
    CUDA_CHECK(cudaMalloc(&ctx->d_lazy_partials,
                          n_streams * inspire::gpu::LAZY_COLLAPSE_STEP_CHUNKS * N * sizeof(uint32_t)));

    // B3: batched-collapse plumbing. Slot-base pointer arrays (packed_b and
    // the two client keys) and a partials pool with one slice per
    // (stream, slot) pair — groups on different streams run concurrently,
    // and within a stream the fwd/conj/limb launches serialize.
    {
        const size_t mb = cfg.max_batch;
        std::vector<uint32_t*> pb(mb);
        std::vector<const uint32_t*> k5(mb), kn(mb);
        for (size_t i = 0; i < mb; i++) {
            pb[i] = ctx->slots[i].d_packed_b;
            k5[i] = ctx->slots[i].d_ksk5_b;
            kn[i] = ctx->slots[i].d_kskneg1_b;
        }
        const size_t pbb = mb * sizeof(uint32_t*);
        CUDA_CHECK(cudaMalloc(&ctx->d_slot_packedb_ptrs, pbb));
        CUDA_CHECK(cudaMalloc(&ctx->d_slot_ksk5_ptrs, pbb));
        CUDA_CHECK(cudaMalloc(&ctx->d_slot_kskneg1_ptrs, pbb));
        CUDA_CHECK(cudaMemcpy(ctx->d_slot_packedb_ptrs, pb.data(), pbb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)ctx->d_slot_ksk5_ptrs, k5.data(), pbb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)ctx->d_slot_kskneg1_ptrs, kn.data(), pbb, cudaMemcpyHostToDevice));

        const size_t slice = inspire::gpu::LAZY_COLLAPSE_STEP_CHUNKS * N;
        CUDA_CHECK(cudaMalloc(&ctx->d_batch_partials,
                              n_streams * mb * slice * sizeof(uint32_t)));
        std::vector<uint32_t*> pp(n_streams * mb);
        for (size_t s = 0; s < n_streams; s++)
            for (size_t b = 0; b < mb; b++)
                pp[s * mb + b] = ctx->d_batch_partials + (s * mb + b) * slice;
        CUDA_CHECK(cudaMalloc(&ctx->d_batch_part_ptrs, pp.size() * sizeof(uint32_t*)));
        CUDA_CHECK(cudaMemcpy(ctx->d_batch_part_ptrs, pp.data(),
                              pp.size() * sizeof(uint32_t*), cudaMemcpyHostToDevice));
        ctx->resident_bytes += n_streams * mb * slice * sizeof(uint32_t);
    }

    // B4: lockstep-Horner plumbing.
    {
        const size_t mb = cfg.max_batch;
        std::vector<uint32_t*> aa(mb), ab(mb);
        std::vector<const uint32_t*> pa(mb), ta(mb), tb(mb), ba(mb), bb(mb);
        for (size_t i = 0; i < mb; i++) {
            aa[i] = ctx->slots[i].d_acc_a;
            ab[i] = ctx->slots[i].d_acc_b;
            pa[i] = ctx->d_packed_a;
            ta[i] = ctx->slots[i].d_rgsw_top_a;
            tb[i] = ctx->slots[i].d_rgsw_top_b;
            ba[i] = ctx->slots[i].d_rgsw_bot_a;
            bb[i] = ctx->slots[i].d_rgsw_bot_b;
        }
        const size_t pbb = mb * sizeof(uint32_t*);
        auto up = [&](auto*& dst, const void* src) {
            CUDA_CHECK(cudaMalloc(&dst, pbb));
            CUDA_CHECK(cudaMemcpy((void*)dst, src, pbb, cudaMemcpyHostToDevice));
        };
        up(ctx->d_slot_acc_a_ptrs, aa.data());
        up(ctx->d_slot_acc_b_ptrs, ab.data());
        up(ctx->d_slot_packeda_ptrs, pa.data());
        up(ctx->d_slot_rgsw_ta_ptrs, ta.data());
        up(ctx->d_slot_rgsw_tb_ptrs, tb.data());
        up(ctx->d_slot_rgsw_ba_ptrs, ba.data());
        up(ctx->d_slot_rgsw_bb_ptrs, bb.data());

        const size_t coeff_sz = 2 * mb * N * sizeof(uint32_t);
        const size_t dig_sz = (size_t)2 * mb * D_EFF * N * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&ctx->d_h_coeff_q0, coeff_sz));
        CUDA_CHECK(cudaMalloc(&ctx->d_h_coeff_q1, coeff_sz));
        CUDA_CHECK(cudaMalloc(&ctx->d_h_dig_q0, dig_sz));
        CUDA_CHECK(cudaMalloc(&ctx->d_h_dig_q1, dig_sz));
        ctx->resident_bytes += 2 * coeff_sz + 2 * dig_sz;
    }

    return ctx;
}

// ============================================================
// Per-query answer path, parameterized by scratch slot. gpu_answer runs it
// on slot 0; gpu_answer_batch does NOT use it — the batch path drives the
// B2/B3/B4 batched kernels directly (see gpu_answer_batch below).
// ============================================================

// Host-side mod reduction + H2D of the query's b vector into a slot.
static void upload_query_b(GpuServerCtx* ctx, QuerySlot& slot, const QueryMessage& qry) {
    const PublicParams& pp = ctx->pp;
    for (size_t i = 0; i < pp.db_rows; i++) {
        slot.h_query_b_q0[i] = (uint32_t)(qry.lwe.b_limb0[i] % Q0);
        slot.h_query_b_q1[i] = (uint32_t)(qry.lwe.b_limb1[i] % Q1);
    }
    CUDA_CHECK(cudaMemcpyAsync(slot.d_query_b_q0, slot.h_query_b_q0,
                               pp.db_rows * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(slot.d_query_b_q1, slot.h_query_b_q1,
                               pp.db_rows * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, 0));
}

// Upload the query's two key-switching-key b-parts into the slot and NTT
// them (the client sends them in coefficient form).
static void upload_ksks(GpuServerCtx* ctx, QuerySlot& slot, const QueryMessage& qry) {
    {
        uint32_t* h_buf = slot.h_ksk5_b;
        for (int j = 0; j < D_EFF; j++) {
            RnsPoly p = qry.ksk_5.b_parts[j];
            if (p.is_ntt) p.to_coeff();
            for (size_t i = 0; i < N; i++) {
                h_buf[j * N + i]             = (uint32_t)p.limbs[0][i];
                h_buf[D_EFF * N + j * N + i] = (uint32_t)p.limbs[1][i];
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(slot.d_ksk5_b, h_buf,
                                   2 * D_EFF * N * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, 0));
        inspire::gpu::gpu_ntt_batch(slot.d_ksk5_b, N, D_EFF,
                                    Q0, ctx->d_fwd_q0, 1024);
        inspire::gpu::gpu_ntt_batch(slot.d_ksk5_b + D_EFF * N, N, D_EFF,
                                    Q1, ctx->d_fwd_q1, 1024);
    }
    {
        uint32_t* h_buf = slot.h_kskneg1_b;
        for (int j = 0; j < D_EFF; j++) {
            RnsPoly p = qry.ksk_neg1.b_parts[j];
            if (p.is_ntt) p.to_coeff();
            for (size_t i = 0; i < N; i++) {
                h_buf[j * N + i]             = (uint32_t)p.limbs[0][i];
                h_buf[D_EFF * N + j * N + i] = (uint32_t)p.limbs[1][i];
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(slot.d_kskneg1_b, h_buf,
                                   2 * D_EFF * N * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, 0));
        inspire::gpu::gpu_ntt_batch(slot.d_kskneg1_b, N, D_EFF,
                                    Q0, ctx->d_fwd_q0, 1024);
        inspire::gpu::gpu_ntt_batch(slot.d_kskneg1_b + D_EFF * N, N, D_EFF,
                                    Q1, ctx->d_fwd_q1, 1024);
    }
}

// Download the slot's accumulator as one RLWE ciphertext (NTT form). Safe to
// return NTT form: the GPU forward NTT and the in-tree CPU NTT share the same
// algorithm and twiddle layout, so the representations are identical.
static RlweCt download_ct(QuerySlot& slot) {
    std::vector<uint32_t> h_a(2 * N), h_b(2 * N);
    CUDA_CHECK(cudaMemcpy(h_a.data(), slot.d_acc_a, 2 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), slot.d_acc_b, 2 * N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    RlweCt ct;
    ct.a = RnsPoly::zero();
    ct.b = RnsPoly::zero();
    for (size_t i = 0; i < N; i++) {
        ct.a.limbs[0][i] = h_a[i];
        ct.a.limbs[1][i] = h_a[N + i];
        ct.b.limbs[0][i] = h_b[i];
        ct.b.limbs[1][i] = h_b[N + i];
    }
    ct.a.is_ntt = true;
    ct.b.is_ntt = true;
    return ct;
}

// skip_matvec: the caller (gpu_answer_batch) has already uploaded the b
// vector and run the batched matvec directly into slot.d_packed_b.
// skip_pack: the caller has already uploaded the keys and run the batched
// pack into slot.d_packed_*; start from Horner.
static std::vector<RlweCt> answer_one(GpuServerCtx* ctx, QuerySlot& slot,
                                      const QueryMessage& qry,
                                      bool skip_matvec = false,
                                      bool skip_pack = false) {
    const PublicParams& pp = ctx->pp;
    size_t half = N / 2;
    size_t n_steps = half - 1;

    // Phase timing (only emits if env var set)
    auto trace = std::getenv("GPU_ANSWER_TRACE") != nullptr;
    auto phase_t0 = std::chrono::high_resolution_clock::now();
    auto mark = [&](const char* name) {
        if (!trace) return;
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double phase_ms = std::chrono::duration<double, std::milli>(t1 - phase_t0).count();
        std::fprintf(stderr, "  [gpu_answer] %s: %.3f ms\n", name, phase_ms);
        phase_t0 = t1;
    };

    // ---- Fine-grained per-kernel-type profiler (GPU_ANSWER_FINE=1) ----
    // Records a CUDA event pair around each kernel group on its own stream so
    // concurrent streams are not serialized; elapsed times are summed per
    // bucket *after* all work completes. The per-bucket sum is total GPU-busy
    // time for that kernel type (it can exceed wall-clock when streams overlap).
    bool fine = std::getenv("GPU_ANSWER_FINE") != nullptr;
    struct FineRec { const char* bucket; cudaEvent_t a, b; };
    std::vector<FineRec> fine_recs;
    std::map<const char*, int> fine_counts;
    auto fine_begin = [&](const char* bucket, cudaStream_t s) -> int {
        if (!fine) return -1;
        FineRec r{bucket, nullptr, nullptr};
        cudaEventCreate(&r.a); cudaEventCreate(&r.b);
        cudaEventRecord(r.a, s);
        fine_recs.push_back(r);
        fine_counts[bucket]++;
        return (int)fine_recs.size() - 1;
    };
    auto fine_end = [&](int idx, cudaStream_t s) {
        if (!fine || idx < 0) return;
        cudaEventRecord(fine_recs[idx].b, s);
    };
    auto fine_report = [&]() {
        if (!fine) return;
        cudaDeviceSynchronize();
        std::map<const char*, float> sums;
        for (auto& r : fine_recs) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, r.a, r.b);
            sums[r.bucket] += ms;
            cudaEventDestroy(r.a); cudaEventDestroy(r.b);
        }
        std::fprintf(stderr, "  [fine] per-kernel-type GPU-busy time (summed over launches):\n");
        for (auto& kv : sums) {
            std::fprintf(stderr, "    %-26s %8.2f ms   (%d launches, %.4f ms/launch)\n",
                         kv.first, kv.second, fine_counts[kv.first],
                         kv.second / std::max(1, fine_counts[kv.first]));
        }
        fine_recs.clear(); fine_counts.clear();
    };

    // ===== Step 1: Mat-vec on GPU =====
    if (!skip_matvec) {
        upload_query_b(ctx, slot, qry);
        // Phase 14: DB is row-major u16; the dual-limb matvec reads it directly.
        if (pp.db_rows >= 32768) {
            inspire::gpu::gpu_matvec_dual(
                slot.d_packed_b, slot.d_packed_b,
                ctx->d_db_rm, slot.d_query_b_q0, slot.d_query_b_q1,
                pp.db_rows, pp.db_cols, Q0, Q1,
                ctx->d_mv_tall_partials0, ctx->d_mv_tall_partials1);
        } else {
            inspire::gpu::gpu_matvec_tensor_batched(
                ctx->d_mv_r0_ptrs, ctx->d_mv_r1_ptrs,
                ctx->d_db_rm, ctx->d_db_byte_sums,
                ctx->d_mv_q0_ptrs, ctx->d_mv_q1_ptrs,
                ctx->d_mv_query_planes, ctx->d_mv_query_sums, ctx->d_mv_gemm_out,
                pp.db_rows, pp.db_cols, Q0, Q1, 1, true, ctx->cublas);
        }
        mark("mat-vec");
    }

    // ===== Step 2: Pack each group =====
    // For each group g:
    //   - b values are already in d_packed_b[g] in coefficient form
    //   - NTT them in place (per limb)
    //   - Run InspiRINGOnline (lazy collapse) to produce packed[g] (NTT form)
    //
    // First, upload ksk5 and ksk_neg1 b-parts, then GPU NTT. The client sends
    // them in coefficient form (ksk_gen_b), so the NTT happens here.

    if (!skip_pack) {
        upload_ksks(ctx, slot, qry);
        mark("upload ksk5/ksk_neg1");

        // Independent of mat-vec/collapse: stage RGSW before the per-group
        // streams start so its H2D copies never wait behind collapse.
        upload_rgsw_part(slot.d_rgsw_top_a, qry.rgsw.top.a_parts, slot.h_rgsw_top_a);
        upload_rgsw_part(slot.d_rgsw_top_b, qry.rgsw.top.b_parts, slot.h_rgsw_top_b);
        upload_rgsw_part(slot.d_rgsw_bot_a, qry.rgsw.bottom.a_parts, slot.h_rgsw_bot_a);
        upload_rgsw_part(slot.d_rgsw_bot_b, qry.rgsw.bottom.b_parts, slot.h_rgsw_bot_b);
        mark("upload RGSW");
    }

    // For each group g, run the lazy collapse pipeline:
    //   1. b_q0 = NTT(d_packed_b[g].q0)
    //   2. b_q1 = NTT(d_packed_b[g].q1)
    //   3. Forward CollapseHalf (uses D_plus, ksk5, perm_fwd) — modifies b
    //   4. Conjugate CollapseHalf (uses D_minus, ksk5, perm_conj) — modifies b
    //   5. Final CollapseOne (uses D_final, ksk_neg1) — modifies b
    //   6. packed[g] = (a, b)
    //
    // The lazy_collapse kernel works on a single limb at a time.

    if (!skip_pack) {
        size_t per_limb_dp = n_steps * D_EFF * N;     // D_plus/D_minus per-limb stride
        size_t per_limb_kf = D_EFF * N;                // ksk5/D_final per-limb stride

        // Per-group dispatch on a pool of streams. Each group's work goes on a
        // single stream so the fwd→conj dependency is preserved within the
        // stream (CUDA streams serialize ops). Different groups run concurrently.
        size_t n_streams = ctx->streams.size();
        for (size_t g = 0; g < pp.n_packed; g++) {
            cudaStream_t s = ctx->streams[g % n_streams];
            uint32_t* d_b_q0 = slot.d_packed_b + g * 2 * N;
            uint32_t* d_b_q1 = slot.d_packed_b + g * 2 * N + N;
            // Partials scratch for this stream (serial within the group's stream).
            uint32_t* d_part = ctx->d_lazy_partials
                + (g % n_streams) * inspire::gpu::LAZY_COLLAPSE_STEP_CHUNKS * N;

            int fp = fine_begin("prep memcpy+NTT", s);
            inspire::gpu::gpu_ntt_forward_stream(d_b_q0, N, Q0, ctx->d_fwd_q0, s);
            inspire::gpu::gpu_ntt_forward_stream(d_b_q1, N, Q1, ctx->d_fwd_q1, s);
            fine_end(fp, s);

            int ff = fine_begin("lazy_collapse fwd", s);
            inspire::gpu::gpu_lazy_collapse_stream(d_b_q0,
                ctx->d_precomp_D_plus[g],
                slot.d_ksk5_b,
                ctx->d_perm_fwd, d_part,
                n_steps, D_EFF, N, Q0, s);
            inspire::gpu::gpu_lazy_collapse_stream(d_b_q1,
                ctx->d_precomp_D_plus[g] + per_limb_dp,
                slot.d_ksk5_b + per_limb_kf,
                ctx->d_perm_fwd, d_part,
                n_steps, D_EFF, N, Q1, s);
            fine_end(ff, s);

            int fc = fine_begin("lazy_collapse conj", s);
            inspire::gpu::gpu_lazy_collapse_stream(d_b_q0,
                ctx->d_precomp_D_minus[g],
                slot.d_ksk5_b,
                ctx->d_perm_conj, d_part,
                n_steps, D_EFF, N, Q0, s);
            inspire::gpu::gpu_lazy_collapse_stream(d_b_q1,
                ctx->d_precomp_D_minus[g] + per_limb_dp,
                slot.d_ksk5_b + per_limb_kf,
                ctx->d_perm_conj, d_part,
                n_steps, D_EFF, N, Q1, s);
            fine_end(fc, s);

            int fi = fine_begin("fused_ip_sub+copy_a", s);
            inspire::gpu::gpu_fused_ip_sub_stream(d_b_q0,
                ctx->d_precomp_D_final[g],
                slot.d_kskneg1_b,
                D_EFF, N, Q0, s);
            inspire::gpu::gpu_fused_ip_sub_stream(d_b_q1,
                ctx->d_precomp_D_final[g] + per_limb_kf,
                slot.d_kskneg1_b + per_limb_kf,
                D_EFF, N, Q1, s);

            fine_end(fi, s);
        }
        // Join stream work on the default stream without blocking the host.
        for (size_t i = 0; i < ctx->streams.size(); i++) {
            CUDA_CHECK(cudaEventRecord(ctx->stream_done_events[i], ctx->streams[i]));
            CUDA_CHECK(cudaStreamWaitEvent(0, ctx->stream_done_events[i], 0));
        }
        mark("collapse all groups");
    }

    // ===== Step 3: Horner evaluation =====
    size_t D_actual = std::min((size_t)pp.D, pp.n_packed);
    size_t c = pp.num_cts;

    std::vector<RlweCt> result(c);

    for (size_t g = 0; g < c; g++) {
        size_t start = g * D_actual;
        size_t end = std::min(start + D_actual, pp.n_packed);
        size_t slice_len = end - start;

        if (slice_len == 1) {
            // Trivial case: just copy packed[start] to result.
            // Copy the packed accumulator out (stays in NTT form).
            uint32_t* d_pa = ctx->d_packed_a + start * 2 * N;
            uint32_t* d_pb = slot.d_packed_b + start * 2 * N;
            CUDA_CHECK(cudaMemcpyAsync(slot.d_acc_a, d_pa, 2 * N * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice, 0));
            CUDA_CHECK(cudaMemcpyAsync(slot.d_acc_b, d_pb, 2 * N * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice, 0));
        } else {
            // Run Horner over packed[start..end-1].
            int fh = fine_begin("horner_eval (ext_prod chain)", 0);
            inspire::gpu::gpu_horner_eval(
                slot.d_acc_a, slot.d_acc_b,
                ctx->d_packed_a + start * 2 * N,
                slot.d_packed_b + start * 2 * N,
                (int)slice_len,
                slot.d_rgsw_top_a, slot.d_rgsw_top_b,
                slot.d_rgsw_bot_a, slot.d_rgsw_bot_b,
                ctx->d_fwd_q0, ctx->d_inv_q0, ctx->inv_n_q0,
                ctx->d_fwd_q1, ctx->d_inv_q1, ctx->inv_n_q1,
                slot.d_scratch_coeff, slot.d_dig,
                slot.d_res_a, slot.d_res_b,
                D_GSW, D_EFF, BASE_LOG, N);
            fine_end(fh, 0);
        }

        // Download (in NTT form; client converts as needed).
        result[g] = download_ct(slot);
    }
    mark("horner + download");

    fine_report();
    return result;
}

// B3: pack all groups for `count` slots with the batched collapse kernels —
// each group's precomp tensor is streamed once and applied to every query.
// Preconditions: every slot's d_packed_b holds its matvec output and its
// keys are uploaded (upload_ksks). Postcondition: slot.d_packed_{a,b} filled.
static void batched_pack(GpuServerCtx* ctx, size_t count) {
    const PublicParams& pp = ctx->pp;
    const size_t n_steps = N / 2 - 1;
    const size_t per_limb_dp = n_steps * D_EFF * N;
    const size_t per_limb_kf = D_EFF * N;
    const size_t n_streams = ctx->streams.size();
    auto* pb_bases = (uint32_t* const*)ctx->d_slot_packedb_ptrs;

    for (size_t g = 0; g < pp.n_packed; g++) {
        cudaStream_t s = ctx->streams[g % n_streams];
        uint32_t* const* part_ptrs =
            ctx->d_batch_part_ptrs + (g % n_streams) * ctx->cfg.max_batch;
        const size_t off_q0 = g * 2 * N;
        const size_t off_q1 = off_q0 + N;

        // Per-query b-poly prep (copy + NTT) on this group's stream.
        for (size_t b = 0; b < count; b++) {
            QuerySlot& sl = ctx->slots[b];
            uint32_t* d_b_q0 = sl.d_packed_b + off_q0;
            uint32_t* d_b_q1 = sl.d_packed_b + off_q1;
            inspire::gpu::gpu_ntt_forward_stream(d_b_q0, N, Q0, ctx->d_fwd_q0, s);
            inspire::gpu::gpu_ntt_forward_stream(d_b_q1, N, Q1, ctx->d_fwd_q1, s);
        }

        // Forward CollapseHalf, both limbs, tensor streamed once per batch.
        inspire::gpu::gpu_lazy_collapse_batched_stream(
            pb_bases, off_q0, ctx->d_precomp_D_plus[g],
            ctx->d_slot_ksk5_ptrs, 0,
            ctx->d_perm_fwd, part_ptrs, n_steps, D_EFF, N, Q0, (int)count, s);
        inspire::gpu::gpu_lazy_collapse_batched_stream(
            pb_bases, off_q1, ctx->d_precomp_D_plus[g] + per_limb_dp,
            ctx->d_slot_ksk5_ptrs, per_limb_kf,
            ctx->d_perm_fwd, part_ptrs, n_steps, D_EFF, N, Q1, (int)count, s);

        // Conjugate CollapseHalf.
        inspire::gpu::gpu_lazy_collapse_batched_stream(
            pb_bases, off_q0, ctx->d_precomp_D_minus[g],
            ctx->d_slot_ksk5_ptrs, 0,
            ctx->d_perm_conj, part_ptrs, n_steps, D_EFF, N, Q0, (int)count, s);
        inspire::gpu::gpu_lazy_collapse_batched_stream(
            pb_bases, off_q1, ctx->d_precomp_D_minus[g] + per_limb_dp,
            ctx->d_slot_ksk5_ptrs, per_limb_kf,
            ctx->d_perm_conj, part_ptrs, n_steps, D_EFF, N, Q1, (int)count, s);

        // Final CollapseOne.
        inspire::gpu::gpu_fused_ip_sub_batched_stream(
            pb_bases, off_q0, ctx->d_precomp_D_final[g],
            ctx->d_slot_kskneg1_ptrs, 0, D_EFF, N, Q0, (int)count, s);
        inspire::gpu::gpu_fused_ip_sub_batched_stream(
            pb_bases, off_q1, ctx->d_precomp_D_final[g] + per_limb_kf,
            ctx->d_slot_kskneg1_ptrs, per_limb_kf, D_EFF, N, Q1, (int)count, s);

    }
    for (size_t i = 0; i < ctx->streams.size(); i++) {
        CUDA_CHECK(cudaEventRecord(ctx->stream_done_events[i], ctx->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(0, ctx->stream_done_events[i], 0));
    }
}

std::vector<RlweCt> gpu_answer(GpuServerCtx* ctx, const QueryMessage& qry) {
    return answer_one(ctx, ctx->slots[0], qry);
}

std::vector<std::vector<RlweCt>>
gpu_answer_batch(GpuServerCtx* ctx, const QueryMessage* queries, size_t count) {
    // All-or-nothing validation before any kernel runs.
    if (count == 0) return {};
    if (count > ctx->cfg.max_batch) {
        fprintf(stderr,
                "gpu_answer_batch: count=%zu exceeds max_batch=%zu "
                "(set GpuServerConfig.max_batch at setup)\n",
                count, ctx->cfg.max_batch);
        std::exit(1);
    }
    for (size_t i = 0; i < count; i++) {
        if (queries[i].lwe.b_limb0.size() != ctx->pp.db_rows ||
            queries[i].lwe.b_limb1.size() != ctx->pp.db_rows) {
            fprintf(stderr, "gpu_answer_batch: query %zu has wrong length "
                            "(%zu, expected db_rows=%zu)\n",
                    i, queries[i].lwe.b_limb0.size(), ctx->pp.db_rows);
            std::exit(1);
        }
    }
    // Phase structure: upload every query's b vector, run ONE batched
    // mat-vec in which all queries share the DB stream (B2), one batched
    // collapse sharing the precomp-tensor stream (B3), and a lockstep
    // batched Horner (B4). The correctness contract — bit-identical to
    // count independent gpu_answer calls — is tested in
    // tests/test_gpu_batch.cu.
    std::vector<std::vector<RlweCt>> out(count);
    if (count == 1) {
        out[0] = answer_one(ctx, ctx->slots[0], queries[0]);
        return out;
    }

    // Stage timing, same env knob as answer_one. The sync is gated on the
    // knob so the untraced hot path is unchanged.
    auto trace = std::getenv("GPU_ANSWER_TRACE") != nullptr;
    auto phase_t0 = std::chrono::high_resolution_clock::now();
    auto mark = [&](const char* name) {
        if (!trace) return;
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double phase_ms = std::chrono::duration<double, std::milli>(t1 - phase_t0).count();
        std::fprintf(stderr, "  [gpu_batch B=%zu] %s: %.3f ms\n", count, name, phase_ms);
        phase_t0 = t1;
    };

    // Stage 1: one tensor-core GEMM for all query byte planes and both limbs.
    // The DB is streamed once for every geometry; there is no tall fallback.
    for (size_t i = 0; i < count; i++)
        upload_query_b(ctx, ctx->slots[i], queries[i]);
    if (count <= 2 && ctx->pp.db_cols >= 38400) {
        inspire::gpu::gpu_matvec_dual_batched(
            ctx->d_mv_r0_ptrs, ctx->d_mv_r1_ptrs, ctx->d_db_rm,
            ctx->d_mv_q0_ptrs, ctx->d_mv_q1_ptrs,
            ctx->pp.db_rows, ctx->pp.db_cols, Q0, Q1, (int)count);
    } else if (count <= 2 && ctx->pp.db_rows >= 32768) {
        for (size_t i = 0; i < count; i++)
            inspire::gpu::gpu_matvec_dual(
                ctx->slots[i].d_packed_b, ctx->slots[i].d_packed_b,
                ctx->d_db_rm,
                ctx->slots[i].d_query_b_q0, ctx->slots[i].d_query_b_q1,
                ctx->pp.db_rows, ctx->pp.db_cols, Q0, Q1,
                ctx->d_mv_tall_partials0, ctx->d_mv_tall_partials1);
    } else {
        inspire::gpu::gpu_matvec_tensor_batched(
            ctx->d_mv_r0_ptrs, ctx->d_mv_r1_ptrs,
            ctx->d_db_rm, ctx->d_db_byte_sums,
            ctx->d_mv_q0_ptrs, ctx->d_mv_q1_ptrs,
            ctx->d_mv_query_planes, ctx->d_mv_query_sums, ctx->d_mv_gemm_out,
            ctx->pp.db_rows, ctx->pp.db_cols, Q0, Q1, (int)count, true, ctx->cublas);
    }
    mark("mat-vec (incl. b upload)");

    // Stage 2: batched pack — every group's precomp tensor is streamed once
    // for the whole batch (geometry-independent win).
    for (size_t i = 0; i < count; i++)
        upload_ksks(ctx, ctx->slots[i], queries[i]);
    for (size_t i = 0; i < count; i++) {
        upload_rgsw_part(ctx->slots[i].d_rgsw_top_a, queries[i].rgsw.top.a_parts,
                         ctx->slots[i].h_rgsw_top_a);
        upload_rgsw_part(ctx->slots[i].d_rgsw_top_b, queries[i].rgsw.top.b_parts,
                         ctx->slots[i].h_rgsw_top_b);
        upload_rgsw_part(ctx->slots[i].d_rgsw_bot_a, queries[i].rgsw.bottom.a_parts,
                         ctx->slots[i].h_rgsw_bot_a);
        upload_rgsw_part(ctx->slots[i].d_rgsw_bot_b, queries[i].rgsw.bottom.b_parts,
                         ctx->slots[i].h_rgsw_bot_b);
    }
    batched_pack(ctx, count);
    mark("pack/collapse (incl. ksk upload)");

    // Stage 3: lockstep batched Horner — all chains advance the same step
    // together, so each step is a fixed 8-launch sequence covering the whole
    // batch (launch count independent of B) — then download.
    {
        const PublicParams& pp = ctx->pp;
        size_t D_actual = std::min((size_t)pp.D, pp.n_packed);
        size_t c = pp.num_cts;

        for (size_t i = 0; i < count; i++) {
            out[i].resize(c);
        }

        for (size_t g = 0; g < c; g++) {
            size_t start = g * D_actual;
            size_t end = std::min(start + D_actual, pp.n_packed);
            size_t L = end - start;

            // Init: acc[b] = packed[start + L - 1].
            for (size_t b = 0; b < count; b++) {
                QuerySlot& sl = ctx->slots[b];
                CUDA_CHECK(cudaMemcpyAsync(sl.d_acc_a, ctx->d_packed_a + (start + L - 1) * 2 * N,
                                           2 * N * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
                CUDA_CHECK(cudaMemcpyAsync(sl.d_acc_b, sl.d_packed_b + (start + L - 1) * 2 * N,
                                           2 * N * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
            }
            for (int st = (int)L - 2; st >= 0; st--) {
                inspire::gpu::gpu_horner_step_batched(
                    ctx->d_slot_acc_a_ptrs, ctx->d_slot_acc_b_ptrs,
                    ctx->d_slot_packeda_ptrs,
                    (const uint32_t* const*)ctx->d_slot_packedb_ptrs,
                    (start + (size_t)st) * 2 * N,
                    ctx->d_slot_rgsw_ta_ptrs, ctx->d_slot_rgsw_tb_ptrs,
                    ctx->d_slot_rgsw_ba_ptrs, ctx->d_slot_rgsw_bb_ptrs,
                    ctx->d_fwd_q0, ctx->d_inv_q0, ctx->inv_n_q0,
                    ctx->d_fwd_q1, ctx->d_inv_q1, ctx->inv_n_q1,
                    ctx->d_h_coeff_q0, ctx->d_h_coeff_q1,
                    ctx->d_h_dig_q0, ctx->d_h_dig_q1,
                    D_GSW, D_EFF, BASE_LOG, N, (int)count);
            }
            for (size_t b = 0; b < count; b++)
                out[b][g] = download_ct(ctx->slots[b]);
        }
    }
    mark("horner (incl. rgsw upload + download)");
    return out;
}

GpuServerCaps gpu_server_caps(const GpuServerCtx* ctx) {
    GpuServerCaps caps;
    caps.max_batch = ctx->cfg.max_batch;
    caps.resident_bytes = ctx->resident_bytes;
    caps.num_cts = ctx->pp.num_cts;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    caps.device_free_bytes = free_b;
    return caps;
}

void gpu_free_server(GpuServerCtx* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_fwd_q0); cudaFree(ctx->d_inv_q0);
    cudaFree(ctx->d_fwd_q1); cudaFree(ctx->d_inv_q1);
    cublasDestroy(ctx->cublas);
    cudaFree(ctx->d_db_rm);
    cudaFree(ctx->d_db_byte_sums);
    cudaFree(ctx->d_mv_query_planes); cudaFree(ctx->d_mv_query_sums);
    cudaFree(ctx->d_mv_gemm_out);
    cudaFree(ctx->d_mv_tall_partials0); cudaFree(ctx->d_mv_tall_partials1);
    cudaFree(ctx->d_perm_fwd); cudaFree(ctx->d_perm_conj);
    for (auto p : ctx->d_precomp_a) cudaFree(p);
    for (auto p : ctx->d_precomp_D_plus) cudaFree(p);
    for (auto p : ctx->d_precomp_D_minus) cudaFree(p);
    for (auto p : ctx->d_precomp_D_final) cudaFree(p);
    cudaFree(ctx->d_packed_a);
    for (auto& sl : ctx->slots) {
        cudaFree(sl.d_query_b_q0); cudaFree(sl.d_query_b_q1);
        cudaFree(sl.d_ksk5_b); cudaFree(sl.d_kskneg1_b);
        cudaFree(sl.d_rgsw_top_a); cudaFree(sl.d_rgsw_top_b);
        cudaFree(sl.d_rgsw_bot_a); cudaFree(sl.d_rgsw_bot_b);
        cudaFree(sl.d_packed_b);
        cudaFree(sl.d_acc_a); cudaFree(sl.d_acc_b);
        cudaFree(sl.d_res_a); cudaFree(sl.d_res_b);
        cudaFree(sl.d_scratch_coeff);
        cudaFree(sl.d_dig);
        cudaFreeHost(sl.h_query_b_q0); cudaFreeHost(sl.h_query_b_q1);
        cudaFreeHost(sl.h_ksk5_b); cudaFreeHost(sl.h_kskneg1_b);
        cudaFreeHost(sl.h_rgsw_top_a); cudaFreeHost(sl.h_rgsw_top_b);
        cudaFreeHost(sl.h_rgsw_bot_a); cudaFreeHost(sl.h_rgsw_bot_b);
    }
    cudaFree(ctx->d_lazy_partials);
    cudaFree((void*)ctx->d_mv_q0_ptrs); cudaFree((void*)ctx->d_mv_q1_ptrs);
    cudaFree(ctx->d_mv_r0_ptrs); cudaFree(ctx->d_mv_r1_ptrs);
    cudaFree(ctx->d_slot_packedb_ptrs);
    cudaFree((void*)ctx->d_slot_ksk5_ptrs); cudaFree((void*)ctx->d_slot_kskneg1_ptrs);
    cudaFree(ctx->d_batch_partials); cudaFree(ctx->d_batch_part_ptrs);
    cudaFree(ctx->d_slot_acc_a_ptrs); cudaFree(ctx->d_slot_acc_b_ptrs);
    cudaFree((void*)ctx->d_slot_packeda_ptrs);
    cudaFree((void*)ctx->d_slot_rgsw_ta_ptrs); cudaFree((void*)ctx->d_slot_rgsw_tb_ptrs);
    cudaFree((void*)ctx->d_slot_rgsw_ba_ptrs); cudaFree((void*)ctx->d_slot_rgsw_bb_ptrs);
    cudaFree(ctx->d_h_coeff_q0); cudaFree(ctx->d_h_coeff_q1);
    cudaFree(ctx->d_h_dig_q0); cudaFree(ctx->d_h_dig_q1);
    for (size_t i = 0; i < ctx->streams.size(); i++) {
        cudaEventDestroy(ctx->stream_done_events[i]);
        cudaStreamDestroy(ctx->streams[i]);
    }
    delete ctx;
}

} // namespace inspire
