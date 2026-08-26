#pragma once
#include "protocol.h"
#include "gpu.cuh"

namespace inspire {

// ============================================================
// GPU server context: holds long-lived data (DB, precomp, twiddles).
// Set up once after Preprocess; reused across many queries.
//
// Generations: one ctx encapsulates one database generation (DB + precomp).
// Rollover = build a new ctx (gpu_preprocess + gpu_setup_server) and route
// the next batch to it; batch boundaries are the natural commit points.
// Use gpu_server_caps().device_free_bytes to decide whether two generations
// fit on the device simultaneously.
// ============================================================
struct GpuServerCtx;

// Batch configuration, fixed at setup time. All per-query scratch is
// allocated for max_batch slots up front: nothing is allocated on the
// request path.
struct GpuServerConfig {
    size_t max_batch = 16;   // gpu_answer_batch accepts at most this many queries
};

// Capability report for the wrapper's scheduler (mechanism, not policy:
// the backend publishes limits; when to close a batch is the caller's call).
struct GpuServerCaps {
    size_t max_batch;          // slots allocated at setup
    size_t resident_bytes;     // encoded DB + precomp + slot pools + tables
    size_t device_free_bytes;  // current free device memory (second generation?)
    size_t num_cts;            // ciphertexts per response
};

// The server-side protocol, in lifecycle order: preprocess → setup_server →
// answer / answer_batch (per request) → free_server.

// GPU-accelerated preprocess. Uses GPU ring_embed internally (99% of offline
// cost). db holds the database as db_rows*db_cols plaintext
// values in [0,P), ROW-MAJOR (db[row*db_cols+col]). No raw byte DB is ever
// materialised on host or device — the encode is just an in-place inverse-DFT,
// so peak memory is the encoded DB, not encoded + raw; and the online matvec
// reads this row-major DB directly (no transpose, single resident copy).
PreprocessData gpu_preprocess(const PublicParams& pp, const uint16_t* db);

// Assemble the long-lived server context from preprocess's output: adopt the
// already-device-resident encoded DB and per-group precomp pointers (ownership
// transfer, no copy), upload the small twiddle/permutation tables, and allocate
// cfg.max_batch per-query scratch slots + streams. Light — the O(precomp size)
// work is in gpu_preprocess; this does not re-upload the DB or precomp.
GpuServerCtx* gpu_setup_server(const PublicParams& pp, const PreprocessData& precomp,
                               const GpuServerConfig& cfg);
// Back-compat overload: default GpuServerConfig.
GpuServerCtx* gpu_setup_server(const PublicParams& pp, const PreprocessData& precomp);

GpuServerCaps gpu_server_caps(const GpuServerCtx* ctx);

// Run the full server-side answer() entirely on GPU for one query.
// Input: query message (CPU-side, in NTT form).
// Output: response (CPU-side, in NTT form, ready for serialization).
std::vector<RlweCt> gpu_answer(GpuServerCtx* ctx, const QueryMessage& qry);

// Batched answer: count <= max_batch queries, each fully self-contained
// (its own keys; key material is never shared across queries — fixed CRS
// plus a reused secret leaks indices, see docs/scheme_description §security).
// Returns one response (num_cts ciphertexts) per query, in input order.
// Blocking; do not call concurrently on the same ctx. All-or-nothing:
// validation errors throw before any kernel runs.
std::vector<std::vector<RlweCt>>
gpu_answer_batch(GpuServerCtx* ctx, const QueryMessage* queries, size_t count);

inline std::vector<std::vector<RlweCt>>
gpu_answer_batch(GpuServerCtx* ctx, const std::vector<QueryMessage>& queries) {
    return gpu_answer_batch(ctx, queries.data(), queries.size());
}

// Free all device memory.
void gpu_free_server(GpuServerCtx* ctx);

} // namespace inspire
