// gpu_answer_batch correctness contract:
//   (1) bit-identical to `count` independent gpu_answer calls, in order;
//   (2) slot-exact against the generated DB via extract();
//   (3) caps report is consistent with the setup config;
//   (4) a second batch on the same ctx reproduces the same answers
//       (slot pools are cleanly reusable).
// Medium geometry (n_packed=4, Horner D=4) so every online stage runs.
#include "protocol.h"
#include "gpu_protocol.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

using namespace inspire;

static bool ct_equal(const RlweCt& x, const RlweCt& y) {
    for (int l = 0; l < 2; l++)
        for (size_t i = 0; i < N; i++)
            if (x.a.limbs[l][i] != y.a.limbs[l][i] ||
                x.b.limbs[l][i] != y.b.limbs[l][i]) return false;
    return true;
}

int main() {
    std::cout << "=== gpu_answer_batch test (Medium: n_packed=4, Horner) ===" << std::endl;
    const size_t N_entries = 524288, entry_bytes = 120, db_rows = 2048;
    PublicParams pp = setup(N_entries, entry_bytes, db_rows);
    const size_t coeffs_per_entry = (pp.w * 8 + P_BITS - 1) / P_BITS;

    std::mt19937_64 rng(42);
    std::vector<uint16_t> db((size_t)pp.db_rows * pp.db_cols);
    for (auto& s : db) s = (uint16_t)(rng() % P);

    auto data = gpu_preprocess(pp, db.data());
    GpuServerConfig cfg;
    cfg.max_batch = 10;
    auto* ctx = gpu_setup_server(pp, data, cfg);

    // (3) caps consistency
    GpuServerCaps caps = gpu_server_caps(ctx);
    if (caps.max_batch != 10 || caps.num_cts != pp.num_cts || caps.resident_bytes == 0) {
        std::cout << "FAIL: caps inconsistent (max_batch=" << caps.max_batch
                  << " num_cts=" << caps.num_cts
                  << " resident=" << caps.resident_bytes << ")" << std::endl;
        return 1;
    }
    std::cout << "  caps: max_batch=" << caps.max_batch
              << " resident=" << caps.resident_bytes / 1e6 << " MB"
              << " device_free=" << caps.device_free_bytes / 1e9 << " GB" << std::endl;

    // 10 queries: 5 distinct indices, each queried twice with independent key
    // material (fresh secret per query — the same-index pair checks that two
    // clients asking for the same record still get bit-independent pipelines).
    // count=10 also exercises the batched-matvec chunk loop (8 + 2).
    const std::vector<size_t> indices = {0, 42, 1000, 100000, 524287,
                                         0, 42, 1000, 100000, 524287};
    std::vector<QueryState> states;
    std::vector<QueryMessage> queries;
    for (size_t idx : indices) {
        auto [qst, qry] = query(pp, idx);
        states.push_back(qst);
        queries.push_back(qry);
    }

    // Reference: independent single-query answers.
    std::vector<std::vector<RlweCt>> ref;
    for (const auto& q : queries) ref.push_back(gpu_answer(ctx, q));

    int fail = 0;

    // (1) batch == singles, bit-exact
    auto batch = gpu_answer_batch(ctx, queries);
    if (batch.size() != queries.size()) { std::cout << "FAIL: batch size" << std::endl; return 1; }
    for (size_t i = 0; i < batch.size(); i++) {
        bool ok = batch[i].size() == ref[i].size();
        for (size_t g = 0; ok && g < batch[i].size(); g++)
            ok = ct_equal(batch[i][g], ref[i][g]);
        if (!ok) { std::cout << "FAIL: batch[" << i << "] != single answer" << std::endl; fail++; }
    }

    // (2) slot-exact via extract
    for (size_t i = 0; i < batch.size(); i++) {
        auto slots = extract(pp, states[i], batch[i]);
        size_t i_star = indices[i] % pp.db_rows;
        size_t col = (indices[i] / pp.db_rows) * coeffs_per_entry;
        bool ok = slots.size() == coeffs_per_entry;
        for (size_t c = 0; ok && c < coeffs_per_entry; c++)
            ok = (slots[c] == (uint16_t)(db[i_star * pp.db_cols + col + c] % P));
        if (!ok) { std::cout << "FAIL: batch[" << i << "] extract mismatch" << std::endl; fail++; }
    }

    // (4) reuse: same batch again, bit-exact against the first run
    auto batch2 = gpu_answer_batch(ctx, queries);
    for (size_t i = 0; i < batch.size(); i++) {
        bool ok = true;
        for (size_t g = 0; ok && g < batch[i].size(); g++)
            ok = ct_equal(batch[i][g], batch2[i][g]);
        if (!ok) { std::cout << "FAIL: rerun batch[" << i << "] differs" << std::endl; fail++; }
    }

    // Partial batches (count < max_batch): 3 covers the BT4 tile, 2 covers BT2.
    auto part3 = gpu_answer_batch(ctx, queries.data(), 3);
    for (size_t i = 0; i < 3; i++) {
        bool ok = true;
        for (size_t g = 0; ok && g < part3[i].size(); g++)
            ok = ct_equal(part3[i][g], ref[i][g]);
        if (!ok) { std::cout << "FAIL: partial-3 batch[" << i << "]" << std::endl; fail++; }
    }
    auto part = gpu_answer_batch(ctx, queries.data(), 2);
    for (size_t i = 0; i < 2; i++) {
        bool ok = true;
        for (size_t g = 0; ok && g < part[i].size(); g++)
            ok = ct_equal(part[i][g], ref[i][g]);
        if (!ok) { std::cout << "FAIL: partial batch[" << i << "]" << std::endl; fail++; }
    }

    gpu_free_server(ctx);
    if (fail == 0) { std::cout << "All gpu_answer_batch checks passed." << std::endl; return 0; }
    return 1;
}
