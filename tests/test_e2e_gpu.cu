// End-to-end GPU answer pipeline test: query -> gpu_answer -> extract,
// checked slot-by-slot against the generated database values.
#include <cstdlib>
#include "protocol.h"
#include "gpu_protocol.h"
#include <iostream>
#include <random>
#include <cstring>
#include <string>
#include <chrono>

using namespace inspire;
using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static int run_test(size_t N_entries, size_t entry_bytes,
                    const std::vector<size_t>& test_indices,
                    const char* label, size_t force_db_rows = 0) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "DB: " << N_entries << " entries × " << entry_bytes << " B"
              << " = " << (N_entries * entry_bytes) / 1024 << " KiB" << std::endl;

    std::mt19937_64 rng(42);

    // Geometry selection (precedence: explicit db_rows env > per-test pin >
    // default). The default db_rows lives in setup() (32768).
    //   INSPIRE_DB_ROWS=<n>   force an exact db_rows
    //   force_db_rows arg     per-test pin (keeps unit-test coverage)
    const char* rows_env = std::getenv("INSPIRE_DB_ROWS");
    PublicParams pp;
    if (rows_env)        pp = setup(N_entries, entry_bytes, (size_t)std::strtoull(rows_env, nullptr, 10));
    else if (force_db_rows) pp = setup(N_entries, entry_bytes, force_db_rows);
    else                 pp = setup(N_entries, entry_bytes);
    std::cout << "  pp: db_rows=" << pp.db_rows << " db_cols=" << pp.db_cols
              << " n_packed=" << pp.n_packed << " D=" << pp.D
              << " D_actual=" << std::min((size_t)pp.D, pp.n_packed)
              << " c=" << pp.num_cts << std::endl;

    // Communication cost (report alongside latency). Query upload grows with
    // db_rows; response download with num_cts. (Analytic, matches setup()'s model.)
    double query_kb = (pp.db_rows * (double)BETA + 6.0 * D_EFF * N * BETA) / 8.0 / 1024.0;
    double resp_kb  = (pp.num_cts * (double)N * (20 + 28)) / 8.0 / 1024.0;
    std::cout << "  comm: query " << query_kb << " KB up, response " << resp_kb
              << " KB down, total " << (query_kb + resp_kb) << " KB" << std::endl;

    // Hint / memory footprint (report alongside latency). The "hint" is the
    // geometry-dependent precomp resident on the GPU: per group D_plus + D_minus
    // (each (N/2-1)*D_EFF*2N u32) + D_final (D_EFF*2N) + a (2N). Scales with
    // n_packed — this is what OOMs the wide geometry at large DBs.
    size_t per_group_u32 = 2 * (N/2 - 1) * D_EFF * 2 * N + D_EFF * 2 * N + 2 * N;
    double hint_gb = (double)pp.n_packed * per_group_u32 * sizeof(uint32_t) / 1e9;
    double encdb_gb = (double)pp.db_rows * pp.db_cols * sizeof(uint16_t) / 1e9;
    std::cout << "  hint: " << hint_gb << " GB precomp + " << encdb_gb
              << " GB DB = " << (hint_gb + encdb_gb) << " GB resident" << std::endl;

    // Slot-native DB: db_rows*db_cols values in [0,P), ROW-MAJOR
    // (db[row*db_cols+col]). The GPU pipeline is the only path; correctness is
    // verified slot-exact against this generated DB.
    size_t coeffs_per_entry = (pp.w * 8 + P_BITS - 1) / P_BITS;
    std::vector<uint16_t> db((size_t)pp.db_rows * pp.db_cols);
    for (auto& s : db) s = (uint16_t)(rng() % P);

    std::cout << "  GPU preprocess..." << std::flush;
    auto t0 = Clock::now();
    auto data = gpu_preprocess(pp, db.data());
    auto t1 = Clock::now();
    std::cout << " " << ms(t0, t1) << " ms" << std::endl;

    std::cout << "  GPU setup_server..." << std::flush;
    t0 = Clock::now();
    auto* gctx = gpu_setup_server(pp, data);
    cudaDeviceSynchronize();
    t1 = Clock::now();
    std::cout << " " << ms(t0, t1) << " ms" << std::endl;

    int pass = 0, fail = 0;
    double gpu_total = 0;

    for (size_t target_idx : test_indices) {
        if (target_idx >= N_entries) continue;
        auto [qst, qry] = query(pp, target_idx);

        // GPU answer
        t0 = Clock::now();
        auto resp_gpu = gpu_answer(gctx, qry);
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double gpu_ms = ms(t0, t1);
        gpu_total += gpu_ms;

        bool correct;
        {
            // Slot-exact check: decrypted slot c must equal the generated DB
            // value at column (col+c), row i_star (row-major db[row*db_cols+col]).
            auto slots = extract(pp, qst, resp_gpu);
            size_t i_star = target_idx % pp.db_rows;
            size_t col = (target_idx / pp.db_rows) * coeffs_per_entry;
            correct = (slots.size() == coeffs_per_entry);
            for (size_t c = 0; correct && c < coeffs_per_entry; c++) {
                uint16_t expect = (uint16_t)(db[i_star * pp.db_cols + (col + c)] % P);
                if (slots[c] != expect) correct = false;
            }
            if (!correct && fail < 3) {
                std::cout << "  FAIL idx=" << target_idx
                          << " k*=" << qst.k_star << " ell*=" << qst.ell_star
                          << " (gpu_ms=" << gpu_ms << ")" << std::endl;
            }
        }
        if (correct) pass++; else fail++;
    }

    std::cout << "  Results: " << pass << " pass, " << fail << " fail" << std::endl;
    std::cout << "  Avg GPU answer time: " << gpu_total / test_indices.size() << " ms" << std::endl;

    gpu_free_server(gctx);
    return fail;
}

int main(int argc, char** argv) {
    std::cout << "=== Full GPU Answer Pipeline e2e Test ===" << std::endl;

    bool only_1gb = argc > 1 && std::string(argv[1]) == "--1gb";
    bool only_4gb = argc > 1 && std::string(argv[1]) == "--4gb";
    bool only_16gb = argc > 1 && std::string(argv[1]) == "--16gb";
    int total_fail = 0;

    if (only_16gb) {
        // 16 GB DB (134,217,728 × 120 B). Use a tall geometry (INSPIRE_DB_ROWS).
        total_fail += run_test(134217728, 120,
                               {0UL, 1UL, 42UL, 1000UL, 1000000UL, 134217727UL},
                               "~16 GB DB");
        if (total_fail == 0) std::cout << "\n16 GB test passed." << std::endl;
        return total_fail == 0 ? 0 : 1;
    }

    if (only_4gb) {
        // 4 GB DB (33,554,432 × 120 B) at the tall default (db_rows=32768,
        // n_packed=32); a wide geometry (db_rows=2048 → n_packed=512) would
        // need ~34 GB of precomp and not fit the card.
        total_fail += run_test(33554432, 120,
                               {0UL, 1UL, 42UL, 1000UL, 1000000UL, 33554431UL},
                               "~4 GB DB");
        if (total_fail == 0) std::cout << "\n4 GB test passed." << std::endl;
        return total_fail == 0 ? 0 : 1;
    }

    // w = 120 bytes = 960 bits = 64 slots × 15 bits per slot (clean packing).
    // Small/Medium pin db_rows to keep their intended coverage (n_packed=1 path
    // and the n_packed=4 Horner path) regardless of the 32768 project default.
    if (!only_1gb) {
        total_fail += run_test(256, 120,
                               {0UL, 42UL, 100UL, 200UL, 255UL},
                               "Small DB (n_packed=1, no Horner)",
                               /*force_db_rows=*/2048);

        total_fail += run_test(262144, 120,
                               {0UL, 42UL, 1000UL, 100000UL, 262143UL},
                               "Medium DB (n_packed=4, Horner D=4)",
                               /*force_db_rows=*/2048);
    }

    total_fail += run_test(8388608, 120,
                           {0UL, 1UL, 42UL, 1000UL, 100000UL, 1000000UL},
                           "~1 GB DB (default db_rows=32768)");

    if (total_fail == 0) {
        std::cout << "\nAll GPU e2e tests passed." << std::endl;
        return 0;
    } else {
        std::cout << "\n" << total_fail << " test cases failed." << std::endl;
        return 1;
    }
}
