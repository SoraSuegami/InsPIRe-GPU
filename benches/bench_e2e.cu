// The benchmark: everything measurable about the pipeline in one binary,
// per database size (1 GB / 4 GB / 16 GB at the default geometry):
//
//   - single-query online latency (median) + communication + resident memory
//   - batch-throughput sweep (gpu_answer_batch)
//   - response-noise margin, measured by decrypting real responses
//   - wire-path CPU timings (query build / pack / unpack / compress /
//     extract), printed once per run — they depend on db_rows, not on the
//     database size
//
//   ./bench_e2e            # 1 GB, 4 GB, 16 GB
//   ./bench_e2e 1 4        # subset (GB sizes)
//
// Env knobs: INSPIRE_DB_ROWS overrides the default geometry;
// INSPIRE_MAX_BATCH sizes the batch slot pool.

#include "capi.h"
#include "protocol.h"
#include "gpu_protocol.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

using namespace inspire;
using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Geometry: explicit INSPIRE_DB_ROWS override, else the setup() default (32768).
static PublicParams pick_pp(size_t N_entries, size_t w) {
    if (const char* r = std::getenv("INSPIRE_DB_ROWS")) {
        size_t rows = (size_t)std::strtoull(r, nullptr, 10);
        if (rows == 0 || (rows & (rows - 1)) != 0) {
            printf("INSPIRE_DB_ROWS=%zu invalid: db_rows must be a power of two.\n", rows);
            exit(1);
        }
        return setup(N_entries, w, rows);
    }
    return setup(N_entries, w);
}

static void bench_size(const char* label, size_t N_entries, size_t w,
                       int warmup, int trials) {
    printf("\n=== %s (%zu entries x %zu B) ===\n", label, N_entries, w);
    PublicParams pp = pick_pp(N_entries, w);

    // Three axes: geometry, communication, hint.
    size_t cpe = (w * 8 + P_BITS - 1) / P_BITS;
    double query_kb = (pp.db_rows * (double)BETA + 6.0 * D_EFF * N * BETA) / 8.0 / 1024.0;
    double resp_kb  = (pp.num_cts * (double)N * (20 + 28)) / 8.0 / 1024.0;
    size_t per_group_u32 = 2 * (N/2 - 1) * D_EFF * 2 * N + D_EFF * 2 * N + 2 * N;
    double hint_gb  = (double)pp.n_packed * per_group_u32 * sizeof(uint32_t) / 1e9;
    double encdb_gb = (double)pp.db_rows * pp.db_cols * sizeof(uint16_t) / 1e9;
    printf("  geom : db_rows=%zu db_cols=%zu n_packed=%zu D_actual=%zu c=%zu\n",
           pp.db_rows, pp.db_cols, pp.n_packed,
           std::min((size_t)pp.D, pp.n_packed), pp.num_cts);
    printf("  comm : %.1f KB up + %.1f KB down = %.1f KB total\n",
           query_kb, resp_kb, query_kb + resp_kb);
    printf("  hint : %.2f GB precomp + %.2f GB DB = %.2f GB resident\n",
           hint_gb, encdb_gb, hint_gb + encdb_gb);

    // Slot-native DB: db_rows*db_cols values in [0,P), row-major. Fast
    // deterministic fill (an LCG) so 16 GB generation isn't the bottleneck.
    std::vector<uint16_t> db((size_t)pp.db_rows * pp.db_cols);
    uint64_t st = 0x9e3779b97f4a7c15ull;
    for (auto& s : db) { st = st * 6364136223846793005ull + 1; s = (uint16_t)((st >> 33) % P); }

    auto t0 = Clock::now();
    auto data = gpu_preprocess(pp, db.data());
    GpuServerConfig scfg;
    scfg.max_batch = 32;    // INSPIRE_MAX_BATCH overrides the slot-pool size
    if (const char* mb = std::getenv("INSPIRE_MAX_BATCH"))
        scfg.max_batch = (size_t)std::strtoull(mb, nullptr, 10);
    auto* ctx = gpu_setup_server(pp, data, scfg);
    cudaDeviceSynchronize();
    printf("  setup: %.1f s (preprocess + server)\n", ms(t0, Clock::now()) / 1000.0);

    // Correctness sanity on a few indices.
    int ok = 0, bad = 0;
    for (size_t idx : {0UL, 1UL, (unsigned long)(N_entries / 2), N_entries - 1}) {
        auto [qst, qry] = query(pp, idx);
        auto resp = gpu_answer(ctx, qry);
        auto slots = extract(pp, qst, resp);
        size_t i_star = idx % pp.db_rows, col = (idx / pp.db_rows) * cpe;
        bool c = slots.size() == cpe;
        for (size_t k = 0; c && k < cpe; k++)   // db row-major: db[row*db_cols+col]
            if (slots[k] != (uint16_t)(db[i_star * pp.db_cols + (col + k)] % P)) c = false;
        c ? ok++ : bad++;
    }
    printf("  check: %d ok, %d bad\n", ok, bad);

    // Warm up, then time. Report the median run (robust to outliers).
    auto [qst, qry] = query(pp, 0);
    for (int i = 0; i < warmup; i++) { auto r = gpu_answer(ctx, qry); }
    cudaDeviceSynchronize();

    std::vector<double> times;
    times.reserve(trials);
    for (int i = 0; i < trials; i++) {
        auto s = Clock::now();
        auto r = gpu_answer(ctx, qry);
        cudaDeviceSynchronize();
        times.push_back(ms(s, Clock::now()));
    }
    std::sort(times.begin(), times.end());
    printf("  >>> latency: %.2f ms median  (%d trials, %d warmup)\n",
           times[times.size() / 2], trials, warmup);

    // Batch throughput sweep: B queries share one DB stream in
    // the batched matvec; packing + Horner still run per query. Each query
    // carries its own fresh key material, as the protocol requires.
    {
        GpuServerCaps caps = gpu_server_caps(ctx);
        std::vector<QueryMessage> qs;
        qs.reserve(caps.max_batch);
        for (size_t b = 0; b < caps.max_batch; b++) {
            auto [s2, q2] = query(pp, (b * 977) % N_entries);
            (void)s2;
            qs.push_back(q2);
        }
        printf("  batch sweep (max_batch=%zu, median of 3):\n", caps.max_batch);
        for (size_t B : {(size_t)1, (size_t)2, (size_t)4, (size_t)8, (size_t)16, (size_t)32}) {
            if (B > caps.max_batch) break;
            { auto r = gpu_answer_batch(ctx, qs.data(), B); }   // warmup
            cudaDeviceSynchronize();
            std::vector<double> ts;
            for (int t = 0; t < 3; t++) {
                auto s = Clock::now();
                auto r = gpu_answer_batch(ctx, qs.data(), B);
                cudaDeviceSynchronize();
                ts.push_back(ms(s, Clock::now()));
            }
            std::sort(ts.begin(), ts.end());
            double bms = ts[ts.size() / 2];
            printf("    B=%2zu: %8.2f ms/batch  %6.2f ms/query  %7.1f q/s\n",
                   B, bms, bms / B, 1000.0 * B / bms);
        }
    }

    // --- Wire-path CPU timings: once per run (they scale with db_rows,
    // not with the database size). ---
    static bool wire_done = false;
    if (!wire_done) {
        wire_done = true;
        ipir_params prm;
        prm.n_entries = pp.N_entries; prm.entry_bytes = pp.w;
        prm.db_rows = pp.db_rows;     prm.db_cols = pp.db_cols;
        prm.n_packed = pp.n_packed;   prm.interp_d = pp.D;
        prm.num_cts = pp.num_cts;     prm.ring_n = N;
        prm.d_eff = (size_t)D_EFF;
        std::memcpy(prm.seed, pp.seed.data(), SEED_BYTES);

        std::vector<uint64_t> flat(ipir_query_u64s(&prm)), flat2(flat.size());
        std::vector<uint8_t> wire_q(ipir_query_packed_bytes(&prm));
        ipir_client_query* cq0 = ipir_query_build(&prm, 1, flat.data());

        const int R = 20;
        double t_build = 0, t_pack = 0, t_unpack = 0, t_comp = 0, t_extc = 0, t_ext = 0;
        auto [wqst, wqry] = query(pp, 2);
        auto wresp = gpu_answer(ctx, wqry);
        for (int it = 0; it < R; it++) {
            auto s = Clock::now();
            { auto qb = query(pp, (size_t)(3 + it)); (void)qb; }
            t_build += ms(s, Clock::now());

            s = Clock::now();
            ipir_query_pack(&prm, flat.data(), wire_q.data());
            t_pack += ms(s, Clock::now());

            s = Clock::now();
            ipir_query_unpack(&prm, wire_q.data(), flat2.data());
            t_unpack += ms(s, Clock::now());

            s = Clock::now();
            auto cresp = compress_response(wresp);
            t_comp += ms(s, Clock::now());

            s = Clock::now();
            auto sl1 = extract_compressed(pp, wqst, cresp);
            t_extc += ms(s, Clock::now());

            s = Clock::now();
            auto sl2 = extract(pp, wqst, wresp);
            t_ext += ms(s, Clock::now());
        }
        if (cq0) ipir_client_query_destroy(cq0);
        printf("  wire-path CPU (per query, db_rows=%zu):\n", pp.db_rows);
        printf("    client query build        : %6.2f ms\n", t_build / R);
        printf("    client query pack (%zu B): %6.2f ms\n",
               ipir_query_packed_bytes(&prm), t_pack / R);
        printf("    server query unpack       : %6.2f ms\n", t_unpack / R);
        printf("    server response compress  : %6.2f ms\n", t_comp / R);
        printf("    client extract_compressed : %6.2f ms (plain extract: %.2f ms)\n",
               t_extc / R, t_ext / R);
    }

    // --- Response-noise margin: decrypt real responses and read the noise
    // off the phase (v = b - a*s); e is the centered residue of the phase
    // mod Delta, exact while |e| < Delta/2. ---
    {
        auto modinv = [](uint64_t a, uint64_t m) {
            int64_t old_r = (int64_t)a, r = (int64_t)m, old_s = 1, sx = 0;
            while (r != 0) {
                int64_t qq = old_r / r, t = r;
                r = old_r - qq * r; old_r = t;
                t = sx; sx = old_s - qq * sx; old_s = t;
            }
            return (uint64_t)((old_s % (int64_t)m + (int64_t)m) % (int64_t)m);
        };
        const uint64_t q1_inv_q0 = modinv(Q1 % Q0, Q0);
        const __int128 Q = (__int128)Q0 * Q1;
        const uint64_t Delta = (uint64_t)(Q / P);

        long double sum_sq = 0.0L;
        uint64_t max_e = 0;
        size_t n_samples = 0;
        std::mt19937_64 nrng(7);
        const int n_queries = 6;
        for (int t = 0; t < n_queries; t++) {
            size_t idx = (size_t)(nrng() % N_entries);
            auto [nqst, nqry] = query(pp, idx);
            auto nresp = gpu_answer(ctx, nqry);
            for (auto& ct : nresp) {
                RnsPoly as = ct.a;
                RnsPoly s_copy = nqst.s_ntt;
                as.mul_inplace(s_copy);
                RnsPoly p = ct.b;
                if (!p.is_ntt) p.to_ntt();
                p.sub_inplace(as);
                p.to_coeff();
                for (size_t i = 0; i < N; i++) {
                    uint64_t v0 = p.limbs[0][i], v1 = p.limbs[1][i];
                    uint64_t diff = (v0 + Q0 - (v1 % Q0)) % Q0;
                    uint64_t h = (uint64_t)(((__int128)diff * q1_inv_q0) % Q0);
                    __int128 v = (__int128)v1 + (__int128)Q1 * h;
                    uint64_t rr = (uint64_t)((v + Delta / 2) % Delta);
                    int64_t e = (int64_t)rr - (int64_t)(Delta / 2);
                    uint64_t ae = (uint64_t)(e < 0 ? -e : e);
                    if (ae > max_e) max_e = ae;
                    sum_sq += (long double)e * (long double)e;
                    n_samples++;
                }
            }
        }
        double sigma = (double)sqrtl(sum_sq / (long double)n_samples);
        double half_delta = (double)(Delta / 2);
        double z = half_delta / (sigma * sqrt(2.0));
        double log2_p_coeff = (z > 26.0)
            ? (-(z * z) / M_LN2 + log2(1.0 / (z * sqrt(M_PI))))
            : log2(erfc(z));
        double log2_p_resp = log2_p_coeff + log2((double)(N * pp.num_cts));
        printf("  noise: sigma 2^%.2f, max|e| 2^%.2f vs Delta/2 = 2^%.2f "
               "(%zu coeffs) -> est. 2^%.0f fail/response\n",
               log2(sigma), log2((double)max_e), log2(half_delta),
               n_samples, log2_p_resp);
    }

    gpu_free_server(ctx);
}

int main(int argc, char** argv) {
    printf("=== inspire-gpu end-to-end benchmark (RTX 5090) ===\n");
    struct Size { const char* label; size_t entries; };
    // 120 B/entry (64 slots): 1 GB = 2^23 entries, 4 GB = 2^25, 16 GB = 2^27.
    Size all[] = {{"1 GB", 8388608}, {"4 GB", 33554432}, {"16 GB", 134217728}};

    std::vector<Size> run;
    if (argc > 1) {
        for (int a = 1; a < argc; a++)
            for (auto& s : all)
                if (std::string(s.label) == std::string(argv[a]) + " GB") run.push_back(s);
    } else {
        for (auto& s : all) run.push_back(s);
    }

    // Entry size knob: INSPIRE_ENTRY_BYTES (default 120 B = 64 slots). An
    // entry must fill a power-of-two number of 15-bit slots, i.e.
    // ceil(8w/15) must divide N; entry counts rescale to keep each tier's
    // total database bytes fixed.
    size_t entry_bytes = 120;
    if (const char* eb = std::getenv("INSPIRE_ENTRY_BYTES"))
        entry_bytes = (size_t)std::strtoull(eb, nullptr, 10);
    size_t cpe = (entry_bytes * 8 + P_BITS - 1) / P_BITS;
    if (entry_bytes == 0 || cpe > N || N % cpe != 0) {
        printf("INSPIRE_ENTRY_BYTES=%zu invalid: an entry must fill a power-of-two\n"
               "number of 15-bit slots (e.g. 15, 30, 60, 120, 240 bytes).\n",
               entry_bytes);
        return 1;
    }

    for (auto& s : run)
        bench_size(s.label, s.entries * 120 / entry_bytes, entry_bytes,
                   /*warmup=*/3, /*trials=*/5);
    printf("\nDone.\n");
    return 0;
}
