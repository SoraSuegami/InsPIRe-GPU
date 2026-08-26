// C ABI facade implementation: marshals flat arrays <-> the C++ types and
// delegates to the gpu_protocol entry points. See capi.h for layouts.
#include "capi.h"
#include "gpu_protocol.h"
#include <cstring>
#include <memory>
#include <new>

using namespace inspire;

struct ipir_server {
    PublicParams pp;
    GpuServerCtx* ctx;
};

static RnsPoly unflatten_poly(const uint64_t* src, bool ntt) {
    // src: [limb0: N][limb1: N]
    RnsPoly p = RnsPoly::zero();
    for (size_t i = 0; i < N; i++) {
        p.limbs[0][i] = src[i];
        p.limbs[1][i] = src[N + i];
    }
    p.is_ntt = ntt;
    return p;
}

// ksk / rgsw-part block: [digit][limb][coeff], d_eff digits.
static std::vector<RnsPoly> unflatten_parts(const uint64_t* src, bool ntt) {
    std::vector<RnsPoly> parts(D_EFF);
    for (int j = 0; j < D_EFF; j++)
        parts[j] = unflatten_poly(src + (size_t)j * 2 * N, ntt);
    return parts;
}

static QueryMessage unmarshal_query(const ipir_query_view& v, size_t db_rows) {
    QueryMessage q;
    q.lwe.b_limb0.assign(v.lwe_b_limb0, v.lwe_b_limb0 + db_rows);
    q.lwe.b_limb1.assign(v.lwe_b_limb1, v.lwe_b_limb1 + db_rows);
    q.ksk_5.b_parts    = unflatten_parts(v.ksk5_b, /*ntt=*/false);
    q.ksk_neg1.b_parts = unflatten_parts(v.kskneg1_b, /*ntt=*/false);
    const size_t blk = (size_t)D_EFF * 2 * N;
    q.rgsw.top.a_parts    = unflatten_parts(v.rgsw + 0 * blk, /*ntt=*/true);
    q.rgsw.top.b_parts    = unflatten_parts(v.rgsw + 1 * blk, /*ntt=*/true);
    q.rgsw.bottom.a_parts = unflatten_parts(v.rgsw + 2 * blk, /*ntt=*/true);
    q.rgsw.bottom.b_parts = unflatten_parts(v.rgsw + 3 * blk, /*ntt=*/true);
    return q;
}

extern "C" {

ipir_server* ipir_server_create(size_t n_entries, size_t entry_bytes,
                                size_t db_rows_or_zero, size_t max_batch,
                                const uint16_t* db,
                                const uint8_t* crs_seed_or_null,
                                ipir_params* out_params) {
    if (!db || !out_params || max_batch == 0) return nullptr;
    try {
    // setup() throws on unsupported geometry; nothing may cross the C ABI.
    std::unique_ptr<ipir_server> srv(new ipir_server);
    srv->pp = db_rows_or_zero ? setup(n_entries, entry_bytes, db_rows_or_zero)
                              : setup(n_entries, entry_bytes);
    // The caller's CRS must be in place before preprocessing: the precomp
    // tensors bake in the CRS-expanded a-parts.
    if (crs_seed_or_null)
        std::memcpy(srv->pp.seed.data(), crs_seed_or_null, SEED_BYTES);

    PreprocessData precomp = gpu_preprocess(srv->pp, db);
    GpuServerConfig cfg;
    cfg.max_batch = max_batch;
    srv->ctx = gpu_setup_server(srv->pp, precomp, cfg);

    out_params->n_entries   = srv->pp.N_entries;
    out_params->entry_bytes = srv->pp.w;
    out_params->db_rows     = srv->pp.db_rows;
    out_params->db_cols     = srv->pp.db_cols;
    out_params->n_packed    = srv->pp.n_packed;
    out_params->interp_d    = srv->pp.D;
    out_params->num_cts     = srv->pp.num_cts;
    out_params->ring_n      = N;
    out_params->d_eff       = (size_t)D_EFF;
    static_assert(SEED_BYTES == sizeof(out_params->seed), "seed size mismatch");
    std::memcpy(out_params->seed, srv->pp.seed.data(), SEED_BYTES);
    return srv.release();
    } catch (...) {
        return nullptr;
    }
}

void ipir_server_caps(const ipir_server* srv, ipir_caps* out) {
    if (!srv || !out) return;
    GpuServerCaps c = gpu_server_caps(srv->ctx);
    out->max_batch         = c.max_batch;
    out->resident_bytes    = c.resident_bytes;
    out->device_free_bytes = c.device_free_bytes;
    out->num_cts           = c.num_cts;
}

int ipir_answer_batch(ipir_server* srv, const ipir_query_view* queries,
                      size_t count, uint64_t* out_resp) {
    if (!srv || !queries || !out_resp) return 1;
    if (count == 0) return 0;
    if (count > gpu_server_caps(srv->ctx).max_batch) return 2;

    std::vector<QueryMessage> qs;
    qs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        const ipir_query_view& v = queries[i];
        if (!v.lwe_b_limb0 || !v.lwe_b_limb1 || !v.ksk5_b || !v.kskneg1_b || !v.rgsw)
            return 3;
        qs.push_back(unmarshal_query(v, srv->pp.db_rows));
    }

    auto resps = gpu_answer_batch(srv->ctx, qs);

    // Flatten: [query][ct][a | b][limb][coeff], NTT form.
    uint64_t* w = out_resp;
    for (size_t i = 0; i < count; i++) {
        for (const RlweCt& ct : resps[i]) {
            for (size_t k = 0; k < N; k++) *w++ = ct.a.limbs[0][k];
            for (size_t k = 0; k < N; k++) *w++ = ct.a.limbs[1][k];
            for (size_t k = 0; k < N; k++) *w++ = ct.b.limbs[0][k];
            for (size_t k = 0; k < N; k++) *w++ = ct.b.limbs[1][k];
        }
    }
    return 0;
}

void ipir_server_destroy(ipir_server* srv) {
    if (!srv) return;
    gpu_free_server(srv->ctx);
    delete srv;
}

}  // extern "C"
