// C ABI round trip: build queries with the C++ client, flatten them through
// the capi layouts, answer via ipir_answer_batch, unflatten, and check
// slot-exact via extract(). Exercises exactly what an FFI consumer does.
#include "capi.h"
#include "protocol.h"
#include <cstring>
#include <iostream>
#include <random>

using namespace inspire;

// [digit][limb][coeff] flatten of a KSK / RGSW part vector (see capi.h).
static void flatten_parts(std::vector<uint64_t>& dst, const std::vector<RnsPoly>& parts,
                          bool want_ntt) {
    for (const RnsPoly& p0 : parts) {
        RnsPoly p = p0;
        if (want_ntt && !p.is_ntt) p.to_ntt();
        if (!want_ntt && p.is_ntt) p.to_coeff();
        for (size_t i = 0; i < N; i++) dst.push_back(p.limbs[0][i]);
        for (size_t i = 0; i < N; i++) dst.push_back(p.limbs[1][i]);
    }
}

int main() {
    std::cout << "=== C ABI (capi) round-trip test ===" << std::endl;
    const size_t N_entries = 524288, entry_bytes = 120, db_rows = 2048;
    const size_t coeffs_per_entry = (entry_bytes * 8 + P_BITS - 1) / P_BITS;

    // Build the slot DB with the same geometry the server will pick.
    PublicParams shape = setup(N_entries, entry_bytes, db_rows);
    std::mt19937_64 rng(7);
    std::vector<uint16_t> db((size_t)shape.db_rows * shape.db_cols);
    for (auto& s : db) s = (uint16_t)(rng() % P);

    ipir_params prm;
    const uint8_t crs_seed[64] = {0x42};
    ipir_server* srv = ipir_server_create(N_entries, entry_bytes, db_rows,
                                          /*max_batch=*/3, db.data(), crs_seed, &prm);
    if (!srv) { std::cout << "FAIL: create" << std::endl; return 1; }
    if (std::memcmp(prm.seed, crs_seed, 64) != 0) {
        std::cout << "FAIL: pinned CRS seed not honored" << std::endl; return 1;
    }
    if (prm.ring_n != N || prm.d_eff != (size_t)D_EFF || prm.db_rows != shape.db_rows ||
        prm.db_cols != shape.db_cols) {
        std::cout << "FAIL: params mismatch" << std::endl; return 1;
    }

    ipir_caps caps;
    ipir_server_caps(srv, &caps);
    if (caps.max_batch != 3) { std::cout << "FAIL: caps" << std::endl; return 1; }

    // Rebuild the client-side PublicParams from ipir_params (what an FFI
    // client does from the published parameters).
    PublicParams pp;
    pp.N_entries = prm.n_entries; pp.w = prm.entry_bytes;
    pp.db_rows = prm.db_rows;     pp.db_cols = prm.db_cols;
    pp.n_packed = prm.n_packed;   pp.D = prm.interp_d;
    pp.num_cts = prm.num_cts;
    std::memcpy(pp.seed.data(), prm.seed, SEED_BYTES);

    const std::vector<size_t> indices = {42, 1000, 524287};
    std::vector<QueryState> states;
    std::vector<std::vector<uint64_t>> ksk5_flat(indices.size()), kskn1_flat(indices.size()),
                                       rgsw_flat(indices.size());
    std::vector<QueryMessage> qmsgs;
    std::vector<ipir_query_view> views(indices.size());

    for (size_t i = 0; i < indices.size(); i++) {
        auto [qst, qry] = query(pp, indices[i]);
        states.push_back(qst);
        qmsgs.push_back(qry);
        flatten_parts(ksk5_flat[i], qmsgs[i].ksk_5.b_parts,    /*want_ntt=*/false);
        flatten_parts(kskn1_flat[i], qmsgs[i].ksk_neg1.b_parts, /*want_ntt=*/false);
        flatten_parts(rgsw_flat[i], qmsgs[i].rgsw.top.a_parts,    true);
        flatten_parts(rgsw_flat[i], qmsgs[i].rgsw.top.b_parts,    true);
        flatten_parts(rgsw_flat[i], qmsgs[i].rgsw.bottom.a_parts, true);
        flatten_parts(rgsw_flat[i], qmsgs[i].rgsw.bottom.b_parts, true);
        views[i].lwe_b_limb0 = qmsgs[i].lwe.b_limb0.data();
        views[i].lwe_b_limb1 = qmsgs[i].lwe.b_limb1.data();
        views[i].ksk5_b      = ksk5_flat[i].data();
        views[i].kskneg1_b   = kskn1_flat[i].data();
        views[i].rgsw        = rgsw_flat[i].data();
    }

    std::vector<uint64_t> out(indices.size() * prm.num_cts * 4 * N);
    int rc = ipir_answer_batch(srv, views.data(), views.size(), out.data());
    if (rc != 0) { std::cout << "FAIL: answer_batch rc=" << rc << std::endl; return 1; }

    int fail = 0;
    const uint64_t* r = out.data();
    for (size_t i = 0; i < indices.size(); i++) {
        std::vector<RlweCt> resp(prm.num_cts);
        for (size_t g = 0; g < prm.num_cts; g++) {
            resp[g].a = RnsPoly::zero(); resp[g].b = RnsPoly::zero();
            for (size_t k = 0; k < N; k++) resp[g].a.limbs[0][k] = *r++;
            for (size_t k = 0; k < N; k++) resp[g].a.limbs[1][k] = *r++;
            for (size_t k = 0; k < N; k++) resp[g].b.limbs[0][k] = *r++;
            for (size_t k = 0; k < N; k++) resp[g].b.limbs[1][k] = *r++;
            resp[g].a.is_ntt = true; resp[g].b.is_ntt = true;
        }
        auto slots = extract(pp, states[i], resp);
        size_t i_star = indices[i] % pp.db_rows;
        size_t col = (indices[i] / pp.db_rows) * coeffs_per_entry;
        bool ok = slots.size() == coeffs_per_entry;
        for (size_t c = 0; ok && c < coeffs_per_entry; c++)
            ok = (slots[c] == (uint16_t)(db[i_star * pp.db_cols + col + c] % P));
        if (!ok) { std::cout << "FAIL: idx=" << indices[i] << " extract mismatch" << std::endl; fail++; }
    }

    // Over-limit must be rejected without touching out_resp.
    std::vector<ipir_query_view> too_many(4, views[0]);
    if (ipir_answer_batch(srv, too_many.data(), 4, out.data()) == 0) {
        std::cout << "FAIL: over-limit batch accepted" << std::endl; fail++;
    }

    // --- Pure-ABI client path: ipir_query_build → answer → ipir_extract ---
    const size_t q_u64s = ipir_query_u64s(&prm);
    if (q_u64s != 2 * prm.db_rows + 6 * prm.d_eff * 2 * prm.ring_n) {
        std::cout << "FAIL: ipir_query_u64s" << std::endl; fail++;
    }
    if (ipir_response_u64s(&prm) != prm.num_cts * 4 * prm.ring_n) {
        std::cout << "FAIL: ipir_response_u64s" << std::endl; fail++;
    }
    if (ipir_entry_slots(&prm) != coeffs_per_entry) {
        std::cout << "FAIL: ipir_entry_slots" << std::endl; fail++;
    }

    const std::vector<size_t> c_indices = {7, 400000};
    std::vector<std::vector<uint64_t>> flats(c_indices.size(),
                                             std::vector<uint64_t>(q_u64s));
    std::vector<ipir_client_query*> cqs(c_indices.size());
    std::vector<ipir_query_view> c_views(c_indices.size());
    for (size_t i = 0; i < c_indices.size(); i++) {
        cqs[i] = ipir_query_build(&prm, c_indices[i], flats[i].data());
        if (!cqs[i]) { std::cout << "FAIL: query_build" << std::endl; return 1; }
        if (ipir_query_view_from_flat(&prm, flats[i].data(), &c_views[i]) != 0) {
            std::cout << "FAIL: view_from_flat" << std::endl; return 1;
        }
    }

    std::vector<uint64_t> c_out(c_indices.size() * ipir_response_u64s(&prm));
    if (ipir_answer_batch(srv, c_views.data(), c_views.size(), c_out.data()) != 0) {
        std::cout << "FAIL: answer_batch (ABI client)" << std::endl; return 1;
    }

    std::vector<uint16_t> c_slots(coeffs_per_entry);
    for (size_t i = 0; i < c_indices.size(); i++) {
        int erc = ipir_extract(&prm, cqs[i],
                               c_out.data() + i * ipir_response_u64s(&prm),
                               c_slots.data());
        if (erc != 0) { std::cout << "FAIL: extract rc=" << erc << std::endl; fail++; continue; }
        size_t i_star = c_indices[i] % pp.db_rows;
        size_t col = (c_indices[i] / pp.db_rows) * coeffs_per_entry;
        for (size_t c = 0; c < coeffs_per_entry; c++)
            if (c_slots[c] != (uint16_t)(db[i_star * pp.db_cols + col + c] % P)) {
                std::cout << "FAIL: ABI-client idx=" << c_indices[i]
                          << " slot mismatch" << std::endl;
                fail++; break;
            }
        ipir_client_query_destroy(cqs[i]);
    }

    // --- Wire packing: lossless BETA-bit CRT packing round-trips exactly, and
    // a packed-then-unpacked query answers identically. ---
    const size_t qp_bytes = ipir_query_packed_bytes(&prm);
    if (qp_bytes != ((prm.db_rows + 6 * prm.d_eff * prm.ring_n) * BETA + 7) / 8) {
        std::cout << "FAIL: query_packed_bytes" << std::endl; fail++;
    }
    std::vector<uint8_t> qp(qp_bytes);
    std::vector<uint64_t> q_rt(q_u64s);
    if (ipir_query_pack(&prm, flats[0].data(), qp.data()) != 0 ||
        ipir_query_unpack(&prm, qp.data(), q_rt.data()) != 0 ||
        q_rt != flats[0]) {
        std::cout << "FAIL: query pack/unpack round trip" << std::endl; fail++;
    }

    const size_t rp_bytes = ipir_response_packed_bytes(&prm);
    std::vector<uint8_t> rp(rp_bytes);
    std::vector<uint64_t> r_rt(ipir_response_u64s(&prm));
    if (ipir_response_pack(&prm, c_out.data(), rp.data()) != 0 ||
        ipir_response_unpack(&prm, rp.data(), r_rt.data()) != 0 ||
        !std::equal(r_rt.begin(), r_rt.end(), c_out.begin())) {
        std::cout << "FAIL: response pack/unpack round trip" << std::endl; fail++;
    }

    // Full wire path with a fresh query: build → pack → unpack → answer →
    // pack → unpack → extract must still hit the right entry.
    {
        const size_t widx = 123456;
        std::vector<uint64_t> wflat(q_u64s);
        ipir_client_query* wq = ipir_query_build(&prm, widx, wflat.data());
        std::vector<uint8_t> wire_q(qp_bytes);
        std::vector<uint64_t> srv_flat(q_u64s);
        ipir_query_pack(&prm, wflat.data(), wire_q.data());
        ipir_query_unpack(&prm, wire_q.data(), srv_flat.data());
        ipir_query_view v;
        ipir_query_view_from_flat(&prm, srv_flat.data(), &v);
        std::vector<uint64_t> wout(ipir_response_u64s(&prm));
        std::vector<uint8_t> wire_r(rp_bytes);
        std::vector<uint64_t> cli_resp(wout.size());
        std::vector<uint16_t> wslots(coeffs_per_entry);
        if (ipir_answer_batch(srv, &v, 1, wout.data()) != 0 ||
            ipir_response_pack(&prm, wout.data(), wire_r.data()) != 0 ||
            ipir_response_unpack(&prm, wire_r.data(), cli_resp.data()) != 0 ||
            ipir_extract(&prm, wq, cli_resp.data(), wslots.data()) != 0) {
            std::cout << "FAIL: packed wire path" << std::endl; fail++;
        } else {
            size_t i_star = widx % pp.db_rows;
            size_t col = (widx / pp.db_rows) * coeffs_per_entry;
            for (size_t c = 0; c < coeffs_per_entry; c++)
                if (wslots[c] != (uint16_t)(db[i_star * pp.db_cols + col + c] % P)) {
                    std::cout << "FAIL: packed wire slot mismatch" << std::endl;
                    fail++; break;
                }

            // Compressed (mod-switched) response: 12 KB/ct must decode to
            // the same slots despite the added rounding noise.
            if (ipir_response_compressed_bytes(&prm) != prm.num_cts * 12288) {
                std::cout << "FAIL: compressed_bytes" << std::endl; fail++;
            }
            std::vector<uint8_t> comp(ipir_response_compressed_bytes(&prm));
            std::vector<uint16_t> comp_slots(coeffs_per_entry);
            if (ipir_response_compress(&prm, wout.data(), comp.data()) != 0 ||
                ipir_extract_compressed(&prm, wq, comp.data(), comp_slots.data()) != 0) {
                std::cout << "FAIL: compressed path rc" << std::endl; fail++;
            } else {
                for (size_t c = 0; c < coeffs_per_entry; c++)
                    if (comp_slots[c] != wslots[c]) {
                        std::cout << "FAIL: compressed slot mismatch at " << c
                                  << std::endl;
                        fail++; break;
                    }
            }
        }
        ipir_client_query_destroy(wq);
    }

    // An out-of-range index must be rejected.
    if (ipir_query_build(&prm, N_entries, flats[0].data()) != nullptr) {
        std::cout << "FAIL: out-of-range index accepted" << std::endl; fail++;
    }

    ipir_server_destroy(srv);
    if (fail == 0) { std::cout << "All capi checks passed." << std::endl; return 0; }
    return 1;
}
