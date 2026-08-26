# inspire-gpu Architecture

Engineering companion to `scheme_description.pdf` (which covers the math: scheme, noise,
parameter selection). This doc covers the code: module layout, data structures, the
GPU pipeline, and the current performance state.

For the history of how the code got here (the `Phase N: …` and online-overhaul
commits), read the git log. This doc describes only the current state.

---

## 1. Scope and non-goals

**Scope.** Single-server InsPIRe PIR (per ePrint 2025/1352) as a C++17 + CUDA library
for **RTX 5090 (sm_120, 32 GB GDDR7)** at deployment. The server operations
(`gpu_preprocess`, `gpu_answer`, `gpu_answer_batch`) run on GPU; the client
operations (`setup`, `query`, `extract`) are CPU. A GPU-accelerated server context
(`GpuServerCtx`) holds long-lived state across many queries.

**Mission framing.** A from-scratch C++/CUDA implementation of the InsPIRe scheme,
holding itself to two standards:

- **Readability**: small codebase, few dependencies, clear module boundaries.
- **Performance**: low per-query and preprocess latency on the target hardware,
  re-measured rather than asserted.

**Non-goals.**
- CPU server path. The server (`preprocess`/`answer`) is GPU-only; only the
  client-side `query` / `extract` and their shared primitives run on CPU,
  and their (sub-ms) performance is not tuned.
- The `inspire_0` / `inspire^2` variants.
- Multi-server / multi-GPU sharding.
- Application layer (cuckoo hashing, keyword PIR, Ethereum integration) — out of scope.
- Database update / live-fold pipeline (sidecar). Out of scope for v1.

## 2. Status (numbers measured 2026-08-24, RTX 5090 — RunPod, Xeon Gold 6530 host)

### Geometry and residency

The **default DB geometry is `db_rows = 32768`** — `setup(N, w)` uses it for every
size (1 GB → n_packed=8, 4 GB → 32, 16 GB → 128); `setup(N, w, db_rows)` overrides.
There is no objective/auto-selection — just the fixed default plus the override.
The database is
**row-major, supplied as slot values**: `db_rows × db_cols` plaintext
values in `[0, P)` (no raw byte DB), encoded in place (inverse-DFT only), and
stored row-major so the online mat-vec reads it directly — **no transpose, one
resident copy.** 16 GB fits in 32 GB because there is a single row-major DB copy
(a column-major encode + GPU transpose would need two copies = ~43 GB).

### Single-query latency (`bench_e2e`, warm, median of 5)

Entries are 120 B = 64 slots by default (sized for the serving layer's
two-account cuckoo buckets); only total DB bytes matter for performance —
the same bytes as 2× as many 60 B entries give the identical geometry and
latencies.

| DB | per-query latency | comm (query↑ + resp↓) | hint (precomp + DB resident) | setup |
|---|---|---|---|---|
| **1 GB** (2²³ items) | **~2.5 ms** | 383 KB (371 + 12) | 1.61 GB | ~3.1 s |
| **4 GB** (2²⁵ items) | **~7.8 ms** | 383 KB | 6.44 GB | ~3.0 s |
| **16 GB** (2²⁷ items) | **~31 ms** | 383 KB | 25.77 GB | ~6.1 s |

Both communication directions are real wire bytes now: queries are 53-bit
CRT-packed (`ipir_query_pack`), responses are modulus-switched to q'
(`ipir_response_compress`, 12,288 B per ciphertext).

(The same suite on an AMD EPYC 7543 host measures 3.2 / 8.7 / ~36 ms; the
two hosts agree within the ±20% gate.)

Comm is constant (db_rows fixed); latency and hint scale with `n_packed`.
Single-query phase split at 16 GB: mat-vec ~14.5 ms, collapse ~9 ms, horner
~10.5 ms.

### Batched throughput (`gpu_answer_batch`, median of 3)

Queries in a batch are independent clients (each carries its own key
material, as the protocol requires); what batching shares is the *data
streams*: one DB pass for the mat-vec, one precomp-tensor pass for the
collapse, and a lockstep Horner whose launch count is independent of B.

| 16 GB DB | batch latency | per-query | throughput |
|---|---|---|---|
| B=1 | 31.0 ms | 31.0 ms | 32 q/s |
| B=4 | 42.4 ms | 10.6 ms | 94 q/s |
| B=8 | 74.7 ms | 9.3 ms | 107 q/s |
| B=16 | 138.8 ms | 8.7 ms | 115 q/s |
| B=32 | 263.3 ms | 8.2 ms | 122 q/s |

1 GB: 402 q/s (B=1) → **619 q/s** (B=16). The curve is at its asymptote by
B≈16: the remaining per-query wall is the batched mat-vec tile (INT32
compute-bound, ~5.8 ms/query at 16 GB), not batch width — see §9.

### Response-noise margin (bench_e2e noise section, measured)

Decrypting real responses and reading the noise off the phase (53-bit
parameter set): at 16 GB (D_actual=128) the measured pipeline sigma is
2^32.56 against a decode threshold of Delta/2 = 2^36, i.e. ~2^-78 failure
per response; at 1 GB, sigma 2^30.58 and ~2^-1323. Sigma scales exactly as
sqrt(D), matching the additive-noise analysis in `scheme_description.pdf`.
The q' modulus-switching compression adds rounding noise dominated by the
body term q/q0'/sqrt(12) ~ 2^31.2; combined, the 16 GB total is ~2^32.8,
an estimated ~2^-53 per response — still past the 2^-40 design target at
every geometry the card holds.

### Wire-path CPU costs (bench_e2e wire section)

Per query, everything besides the GPU answer (1 GB tier, db_rows = 32768,
Xeon Gold 6530):

| Step | Side | Time |
|---|---|---|
| query build (keys + encryption) | client | 31 ms (~42 ms through the FFI, which also writes the 918 KB flat buffer) |
| query pack (371 KB) | client | 0.5 ms |
| query unpack | server | 0.4 ms |
| response compress (12 KB) | server | 0.8 ms |
| extract_compressed | client | 1.5 ms (plain extract: 0.5 ms) |

Query build dominates the client's CPU cost and scales with db_rows.
Untapped: the CRS-derived a-parts it expands are identical for every query
(fixed seed), so a client doing repeated lookups could cache the expansion.

## 3. Specification and correctness references

The authoritative references for the scheme and its parameters are:

| Artifact | Role |
|---|---|
| [ePrint 2025/1352](https://eprint.iacr.org/2025/1352) | The InsPIRe paper — the scheme itself. |
| `docs/scheme_description.{tex,pdf}` | This project's derivation of the math, noise, and parameter selection. |

Investigation order on a correctness discrepancy:
1. `scheme_description.tex` for what the math says.
2. The bit-exact CPU oracle tests (§10) to localize the failing primitive.

> **Note — slot values vs. the spec's byte packing.** `scheme_description.tex`
> describes `EncodeDatabase` / `Decode` / `Extract` with a byte↔slot packing
> layer (`Encode_p` bit-packs `w`-byte records into 15-bit slots in `[0, P)`).
> `inspire-gpu` takes slot values directly: the DB is supplied already as values in
> `[0, P)`, and `gpu_encode_inverse_dft` applies *only* the inverse-DFT — the
> byte↔slot packing is the caller's responsibility and is not implemented
> here. The spec's packing description is correct
> as a scheme-level I/O contract; it just lives outside this codebase.

## 4. Code layout

```
inspire-gpu/
├── CMakeLists.txt
├── docs/
│   ├── scheme_description.{tex,pdf}   # the math
│   └── ARCHITECTURE.md                # this doc
├── src/
│   ├── params.{h,cpp}                 # N, Q0, Q1, P, D_GSW, BASE_LOG; setup()
│   ├── ring.{h,cpp}                   # RnsPoly, automorphism, gadget decomp, sampling
│   ├── ntt.{h,cpp}                    # in-tree scalar negacyclic NTT
│   ├── crypto.{h,cpp}                 # sk_gen, rlwe_dec, lwe_enc_b, ksk_gen_b, rgsw_enc_b, ext_prod, horner_eval
│   ├── protocol.{h,cpp}               # query, extract (client side)
│   ├── capi.h                         # C ABI (ipir_*) — declarations for both halves
│   ├── capi.cu                        # server half (GPU library)
│   ├── capi_client.cpp                # client half + wire packing/compression (CPU lib)
│   ├── gpu.cuh                        # public CUDA API (sectioned by phase)
│   ├── gpu_common.cuh                 # shared __device__ helpers (mod/Barrett/Shoup-Harvey/NTT primitives)
│   ├── gpu_primitives.cu             # context, NTT, poly ops, gadget decomp, memory helpers
│   ├── gpu_preprocess.cu             # encode, a-side matmul, ring_embed, automorphisms, KSK perm, collapse_fused
│   ├── gpu_online.cu                 # matvec, lazy_collapse, fused_ip_sub, ext_prod, horner
│   ├── gpu_protocol.{h,cu}            # GpuServerCtx; gpu_preprocess, gpu_setup_server, gpu_answer
├── tests/
│   ├── test_ring, test_crypto                       # CPU client-primitive unit tests
│   ├── test_gpu_ntt, test_gpu_decomp,
│   ├── test_gpu_extprod, test_gpu_horner            # GPU per-kernel, bit-exact vs CPU primitive
│   ├── test_e2e_gpu                                 # Full pipeline, slot-exact; Small/Medium/1 GB
│   ├── test_gpu_batch                               # gpu_answer_batch bit-identical to B singles
│   └── test_capi                                    # C ABI round trip vs C++ API
├── benches/
│   └── bench_e2e             # latency + batch sweep + noise margin + wire timings
```

### Module responsibilities

The library is **GPU-only on the server side**; the CPU code is just the client
(query/extract) plus the primitives it shares with the GPU correctness tests.

- **`params`**: fixed crypto constants + `setup(N_entries, w[, db_rows])` → `PublicParams`
  (default geometry db_rows=32768).
- **`ring` / `ntt`**: math layer. `RnsPoly`, in-place arithmetic, NTT, automorphism,
  gadget decomposition, discrete-Gaussian / ternary sampling, seed expansion (SHAKE-256
  via OpenSSL). Used by the client side and as the bit-exact oracle for the GPU
  per-kernel tests (the GPU upload narrows `uint64_t` → `uint32_t` at the boundary).
- **`crypto`**: per-algorithm wrappers (`sk_gen`, `rlwe_dec`, `lwe_enc_b`, `ksk_gen_b`,
  `rgsw_enc_b`, `ext_prod`, `horner_eval`). Thin over `ring`.
- **`protocol` (client side, CPU — always sub-ms)**:
  - `query` — builds the encrypted query message.
  - `extract` — decrypt the response to the queried entry's plaintext slots in
    [0,P) (any byte↔slot packing is caller-side).
  (Server `preprocess`/`answer` are GPU-only — `gpu_preprocess`/`gpu_answer`.)
- **`gpu` family**: device-side kernels, split by pipeline phase. `gpu.cuh` is the
  public API, sectioned to match these TUs.
  - `gpu_common.cuh`: shared device helpers (mod/Barrett/Shoup-Harvey/NTT primitives).
  - `gpu_primitives.cu`: phase-agnostic building blocks — context init, NTT
    (fwd/inv/batch/stream), pointwise poly ops, gadget decomp, memory helpers.
  - `gpu_preprocess.cu`: server one-time setup — `gpu_encode_inverse_dft`
    (row-major inverse-DFT on slot values, no byte-pack/transpose), a-side matmul,
    ring_embed, automorphisms, KSK perm, and the fused `collapse_fused_kernel`.
  - `gpu_online.cu`: per-query hot path — matvec, online lazy-collapse
    (partial/reduce) + `fused_ip_sub`, ext_prod, horner.
- **`gpu_protocol`**: the **server-side** protocol — the GPU mirror of the
  client-side `protocol` (`query`/`extract`). Owns `GpuServerCtx` (DB, precomp,
  twiddles, per-query scratch, per-group streams) and exposes the operations in
  lifecycle order: `gpu_preprocess` (full preprocess on device) → `gpu_setup_server`
  (build the per-query-reusable context) → `gpu_answer` (upload → matvec →
  online-collapse → horner → download) → `gpu_free_server`.

## 5. Build, dependencies, hardware target

- **C++17**, `-O3 -march=native` in Release. `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for tooling.
- **CUDA**: `CMAKE_CUDA_ARCHITECTURES` defaults to **`"120"`** (RTX 5090 / Blackwell);
  override with `-DCMAKE_CUDA_ARCHITECTURES=86` (or any list) when targeting other GPUs.
- **OpenSSL** for SHAKE-256 seed expansion (`EVP_shake256`) — the only external
  runtime dep.
- **In-tree NTT.** The scalar NTT (`src/ntt.{h,cpp}`) is the only NTT impl on the
  CPU side; it shares the same algorithm and twiddle layout as the GPU NTT
  (`get_ntt_twiddles()` exports the GPU-friendly tables).

```
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 6. Data structures and layout conventions

- **CPU `RnsPoly`** (`ring.h`): `std::array<std::vector<uint64_t>, 2>` per-limb storage,
  `is_ntt` flag tracks domain.
- **GPU polynomials** (`gpu.cuh`, `gpu_protocol.cu`): `uint32_t` in NTT form, packed
  `[limb0: N][limb1: N]` contiguously. A "vector of d polynomials" is
  `[limb0: d×N][limb1: d×N]`.
- **Conversion is a narrow cast at upload time** — the CPU stores `uint64_t` (a
  historical layout choice); the GPU narrows to `uint32_t` at upload. Never
  round-tripped in the hot path.
- **Encoded DB on GPU**: row-major `uint16_t[db_rows × db_cols]` (`db[row*db_cols+col]`),
  P=65535, 15-bit-per-slot packing. `gpu_encode_inverse_dft` writes it row-major directly; the
  online mat-vec reads it as-is (no transpose, single resident copy). The DB is
  supplied as plaintext slot values in `[0, P)`, not raw bytes.

Key types:

| Type | Defined in | Lifetime |
|---|---|---|
| `PublicParams` | `params.h` | Computed once by `setup()`; passed by const-ref everywhere. |
| `RnsPoly`, `RlweCt` | `ring.h` | Value types; std::vector-based. |
| `LweQuery`, `Ksk`, `RgswCt` | `crypto.h` | Per-query messages (client side). |
| `QueryState`, `QueryMessage` | `protocol.h` | Client output of `query()`. |
| `PreprocessData` | `protocol.h` | Server-side. After `gpu_preprocess`, holds `d_db_col` (the device-resident **row-major** u16 DB — name is legacy) and `d_precomp_*` (per-group device pointers). |
| `GpuServerCtx` | opaque, in `gpu_protocol.cu` | Long-lived GPU state. Owns DB, precomp, twiddles, per-query scratch, streams. |

Invariants:
- `Precomp::a` and all `D_*` polynomials are in NTT form.
- `RgswCt` `a_parts`/`b_parts` are in NTT form.
- `Ksk::b_parts` are stored in **coefficient form** — `expand_ksk_b` does the NTT.
- DB on GPU is row-major `uint16_t[db_rows × db_cols]`.

## 7. GPU pipeline

```cpp
// db: db_rows*db_cols plaintext values in [0,P), ROW-MAJOR (db[row*db_cols+col]).
PreprocessData      gpu_preprocess(const PublicParams& pp, const uint16_t* db);
GpuServerCtx*       gpu_setup_server(const PublicParams& pp, const PreprocessData& precomp,
                                     const GpuServerConfig& cfg = {});  // cfg.max_batch slots
GpuServerCaps       gpu_server_caps(const GpuServerCtx*);   // limits for the caller's scheduler
std::vector<RlweCt> gpu_answer(GpuServerCtx* ctx, const QueryMessage& qry);
std::vector<std::vector<RlweCt>>
                    gpu_answer_batch(GpuServerCtx*, const QueryMessage* qs, size_t count);
void                gpu_free_server(GpuServerCtx* ctx);
```

`capi.h` mirrors both sides over `extern "C"` for FFI consumers: the server
lifecycle (`ipir_server_create` / `ipir_server_caps` / `ipir_answer_batch` /
`ipir_server_destroy`, GPU library) and the client plus wire encodings
(`ipir_query_build` / `ipir_extract` / `ipir_query_pack` /
`ipir_response_compress` / `ipir_extract_compressed`, CPU library); layouts
are documented in the header and round-tripped by `test_capi`.

### Serving contract

- **Nothing is allocated on the request path.** `gpu_setup_server` sizes
  `cfg.max_batch` query slots (scratch + pointer tables) up front;
  `gpu_answer_batch` rejects `count > max_batch`.
- **Every query is self-contained** (its own LWE vector, RGSW, and two KSK
  b-parts). Key material cannot be shared across queries: with the CRS-fixed
  random components, two ciphertexts under one secret leak the difference of
  their payloads, and the paper's App. F.1 security argument requires a fresh
  secret per query. The server must also never cache keys across calls.
- **A generation = a ctx.** Database rollover is: build a new
  `PreprocessData`, `gpu_setup_server` it (memory permitting — see
  `caps.device_free_bytes`), route the next batch to the new ctx, free the
  old one. There is no in-place update path: the collapse precomp bakes in
  gadget decompositions of DB-derived values, which are not incrementally
  updatable.
- Calls on one ctx must be externally serialized (one batch at a time).

### `gpu_preprocess` — one-time work

```
encode: upload row-major slot DB (one u16 copy) + in-place inverse-DFT (no byte-pack, no transpose)
CRS derivation (4 CRS polys, CPU NTT)
GpuRingEmbedHelper init: build mono32 table (CPU NTT × 4096 polys → upload)
GpuCollapseHelper init: SHAKE256 + upload + GPU NTT + batched perm → device KSK
per-group loop:
  alloc precomp_a / D_plus / D_minus / D_final
  GPU a-side matmul → helper.d_a32
  GPU ring_embed + Phase 11 batched per-k auto_perm → helper.d_a_post_*
  cudaMemcpyAsync into per-group staging buffers (input to batched collapse)
batched collapse: one kernel launch covering all groups (Phase 10)
```

Preprocess total at the default geometry (db_rows=32768): ~2.6 s @ 1 GB, ~2.9 s
@ 4 GB, ~5.8 s @ 16 GB. The row-major encode (upload one u16 copy + in-place
inverse-DFT, coalesced) replaced the old byte-pack + column-major encode + GPU
transpose; it is both lower-memory (one DB copy) and faster.

### `gpu_answer` — per query

```
upload query b → matvec → upload + NTT KSK_b
for each group (streams round-robin):
    lazy_collapse_stream × {fwd, conj} × {limb 0, limb 1}
    fused_ip_sub_stream × {limb 0, limb 1}
upload RGSW
horner_eval D
download response
```

### `gpu_answer_batch` — B queries, three stages

```
Stage 1  upload all b vectors; batched mat-vec (register tiles of 4 share one
         DB stream) on wide geometries, per-query row-split dispatch on tall
Stage 2  upload + NTT all KSKs; batched pack: per group, the precomp tensor
         streams ONCE while register tiles of up to 8 apply it to every
         query's b-side (per-(stream,slot) partials pool)
Stage 3  upload RGSWs; lockstep batched Horner: all chains advance the same
         step together — gather → batched INTT → strided gadget decomp →
         batched NTT → per-query MAC = 8 launches per step, independent of B
         — then download
```

Every batched kernel preserves the single-query accumulation order, so
`gpu_answer_batch` is bit-identical to `count` independent `gpu_answer`
calls (`test_gpu_batch` enforces this).

Per-phase timing at the default geometry (db_rows=32768), measured with
`GPU_ANSWER_TRACE=1` / `GPU_ANSWER_FINE=1`. All three phases scale with `n_packed`
(= D_actual = horner chain length):

| Phase | 1 GB (n_packed=8) | 16 GB (n_packed=128) | Notes |
|---|---|---|---|
| mat-vec | ~1.5 ms | ~14.5 ms | reads whole DB; bandwidth-bound; one-thread-per-column when columns fill the SMs |
| upload ksk5/neg1 | ~0.2 ms | ~0.2 ms | tiny |
| collapse (all groups) | ~0.65 ms | ~9 ms | lazy-collapse partial+reduce (grid over coeffs × step-chunks); fills the GPU |
| upload RGSW | ~0.05 ms | ~0.05 ms | tiny |
| horner + download | ~1.5 ms | ~10.5 ms | per-iter NTTs batched per modulus @ 1024 threads; `+packed` folded into the MAC |
| TOTAL | **~2.9 ms** | **~34 ms** | |

All three online phases are optimized (collapse step-parallelism, horner NTT
batching + 1024-thread NTTs + fused add, mat-vec dispatch). Online Horner is
GPU-compute bound, not launch-bound (launch/latency overhead ≈ 1%).
Remaining headroom: mat-vec is ~55–67% of HBM bandwidth (vectorized u16 loads
could approach the ~2.4 ms/1 GB floor); horner is a serial dependent chain whose
floor is set by `D_actual` length.

### Streams

`GpuServerCtx` holds a pool of CUDA streams. Per-query `gpu_answer` round-robins the
per-group lazy-collapse + final-step work across them so the kernels overlap on the
GPU's SMs. After the collapse fusion, the explicit streams become moot.

## 8. CPU code (client side only)

The server pipeline is **entirely on GPU** — there is no CPU `preprocess`/`answer`
anymore (the CPU InspiRING reference in `packing.cpp` and `protocol.cpp::preprocess`
were removed). The remaining CPU code is:
- **`query()` / `extract()`** — the client-side endpoints,
  always used, sub-millisecond.
- **`ring` / `ntt` / `crypto` primitives** — used by the client and as the
  bit-exact oracle for the GPU per-kernel tests.

## 9. Optimization pointer

Remaining opportunities, in priority order (all sized against the batched
profile at 16 GB, B=16: per-query ≈ 8.7 ms = mat-vec 5.8 + collapse 1.1 +
horner 0.7 + transfers/host ~1.1):

1. **Tensor-core INT8 mat-vec** — the batched mat-vec is INT32 compute-bound
   (that is why its tile stops at 4). Decomposing operands into INT8 planes
   and recombining with shifts (the VIPIR recipe, adapted to our two 27-bit
   primes with K-tiled lazy accumulation) moves it back to bandwidth-bound:
   one 16 GB stream per batch ≈ 9.6 ms, i.e. ~1 ms/query at B=16. Expected
   endpoint ≈ 4 ms/query (~250 q/s/card). A `__dp4a` variant is the lower-risk
   first step. This also removes the tall-geometry fallback (a GEMM handles
   any shape), letting 1 GB amortize its DB stream too.
2. **Batched b-poly prep in Stage 2** — the per-(group, query) copy+NTT prep
   is 4·n_packed·B launches (8k at 16 GB / B=16); the lockstep-Horner
   machinery (gather kernel + pointer-array NTT) collapses it to ~3. Matters
   from B≈32 up, where submission overhead stops hiding.
3. **Upload/compute pipelining** — overlap the ~22 MB/batch of H2D (b vectors,
   KSKs, RGSWs) and host-side marshalling with Stage 1; ~0.5–1 ms/query.
4. **`GpuRingEmbedHelper` init on GPU** — ~400 ms off preprocess (mono32 table
   generation still on CPU).
5. **3-limb RNS-native gadget** — algorithmic. Eliminates the per-step NTT chain in
   `collapse_fused_kernel`; ~25% steady-state precomp memory, ~2–3× collapse. Bigger
   change; requires retuning the ciphertext modulus.

(Single-query mat-vec vectorization was tried and lost — occupancy-starved;
see git history. The batched profile supersedes that analysis.)

## 10. Testing and correctness oracle

```
test_ring, test_crypto                     # CPU client-primitive unit tests
test_gpu_ntt, test_gpu_decomp,
test_gpu_extprod, test_gpu_horner          # bit-exact vs CPU primitive
test_e2e_gpu                               # Full pipeline, slot-exact; Small + Medium + 1 GB DB
test_gpu_batch                             # batch path bit-identical to B independent singles
test_capi                                  # C ABI facade round trip
```

(9 ctest tests. There is no standalone collapse test — collapse is covered
transitively by `test_e2e_gpu`'s slot-exact check.)

Run with:

```
ctest --test-dir build --output-on-failure
./build/test_e2e_gpu --4gb            # single size; or --1gb / --16gb
INSPIRE_DB_ROWS=65536 ./build/test_e2e_gpu --4gb   # override geometry
```

Profiling is in-situ on the real pipeline (no standalone microbenches):
`INSPIRE_PROFILE_PREPROCESS=1` for per-phase preprocess timing,
`GPU_ANSWER_TRACE=1` / `GPU_ANSWER_FINE=1` for per-phase online-query timing.

## 11. Benchmark contract

### Hardware-of-record

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB GDDR7 (sm_120) |
| CPU | Xeon Gold 6530 (primary) / AMD EPYC 7543 (cross-check, ±20% gate) |
| OS | Linux 6.8 / 6.14 |

All numbers in this doc are measured on these configurations; the two hosts
agree within the ±20% gate. Numbers measured on
any other GPU/host are informational only, not contracts.

### Performance gates

For a change to be declared a "win," it must:
1. **Not regress** the per-query latency or preprocess time at 1 GB DB.
2. Keep all 9 `ctest` tests passing.
3. Match or beat the cited "expected gain" within ±20%, on this hardware.

If an optimization doesn't reproduce its expected gain on sm_120, mention it in
the commit message and decide whether to keep, retune, or revert.

### Codebase-quality gates

A change is also evaluated on:
- **LOC delta** (smaller-is-better; dead code removal is encouraged).
- **Whether it introduces a new external dependency** (avoid unless justified).
- **Whether it maintains the GPU/CPU separation** (CPU is reference; GPU is hot).
- **Whether the test coverage stays meaningful** (every new kernel needs a
  bit-exact CPU oracle test or it lacks a baseline for regressions).

## 12. Open issues

- **Tensor-core mat-vec** is the top remaining online lever at scale — see §9
  item 1; §9 items 2–3 are the cheap follow-ups.
- **Docs drift:** this doc's Status/per-phase numbers are a snapshot — refresh
  them when the kernels or geometry change.
- The InsPIRe^(2) variant spec is a reviewed draft kept OUT of this repo
  (ask Keewoo for `scheme_description_inspire2.pdf`); its parameter script
  and any implementation are future work.
