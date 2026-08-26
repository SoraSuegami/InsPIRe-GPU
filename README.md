# inspire-gpu

CUDA server implementation of the **InsPIRe**[^1] PIR scheme, with a C++
client, a batch serving API, and a C ABI. The server pipeline (preprocess +
answer) runs entirely on the GPU; the client side (query, extract) is plain
CPU code. The public parameters (database geometry and the CRS seed) come
from a one-time `setup()` whose output both sides share.

[^1]: Rasoul Akhavan Mahdavi, Sarvar Patel, Joon Young Seo, and Kevin Yeo.
    *InsPIRe: Communication-Efficient PIR with Server-side Preprocessing.*
    IEEE S&P 2026. [ePrint 2025/1352](https://eprint.iacr.org/2025/1352).

## Disclaimer

The author is a cryptographer working on PIR but with only a high-level
understanding of GPU architecture. The CUDA code is entirely
vibe-coded, written by an AI from high-level instructions; the author's
best effort went into anchoring it, through the specs in `docs/` and the
bit-exact test suite. Privacy is the exception: in a doubly-stateless PIR
it rests solely on the query the client sends, and the client side is
simple, CPU-only code, which the author could and did review.

## Why InsPIRe?

**Single server.** No non-collusion assumption.

**Doubly-stateless[^2]: neither side keeps state for the other.** Beyond
keeping the API clean, this buys the following:

- **Stateless client.** Client has no database-dependent hint to maintain; a
  cold client can query immediately. With a database-dependent client-side
  hint, every update makes it stale and entails per-client interaction and
  work.
- **Stateless server.** Server stores no per-client state such as key
  material. Stored per-client keys make every query linkable to its client,
  so anonymity becomes unachievable in principle, even behind tools like
  Tor.

**Mild server-memory blowup (under our parameter choices).** The database is
not stored NTT-transformed over the ciphertext modulus. The resident
footprint is about 1.6× the plaintext database: a 16 GB database serves from
25.8 GB of VRAM on a single 32 GB RTX 5090, whereas schemes that need the
NTT-precomputed database usually pay around 4×.

**Preprocessing, the usual objection, is manageable.** InsPIRe pays for the
properties above with a relatively heavy server-side preprocess. On a GPU it
drops to seconds (the Preprocessing column below), and the remaining refresh delay
can be hidden by the sidecar pattern[^2]: a small secondary instance — as
simple as a plain broadcast — absorbs
fresh updates while the main engine serves the previous snapshot, and
periodically the accumulated updates are folded in by re-preprocessing the
main database. Serving stays current with zero downtime.

[^2]: Ali Atiia and Keewoo Lee.
    [*Sharded PIR Design for the Ethereum State.*](https://ethresear.ch/t/sharded-pir-design-for-the-ethereum-state/24552)
    ethresear.ch, 2026. The sidecar pattern is Section 5.3.

## Scope

This is the PIR engine, not the service around it. The following are the
caller's (or a wrapper repo's) responsibility, by design:

- **Keyword lookup.** The API retrieves entries **by position**. Mapping keys
  to positions (for Ethereum state: address → index, e.g. via cuckoo
  hashing) is an outer layer.
- **Database updates.** A refresh is: run `gpu_preprocess` on the new
  snapshot (seconds; see the Preprocessing column), stand up a new context, swap,
  free the old one. Zero-downtime orchestration on top of that swap is the
  outer layer's job.
- **Networking.** In-process C++/C API only; transport is the caller's.
  The wire encodings themselves are provided in `capi.h` (see the API
  section).

Deferred, not implemented: tensor-core mat-vec is the main remaining speed
lever.

## API

C++ consumers use two headers; FFI consumers use one, `capi.h` (below).
Everything else is an implementation detail.

| Header | Side | What it exposes |
|---|---|---|
| [`src/protocol.h`](src/protocol.h) | **Client** (CPU) | `query()` builds the encrypted query; `extract()` decrypts the response to the retrieved entry's plaintext slots. |
| [`src/gpu_protocol.h`](src/gpu_protocol.h) | **Server** (GPU) | `gpu_preprocess()` → `gpu_setup_server()` → `gpu_answer()` (per query) → `gpu_free_server()`. |

### Typical flow

```cpp
PublicParams pp = setup(N_entries, w);              // shared: both sides use the same pp

// Server, one-time:
PreprocessData precomp = gpu_preprocess(pp, db);  // db: db_rows*db_cols values in [0,P), row-major
GpuServerCtx*  ctx     = gpu_setup_server(pp, precomp);

// Per query:
auto [qst, qry]   = query(pp, idx);                 // client
auto resp         = gpu_answer(ctx, qry);           // server
auto slots        = extract(pp, qst, resp);         // client → the entry's slots in [0,P)

gpu_free_server(ctx);
```

### The `db` buffer

The logical database is `N_entries` entries of `w` bytes each; `db` is
that database laid out as a row-major `u16` array of
`db_rows × db_cols` values, each in `[0, P)` (`P = 65535`), with `db_rows`
and `db_cols` computed by `setup()`. Each entry occupies
`ceil(8w/15)` consecutive values (15 payload bits per value); packing an
entry's bytes into its values is the caller's responsibility. Entry `i`
sits in row `i mod db_rows`, starting at column
`(i div db_rows) * ceil(8w/15)`; columns past the last entry are padding
and do not affect any other entry's result.

### Batched serving

Queries from independent clients batched together share the two big data
passes (one over the database, one over the
precomputed tensors) and advance their polynomial evaluations in step:
from 32 to 115 q/s at 16 GB on one card (see the benchmark table). Per-query
scratch buffers are pre-allocated at setup (`cfg.max_batch`); nothing is
allocated on the request path, and `gpu_server_caps` publishes the limits
for the caller's scheduler:

```cpp
GpuServerConfig cfg;  cfg.max_batch = 32;
GpuServerCtx* ctx = gpu_setup_server(pp, precomp, cfg);
GpuServerCaps caps = gpu_server_caps(ctx);          // max_batch, resident_bytes, free bytes

std::vector<QueryMessage> qs = ...;                 // one per client; keys are per query
auto resps = gpu_answer_batch(ctx, qs.data(), qs.size());
```

Results are bit-identical to independent `gpu_answer` calls.

### C ABI

For FFI consumers (a Rust wrapper, for example), [`src/capi.h`](src/capi.h)
is the entire interface: server and client over `extern "C"`, plus the wire
encodings, lossless query packing (371 KB) and response compression by
modulus switching (12 KB per ciphertext, or 26.5 KB lossless). The client
half lives in the CPU library, so a client-only consumer links no CUDA.

`ipir_server_create` takes an optional 64-byte CRS seed (NULL draws one
fresh). Pinning the seed keeps queries valid across database rebuilds and
lets several servers — replicas, or the two sides of a rolling swap —
accept the same queries; the CRS is public randomness, so privacy rests on
the fresh per-query secret either way.

## Benchmark results

Measured warm on one RTX 5090 (platform details below), 120-byte entries
at the default geometry `db_rows = 32768`.

### Single query

Median of 5 trials:

| Database | Number of entries | Per-query latency | Communication (↑query + ↓resp) | Resident memory (precomp + DB) | Preprocessing |
|---|---|---|---|---|---|
| **1 GB**  | 2²³ | **~2.6 ms** | 383 KB (371 + 12) | 1.61 GB  | ~3.3 s |
| **4 GB**  | 2²⁵ | **~7.9 ms** | 383 KB             | 6.44 GB  | ~4.0 s |
| **16 GB** | 2²⁷ | **~31 ms**  | 383 KB             | 25.77 GB | ~7.7 s |

### Batched throughput

`gpu_answer_batch` (median of 3), 16 GB database:

| Batch | Batch latency | Per-query | Throughput |
|---|---|---|---|
| B=1  | 31.1 ms  | 31.1 ms | 32 q/s |
| B=2  | 34.4 ms  | 17.2 ms | 58 q/s |
| B=4  | 44.4 ms  | 11.1 ms | 90 q/s |
| B=8  | 76.7 ms  | 9.6 ms  | 104 q/s |
| B=16 | 143.4 ms | 9.0 ms  | 112 q/s |
| B=32 | 278.5 ms | 8.7 ms  | **115 q/s** |

Batching reaches **258 q/s** at 4 GB and **579 q/s** at 1 GB. All sizes fit
and run on a single 32 GB card (peak 29.4 GB at 16 GB). The remaining
per-query wall is the mat-vec stage, about 70% of the batched per-query
cost; tensor-core INT8 mat-vec is the main untapped lever.

Client-side costs are CPU-only: building a query takes ~31 ms, and packing,
compression, and extraction each cost a few milliseconds or less.

### Correctness margin

Measured by decrypting real responses (part of `./build/bench_e2e`): at
16 GB the response noise has sigma = 2^32.6 (largest
observed |e|: 2^34.6) against the decode threshold Delta/2 = 2^36, an
estimated failure probability of about 2^-78 per response; the q′ response
compression adds rounding noise for a combined ~2^-53, still past the
2^-40 design target. Sigma scales as the square root of the database size,
matching the additive-noise analysis; noise is not a binding constraint at
any size that fits the card.

## Tested platform

| Component | Spec |
|---|---|
| GPU | **NVIDIA RTX 5090**, 32 GB GDDR7 (Blackwell, **sm_120**) |
| CPU | Xeon Gold 6530 |
| OS  | Linux 6.8 (Ubuntu 24.04) |
| Toolchain | CUDA 12.9 and 13.3 (both tested), C++17, OpenSSL (SHAKE-256) |

`CMAKE_CUDA_ARCHITECTURES` defaults to `120`. Override for other GPUs, e.g.
`-DCMAKE_CUDA_ARCHITECTURES=86`.

The client half is plain C++17 + OpenSSL and is portable:
`-DINSPIRE_CLIENT_ONLY=ON` builds it, with its tests, on machines without
CUDA, including macOS/ARM. It has been exercised from a Mac against a
remote GPU server.

## Build & run

The only external dependency is **OpenSSL** (SHAKE-256 for CRS seed
expansion); the NTT is implemented in this repo, not an external library.

```bash
cmake -S . -B build && cmake --build build -j

# Client-only build, for machines without CUDA (CPU library + its tests):
cmake -S . -B build-cpu -DINSPIRE_CLIENT_ONLY=ON && cmake --build build-cpu -j

# Run the test suite (CPU unit + GPU bit-exact-vs-CPU + end-to-end):
ctest --test-dir build --output-on-failure

# Benchmark (1/4/16 GB): latency, batch sweep, noise margin, wire timings:
./build/bench_e2e
./build/bench_e2e 1 4                     # subset, by GB size
INSPIRE_DB_ROWS=65536 ./build/bench_e2e   # geometry override (a power of two)
INSPIRE_ENTRY_BYTES=60 ./build/bench_e2e  # entry size (default 120; see below)
INSPIRE_MAX_BATCH=8 ./build/bench_e2e     # cap the batch sweep (default 32)
```

Entry sizes that fill their slots exactly are 15, 30, 60, 120, 240, ...
bytes; pad records up to the next such size.

## File structure

```
inspire-gpu/
├── CMakeLists.txt
├── src/
│   ├── params.{h,cpp}        # scheme constants + setup() → PublicParams
│   ├── ring.{h,cpp}          # RnsPoly, automorphism, gadget decomp, sampling
│   ├── ntt.{h,cpp}           # scalar negacyclic NTT, implemented here
│   ├── crypto.{h,cpp}        # sk_gen, rlwe_dec, lwe/ksk/rgsw enc, ext_prod, horner
│   ├── protocol.{h,cpp}      # CLIENT API: query, extract
│   ├── capi.h                # C ABI (ipir_*) — declarations for both halves
│   ├── capi.cu               # server half (GPU library)
│   ├── capi_client.cpp       # client half + wire packing/compression (CPU-only)
│   ├── gpu.cuh               # public CUDA kernel API (sectioned by phase)
│   ├── gpu_common.cuh        # shared __device__ helpers (mod/Barrett/NTT)
│   ├── gpu_primitives.cu     # context, NTT, poly ops, gadget decomp, memory
│   ├── gpu_preprocess.cu     # encode, a-side matmul, ring_embed, collapse_fused
│   ├── gpu_online.cu         # matvec, lazy_collapse, ext_prod, horner
│   └── gpu_protocol.{h,cu}   # SERVER API: preprocess, setup_server, answer(+_batch), caps
├── tests/                    # ctest suite (CPU unit + GPU bit-exact + e2e)
├── benches/
│   └── bench_e2e.cu          # latency, batch sweep, noise margin, wire timings
└── docs/
    ├── ARCHITECTURE.md       # engineering notes
    └── scheme_description.{tex,pdf}  # scheme notes
```

## Parameters

The parameters ([`src/params.h`](src/params.h)) are set for 128-bit
security: ring degree N = 2048, log₂ q ≤ 53, σ = 3.19. Verified with the
[lattice estimator](https://github.com/malb/lattice-estimator) at commit
`53da598`.

The database geometry (`db_rows`) is the one free knob. It trades
communication against computation and server memory: the query carries one
value per row, so shrinking `db_rows` shrinks the query, but the same
database then spreads over more columns, and per-query computation and
precomputed data both grow with the column count. We consider
communication the scarce resource, especially with a GPU doing the
computation, yet it cannot be pushed arbitrarily low for exactly that
reason; `db_rows = 2^15` is the point where the precomputed data is about
half the database itself, which we judged the right memory blowup.

