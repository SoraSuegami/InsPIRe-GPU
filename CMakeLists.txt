cmake_minimum_required(VERSION 3.20)

# -DINSPIRE_CLIENT_ONLY=ON builds just the CPU library (client API + wire
# encodings) and its tests — no CUDA toolchain required, so it configures on
# any machine, including macOS/ARM.
option(INSPIRE_CLIENT_ONLY "Build the CPU client library and tests only (no CUDA)" OFF)

project(inspire-gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" INSPIRE_HAVE_MARCH_NATIVE)
if(INSPIRE_HAVE_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

if(NOT INSPIRE_CLIENT_ONLY)
  # Default CUDA target: sm_120 (RTX 5090 / Blackwell consumer). Override with
  # -DCMAKE_CUDA_ARCHITECTURES=86 (or any list) when targeting other GPUs.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120")
  endif()
  enable_language(CUDA)
endif()

find_package(OpenSSL REQUIRED)

# Core library. The NTT lives in src/ntt.{h,cpp} as portable scalar C++,
# so the library builds on ARM as well as x86.
add_library(inspire
  src/params.cpp
  src/ntt.cpp
  src/ring.cpp
  src/crypto.cpp
  src/protocol.cpp
  src/capi_client.cpp   # client half of the C ABI (no CUDA; see capi.h)
)
target_include_directories(inspire PUBLIC src)
target_link_libraries(inspire PUBLIC OpenSSL::Crypto)

# GPU library
if(NOT INSPIRE_CLIENT_ONLY)
add_library(inspire_gpu
    src/gpu_primitives.cu      # NTT, decomp, poly ops, mem/init (phase-agnostic)
    src/gpu_preprocess.cu      # encode, a-side matmul, ring_embed, automorphism, collapse_fused
    src/gpu_online.cu          # matvec, lazy_collapse, ext_prod, horner, fused_ip_sub
    src/gpu_protocol.cu        # orchestration: GpuServerCtx, gpu_preprocess/setup_server/answer(_batch)
    src/capi.cu)               # C ABI facade for FFI consumers (see src/capi.h)
target_include_directories(inspire_gpu PUBLIC src)
target_link_libraries(inspire_gpu PUBLIC inspire)
endif()

# ------------------------------------------------------------------
# Tests (registered with CTest; `ctest --output-on-failure` to run all)
# ------------------------------------------------------------------
enable_testing()

function(inspire_add_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} inspire ${ARGN})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# CPU unit tests (exercise the client-side primitives used by query()/extract()).
inspire_add_test(test_ring    tests/test_ring.cpp)
inspire_add_test(test_crypto  tests/test_crypto.cpp)

# GPU per-kernel tests (bit-exact vs CPU oracle)
if(NOT INSPIRE_CLIENT_ONLY)
inspire_add_test(test_gpu_ntt        tests/test_gpu_ntt.cu        inspire_gpu)
inspire_add_test(test_gpu_decomp     tests/test_gpu_decomp.cu     inspire_gpu)
inspire_add_test(test_gpu_extprod    tests/test_gpu_extprod.cu    inspire_gpu)
inspire_add_test(test_gpu_horner     tests/test_gpu_horner.cu     inspire_gpu)
inspire_add_test(test_e2e_gpu        tests/test_e2e_gpu.cu        inspire_gpu)
inspire_add_test(test_gpu_batch      tests/test_gpu_batch.cu      inspire_gpu)
inspire_add_test(test_capi           tests/test_capi.cu           inspire_gpu)

# ------------------------------------------------------------------
# Benchmark (not a ctest — manual perf run). One binary plus the env-gated
# traces (INSPIRE_PROFILE_PREPROCESS / GPU_ANSWER_TRACE / _FINE) is the whole
# profiling toolkit: per-size latency, batch sweep, measured noise margin,
# and wire-path CPU timings.
add_executable(bench_e2e     benches/bench_e2e.cu)
target_include_directories(bench_e2e PRIVATE src)
target_link_libraries(bench_e2e inspire inspire_gpu)

endif()
