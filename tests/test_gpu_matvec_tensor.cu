#include "gpu.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_OK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        std::cerr << "CUDA failure: " << cudaGetErrorString(e_) << std::endl; \
        return 1; \
    } \
} while (0)

#define CUBLAS_OK(call) do { \
    cublasStatus_t s_ = (call); \
    if (s_ != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS failure: " << (int)s_ << std::endl; \
        return 1; \
    } \
} while (0)

using namespace inspire::gpu;

int main() {
    const size_t rows = std::getenv("INSPIRE_MATVEC_TEST_ROWS")
        ? std::strtoull(std::getenv("INSPIRE_MATVEC_TEST_ROWS"), nullptr, 10)
        : 2048;
    const size_t cols = std::getenv("INSPIRE_MATVEC_TEST_COLS")
        ? std::strtoull(std::getenv("INSPIRE_MATVEC_TEST_COLS"), nullptr, 10)
        : 2048;
    const int batch = std::getenv("INSPIRE_MATVEC_TEST_BATCH")
        ? std::atoi(std::getenv("INSPIRE_MATVEC_TEST_BATCH"))
        : 3;

    std::mt19937_64 rng(0x5090c0deULL);
    std::vector<uint16_t> db(rows * cols);
    for (auto& x : db) x = (uint16_t)(rng() % 65535u);
    const uint16_t edge_db[] = {0, 1, 127, 128, 255, 256, 32768, 65534};
    for (size_t i = 0; i < sizeof(edge_db) / sizeof(edge_db[0]); i++) db[i] = edge_db[i];

    std::vector<uint32_t> q0(batch * rows), q1(batch * rows);
    for (auto& x : q0) x = (uint32_t)(rng() % Q0);
    for (auto& x : q1) x = (uint32_t)(rng() % Q1);
    const uint32_t edge_q0[] = {0, 1, 255, 256, 65535, Q0 - 1};
    const uint32_t edge_q1[] = {0, 1, 255, 256, 65535, Q1 - 1};
    for (int b = 0; b < batch; b++) {
        for (size_t i = 0; i < sizeof(edge_q0) / sizeof(edge_q0[0]); i++) {
            q0[(size_t)b * rows + i] = edge_q0[i];
            q1[(size_t)b * rows + i] = edge_q1[i];
        }
    }

    uint16_t* d_db = nullptr;
    int32_t *d_db_sums = nullptr, *d_q_sums = nullptr, *d_gemm = nullptr;
    int8_t* d_q_planes = nullptr;
    uint32_t *d_q0 = nullptr, *d_q1 = nullptr, *d_r0 = nullptr, *d_r1 = nullptr;
    const uint32_t **d_q0_ptrs = nullptr, **d_q1_ptrs = nullptr;
    uint32_t **d_r0_ptrs = nullptr, **d_r1_ptrs = nullptr;

    CUDA_OK(cudaMalloc(&d_db, db.size() * sizeof(uint16_t)));
    CUDA_OK(cudaMemcpy(d_db, db.data(), db.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_db_sums, 2 * cols * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&d_q0, q0.size() * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_q1, q1.size() * sizeof(uint32_t)));
    CUDA_OK(cudaMemcpy(d_q0, q0.data(), q0.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_q1, q1.data(), q1.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_r0, (size_t)batch * cols * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_r1, (size_t)batch * cols * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_q_planes, (size_t)batch * 8 * rows * sizeof(int8_t)));
    CUDA_OK(cudaMalloc(&d_q_sums, (size_t)batch * 8 * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&d_gemm, (size_t)batch * 8 * 2 * cols * sizeof(int32_t)));

    std::vector<const uint32_t*> h_q0_ptrs(batch), h_q1_ptrs(batch);
    std::vector<uint32_t*> h_r0_ptrs(batch), h_r1_ptrs(batch);
    for (int b = 0; b < batch; b++) {
        h_q0_ptrs[b] = d_q0 + (size_t)b * rows;
        h_q1_ptrs[b] = d_q1 + (size_t)b * rows;
        h_r0_ptrs[b] = d_r0 + (size_t)b * cols;
        h_r1_ptrs[b] = d_r1 + (size_t)b * cols;
    }
    CUDA_OK(cudaMalloc(&d_q0_ptrs, batch * sizeof(uint32_t*)));
    CUDA_OK(cudaMalloc(&d_q1_ptrs, batch * sizeof(uint32_t*)));
    CUDA_OK(cudaMalloc(&d_r0_ptrs, batch * sizeof(uint32_t*)));
    CUDA_OK(cudaMalloc(&d_r1_ptrs, batch * sizeof(uint32_t*)));
    CUDA_OK(cudaMemcpy((void*)d_q0_ptrs, h_q0_ptrs.data(), batch * sizeof(uint32_t*), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy((void*)d_q1_ptrs, h_q1_ptrs.data(), batch * sizeof(uint32_t*), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_r0_ptrs, h_r0_ptrs.data(), batch * sizeof(uint32_t*), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_r1_ptrs, h_r1_ptrs.data(), batch * sizeof(uint32_t*), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_OK(cublasCreate(&handle));
    CUBLAS_OK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    gpu_matvec_tensor_prepare_db(d_db, d_db_sums, rows, cols);
    gpu_matvec_tensor_batched(
        d_r0_ptrs, d_r1_ptrs, d_db, d_db_sums,
        d_q0_ptrs, d_q1_ptrs, d_q_planes, d_q_sums, d_gemm,
        rows, cols, Q0, Q1, batch, false, handle);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<uint32_t> got0((size_t)batch * cols), got1((size_t)batch * cols);
    CUDA_OK(cudaMemcpy(got0.data(), d_r0, got0.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(got1.data(), d_r1, got1.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int failures = 0;
    const bool sample_only = std::getenv("INSPIRE_MATVEC_TEST_SAMPLE_ONLY") != nullptr;
    std::vector<size_t> checked_cols;
    if (sample_only) {
        checked_cols = {0, 1, 17, cols / 4, cols / 2, cols - 2, cols - 1};
    } else {
        checked_cols.resize(cols);
        for (size_t j = 0; j < cols; j++) checked_cols[j] = j;
    }
    for (int b = 0; b < batch; b++) {
        for (size_t j : checked_cols) {
            uint64_t a0 = 0, a1 = 0;
            for (size_t i = 0; i < rows; i++) {
                uint64_t d = db[i * cols + j];
                a0 += d * q0[(size_t)b * rows + i];
                a1 += d * q1[(size_t)b * rows + i];
                if ((i & 0xfff) == 0xfff) { a0 %= Q0; a1 %= Q1; }
            }
            uint32_t e0 = (uint32_t)(a0 % Q0), e1 = (uint32_t)(a1 % Q1);
            size_t out = (size_t)b * cols + j;
            if (got0[out] != e0 || got1[out] != e1) {
                if (failures++ < 8)
                    std::cerr << "mismatch b=" << b << " j=" << j
                              << " got=(" << got0[out] << ',' << got1[out]
                              << ") expected=(" << e0 << ',' << e1 << ")\n";
            }
        }
    }

    cublasDestroy(handle);
    cudaFree(d_db); cudaFree(d_db_sums);
    cudaFree(d_q0); cudaFree(d_q1); cudaFree(d_r0); cudaFree(d_r1);
    cudaFree((void*)d_q0_ptrs); cudaFree((void*)d_q1_ptrs);
    cudaFree(d_r0_ptrs); cudaFree(d_r1_ptrs);
    cudaFree(d_q_planes); cudaFree(d_q_sums); cudaFree(d_gemm);

    if (failures) return 1;
    std::cout << "Tensor mat-vec is bit-exact for " << batch
              << " queries x " << checked_cols.size() << '/' << cols
              << " checked outputs." << std::endl;
    return 0;
}
