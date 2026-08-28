#include "gpu.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

#define CUDA_OK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        std::cerr << "CUDA failure: " << cudaGetErrorString(e_) << std::endl; \
        return 1; \
    } \
} while (0)

using namespace inspire::gpu;

int main() {
    constexpr size_t rows = TENSOR_MATVEC_MAX_ROWS + 1;
    constexpr size_t cols = N;
    constexpr int batch = 3;
    static_assert(rows == 131072,
                  "the adversarial boundary must make 16384*rows overflow int32");

    uint16_t* d_db = nullptr;
    int32_t* d_db_sums = nullptr;
    uint32_t *d_q0 = nullptr, *d_q1 = nullptr, *d_out = nullptr;
    const uint32_t **d_q0_ptrs = nullptr, **d_q1_ptrs = nullptr;
    uint32_t **d_out0_ptrs = nullptr, **d_out1_ptrs = nullptr;

    const size_t db_words = rows * cols;
    CUDA_OK(cudaMalloc(&d_db, db_words * sizeof(uint16_t)));
    CUDA_OK(cudaMemset(d_db, 0, db_words * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_db_sums, 2 * cols * sizeof(int32_t)));
    gpu_matvec_tensor_prepare_db(d_db, d_db_sums, rows, cols);

    CUDA_OK(cudaMalloc(&d_q0, (size_t)batch * rows * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_q1, (size_t)batch * rows * sizeof(uint32_t)));
    CUDA_OK(cudaMemset(d_q0, 0, (size_t)batch * rows * sizeof(uint32_t)));
    CUDA_OK(cudaMemset(d_q1, 0, (size_t)batch * rows * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_out, (size_t)batch * 2 * cols * sizeof(uint32_t)));
    CUDA_OK(cudaMemset(d_out, 0xff, (size_t)batch * 2 * cols * sizeof(uint32_t)));

    std::vector<const uint32_t*> q0(batch), q1(batch);
    std::vector<uint32_t*> out0(batch), out1(batch);
    for (int b = 0; b < batch; b++) {
        q0[b] = d_q0 + (size_t)b * rows;
        q1[b] = d_q1 + (size_t)b * rows;
        out0[b] = d_out + (size_t)b * 2 * cols;
        out1[b] = out0[b];
    }
    const size_t ptr_bytes = batch * sizeof(uint32_t*);
    CUDA_OK(cudaMalloc(&d_q0_ptrs, ptr_bytes));
    CUDA_OK(cudaMalloc(&d_q1_ptrs, ptr_bytes));
    CUDA_OK(cudaMalloc(&d_out0_ptrs, ptr_bytes));
    CUDA_OK(cudaMalloc(&d_out1_ptrs, ptr_bytes));
    CUDA_OK(cudaMemcpy((void*)d_q0_ptrs, q0.data(), ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy((void*)d_q1_ptrs, q1.data(), ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_out0_ptrs, out0.data(), ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_out1_ptrs, out1.data(), ptr_bytes, cudaMemcpyHostToDevice));

    gpu_matvec_centered_packed_batched(
        d_out0_ptrs, d_out1_ptrs, reinterpret_cast<const int8_t*>(d_db),
        d_q0_ptrs, d_q1_ptrs, rows, cols, Q0, Q1, batch);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<uint32_t> got((size_t)batch * 2 * cols);
    CUDA_OK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
    for (uint32_t x : got) {
        if (x != 0) {
            std::cerr << "overflow fallback mismatch: expected zero, got " << x << std::endl;
            return 1;
        }
    }

    cudaFree(d_db); cudaFree(d_db_sums);
    cudaFree(d_q0); cudaFree(d_q1); cudaFree(d_out);
    cudaFree((void*)d_q0_ptrs); cudaFree((void*)d_q1_ptrs);
    cudaFree(d_out0_ptrs); cudaFree(d_out1_ptrs);
    std::cout << "Exact scalar fallback passed at the INT8 overflow boundary." << std::endl;
    return 0;
}
