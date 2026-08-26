// Test GPU gadget decomposition matches CPU decomp() bit-for-bit.
#include "check.h"
#include "ring.h"
#include "gpu.cuh"
#include <iostream>
#include <random>
#include <vector>

using namespace inspire;

int main() {
    std::cout << "=== GPU gadget_decomp vs CPU decomp ===" << std::endl;

    std::mt19937_64 rng(42);

    // Random RnsPoly in coefficient form
    RnsPoly p;
    p.is_ntt = false;
    for (size_t i = 0; i < N; i++) {
        p.limbs[0][i] = rng() % Q0;
        p.limbs[1][i] = rng() % Q1;
    }

    // CPU decompose
    auto cpu_digits = decomp(p);
    if (cpu_digits.size() != (size_t)D_EFF) {
        std::cerr << "CPU decomp returned " << cpu_digits.size()
                  << " digits, expected " << D_EFF << std::endl;
        return 1;
    }

    // Upload p to GPU as uint32
    std::vector<uint32_t> h_a0(N), h_a1(N);
    for (size_t i = 0; i < N; i++) {
        h_a0[i] = (uint32_t)p.limbs[0][i];
        h_a1[i] = (uint32_t)p.limbs[1][i];
    }
    uint32_t *d_a0, *d_a1, *d_out0, *d_out1;
    cudaMalloc(&d_a0, N * sizeof(uint32_t));
    cudaMalloc(&d_a1, N * sizeof(uint32_t));
    cudaMalloc(&d_out0, D_EFF * N * sizeof(uint32_t));
    cudaMalloc(&d_out1, D_EFF * N * sizeof(uint32_t));
    cudaMemcpy(d_a0, h_a0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a1, h_a1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    inspire::gpu::gpu_gadget_decomp(d_a0, d_a1, d_out0, d_out1,
                                    N, D_GSW, D_EFF, BASE_LOG);
    cudaDeviceSynchronize();

    // Download
    std::vector<uint32_t> h_out0(D_EFF * N), h_out1(D_EFF * N);
    cudaMemcpy(h_out0.data(), d_out0, D_EFF * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out1.data(), d_out1, D_EFF * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Compare
    int mm = 0;
    for (int j = 0; j < D_EFF; j++) {
        for (size_t i = 0; i < N; i++) {
            uint32_t cpu0 = (uint32_t)cpu_digits[j].limbs[0][i];
            uint32_t cpu1 = (uint32_t)cpu_digits[j].limbs[1][i];
            uint32_t gpu0 = h_out0[j * N + i];
            uint32_t gpu1 = h_out1[j * N + i];
            if (cpu0 != gpu0 || cpu1 != gpu1) {
                if (mm < 5) {
                    std::cout << "  MISMATCH at j=" << j << " i=" << i
                              << ": CPU=(" << cpu0 << "," << cpu1
                              << ") GPU=(" << gpu0 << "," << gpu1 << ")" << std::endl;
                }
                mm++;
            }
        }
    }
    std::cout << "Decomp test: " << (mm == 0 ? "PASS" : "FAIL")
              << " (" << mm << " mismatches out of " << D_EFF * N << ")" << std::endl;

    cudaFree(d_a0); cudaFree(d_a1); cudaFree(d_out0); cudaFree(d_out1);
    return mm == 0 ? 0 : 1;
}
