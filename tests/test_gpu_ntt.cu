#include "ring.h"
#include "ntt.h"
#include "gpu.cuh"
#include "check.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cstring>

using namespace inspire;
// Note: don't "using namespace inspire::gpu" to avoid ambiguity with N, Q0, Q1

int main() {
    std::cout << "=== GPU NTT Correctness Test ===" << std::endl;

    // Extract the CPU NTT twiddle factors
    std::vector<uint32_t> fwd0, inv0, fwd1, inv1;
    uint32_t inv_n0, inv_n1;
    get_ntt_twiddles(0, fwd0, inv0, inv_n0);
    get_ntt_twiddles(1, fwd1, inv1, inv_n1);

    std::cout << "Twiddles extracted (in-tree NTT):" << std::endl;
    std::cout << "  Q0=" << Q0 << ": fwd[1]=" << fwd0[1] << " inv[1]=" << inv0[1]
              << " inv_n=" << inv_n0 << std::endl;
    std::cout << "  Q1=" << Q1 << ": fwd[1]=" << fwd1[1] << " inv[1]=" << inv1[1]
              << " inv_n=" << inv_n1 << std::endl;

    // Upload twiddles to GPU
    uint32_t *d_fwd0, *d_inv0;
    cudaMalloc(&d_fwd0, N * sizeof(uint32_t));
    cudaMalloc(&d_inv0, N * sizeof(uint32_t));
    cudaMemcpy(d_fwd0, fwd0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv0, inv0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Test 1: Forward NTT
    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_poly_cpu(N);
    std::vector<uint32_t> h_poly_gpu(N);
    for (size_t i = 0; i < N; i++) {
        uint32_t v = rng() % Q0;
        h_poly_cpu[i] = v;
        h_poly_gpu[i] = v;
    }

    // CPU NTT (in-tree)
    ntt_forward(h_poly_cpu.data(), 0);

    // GPU NTT
    uint32_t* d_poly;
    cudaMalloc(&d_poly, N * sizeof(uint32_t));
    cudaMemcpy(d_poly, h_poly_gpu.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    inspire::gpu::gpu_ntt_forward(d_poly, N, Q0, d_fwd0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_poly_gpu.data(), d_poly, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Compare
    int mismatches = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_poly_cpu[i] != h_poly_gpu[i]) {
            if (mismatches < 5) {
                std::cout << "  MISMATCH at " << i << ": CPU=" << h_poly_cpu[i]
                          << " GPU=" << h_poly_gpu[i] << std::endl;
            }
            mismatches++;
        }
    }

    INSPIRE_CHECK(mismatches == 0, "Test 1 (Forward NTT)");
    std::cout << "Test 1 (Forward NTT): " << (mismatches == 0 ? "PASS" : "FAIL")
              << " (" << mismatches << " mismatches)" << std::endl;

    // Test 2a: GPU inverse vs CPU inverse
    // Take the GPU forward NTT result and apply both inversions
    std::vector<uint64_t> ntt_result_cpu(N);
    std::vector<uint32_t> ntt_result_gpu(N);
    for (size_t i = 0; i < N; i++) {
        uint32_t v = rng() % Q0;
        h_poly_gpu[i] = v;
        h_poly_cpu[i] = v;
    }
    // Do CPU forward + CPU inverse (should be identity)
    ntt_forward(h_poly_cpu.data(), 0);
    for (size_t i = 0; i < N; i++) ntt_result_cpu[i] = h_poly_cpu[i];
    ntt_inverse(h_poly_cpu.data(), 0);

    // Do GPU forward + GPU inverse
    cudaMemcpy(d_poly, h_poly_gpu.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    inspire::gpu::gpu_ntt_forward(d_poly, N, Q0, d_fwd0);

    // Check GPU forward matches CPU forward
    cudaDeviceSynchronize();
    cudaMemcpy(ntt_result_gpu.data(), d_poly, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    inspire::gpu::gpu_ntt_inverse(d_poly, N, Q0, d_inv0, inv_n0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_poly_gpu.data(), d_poly, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    mismatches = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_poly_cpu[i] != h_poly_gpu[i]) {
            if (mismatches < 3)
                std::cout << "  INV MISMATCH at " << i << ": CPU=" << h_poly_cpu[i]
                          << " GPU=" << h_poly_gpu[i] << std::endl;
            mismatches++;
        }
    }
    INSPIRE_CHECK(mismatches == 0, "Test 2 (INTT round-trip)");
    std::cout << "Test 2 (INTT round-trip): " << (mismatches == 0 ? "PASS" : "FAIL")
              << " (" << mismatches << " mismatches)" << std::endl;

    // (former Test 2c tested an identity, "forward kernel with inverse
    // twiddles plus n^{-1} scale equals INTT", that only holds for the old
    // CPU library's bit-reversal layout. Our convention uses proper GS
    // butterflies for INTT (Test 2 covers correctness), so 2c was removed.)

    // Test 2b: Apply CPU inverse to GPU-forward result
    std::vector<uint64_t> cpu_inv(N);
    for (size_t i = 0; i < N; i++) cpu_inv[i] = ntt_result_gpu[i];
    ntt_inverse(cpu_inv.data(), 0);

    mismatches = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_poly_cpu[i] != (uint32_t)cpu_inv[i]) mismatches++;
    }
    INSPIRE_CHECK(mismatches == 0, "Test 2b (CPU-inv of GPU-fwd)");
    std::cout << "Test 2b (CPU-inv of GPU-fwd): " << (mismatches == 0 ? "PASS" : "FAIL")
              << " (" << mismatches << " mismatches)" << std::endl;

    // Test 3: Batch NTT (256 polynomials)
    size_t batch = 256;
    std::vector<uint64_t> h_batch_cpu(batch * N);
    std::vector<uint32_t> h_batch_gpu(batch * N);
    for (size_t i = 0; i < batch * N; i++) {
        uint32_t v = rng() % Q0;
        h_batch_cpu[i] = v;
        h_batch_gpu[i] = v;
    }

    // CPU batch NTT
    for (size_t b = 0; b < batch; b++)
        ntt_forward(h_batch_cpu.data() + b * N, 0);

    // GPU batch NTT
    uint32_t* d_batch;
    cudaMalloc(&d_batch, batch * N * sizeof(uint32_t));
    cudaMemcpy(d_batch, h_batch_gpu.data(), batch * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    inspire::gpu::gpu_ntt_batch(d_batch, N, batch, Q0, d_fwd0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_batch_gpu.data(), d_batch, batch * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    mismatches = 0;
    for (size_t i = 0; i < batch * N; i++) {
        if (h_batch_cpu[i] != h_batch_gpu[i]) mismatches++;
    }
    INSPIRE_CHECK(mismatches == 0, "Test 3 (Batch NTT)");
    std::cout << "Test 3 (Batch NTT " << batch << "): " << (mismatches == 0 ? "PASS" : "FAIL")
              << " (" << mismatches << " mismatches)" << std::endl;

    // Benchmark: batch NTT timing
    std::cout << "\n=== GPU Batch NTT Benchmark ===" << std::endl;
    for (size_t bs : {1, 64, 128, 256, 512}) {
        uint32_t* d_bench;
        cudaMalloc(&d_bench, bs * N * sizeof(uint32_t));
        // Fill with random data
        std::vector<uint32_t> h_bench(bs * N);
        for (auto& v : h_bench) v = rng() % Q0;
        cudaMemcpy(d_bench, h_bench.data(), bs * N * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Warmup
        inspire::gpu::gpu_ntt_batch(d_bench, N, bs, Q0, d_fwd0);
        cudaDeviceSynchronize();

        // Reset + benchmark
        cudaMemcpy(d_bench, h_bench.data(), bs * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        auto t0 = std::chrono::high_resolution_clock::now();
        inspire::gpu::gpu_ntt_batch(d_bench, N, bs, Q0, d_fwd0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  Batch " << bs << " x " << N << ": " << ms << " ms ("
                  << ms / bs * 1000 << " us/poly)" << std::endl;
        cudaFree(d_bench);
    }

    cudaFree(d_fwd0); cudaFree(d_inv0);
    cudaFree(d_poly); cudaFree(d_batch);

    return inspire_test_status();
}
