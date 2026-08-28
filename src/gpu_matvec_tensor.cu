// Tensor-core online mat-vec.  The encoded u16 database is reinterpreted in
// place as an interleaved pair of centered INT8 digit planes.  In column-major
// terms this is a (2*db_cols) x db_rows matrix, so one INT8 GEMM evaluates all
// four byte planes of both RNS query limbs for every query in a batch.

#include "gpu_common.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", \
                     __FILE__, __LINE__, (int)status_); \
        std::exit(1); \
    } \
} while (0)

namespace inspire {
namespace gpu {

__global__ void center_db_bytes_kernel(uint8_t* db_bytes, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        uint8_t u = db_bytes[i];
        reinterpret_cast<int8_t*>(db_bytes)[i] = (int8_t)((int)u - 128);
    }
}

__global__ void sum_centered_db_rows_kernel(const int8_t* db,
                                            int32_t* sums,
                                            size_t db_rows,
                                            size_t byte_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= byte_cols) return;
    int32_t acc = 0;
    for (size_t i = 0; i < db_rows; i++)
        acc += (int32_t)db[i * byte_cols + j];
    sums[j] = acc;
}

void gpu_matvec_tensor_prepare_db(uint16_t* d_db_rm, int32_t* d_byte_sums,
                                  size_t db_rows, size_t db_cols) {
    const size_t byte_cols = 2 * db_cols;
    const size_t byte_count = db_rows * byte_cols;
    const size_t threads = 256;
    center_db_bytes_kernel<<<(byte_count + threads - 1) / threads, threads>>>(
        reinterpret_cast<uint8_t*>(d_db_rm), byte_count);
    sum_centered_db_rows_kernel<<<(byte_cols + threads - 1) / threads, threads>>>(
        reinterpret_cast<const int8_t*>(d_db_rm), d_byte_sums,
        db_rows, byte_cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void decompose_query_bytes_kernel(
        const uint32_t* const* q0_ptrs,
        const uint32_t* const* q1_ptrs,
        int8_t* planes, size_t db_rows, int batch) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int plane_col = (int)blockIdx.y;
    if (i >= db_rows || plane_col >= batch * 8) return;
    int b = plane_col / 8;
    int rem = plane_col % 8;
    int limb = rem / 4;
    int digit = rem % 4;
    uint32_t v = limb == 0 ? q0_ptrs[b][i] : q1_ptrs[b][i];
    uint32_t u = (v >> (8 * digit)) & 0xffu;
    planes[(size_t)plane_col * db_rows + i] = (int8_t)((int)u - 128);
}

__global__ void sum_query_planes_kernel(const int8_t* planes,
                                        int32_t* sums,
                                        size_t db_rows,
                                        int plane_cols) {
    int col = (int)blockIdx.x;
    if (col >= plane_cols) return;
    int32_t local = 0;
    const int8_t* p = planes + (size_t)col * db_rows;
    for (size_t i = threadIdx.x; i < db_rows; i += blockDim.x)
        local += (int32_t)p[i];

    __shared__ int32_t sh[256];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[col] = sh[0];
}

__device__ __forceinline__ uint32_t pow256_mod(int exp, uint32_t q) {
    uint64_t r = 1;
    for (int i = 0; i < exp; i++) r = (r * 256u) % q;
    return (uint32_t)r;
}

__global__ void recompose_tensor_matvec_kernel(
        uint32_t* const* out0,
        uint32_t* const* out1,
        const int32_t* gemm,
        const int32_t* db_sums,
        const int32_t* query_sums,
        size_t db_rows, size_t db_cols,
        uint32_t q0, uint32_t q1,
        int batch) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = (int)blockIdx.y;
    if (j >= db_cols || b >= batch) return;
    const size_t m = 2 * db_cols;

    uint32_t residues[2] = {0, 0};
    const uint32_t moduli[2] = {q0, q1};
    const uint32_t mus[2] = {Q0_BARRETT_MU, Q1_BARRETT_MU};
    #pragma unroll
    for (int limb = 0; limb < 2; limb++) {
        uint32_t q = moduli[limb];
        uint32_t mu = mus[limb];
        uint32_t acc = 0;
        #pragma unroll
        for (int digit = 0; digit < 4; digit++) {
            int col = b * 8 + limb * 4 + digit;
            int32_t qs = query_sums[col];
            #pragma unroll
            for (int byte = 0; byte < 2; byte++) {
                size_t row = 2 * j + (size_t)byte;
                int64_t dot_unsigned = (int64_t)gemm[(size_t)col * m + row]
                    + 128ll * (int64_t)db_sums[row]
                    + 128ll * (int64_t)qs
                    + 16384ll * (int64_t)db_rows;
                uint32_t dot_mod = barrett_mod((uint64_t)dot_unsigned, q, mu);
                uint32_t scale = pow256_mod(byte + digit, q);
                uint32_t term = barrett_mod((uint64_t)dot_mod * scale, q, mu);
                acc = mod_add(acc, term, q);
            }
        }
        residues[limb] = acc;
    }
    out0[b][j] = residues[0];
    out1[b][j] = residues[1];
}

void gpu_matvec_tensor_batched(
    uint32_t* const* d_results0, uint32_t* const* d_results1,
    const uint16_t* d_centered_db, const int32_t* d_db_byte_sums,
    const uint32_t* const* d_queries0, const uint32_t* const* d_queries1,
    int8_t* d_query_planes, int32_t* d_query_sums, int32_t* d_gemm_out,
    size_t db_rows, size_t db_cols, uint32_t q0, uint32_t q1,
    int batch, cublasHandle_t handle) {
    const int plane_cols = batch * 8;
    const size_t threads = 256;
    dim3 qgrid((db_rows + threads - 1) / threads, plane_cols);
    decompose_query_bytes_kernel<<<qgrid, threads>>>(
        d_queries0, d_queries1, d_query_planes, db_rows, batch);
    sum_query_planes_kernel<<<plane_cols, 256>>>(
        d_query_planes, d_query_sums, db_rows, plane_cols);

    const int m = (int)(2 * db_cols);
    const int n = plane_cols;
    const int k = (int)db_rows;
    const int alpha = 1;
    const int beta = 0;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        reinterpret_cast<const int8_t*>(d_centered_db), CUDA_R_8I, m,
        d_query_planes, CUDA_R_8I, k,
        &beta,
        d_gemm_out, CUDA_R_32I, m,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    dim3 out_grid((db_cols + threads - 1) / threads, batch);
    recompose_tensor_matvec_kernel<<<out_grid, threads>>>(
        d_results0, d_results1, d_gemm_out,
        d_db_byte_sums, d_query_sums,
        db_rows, db_cols, q0, q1, batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace inspire
