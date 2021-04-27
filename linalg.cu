#include "cuda_utils.h"
#include "linalg.h"
#include <stdio.h>

#ifdef CUDA
extern cublasHandle_t cublas_handle;
#endif

// #define TILE_DIM 32
// #define TILE_N 16
// #define TILE_TB_HEIGHT 8
// #define TILE_M (TILE_N*TILE_TB_HEIGHT)

// int BLOCK_SIZE = 32;

/*
 *  Naive matrix multiply kernels.
 */
__global__ void dgemm_naive(const double *A, const double *B,
                            double *C,
                            const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void dgemm_ta_naive(const double *A, const double *B,
                               double *C,
                               const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0;
        for (int k = 0; k < K; k++)
            sum += A[k * M + row] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void dgemm_tb_naive(const double *A, const double *B, const double *C,
                               double *D,
                               const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[col * K + k];
        D[row * N + col] = sum + C[row * N + col];
    }
}

/*
 *  Optimized matrix multiply kernels using shared memory.
 */
// A: M x K, B: K x N, C: M x N
template <int BLOCK_SIZE> __global__ void dgemm_optimized(
    const double *A, const double *B,
    double *C,
    const int M, const int N, const int K)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = K * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + K - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * N;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    double Csub = 0;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        if (a + K * ty + tx < M * K)
            As[ty][tx] = A[a + K * ty + tx];

        if (b + N * ty + tx < K * N)
            Bs[ty][tx] = B[b + N * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    if (c + N * ty + tx < M*N)
        C[c + N * ty + tx] = Csub;
}

__global__ void dgemm_ta_optimized(const double *A, const double *B,
                                   double *C,
                                   const int M, const int N, const int K) {
    /*
     * FILLME: fill the code.
     */
}

__global__ void dgemm_tb_optimized(const double *A, const double *B, const double *C,
                                   double *D,
                                   const int M, const int N, const int K) {
    /*
     * FILLME: fill the code.
     */
}

// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifndef CUBLAS
    const int BLOCK_SIZE = 32;
#if defined(_GPU_GEMM_NAIVE)
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dgemm_naive <<< grid, block>>>(A, B, C, M, N, K);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
    /*
     *  FILLME: Set up the blocks, grid and the shared memory size.
     */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x,
              (K + block.y - 1) / block.y);
    size_t shmem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    dgemm_optimized<BLOCK_SIZE> <<< grid, block>>>(A, B, C, M, N, K);

    // dim3 block(TILE_N, TILE_TB_HEIGHT);
    // dim3 grid(M/TILE_M, N/TILE_N);
    // size_t shmem_size = TILE_TB_HEIGHT * TILE_N * sizeof(double);
    // dgemm_optimized <<< grid, block, shmem_size>>>(B, A, C, N, M, K);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#endif
#else
// Matrices are stored in row-major order, but cuBLAS assumes column-major
// order. We want to compute:
//         A * B = (A^T)^T * (B^T)^T = A'^T * B'^T = (B' * A')^T
    /*
     *  FILLME: Use cublasDgemm()
     */
    cublasStatus_t stat;
    double alpha = 1;
    double beta = 0;
    stat = cublasDgemm(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K,
                       &alpha,
                       B, N,
                       A, K,
                       &beta,
                       C, N);
    checkCublasErrors(stat);
#endif
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_ta_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifndef CUBLAS
    const int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dgemm_ta_naive <<< grid, block>>>(A, B, C, M, N, K);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
    /*
     *  FILLME: Set up the blocks, grid and the shared memory size.
     */
    dim3 block(1, 1);
    dim3 grid(1, 1);
    size_t shmem_size = 0;
    dgemm_ta_optimized <<< grid, block, shmem_size>>>(A, B, C, M, N, K);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#endif
#else
// Matrices are stored in row-major order, but cuBLAS assumes column-major
// order. We want to compute:
//         A^T * B = A^T * (B^T)^T = A' * B'^T = (B'*A'^T)^T
    /*
     *  FILLME: Use cublasDgemm()
     */
    cublasStatus_t stat;
    double alpha = 1;
    double beta = 0;
    stat = cublasDgemm(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       N, M, K,
                       &alpha,
                       B, N,
                       A, M, // A, M
                       &beta,
                       C, N);
    checkCublasErrors(stat);

#endif
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
void dgemm_tb_gpu(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K) {
#ifndef CUBLAS
    const int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dgemm_tb_naive <<< grid, block>>>(A, B, C, D, M, N, K);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
    /*
     *  FILLME: Set up the blocks, grid and the shared memory size.
     */
    dim3 block(1, 1);
    dim3 grid(1, 1);
    size_t shmem_size = 0;
    dgemm_tb_optimized <<< grid, block, shmem_size>>>(A, B, C, D, M, N, K);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
#endif
#else
// D = A * B^T
// Matrices are stored in row-major order, but cuBLAS assumes column-major
// order. We want to compute:
//         C = A * B^T = (A^T)^T * B^T  = A'^T * B' = (B'^T * A')^T
    /*
     *  FILLME: Use cublasDgemm()
     */
    cublasStatus_t stat;
    double alpha = 1;
    double beta = 0;
    stat = cublasDgemm(cublas_handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       N, M, K,
                       &alpha,
                       B, K, // B, K
                       A, K,
                       &beta,
                       D, N);
    checkCublasErrors(stat);

// D = C + D
    /*
     *  FILLME: Use cublasDgeam()
     */
    // C in row-major is M x N, so in col-major it is N x M
    alpha = 1;
    beta = 1;
    stat = cublasDgeam(cublas_handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N,
                       &alpha,
                       C, M,
                       &beta,
                       D, M,
                       D, M);
    checkCublasErrors(stat);

#endif
}


/*
 *  Optimized matrix multiply kernels using shared memory.
 */
// __global__ void dgemm_optimized(const double *A, const double *B,
// double *C,
// const int M, const int N, const int K) {
/*
 * FILLME: fill the code.
 */

/*
// from https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic/18856054
double CValue = 0;
const int ARows = M;
const int ACols = K;
const int BRows = K;
const int BCols = N;
const int CRows = M;
const int CCols = N;

int Row = blockIdx.y*TILE_DIM + threadIdx.y;
int Col = blockIdx.x*TILE_DIM + threadIdx.x;

__shared__ double As[TILE_DIM][TILE_DIM];
__shared__ double Bs[TILE_DIM][TILE_DIM];

for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

     if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
         As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
     else
         As[threadIdx.y][threadIdx.x] = 0.0;

     if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
         Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
     else
         Bs[threadIdx.y][threadIdx.x] = 0.0;

     __syncthreads();

     for (int n = 0; n < TILE_DIM; ++n)
         CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

     __syncthreads();
}

if (Row < CRows && Col < CCols)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
       (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
*/
// from parboil sgemm
/*
// Partial results
double c[TILE_N];
for (int i=0; i < TILE_N; i++) c[i] = 0.0;
int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
int m = blockIdx.x * TILE_M + mid;
int n = blockIdx.y * TILE_N + threadIdx.x;
__shared__ double b_s[TILE_TB_HEIGHT][TILE_N];
for (int i = 0; i < K; i+=TILE_TB_HEIGHT) {
    double a;
    b_s[threadIdx.y][threadIdx.x]=B[n + (i+threadIdx.y)*N];
    __syncthreads();
    for (int j = 0; j < TILE_TB_HEIGHT; j++) {
        a = A[m + (i+j)*K];
        for (int kk = 0; kk < TILE_N; kk++)
            c[kk] += a * b_s[j][kk];
    }
    __syncthreads();
}
int t = N*blockIdx.y * TILE_N + m;
for (int i = 0; i < TILE_N; i++) {
    // C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
    C[t+i*N] = c[i];
}
*/

// }
