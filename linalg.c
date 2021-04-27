#include "linalg.h"

// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifdef CBLAS
    /*
     *  FILLME: Use cblas_dgemm()
     */
    double alpha, beta;
    alpha = 1;
    beta = 0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
    // A: K elements per row, M rows
#else
    int i, j, k;
    double sum;
    /*
     * FILLME: Parallelize the code.
     */
    #pragma omp parallel for private(i,j,k,sum) collapse(2)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
#endif
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_ta(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifdef CBLAS
    /*
     *  FILLME: Use cblas_dgemm()
     */
    double alpha, beta;
    alpha = 1;
    beta = 0;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, alpha, A, M, B, N, beta, C, N);
    // A: K elements per row, M rows
#else
    int i, j, k;
    double sum;
    /*
     * FILLME: Parallelize the code.
     */
    #pragma omp parallel for private(i,j,k,sum) collapse(2)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[k * M + i] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
#endif
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
void dgemm_tb(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K) {
#ifdef CBLAS
    /*
     *  FILLME: Use cblas_dgemm()
     */
    double alpha, beta;
    
    // first I do: D = C
    memcpy(D, C, M * N * sizeof(double));
    
    // then I do: D = 1 * A*B' + 1 * D
    alpha = 1;
    beta = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, K, B, K, beta, D, N);
    // A: K elements per row, M rows
#else
    int i, j, k;
    double sum;
    /*
     * FILLME: Parallelize the code.
     */
    #pragma omp parallel for private(i,j,k,sum) collapse(2)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[i * K + k] * B[j * K + k];
            D[i * N + j] = sum + C[i * N + j];
        }
    }
#endif
}

void hadamard2D(double *out, const double *in1, const double *in2, const int M, const int N) {
    int i, j;
    double res;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            res = in1[i * N + j] * in2[i * N + j];
            out[i * N + j] = res;
        }
}

void sumRows(double *out, const double *in, const int M, const int N) {
    memset(out, 0, N * sizeof(double));
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++)
            out[j] += in[i * N + j];
    }
}

void gradientb(double *out, const double *in, const int M, const int N, const double lr) {
    int i, j;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            out[i * N + j] -= (lr) * in[j];
}

void gradientW(double *out, const double *in, const int M, const int N, const double lr) {
    int i, j;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            out[i * N + j] -= (lr) * in[i * N + j];
}