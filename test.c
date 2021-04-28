#include "linalg.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#ifdef CUDA
#include "cuda_utils.h"
cublasHandle_t cublas_handle;
#endif

int validate_ref = 1;
int debug = 0;
xtimer_t dgemm_ref_timer, dgemm_ref_tb_timer, dgemm_ref_ta_timer;
xtimer_t dgemm_timer, dgemm_tb_timer, dgemm_ta_timer;

// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
static void ref_dgemm(const double *A, const double *B, double *C, const int M, const int N, const int K) {
    int i, j, k;
    double sum;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
static void ref_dgemm_ta(const double *A, const double *B, double *C, const int M, const int N, const int K) {
    int i, j, k;
    double sum;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[k * M + i] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
static void ref_dgemm_tb(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K) {
    int i, j, k;
    double sum;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[i * K + k] * B[j * K + k];
            D[i * N + j] = sum + C[i * N + j];
        }
    }
}

static void rand_init(double *data, const int N) {
    for (int i = 0; i < N; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
}


static void inc_init(double *data, const int N) {
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }
}


static bool is_matching(const double *a, const double *b, const int N) {
    if (validate_ref){
        for (int i = 0; i < N; i++) {
            double diff = fabs(a[i] - b[i]);
            if (diff > 1e-8) { // double has just 6.5 significant digits
                printf("Mismatch at %d: %lf, %lf (%lf)\n", i, a[i], b[i], diff);
                return false;
            }
        }
    }
    return true;
}

static void print_mat(const double *a, const int M, const int N) {
    printf("---------------\n");
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            printf("%.4lf, ", a[r * N + c]);
        }
        printf("\n");
    }
}

static bool test_dgemm(int M, int N, int K) {
    bool status = true;
    double *a = (double *)malloc(M * K * sizeof(double));
    double *b = (double *)malloc(K * N * sizeof(double));
    double *c = (double *)calloc(M * N, sizeof(double));
    double *c_ref = (double *)calloc(M * N, sizeof(double));
#ifdef CUDA
    double *dev_a, *dev_b, *dev_c;
    checkCudaErrors(cudaMalloc((void **)&dev_a, M * K * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_b, K * N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, M * N * sizeof(double)));
#endif

    if(debug){
        inc_init(a, M * K);
        inc_init(b, K * N);
    }else{
        rand_init(a, M * K);
        rand_init(b, K * N);
    }
#ifdef CUDA
    checkCudaErrors(cudaMemcpy(dev_a, a, M * K * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, K * N * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // Reference implementation
    if (validate_ref){
        timer_start(&dgemm_ref_timer);
        ref_dgemm(a, b, c_ref, M, N, K);
        timer_stop(&dgemm_ref_timer);
        timer_elapsed_time(&dgemm_ref_timer);
    }
    // Evaluated implementation
    timer_start(&dgemm_timer);

#ifdef CUDA
    dgemm_gpu(dev_a, dev_b, dev_c, M, N, K);
    checkCudaErrors(cudaMemcpy(c, dev_c, M * N * sizeof(double), cudaMemcpyDeviceToHost));
#else
    dgemm(a, b, c, M, N, K);
#endif
    timer_stop(&dgemm_timer);
    timer_elapsed_time(&dgemm_timer);

    status = is_matching(c, c_ref, M * N);
    if (debug){
        printf("Referece Matrix:\n");
        print_mat(c_ref, M, N);
        printf("Calculated Matrix:\n");
        print_mat(c, M, N);
    }
    // Cleanup
#ifdef CUDA
    checkCudaErrors(cudaFree(dev_a));
    checkCudaErrors(cudaFree(dev_b));
    checkCudaErrors(cudaFree(dev_c));
#endif
    free(a);
    free(b);
    free(c);
    free(c_ref);
    return status;
}

static bool test_dgemm_ta(int M, int N, int K) {
    bool status = true;
    double *a = (double *)malloc(M * K * sizeof(double));
    double *b = (double *)malloc(K * N * sizeof(double));
    double *c = (double *)calloc(M * N, sizeof(double));
    double *c_ref = (double *)calloc(M * N, sizeof(double));
#ifdef CUDA
    double *dev_a, *dev_b, *dev_c;
    checkCudaErrors(cudaMalloc((void **)&dev_a, M * K * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_b, K * N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, M * N * sizeof(double)));
#endif
    if(debug){
        inc_init(a, M * K);
        inc_init(b, K * N);
    }else{
        rand_init(a, M * K);
        rand_init(b, K * N);
    }
#ifdef CUDA
    checkCudaErrors(cudaMemcpy(dev_a, a, M * K * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, K * N * sizeof(double), cudaMemcpyHostToDevice));
#endif
    if (validate_ref){
        timer_start(&dgemm_ref_ta_timer);

        // Reference implementation
        ref_dgemm_ta(a, b, c_ref, M, N, K);

        timer_stop(&dgemm_ref_ta_timer);
        timer_elapsed_time(&dgemm_ref_ta_timer);
    }

    // Evaluated implementation
    timer_start(&dgemm_ta_timer);

#ifdef CUDA
    dgemm_ta_gpu(dev_a, dev_b, dev_c, M, N, K);
    checkCudaErrors(cudaMemcpy(c, dev_c, M * N * sizeof(double), cudaMemcpyDeviceToHost));
#else
    dgemm_ta(a, b, c, M, N, K);
#endif
    timer_stop(&dgemm_ta_timer);
    timer_elapsed_time(&dgemm_ta_timer);

    status = is_matching(c, c_ref, M * N);
    if (debug){
        printf("Referece Matrix:\n");
        print_mat(c_ref, M, N);
        printf("Calculated Matrix:\n");
        print_mat(c, M, N);
    }
    // Cleanup
#ifdef CUDA
    checkCudaErrors(cudaFree(dev_a));
    checkCudaErrors(cudaFree(dev_b));
    checkCudaErrors(cudaFree(dev_c));
#endif
    free(a);
    free(b);
    free(c);
    free(c_ref);
    return status;
}

static bool test_dgemm_tb(int M, int N, int K) {
    bool status = true;
    double *a = (double *)malloc(M * K * sizeof(double));
    double *b = (double *)malloc(K * N * sizeof(double));
    double *c = (double *)malloc(M * N * sizeof(double));
    double *d = (double *)calloc(M * N, sizeof(double));
    double *d_ref = (double *)calloc(M * N, sizeof(double));
#ifdef CUDA
    double *dev_a, *dev_b, *dev_c, *dev_d;
    checkCudaErrors(cudaMalloc((void **)&dev_a, M * K * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_b, K * N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, M * N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_d, M * N * sizeof(double)));
#endif
    if(debug){
        inc_init(a, M * K);
        inc_init(b, K * N);
        inc_init(c, M * N);
    }else{
        rand_init(a, M * K);
        rand_init(b, K * N);
        rand_init(c, M * N);
    }
#ifdef CUDA
    checkCudaErrors(cudaMemcpy(dev_a, a, M * K * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, K * N * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_c, c, M * N * sizeof(double), cudaMemcpyHostToDevice));
#endif
    if (validate_ref) {
        timer_start(&dgemm_ref_tb_timer);

        // Reference implementation
        ref_dgemm_tb(a, b, c, d_ref, M, N, K);
        timer_stop(&dgemm_ref_tb_timer);
    timer_elapsed_time(&dgemm_ref_tb_timer);
    }

    // Evaluated implementation
    timer_start(&dgemm_tb_timer);

#ifdef CUDA
    dgemm_tb_gpu(dev_a, dev_b, dev_c, dev_d, M, N, K);
    checkCudaErrors(cudaMemcpy(d, dev_d, M * N * sizeof(double), cudaMemcpyDeviceToHost));
#else
    dgemm_tb(a, b, c, d, M, N, K);
#endif
    timer_stop(&dgemm_tb_timer);
    timer_elapsed_time(&dgemm_tb_timer);

    status = is_matching(d, d_ref, M * N);
    if (debug){
        printf("Referece Matrix:\n");
        print_mat(d_ref, M, N);
        printf("Calculated Matrix:\n");
        print_mat(d, M, N);
    }
// Cleanup
#ifdef CUDA
    checkCudaErrors(cudaFree(dev_a));
    checkCudaErrors(cudaFree(dev_b));
    checkCudaErrors(cudaFree(dev_c));
    checkCudaErrors(cudaFree(dev_d));
#endif
    free(a);
    free(b);
    free(c);
    free(d);
    free(d_ref);
    return status;
}

int main(int argc, char **argv) {
#ifdef CUDA
    checkCublasErrors(cublasCreate(&cublas_handle));
#endif
    int M = 1<<9; 
    int N = 1<<10;
    int K = 1<<11;
    validate_ref = 1;
    debug = 0;
    if(argc > 1){
        M = atoi(argv[1]);        
    }
    if(argc > 2){
        N = atoi(argv[2]);        
    }
    if(argc > 3){
        K = atoi(argv[3]);        
    }
    if(argc > 4){
        validate_ref = atoi(argv[4]);        
    }
    if(argc > 5){
        debug = atoi(argv[5]);
    }
    printf("M:%d, N:%d, K:%d, validate:%d, debug: %d\n", M, N, K, validate_ref, debug);

    timer_clear(&dgemm_timer);
    timer_clear(&dgemm_ta_timer);
    timer_clear(&dgemm_tb_timer);
    timer_clear(&dgemm_ref_timer);
    timer_clear(&dgemm_ref_ta_timer);
    timer_clear(&dgemm_ref_tb_timer);

    bool status;
    printf("Testing DGEMM... ");
    status = test_dgemm(M, N, K);
    if (status) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }
    printf("DGEMM REF time: %.6lf\n", timer_elapsed_time(&dgemm_ref_timer));
    printf("DGEMM time: %.6lf\n", timer_elapsed_time(&dgemm_timer));

    printf("Testing DGEMM_TA... ");
    status = test_dgemm_ta(M, N, K);
    if (status) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }
    printf("DGEMM_TA REF time: %.6lf\n", timer_elapsed_time(&dgemm_ref_ta_timer));
    printf("DGEMM_TA time: %.6lf\n", timer_elapsed_time(&dgemm_ta_timer));

    printf("Testing DGEMM_TB... ");
    status = test_dgemm_tb(M, N, K);
    if (status) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }
    printf("DGEMM_TB REF time: %.6lf\n", timer_elapsed_time(&dgemm_ref_tb_timer));
    printf("DGEMM_TB time: %.6lf\n", timer_elapsed_time(&dgemm_tb_timer));

    return 0;
}
