#ifndef __LINALG_H__
#define __LINALG_H__

#include <math.h>
#include <string.h>

#ifdef CBLAS
#include <cblas.h>
#endif

#ifdef CUDA
#include "cuda_utils.h"
#endif

// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm(const double *A, const double *B, double *C, const int M, const int N, const int K);
#ifdef CUDA
__BEGIN_C_DECLS
void dgemm_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K);
__END_C_DECLS
#endif

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_ta(const double *A, const double *B, double *C, const int M, const int N, const int K);
#ifdef CUDA
__BEGIN_C_DECLS
void dgemm_ta_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K);
__END_C_DECLS
#endif

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
void dgemm_tb(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K);
#ifdef CUDA
__BEGIN_C_DECLS
void dgemm_tb_gpu(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K);
__END_C_DECLS
#endif

// Computes out = in1 hadamard in2, i.e. the Hadamard product of 2D arrays in1 and in2
void hadamard2D(double *out, const double *in1, const double *in2, const int M, const int N);

// Sums the rows of the 2D array in the 1D vector out
void sumRows(double *out, const double *in, const int M, const int N);

// Bias updates
void gradientb(double *out, const double *in, const int M, const int N, const double lr);

// Weights updates
void gradientW(double *out, const double *in, const int M, const int N, const double lr);

#endif // __LINALG_H__
