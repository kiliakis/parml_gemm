#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include "decls.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

__BEGIN_C_DECLS

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line);
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// This will output the proper cuBLAS status string in the event
// that a cuBLAS host call returns an error
void check_cublas(cublasStatus_t result, char const *const func,
                  const char *const file, int const line);
#define checkCublasErrors(val) check_cublas((val), #val, __FILE__, __LINE__)

__END_C_DECLS

#endif
