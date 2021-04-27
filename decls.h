/*
 *  decls.h -- NVCC forces C++ compilation of .cu files, so we
 *             need to declare C functions using extern "C" to
 *             avoid name mangling during linkage
 */
#ifndef __DECLS_H__
#define __DECLS_H__

#undef __BEGIN_C_DECLS
#undef __END_C_DECLS

#if defined(__cplusplus) || defined(__CUDACC__)
#define __BEGIN_C_DECLS extern "C" {
#define __END_C_DECLS }
#else
#define __BEGIN_C_DECLS
#define __END_C_DECLS
#endif /* __cplusplus || __CUDACC__ */

#endif /* __DECLS_H__ */
