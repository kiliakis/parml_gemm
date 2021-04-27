#!/bin/bash

export OPENBLAS_ROOT=/home/kiliakis/install
export LD_LIBRARY_PATH=${OPENBLAS_ROOT}/lib:$LD_LIBRARY_PATH

# Replace with your path
# cd ${HOME}/kostis/ex1_omp_cuda

nthreads="1 2 4 8 16 32"
for t in $nthreads; do
	echo "NUM_THREADS $t"
	OPENBLAS_NUM_THREADS=$t ./test_blas 512 1024 2048 0
done
