#!/bin/bash

export OPENBLAS_ROOT=/home/kiliakis/install
export LD_LIBRARY_PATH=${OPENBLAS_ROOT}/lib:$LD_LIBRARY_PATH

# Replace with your path
# cd ${HOME}/kostis/ex1_omp_cuda


nthreads="1 2 4 8 16"
for t in $nthreads; do
	echo "NUM_THREADS $t"
	GOMP_CPU_AFFINITY=0-$(($t-1)) OMP_NUM_THREADS=$t ./test_omp 512 1024 2048 0
done
