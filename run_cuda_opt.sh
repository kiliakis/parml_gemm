#!/bin/bash


# export PATH=/usr/local/cuda-9.2/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID


# Replace with your path
# cd ${HOME}/kostis/ex1_omp_cuda

./src/test_cuda_opt
