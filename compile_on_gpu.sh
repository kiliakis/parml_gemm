#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Replace with your path
# cd ${HOME}/kostis/ex1_omp_cuda/src


# make -f Makefile.gpu clean
make -f Makefile.gpu all


