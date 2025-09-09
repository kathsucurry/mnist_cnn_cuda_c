#include <cuda_runtime.h>
#include <stdio.h> 

#include "../includes/common.cuh"


void gpu_assert(cudaError_t code, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %u %s; line %d\n", code, cudaGetErrorString(code), line);
      exit(code);
   }
}


void *malloc_check(size_t size) {
    void *malloc_return = malloc(size);
    if (!malloc_return) {
        fprintf(stderr, "Malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    return malloc_return;
}


void *calloc_check(size_t num, size_t size) {
    void *calloc_return = calloc(num, size);
    if (!calloc_return) {
        fprintf(stderr, "Calloc failed!\n");
        exit(EXIT_FAILURE);
    }
    return calloc_return;
}