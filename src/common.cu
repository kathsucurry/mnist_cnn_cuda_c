#include <cuda_runtime.h>
#include <stdio.h> 


void gpu_assert(cudaError_t code, int line) {
   if (code != cudaSuccess) {
      printf("GPUassert: %u %s; line %d\n", code, cudaGetErrorString(code), line);
      exit(code);
   }
}


void *malloc_check(size_t size) {
    void *malloc_return = malloc(size);
    if (!malloc_return) {
        printf("Malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    return malloc_return;
}
