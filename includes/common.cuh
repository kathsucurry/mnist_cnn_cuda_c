#ifndef COMMON
#define COMMON


#define MAX_PIXEL_VALUE 255
#define LABEL_SIZE 10
#define DATASET_SPLIT_TRAIN_PROPORTION 0.6
#define BATCH_SIZE 256
#define gpu_error_check(ans) { gpu_assert((ans), __LINE__); }


void gpu_assert(cudaError_t code, int line);
void *malloc_check(size_t size);


#endif