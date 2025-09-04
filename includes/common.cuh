#ifndef COMMON
#define COMMON


#define MAX_PIXEL_VALUE 255
#define LABEL_SIZE 10
#define DATASET_SPLIT_TRAIN_PROPORTION 0.6
#define BATCH_SIZE 256
#define gpu_error_check(ans) { gpu_assert((ans), __LINE__); }


/**
 * Checks if a CUDA-specific command succeeds. Aborts the program if the command fails.
 * @param code The CUDA error code returned by the command.
 * @param line The line corresponding to the command.
 */
void gpu_assert(cudaError_t code, int line);


/**
 * Performs a memory allocation and checks if the allocation succeeds. Aborts the program if
 *  the allocation fails.
 * @param size The number of bytes to be allocated.
 */
void *malloc_check(size_t size);


#endif