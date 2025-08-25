#ifndef KERNELS
#define KERNELS


#include <stdint.h>

#define TILE_WIDTH 16
#define THREAD_COARSENING_FACTOR 8
#define eps 1e-6

enum pooling_type { MAX, MEAN };


/**
 * Conv2d forward kernel implementation, following the method in PMPP chapter 16.3 (Fig. 16.13,14).
 * 
 * @param Y The output tensor stored in row-major format.
 * @param X The input tensor stored in row-major format.
 * @param filters The filters tensor stored in row-major format.
 * @param kernel_length The length of each (square) kernel.
 * @param in_channels The number of input channels.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the output indices.
 * @param in_height The input height.
 * @param in_width The input width.
 */
__global__ void Conv2DForwardKernel(
    float *Y, float *X, float *filters,
    uint32_t kernel_length,
    uint32_t in_channels,
    uint32_t num_tiles_h,
    uint32_t in_height, uint32_t in_width
);


/**
 * Conv2d backward kernel implementation for computing dX.
 * 
 * @param dX The gradient to be computed; stored in row-major format.
 * @param dY The gradient needed for computing dX; stored in row-major format.
 * @param W The weights needed for computing dX; stored in row-major format.
 * @param kernel_length The length of each (square) kernel.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the input features indices.
 * @param in_height The input height.
 * @param in_width The input width.
 */
__global__ void Conv2DBackwardXGradKernel(
    float *dX, float *dY, float *W,
    uint32_t kernel_length,
    uint32_t out_channels,
    uint32_t num_tiles_h,
    uint32_t in_height, uint32_t in_width
);


/**
 * Conv2d backward kernel implementation for computing dW.
 * 
 * @param dW The weight gradient to be computed; stored in row-major format.
 * @param dY The gradient needed for computing dW; stored in row-major format.
 * @param X The X tensor needed for computing dW; stored in row-major format.
 * @param kernel_length The length of each (square) kernel.
 * @param num_samples The number of samples.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the filter indices.
 * @param out_height The output height.
 * @param out_width The output width.
 */
__global__ void Conv2DBackwardWGradKernel(
    float *dW, float *dY, float *X,
    uint32_t num_samples,
    uint32_t kernel_length,
    uint32_t num_tiles_h,
    uint32_t out_height, uint32_t out_width
);


/**
 * Sigmoid forward kernel implementation.
 * 
 * @param Y The output tensor stored in row-major format.
 * @param X The input tensor stored in row-major format.
 * @param grad The gradient tensor to be computed simultanously; stored in row-major format.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the output indices.
 * @param out_height The output height.
 * @param out_width The output width.
 */
__global__ void SigmoidForwardKernel(
    float *Y, float *X, float *grad,
    uint32_t num_tiles_h,
    uint32_t out_height, uint32_t out_width
);


/**
 * Performs element-wise multiplication and stores the product in X1.
 * 
 * @param X1 The input tensor #1, to be replaced with the product of X1 and X2.
 * @param X2 The input tensor #2.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the output indices.
 * @param feature_height The feature height.
 * @param feature_width The feature width.
 */
__global__ void MultiplyKernel(
    float *X1, float *X2,
    uint32_t num_tiles_h,
    uint32_t feature_height, uint32_t feature_width
);


/**
 * Mean/Max pool forward kernel implementation.
 *  Assumes that the stride is always the kernel_length and the input width and height are
 *  always divisible by kernel_length.
 * 
 * @param Y The output tensor stored in row-major format.
 * @param X The input tensor stored in row-major format.
 * @param pool_type either MAX or MEAN; specifies the pooling type.
 * @param grad The gradient tensor to be computed simultanously; stored in row-major format.
 * @param kernel_length The length of the kernel.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the output indices.
 * @param in_height The input height.
 * @param in_width The input width.
 * @param out_height The output height.
 * @param out_width The output width.
 */
__global__ void PoolForwardKernel(
    float *Y, float *X, 
    pooling_type pool_type,
    float *grad,
    uint32_t kernel_length,
    uint32_t num_tiles_h,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
);


/**
 * Mean/Max pool backward kernel implementation.
 *  Assumes that the stride is always the kernel_length and the input width and height are
 *  always divisible by kernel_length.
 * 
 * @param dX The gradient to be computed; stored in row-major format.
 * @param dY The gradient needed for computing dX; stored in row-major format.
 * @param kernel_length The length of each (square) kernel.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the gradient indices.
 * @param dX_height The height of the gradient.
 * @param dX_width The width of the gradient.
 * @param dY_height The height of the next layer's gradient.
 * @param dY_width The width of the next layer's gradient.
 */
__global__ void PoolBackwardKernel(
    float *dX, float *dY,
    uint32_t kernel_length,
    uint32_t num_tiles_h,
    uint32_t dX_height, uint32_t dX_width,
    uint32_t dY_height, uint32_t dY_width

);


/**
 * Linear forward kernel implementation where given X and W, performs matrix multiplication (X, W.T).
 *  Since it performs matrix multiplication on the transposed W, it takes advantage of the W.T's
 *  column-major format that allows memory coalescing.
 * 
 * @param Y The output tensor stored in row-major format.
 * @param X The X tensor.
 * @param W The W tensor.
 * @param num_samples The number of samples.
 * @param in_features The number of input features.
 * @param out_features The number of output features.
 */
__global__ void LinearForwardKernel(
    float *Y, float *X, float *W,
    uint32_t num_samples,
    uint32_t in_features, uint32_t out_features
);


/**
 * Performs matrix multiplication (X, A).
 * 
 * @param Y The output tensor stored in row-major format.
 * @param X The X matrix.
 * @param A The A matrix.
 * @param X_height The height of X.
 * @param X_width The width of X (or the height of A).
 * @param A_width The width of A.
 */
__global__ void MatMulKernel(   
    float *Y, float *X, float *A,
    uint32_t X_height, uint32_t X_width, uint32_t A_width
);


/**
 * Transposes a matrix.
 * 
 * @param Y The output matrix stored in row-major format.
 * @param X The input matrix stored in row-major format.
 * @param height The height of input matrix.
 * @param width The width of input matrix.
 */
__global__ void TransposeMatrixKernel(   
    float *Y, float *X,
    uint32_t height, uint32_t width
);


/**
 * Updates the weights of a linear layer.
 * 
 * @param W The weight tensor to be updated; stored in row-major format.
 * @param dW The weight gradient tensor; stored in row-major format and has the same dimension as W.
 * @param height The height of the weight tensor.
 * @param width The width of the weight tensor.
 * @param lr The learning rate.
 */
__global__ void UpdateLinearWeightsKernel(float *W, float *dW, uint32_t height, uint32_t width, float lr);


/**
 * Updates the filters of a conv2d layer.
 * 
 * @param W The filter tensor to be updated; stored in row-major format.
 * @param dW The filter gradient tensor; stored in row-major format and has the same dimension as W.
 * @param height The height of the filter tensor.
 * @param width The width of the filter tensor.
 * @param num_tiles_h The number of tiles horizontally in the y-axis of the grid; used for computing the filter indices.
 * @param lr The learning rate.
 */
__global__ void UpdateConv2DWeightsKernel(
    float *W, float *dW, uint32_t height, uint32_t width, uint32_t num_tiles_h, float lr
);


/**
 * Part of the softmax computation: calculates the exponent and sums by row.
 * 
 * @param exp_X The exponential of the elements in X.
 * @param sub_exp_X The sum of exp(X) by row.
 * @param X The input tensor; stored in row-major format.
 * @param height The height of X.
 * @param width The width of X.
 */
__global__ void CalcExpAndSumByRowKernel(
    float *exp_X, float *sum_exp_X, float *X, uint32_t height, uint32_t width
);


/**
 * Part of the softmax computation: normalizes X by row.
 * 
 * @param X The input tensor, will be replaced by the normalization output.
 * @param sum_row The sum of X by row.
 * @param height The height of X.
 * @param width The width of X.
 */
__global__ void NormalizeKernel(float *X, float *sum_row, uint32_t height, uint32_t width);


/**
 * Performs NLL loss on log(X).
 * 
 * @param out The loss output.
 * @param X The X tensor stored in row-major format; dimension: [num_samples, num_features].
 * @param y The one-hot encodings of the true labels stored in row-major format; dimension: [num_samples, num_features].
 * @param num_samples The number of samples.
 * @param num_features The number of features/label categories.
 */
__global__ void NegativeLogLikelihoodLogKernel(
    float *out, const float *X, const uint8_t *y, uint32_t num_samples, uint32_t num_features
);


/**
 * Computes the gradient of softmax layer.
 * 
 * @param dX The gradient to be computed.
 * @param softmax_output The output of the softmax activation.
 * @param y The one-hot encodings of the true labels stored in row-major format.
 * @param num_samples The number of samples.
 * @param num_features The number of features/label categories.
 */
__global__ void SoftmaxGradientKernel(
    float *dX, const float *softmax_output, const uint8_t *y, uint32_t num_samples, uint32_t num_features
);


/**
 * Get the count of accurate predictions.
 * 
 * @param count The count output of accurate predictions.
 * @param input_tensor Normalized softmax output.
 * @param y The one-hot encodings of the true labels stored in row-major format.
 * @param num_samples The number of samples.
 * @param num_features The number of features/label categories.
 */
__global__ void GetAccuratePredCountKernel(
    uint32_t *count, const float *input_tensor, const uint8_t *y, uint32_t num_samples, uint32_t num_features
);


#endif