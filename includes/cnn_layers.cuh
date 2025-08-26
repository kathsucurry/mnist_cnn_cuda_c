#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>

#include "kernel_functions.cuh"


/** @struct Tensor
 * Stores a multi-dimensional float array.
 * 
 * @var Tensor::dim_size
 *  The size of dimension. For instance, a dimension of [2, 2, 2] would have a size of 3.
 * @var Tensor::dim
 *  The dimension of the tensor.
 * @var Tensor::values_d
 *  The float values of the tensor that is allocated in device in row-major format.
 */
typedef struct {
    uint8_t dim_size;
    uint32_t *dim;
    float *values_d;
} Tensor;


/** @struct NetworkWeights.
 * Stores the weights of the network, assuming there is only one conv2d layer and one linear layer.
 * 
 * @var NetworkWeights::conv2d_weight
 *  The weights of the conv2d layer.
 * @var NetworkWeights::linear_weight
 *  The weights of the linear layer.
 */
typedef struct {
    Tensor *conv2d_weight;
    Tensor *linear_weight;
} NetworkWeights;


/** @struct LayerGradients.
 * Stores gradients/weights/X values of a layer. Note that dX is the dY of the previous layer.
 * 
 * @var LayerGradients::dW_or_W
 *  The weights or weight gradients of the layer.
 * @var LayerGradients::dX_or_X
 *  The weights or the input gradients of the layer.
 * @var LayerGradients::is_grad
 *  True if it stores the gradients of W and X, else, it stores the values of W and X.
 */
typedef struct {
    Tensor *dW_or_W;
    Tensor *dX_or_X;
    bool is_grad;
} LayerGradients;


/** @struct NetworkOutputs.
 * The network outputs, containing the output of the last layer and the gradient values.
 * 
 * @var NetworkOutputs::gradients
 *  The gradient values of each layer.
 * @var NetworkOutputs::num_layers
 *  The number of layers within the network.
 * @var NetworkOutputs::output
 *  The tensor output of the last layer.
 */
typedef struct {
    LayerGradients *gradients;
    uint32_t num_layers;
    Tensor *output;
    float *layer_durations_ms; // Time taken by each layer in milliseconds.
} NetworkOutputs;


/** @struct EpochOutput.
 * The output of one training epoch.
 * 
 * @var EpochOutput::loss
 *  The loss value.
 * @var EpochOutput::accuracy_percent
 *  The accuracy value in percentage.
 */
typedef struct {
    float loss;
    float accuracy_percent;
} EpochOutput;


/**
 * Computes the tensor size.
 * 
 * @param dim The dimension array.
 * @param dim_size The size of the dimension array.
 * @return The size of the tensor.
 */
uint32_t get_tensor_size(const uint32_t *dim, const uint8_t dim_size);


/**
 * Generates a tensor object, which can be freed using `free_tensor` function.
 * 
 * @param X The values of the tensor stored in row-major format.
 * @param dim The dimension array of the tensor.
 * @param dim_size The size of the dimension array.
 * @return The newly generated tensor.
 */
Tensor *generate_tensor(float *X, uint32_t *dim, uint8_t dim_size);


/**
 * Deep copies a tensor.
 * 
 * @param tensor The tensor to be copied.
 * @return A new tensor copied from the input tensor.
 */
Tensor *deep_copy_tensor(Tensor *tensor);


/**
 * Deallocates a tensor.
 * 
 * @param tensor The tensor to be deallocated.
 */
void free_tensor(Tensor *tensor);


/**
 * Deallocates a NetworkWeights object.
 * 
 * @param weights The network weights to be deallocated.
 */
void free_network_weights(NetworkWeights *weights);


/**
 * Deallocates a NetworkOutputs object.
 *  Make sure to deallocate the gradient content if not empty.
 * 
 * @param output The network output to be deallocated.
 * @param include_grad Whether the network output contains gradient information (i.e., if gradients are computed).
 */
void free_network_outputs(NetworkOutputs *output, bool include_grad);

/**
 * Stores filter weights in the constant variable.
 *  Here, we only assume one convolutional layer. To further expand this, we can
 *  store the current length (or the current start index for the next layer weights) of the constant and save
 *  the start index every time we store a new set of layer weights.
 * 
 * @param filters_d The filter tensor to be stored in the constant variable.
 * @param size The size of the tensor.
 * @return The cuda error code.
 */
cudaError_t update_conv2d_const_filters(float *filters_d, uint32_t size);

/**
 * Initializes conv2d layer weights using uniform Xaiver initialization.
 *  For simplicity, assume that stride is always 1.
 * 
 * @param in_channels The size of input channels.
 * @param out_channels The size of output channels.
 * @param filter_length The size/length of the filter.
 * @param seed The randomization seed
 * @return A weight tensor with dimension [out_channels, in_channels, filter_length, filter_length].
 */
Tensor *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_length, uint32_t seed);


/**
 * Initializes linear layer weights using uniform Xaiver initialization.
 * 
 * @param in_features The size of input features.
 * @param out_features The size of output features.
 * @param seed The randomization seed
 * @return A weight tensor with dimension [out_features, in_features].
 */
Tensor *initialize_linear_layer_weights(uint32_t in_features, uint32_t out_features, uint32_t seed);

/* Forward layer functions */

/**
 * Performs conv2d forward pass.
 *  The function replaces X tensor in-place with the output of the conv2d; the new dimension: [num_samples, out_channels, out_height, out_width] where
 *  `out_height = in_height - filter_length + 1` and `out_width  = in_width - filter_length + 1`.
 * 
 * @param X The X tensor with dimension [num_samples, in_channels, in_height, in_width], to be updated in-place with the output of the conv2d forward pass.
 * @param filters The filter tensor with dimension [out_channels, in_channels, filter_length, filter_length].
 * @param grad The gradient tensor for storing W and X; dW and dX will be computed during backward pass.
 * @param compute_grad Whether to compute the gradients (e.g., it's not needed for evaluation purposes).
 * @return The total runtime of the kernel run.
 */
float run_conv2d_forward(Tensor *X, Tensor *filters, LayerGradients *grad, bool compute_grad);


/**
 * Performs conv2d backward pass: computing dW and dX, then updating the weights given learning rate.
 * 
 * @param conv2d_weights The current weights of the conv2d layer.
 * @param grad Contains W and X of the layer for computing dW and dX.
 * @param next_layer_grad The gradients from the next layer for obtaining dY.
 * @param lr The learning rate for updating the weights.
 * @return The total runtime of the kernel run.
 */
float run_conv2d_backward(Tensor *conv2d_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr);


/**
 * Performs sigmoid activation function.
 *  The function replaces X tensor in-place with the output of the sigmoid; no dimension change occurs.
 * 
 * @param X The input tensor, to be updated in-place with the output of the sigmoid function.
 * @param grad The gradient tensor for storing the layer's dX.
 * @param compute_grad Whether to compute the gradients (e.g., it's not needed for evaluation purposes).
 * @return The total runtime of the kernel run.
 */
float run_sigmoid_forward(Tensor *X, LayerGradients *grad, bool compute_grad);


/**
 * Performs sigmoid backward pass to compute dX using chain rule.
 * 
 * @param grad Contains dX of the layer before performing chain rule.
 * @param next_layer_grad The gradients from the next layer for obtaining dY.
 * @return The total runtime of the kernel run.
 */
float run_sigmoid_backward(LayerGradients *grad, LayerGradients *next_layer_grad);


/**
 * Performs pooling forward pass.
 *  Assumes the stride is always the kernel size.
 * 
 * @param X The input tensor, to be updated in-place with the output of the pooling function.
 * @param kernel_length The kernel length.
 * @param pool_type Either MAX or MEAN pooling.
 * @param grad The gradient tensor for storing the layer's dX.
 * @param compute_grad Whether to compute the gradients (e.g., it's not needed for evaluation purposes).
 * @return The total runtime of the kernel run.
 */
float run_pooling_forward(Tensor *X, uint32_t kernel_length, pooling_type pool_type, LayerGradients *grad, bool compute_grad);


/**
 * Performs pooling backward pass to compute dX using chain rule.
 * 
 * @param kernel_length The length of the kernel.
 * @param grad Contains dX of the layer before performing chain rule.
 * @param next_layer_grad The gradients from the next layer for obtaining dY.
 * @return The total runtime of the kernel run.
 */
float run_pooling_backward(uint32_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad);


/**
 * Reshapes the input into a one-dimensional tensor.
 * 
 * @param X The input tensor. It will be replaced with a [num_samples, size]-dimensional tensor where size = the
 *  input tensor size / num_samples.
 * @return The total runtime of the kernel run.
 */
float run_flatten_forward(Tensor *X);


/**
 * Reshapes dY with dimension [num_samples, num_features] to [num_samples, num_channels, kernel_length, kernel_length].
 *  The number of channels is derived by num_features / (kernel_length * kernel_length).
 * 
 * @param num_samples The number of samples.
 * @param kernel_length The kernel length.
 * @param grad Contains dX of the layer before performing chain rule.
 * @param next_layer_grad The gradients from the next layer for obtaining dY.
 * @return The total runtime of the kernel run.
 */
float run_flatten_backward(uint32_t num_samples, uint8_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad);


/**
 * Performs linear forward pass.
 * 
 * @param X The X tensor with dimension [num_samples, in_features], to be updated in-place with the output of the linear pass.
 *  The output dimension would be [num_samples, out_features].
 * @param linear_weights The weights of the linear layer with dimension [out_features, in_features].
 * @param grad The gradient tensor for storing dW and dX, which is essentially X and W, respectively.
 * @param compute_grad Whether to compute the gradients (e.g., it's not needed for evaluation purposes).
 * @return The total runtime of the kernel run.
 */
float run_linear_forward(Tensor *X, Tensor *linear_weights, LayerGradients *grad, bool compute_grad);


/**
 * Performs linear backward pass: computing dW and dX through the chain rule, then updating the weights given learning rate.
 * 
 * @param linear_weights The current weights of the linear layer.
 * @param grad Contains dW and dX of the layer.
 * @param next_layer_grad The gradients from the next layer for obtaining dY.
 * @param lr The learning rate for updating the weights.
 * @return The total runtime of the kernel run.
 */
float run_linear_backward(Tensor *linear_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr);


/**
 * Performs softmax activation function and computes the gradients.
 * 
 * @param X The input tensor, to be updated in-place with the output of the softmax function.
 * @param y_d The one-hot encodings of the labels, to be used for computing gradients.
 * @param grad The gradient tensor for storing the layer's dX.
 * @param compute_grad Whether to compute the gradients (e.g., it's not needed for evaluation purposes).
 * @return The total runtime of the kernel run.
 */
float run_softmax_forward(Tensor *X, uint8_t *y_d, LayerGradients *grad, bool compute_grad);


/**
 * Computes the negative log likelihood(log(tensor)) loss to obtain the cross-entropy loss.
 *  Recall that cross-entropy loss = negative log likelihood(log softmax). In this project specifically,
 *  softmax activation has been run (instead of log softmax for easier gradient computation), so we still
 *  need to perform log on the softmax output to get the input to the NLL loss computation.
 * 
 * @param input_tensor The input tensor (i.e., the output of the softmax activation function); dimension: [num_samples, label_size].
 * @param y_d The one-hot encodings of the labels; dimension: [num_samples, label_size].
 * @return The NLL loss value.
 */
float *compute_negative_log_likelihood_log_loss(Tensor *input_tensor, uint8_t *y_d);


/**
 * Get the count of accurate predictions.
 * 
 * @param input_tensor The input tensor (i.e., the output of the softmax activation function); dimension: [num_samples, label_size].
 * @param y_d The one-hot encodings of the labels; dimension: [num_samples, label_size].
 * @return The number of accurate predictions.
 */
uint32_t *get_accurate_predictions_count(Tensor *input_tensor, uint8_t *y_d);

#endif
