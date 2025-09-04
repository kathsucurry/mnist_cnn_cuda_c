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
    bool is_grad; // True if dW/dX is stored, else it saves W/X for later computation in the chain rule.
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
 * Computes the tensor size
 */
uint32_t get_tensor_size(const uint8_t dim_size, const uint32_t *dim);
Tensor *initialize_tensor(float *X, uint8_t dim_size, uint32_t *dim);
Tensor *deep_copy_tensor(Tensor *tensor);
void free_tensor(Tensor *tensor);
void free_layer_gradients(LayerGradients *gradients);
void free_network_weights(NetworkWeights *weights);
void free_network_outputs(NetworkOutputs *output, bool include_grad);

Tensor *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size, uint32_t seed);
Tensor *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels, uint32_t seed);

/* Forward layer functions */

void run_conv2d_forward(Tensor *output, Tensor *filters, LayerGradients *grad, bool compute_grad);
void run_conv2d_backward(Tensor *conv2d_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float learning_rate);

void run_sigmoid_forward(Tensor *tensor, LayerGradients *grad, bool compute_grad);
void run_sigmoid_backward(LayerGradients *grad, LayerGradients *next_layer_grad);

void run_pooling_forward(Tensor *tensor, uint32_t kernel_length, pooling_type pool_type, LayerGradients *grad, bool compute_grad);
void run_pooling_backward(uint32_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad);

void run_flatten_forward(Tensor *tensor);
void run_flatten_backward(uint32_t num_samples, uint8_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad);

void run_linear_forward(Tensor *X, Tensor *linear_weights, LayerGradients *grad, bool compute_grad);
void run_linear_backward(Tensor *linear_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr);

void run_softmax_forward(Tensor *tensor, uint8_t *y_d, LayerGradients *grad, bool compute_grad);

float *compute_negative_log_likelihood_log_lost(Tensor *tensor, uint8_t *y_d);

uint32_t *get_accurate_predictions(Tensor *logits, uint8_t *y_d);

#endif
