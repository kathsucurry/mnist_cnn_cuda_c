#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "../includes/cnn_layers.cuh"
#include "../includes/kernel_functions.cuh"
#include "../includes/common.cuh"


uint32_t get_tensor_size(const uint32_t *dim, const uint8_t dim_size) {
    uint32_t size = 1;
    for (uint8_t i = 0; i < dim_size; ++i)
        size *= dim[i];
    return size;
}


Tensor *generate_tensor(float *X, uint32_t *dim, uint8_t dim_size) {
    Tensor *tensor = (Tensor *)malloc_check(sizeof(Tensor));
    tensor->dim_size = dim_size;
    tensor->dim = dim;

    uint32_t size = get_tensor_size(dim, dim_size);
    float *values_d;
    gpu_error_check(cudaMalloc((void**)&values_d, size * sizeof(float)));
    gpu_error_check(cudaMemcpy(values_d, X, size * sizeof(float), cudaMemcpyHostToDevice));

    tensor->values_d = values_d;
    return tensor;
}


Tensor *deep_copy_tensor(Tensor *tensor) {
    Tensor *new_tensor = (Tensor *)malloc_check(sizeof(Tensor));
    new_tensor->dim_size = tensor->dim_size;

    new_tensor->dim = (uint32_t *)malloc_check(new_tensor->dim_size * sizeof(uint32_t));
    memcpy(new_tensor->dim, tensor->dim, new_tensor->dim_size * sizeof(uint32_t));

    uint32_t out_size = get_tensor_size(new_tensor->dim, new_tensor->dim_size);
    float *new_tensor_values_d;
    gpu_error_check(cudaMalloc((void**)&new_tensor_values_d, out_size * sizeof(float)));
    gpu_error_check(cudaMemcpy(new_tensor_values_d, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToDevice));

    new_tensor->values_d = new_tensor_values_d;
    return new_tensor;
}


void free_tensor(Tensor *tensor) {
    if (!tensor)
        return;
    gpu_error_check(cudaFree(tensor->values_d));
    free(tensor->dim);
    free(tensor);
}


void free_layer_gradients(LayerGradients *gradients) {
    free_tensor(gradients->dW_or_W);
    free_tensor(gradients->dX_or_X);
    free(gradients);
}


void free_network_weights(NetworkWeights *weights) {
    free_tensor(weights->conv2d_weight);
    free_tensor(weights->linear_weight);
    free(weights);
}


void free_network_outputs(NetworkOutputs *output, bool include_grad) {
    if (include_grad)
        for (uint32_t layer_index = 0; layer_index < output->num_layers; ++layer_index) {
            LayerGradients gradient = output->gradients[layer_index];
            free_tensor(gradient.dW_or_W);
            free_tensor(gradient.dX_or_X);
        }
    free(output->gradients);
    free_tensor(output->output);
    free(output);
}


float *_uniform_xavier_initialization(uint32_t fan_in, uint32_t fan_out, uint32_t size, uint32_t seed) {
    // Assume gain = 1.
    srand(seed);
    float x = sqrtf(6.0 / (fan_in + fan_out));
    float *array = (float *)malloc_check(size * sizeof(float));
    for (uint32_t i = 0; i < size; ++i)
        array[i] = x * 2 * (rand() * 1.0 / RAND_MAX) - x; 
    return array;
}


cudaError_t update_conv2d_const_filters(float *filters_d, uint32_t size) {
    return cudaMemcpyToSymbol(const_conv2d_filters, filters_d, size * sizeof(float), 0, cudaMemcpyDeviceToDevice);
}


/**
 * Store filter weights in the constant variable.
 * 
 * Here, we only assume one convolutional layer. To further expand this, we can
 * store the current length (or the current start index for the next layer weights) of the constant and save
 * the start index every time we store a new set of layer weights.
 * 
 */
cudaError_t update_conv2d_const_filters(float *filters_d, uint32_t size) {
    return cudaMemcpyToSymbol(const_conv2d_filters, filters_d, size * sizeof(float), 0, cudaMemcpyDeviceToDevice);
}


Tensor *initialize_conv_layer_weights(
    uint32_t in_channels,
    uint32_t out_channels,
    uint8_t filter_length,
    uint32_t seed
) {
    Tensor *conv_weight = (Tensor *)malloc_check(sizeof(Tensor));
    // Dimensions = out_channels * in_channels * filter_length * filter_length.
    conv_weight->dim_size = 4;
    conv_weight->dim = (uint32_t *)malloc_check(conv_weight->dim_size * sizeof(uint32_t));
    conv_weight->dim[0] = out_channels;
    conv_weight->dim[1] = in_channels;
    conv_weight->dim[2] = filter_length;
    conv_weight->dim[3] = filter_length;

    uint32_t weight_size = out_channels * in_channels * filter_length * filter_length;
    uint32_t fan_in = in_channels * filter_length * filter_length;
    uint32_t fan_out = out_channels * filter_length * filter_length;

    float *filters = _uniform_xavier_initialization(fan_in, fan_out, weight_size, seed);
    float *filters_d;
    gpu_error_check(cudaMalloc((void**)&filters_d, weight_size * sizeof(float)));
    gpu_error_check(cudaMemcpy(filters_d, filters, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    conv_weight->values_d = filters_d;
    
    // Store in constant memory for faster kernel run.
    update_conv2d_const_filters(filters_d, weight_size);
    
    free(filters);

    return conv_weight;
}


Tensor *initialize_linear_layer_weights(uint32_t in_features, uint32_t out_features, uint32_t seed) {
    Tensor *linear_weight = (Tensor *)malloc_check(sizeof(Tensor));
    linear_weight->dim_size = 2;

    linear_weight->dim = (uint32_t *)malloc_check(linear_weight->dim_size * sizeof(uint32_t));
    linear_weight->dim[0] = out_features;
    linear_weight->dim[1] = in_features;
    uint32_t weight_size = out_features * in_features;

    float *weights = _uniform_xavier_initialization(in_features, out_features, weight_size, seed);
    float *weights_d;
    gpu_error_check(cudaMalloc((void**)&weights_d, weight_size * sizeof(float)));
    gpu_error_check(cudaMemcpy(weights_d, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    linear_weight->values_d = weights_d;
    free(weights);

    return linear_weight;
}


float run_conv2d_forward(Tensor *X, Tensor *filters, LayerGradients *grad, bool compute_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    uint32_t num_samples = X->dim[0];
    uint32_t in_channels = X->dim[1];
    uint32_t in_height   = X->dim[2];
    uint32_t in_width    = X->dim[3];

    uint32_t filter_length = filters->dim[filters->dim_size - 1];
    uint32_t out_height    = in_height - filter_length + 1;
    uint32_t out_width     = in_width - filter_length + 1;
    uint32_t out_channels  = filters->dim[0];
    uint32_t out_size      = num_samples * out_channels * out_height * out_width;

    if (compute_grad) {
        // Store tensors for backprop later.
        grad->dW_or_W = NULL;
        grad->dX_or_X = deep_copy_tensor(X);
        grad->is_grad = false;
    }

    float *Y_d;
    gpu_error_check(cudaMalloc((void**)&Y_d, out_size * sizeof(float)));

    uint32_t grid_width = ceil(out_width * 1.0 / TILE_WIDTH_L);
    uint32_t grid_height = ceil(out_height * 1.0 / TILE_WIDTH_L);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH_L, TILE_WIDTH_L, 1);
    dim3 dimGrid(out_channels, out_tiles_num, num_samples);
    
    cudaEventRecord(start);
    Conv2DForwardSimpleKernel<<<dimGrid, dimBlock>>>(
        Y_d, X->values_d,
        filter_length,
        in_channels,
        grid_width,
        in_height, in_width
    );
    gpu_error_check(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Update X (output) tensor.
    X->dim_size = 4;
    
    free(X->dim);
    X->dim = (uint32_t *)malloc_check(X->dim_size * sizeof(uint32_t));
    X->dim[0] = num_samples;
    X->dim[1] = out_channels;
    X->dim[2] = out_height;
    X->dim[3] = out_width;
    
    gpu_error_check(cudaFree(X->values_d));
    X->values_d = Y_d;

    return time_spent_ms;
}


float run_conv2d_backward(Tensor *conv2d_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Recall that in grad stores W in dW and X in dX (i.e., not gradient values).
    // num samples, out channels, filter length
    Tensor *dY = next_layer_grad->dX_or_X;
    Tensor *X  = grad->dX_or_X;
    uint32_t num_samples   = X->dim[0];
    uint32_t in_channels   = conv2d_weights->dim[1];
    uint32_t out_channels  = conv2d_weights->dim[0];
    uint32_t filter_length = conv2d_weights->dim[conv2d_weights->dim_size - 1]; 
    uint32_t in_height     = X->dim[2];
    uint32_t in_width      = X->dim[3];
    uint32_t out_height    = in_height - filter_length + 1;
    uint32_t out_width     = in_width - filter_length + 1;
    uint32_t in_size       = num_samples * in_channels * in_height * in_width;
    uint32_t weight_size   = out_channels * in_channels * filter_length * filter_length;

    // Calculate dX.
    float *dX_d;
    gpu_error_check(cudaMalloc((void**)&dX_d, in_size * sizeof(float)));
    cudaMemset(dX_d, 0, in_size * sizeof(float));

    uint32_t grid_width = ceil(in_width * 1.0 / TILE_WIDTH);
    uint32_t grid_height = ceil(in_height * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGriddX(in_channels, out_tiles_num, num_samples);
    cudaEventRecord(start);
    Conv2DBackwardXGradKernel<<<dimGriddX, dimBlock>>>(
        dX_d, dY->values_d, conv2d_weights->values_d,
        filter_length,
        out_channels,
        grid_width,
        in_height, in_width
    );
    gpu_error_check(cudaGetLastError());

    Tensor *dX   = (Tensor *)malloc_check(sizeof(Tensor));
    dX->values_d = dX_d;
    dX->dim_size = 4;
    dX->dim      = (uint32_t *)malloc_check(4 * sizeof(uint32_t));
    memcpy(dX->dim, X->dim, 4 * sizeof(uint32_t));

    // Calculate dW.
    float *dW_d;
    gpu_error_check(cudaMalloc((void**)&dW_d, weight_size * sizeof(float)));
    cudaMemset(dW_d, 0, weight_size * sizeof(float));

    grid_width  = ceil(filter_length * 1.0 / TILE_WIDTH);
    grid_height = ceil(filter_length * 1.0 / TILE_WIDTH);
    out_tiles_num = grid_width * grid_height;

    dim3 dimGriddW(in_channels, out_tiles_num, out_channels);
    Conv2DBackwardWGradKernel<<<dimGriddW, dimBlock>>>(
        dW_d, dY->values_d, X->values_d,
        num_samples,
        filter_length,
        grid_width,
        out_height, out_width
    );
    gpu_error_check(cudaGetLastError());

    Tensor *dW   = (Tensor *)malloc_check(sizeof(Tensor));
    dW->values_d = dW_d;
    dW->dim_size = 4;
    dW->dim      = (uint32_t *)malloc_check(4 * sizeof(uint32_t));
    memcpy(dW->dim, conv2d_weights->dim, 4 * sizeof(uint32_t));

    // Update W.
    UpdateConv2DWeightsKernel<<<dimGriddW, dimBlock>>>(
        conv2d_weights->values_d, dW_d, filter_length, filter_length, grid_width, lr
    );
    gpu_error_check(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Update grad.
    free_tensor(X);
    grad->dX_or_X = dX;
    grad->dW_or_W = dW;
    grad->is_grad = true;

    return time_spent_ms;
}


float run_sigmoid_forward(Tensor *X, LayerGradients *grad, bool compute_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    uint32_t num_samples    = X->dim[0];
    uint32_t num_channels   = X->dim[1];
    uint32_t feature_height = X->dim[2];
    uint32_t feature_width  = X->dim[3];
    uint32_t out_size       = num_samples * num_channels * feature_height * feature_width;

    float *Y_d;
    gpu_error_check(cudaMalloc((void**)&Y_d, out_size * sizeof(float)));

    // Prepare gradients.
    float *grad_values_d;
    gpu_error_check(cudaMalloc((void**)&grad_values_d, out_size * sizeof(float)));
    cudaMemset(grad_values_d, 0, out_size * sizeof(float));

    // Set tile width.
    uint32_t grid_height = ceil(feature_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(feature_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);

    cudaEventRecord(start);
    SigmoidForwardKernel<<<dimGrid, dimBlock>>>(
        Y_d, X->values_d, 
        grad_values_d,
        grid_width,
        feature_height, feature_width
    );
    gpu_error_check(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (compute_grad) {
        Tensor *dX = (Tensor *)malloc_check(sizeof(Tensor));
        dX->dim_size = X->dim_size;
        dX->dim = (uint32_t *)malloc_check(dX->dim_size * sizeof(uint32_t));
        memcpy(dX->dim, X->dim, dX->dim_size * sizeof(uint32_t));
        dX->values_d = grad_values_d;

        grad->dW_or_W = NULL;
        grad->dX_or_X = dX;
        grad->is_grad = true;
    } else {
        gpu_error_check(cudaFree(grad_values_d));
    }

    // Update tensor.
    gpu_error_check(cudaFree(X->values_d));
    X->values_d = Y_d;

    return time_spent_ms;
}


float run_sigmoid_backward(LayerGradients *grad, LayerGradients *next_layer_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Tensor *dX = grad->dX_or_X;
    uint32_t num_samples    = dX->dim[0];
    uint32_t num_channels   = dX->dim[1];
    uint32_t feature_height = dX->dim[2];
    uint32_t feature_width  = dX->dim[3];

    uint32_t grid_height = ceil(feature_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(feature_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);
    cudaEventRecord(start);
    MultiplyKernel<<<dimGrid, dimBlock>>>(
        dX->values_d, next_layer_grad->dX_or_X->values_d,
        grid_width,
        feature_height, feature_width
    );
    gpu_error_check(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_spent_ms;
}


float run_pooling_forward(Tensor *X, uint32_t kernel_length, pooling_type pool_type, LayerGradients *grad, bool compute_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t num_samples    = X->dim[0];
    uint32_t num_channels   = X->dim[1];
    uint32_t feature_height = X->dim[2];
    uint32_t feature_width  = X->dim[3];
    uint32_t out_height     = feature_height / kernel_length;
    uint32_t out_width      = feature_width / kernel_length;
    uint32_t in_size        = num_samples * num_channels * feature_height * feature_width;
    uint32_t out_size       = num_samples * num_channels * out_height * out_width;

    float *Y_d;
    gpu_error_check(cudaMalloc((void**)&Y_d, out_size * sizeof(float)));

    uint32_t grid_height = ceil(out_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(out_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);

    if (pool_type != MEAN && pool_type != MAX) {
        fprintf(stderr, "The inputted pooling type is currently not implemented.");
        free_tensor(X);
        gpu_error_check(cudaFree(Y_d));
        exit(EXIT_FAILURE);
    }

    // Store gradients.
    float *grad_values_d;
    gpu_error_check(cudaMalloc((void**)&grad_values_d, in_size * sizeof(float)));
    cudaMemset(grad_values_d, 0, in_size * sizeof(float));

    cudaEventRecord(start);
    PoolForwardKernel<<<dimGrid, dimBlock>>>(
        Y_d, X->values_d, 
        pool_type,
        grad_values_d,
        kernel_length,
        grid_width,
        feature_height, feature_width,
        out_height, out_width
    );
    gpu_error_check(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (compute_grad) {
        Tensor *dX = (Tensor *)malloc_check(sizeof(Tensor));
        dX->dim_size = X->dim_size;
        dX->dim = (uint32_t *)malloc_check(dX->dim_size * sizeof(uint32_t));
        memcpy(dX->dim, X->dim, dX->dim_size * sizeof(uint32_t));
        dX->values_d = grad_values_d;

        grad->dW_or_W = NULL;
        grad->dX_or_X = dX;
        grad->is_grad = true;
    } else
        gpu_error_check(cudaFree(grad_values_d));

    // Update tensor.
    X->dim[2] = out_height;
    X->dim[3] = out_width;
    gpu_error_check(cudaFree(X->values_d));
    X->values_d = Y_d;
    
    return time_spent_ms;
}


float run_pooling_backward(uint32_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Tensor *dX = grad->dX_or_X;
    uint32_t num_samples  = dX->dim[0];
    uint32_t num_channels = dX->dim[1];
    uint32_t grad_height  = dX->dim[2];
    uint32_t grad_width   = dX->dim[3];
    uint32_t next_layer_grad_height = grad_height / kernel_length;
    uint32_t next_layer_grad_width  =  grad_width / kernel_length;
    
    uint32_t grid_height = ceil(grad_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(grad_width * 1.0 / TILE_WIDTH);
    uint32_t grad_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, grad_tiles_num, num_samples);

    cudaEventRecord(start);
    PoolBackwardKernel<<<dimGrid, dimBlock>>>(
        dX->values_d, next_layer_grad->dX_or_X->values_d,
        kernel_length,
        grid_width,
        grad_height, grad_width,
        next_layer_grad_height, next_layer_grad_width
    );
    gpu_error_check(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_spent_ms;
}


float run_flatten_forward(Tensor *X) {
    // Make sure to keep the sample dimension.
    uint32_t num_samples = X->dim[0];
    uint32_t size = get_tensor_size(X->dim, X->dim_size) / num_samples;
    X->dim_size = 2;
    free(X->dim);

    X->dim = (uint32_t *)malloc_check(X->dim_size * sizeof(uint32_t));
    X->dim[0] = num_samples;
    X->dim[1] = size;

    return 0;
}


float run_flatten_backward(uint32_t num_samples, uint8_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad) {
    grad->dW_or_W = NULL;
    
    // Update dX.
    Tensor *dX = deep_copy_tensor(next_layer_grad->dX_or_X);
    // Derive dimensions from num_samples and kernel_length:
    // out_size = num_samples * num_channels * kernel_length**2.
    uint32_t out_size     = get_tensor_size(dX->dim, dX->dim_size);
    uint32_t num_channels = out_size / (num_samples * kernel_length * kernel_length);
    dX->dim_size = 4;

    free(dX->dim);
    dX->dim = (uint32_t *)malloc_check(dX->dim_size * sizeof(uint32_t));
    dX->dim[0] = num_samples;
    dX->dim[1] = num_channels;
    dX->dim[2] = kernel_length;
    dX->dim[3] = kernel_length;

    grad->dX_or_X = dX;
    grad->is_grad = true;

    return 0;
}


float run_linear_forward(Tensor *X, Tensor *linear_weights, LayerGradients *grad, bool compute_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t in_features  = linear_weights->dim[1];
    uint32_t out_features = linear_weights->dim[0];
    uint32_t num_samples  = X->dim[0];
    uint32_t out_size     = num_samples * out_features;

    // Run linear layer.
    float *Y_d;
    gpu_error_check(cudaMalloc((void**)&Y_d, out_size * sizeof(float)));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(out_features * 1.0 / (TILE_WIDTH * THREAD_COARSENING_FACTOR)), ceil(num_samples * 1.0 / TILE_WIDTH));
    cudaEventRecord(start);
    LinearForwardKernel<<<dimGrid, dimBlock>>>(
        Y_d,
        X->values_d,
        linear_weights->values_d,
        num_samples,
        in_features, out_features
    );
    gpu_error_check(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (compute_grad) {
        // Store gradients.
        Tensor *dW = (Tensor *)malloc_check(sizeof(Tensor));
        Tensor *dX = (Tensor *)malloc_check(sizeof(Tensor));
        
        float *dW_values, *dX_values;
        gpu_error_check(cudaMalloc((void**)&dW_values, num_samples * in_features * sizeof(float)));
        gpu_error_check(cudaMemcpy(dW_values, X->values_d, num_samples * in_features * sizeof(float), cudaMemcpyHostToDevice));
        gpu_error_check(cudaMalloc((void**)&dX_values, out_features * in_features * sizeof(float)));
        gpu_error_check(cudaMemcpy(dX_values, linear_weights->values_d, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice));

        dW->dim = (uint32_t *)malloc_check(2 * sizeof(uint32_t));
        dW->dim[0]   = num_samples;
        dW->dim[1]   = in_features;
        dW->dim_size = 2;
        dW->values_d = dW_values;

        dX->dim = (uint32_t *)malloc_check(2 * sizeof(uint32_t));
        dX->dim[0]   = out_features;
        dX->dim[1]   = in_features;
        dX->dim_size = 2;
        dX->values_d = dX_values;

        grad->dW_or_W = dW;
        grad->dX_or_X = dX;
        grad->is_grad = true;
    }

    // Update tensor.
    X->dim[1] = out_features;
    gpu_error_check(cudaFree(X->values_d));
    X->values_d = Y_d;

    return time_spent_ms;
}


float run_linear_backward(Tensor *linear_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Recall dY has dimension [num_samples x out_features].
    Tensor *dY = next_layer_grad->dX_or_X;
    Tensor *dW = grad->dW_or_W;
    Tensor *dX = grad->dX_or_X;

    uint32_t num_samples  = dY->dim[0];
    uint32_t out_features = dY->dim[1];
    uint32_t in_features  = dW->dim[1];

    float *dYT, *updated_dW_d, *updated_dX_d;
    gpu_error_check(cudaMalloc((void**)&dYT, num_samples * out_features * sizeof(float)));
    cudaMemset(dYT, 0, num_samples * out_features * sizeof(float));
    
    // Update dW = dY.T @ dW.
    // Transpose dY.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGridTranspose(ceil(num_samples * 1.0 / TILE_WIDTH), ceil(out_features * 1.0 / TILE_WIDTH));
    cudaEventRecord(start);
    TransposeMatrixKernel<<<dimGridTranspose, dimBlock>>>(dYT, dY->values_d, out_features, num_samples);
    gpu_error_check(cudaGetLastError());

    gpu_error_check(cudaMalloc((void**)&updated_dW_d, out_features * in_features * sizeof(float)));
    dim3 dimGridUpdateDW(ceil(in_features * 1.0 / TILE_WIDTH / THREAD_COARSENING_FACTOR), ceil(out_features * 1.0 / TILE_WIDTH));
    MatMulKernel<<<dimGridUpdateDW, dimBlock>>>(updated_dW_d, dYT, dW->values_d, out_features, num_samples, in_features);
    gpu_error_check(cudaGetLastError());

    gpu_error_check(cudaFree(dYT));
    gpu_error_check(cudaFree(dW->values_d));
    dW->values_d = updated_dW_d;
    dW->dim[0]   = out_features;

    // Update dX = dY @ dX.
    dim3 dimGridUpdateDX(ceil(in_features * 1.0 / TILE_WIDTH / THREAD_COARSENING_FACTOR), ceil(num_samples * 1.0 / TILE_WIDTH));
    gpu_error_check(cudaMalloc((void**)&updated_dX_d, num_samples * in_features * sizeof(float)));
    MatMulKernel<<<dimGridUpdateDX, dimBlock>>>(updated_dX_d, dY->values_d, dX->values_d, num_samples, out_features, in_features);
    gpu_error_check(cudaGetLastError());
    
    gpu_error_check(cudaFree(dX->values_d));
    dX->values_d = updated_dX_d;
    dX->dim[0]   = num_samples;

    // Update weights W = W - lr * dW.
    dim3 dimGridUpdateW(ceil(in_features * 1.0 / TILE_WIDTH), ceil(out_features * 1.0 / TILE_WIDTH));
    UpdateLinearWeightsKernel<<<dimGridUpdateW, dimBlock>>>(linear_weights->values_d, updated_dW_d, out_features, in_features, lr);
    gpu_error_check(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_spent_ms;
}


/**
 * Perform softmax function on a 2D tensor across column.
 * 
 */
float run_softmax_forward(Tensor *X, uint8_t *y_d, LayerGradients *grad, bool compute_grad) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (X->dim_size != 2) {
        fprintf(stderr, "The input tensor must have 2 dimensions to perform softmax function.\n");
        free_tensor(X);
        X = NULL;
        return 0;
    }

    uint32_t num_samples  = X->dim[0];
    uint32_t num_features = X->dim[1];
    uint32_t out_size     = num_samples * num_features;

    float *X_output_d, *X_exp_sum_d;
    gpu_error_check(cudaMalloc((void**)&X_output_d, out_size * sizeof(float)));
    gpu_error_check(cudaMalloc((void**)&X_exp_sum_d, num_samples * sizeof(float)));
    cudaMemset(X_exp_sum_d, 0, num_samples * sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(num_features * 1.0 / TILE_WIDTH), ceil(num_samples * 1.0 / TILE_WIDTH));
    cudaEventRecord(start);
    CalcExpAndSumByRowKernel<<<dimGrid, dimBlock>>>(
        X_output_d, X_exp_sum_d, X->values_d, num_samples, num_features
    );
    gpu_error_check(cudaGetLastError());

    NormalizeKernel<<<dimGrid, dimBlock>>>(X_output_d, X_exp_sum_d, num_samples, num_features);
    gpu_error_check(cudaGetLastError());

    gpu_error_check(cudaFree(X_exp_sum_d));
    
    // Update X.
    gpu_error_check(cudaFree(X->values_d));
    X->values_d = X_output_d;

    if (compute_grad) {
        // Compute gradients assuming cross entropy loss.
        float *dX_d;
        gpu_error_check(cudaMalloc((void**)&dX_d, out_size * sizeof(float)));
        cudaMemset(dX_d, 0, out_size * sizeof(float));
        SoftmaxGradientKernel<<<dimGrid, dimBlock>>>(dX_d, X_output_d, y_d, num_samples, num_features);
        gpu_error_check(cudaGetLastError());

        Tensor *dX = (Tensor *)malloc_check(sizeof(Tensor));
        dX->dim_size = 2;
        dX->dim = (uint32_t *)malloc_check(2 * sizeof(uint32_t));
        dX->dim[0] = X->dim[0];
        dX->dim[1] = X->dim[1];
        dX->values_d = dX_d;

        grad->dW_or_W = NULL;
        grad->dX_or_X = dX;
        grad->is_grad = true;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_spent_ms = 0;

    cudaEventElapsedTime(&time_spent_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_spent_ms;
}


float *compute_negative_log_likelihood_log_loss(Tensor *input_tensor, uint8_t *y_d) {
    uint32_t num_samples  = input_tensor->dim[0];
    uint32_t num_features = input_tensor->dim[1];

    float *out_d;
    gpu_error_check(cudaMalloc((void**)&out_d, sizeof(float)));
    cudaMemset(out_d, 0, sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(LABEL_SIZE * 1.0 / (TILE_WIDTH * THREAD_COARSENING_FACTOR)), ceil(num_samples * 1.0 / TILE_WIDTH));
    NegativeLogLikelihoodLogKernel<<<dimGrid, dimBlock>>>(out_d, input_tensor->values_d, y_d, num_samples, num_features);
    gpu_error_check(cudaGetLastError());

    float *out = (float *)malloc_check(sizeof(float));
    gpu_error_check(cudaMemcpy(out, out_d, sizeof(float), cudaMemcpyDeviceToHost));
    gpu_error_check(cudaFree(out_d));

    return out;
}


uint32_t *get_accurate_predictions_count(Tensor *input_tensor, uint8_t *y_d) {
    uint32_t num_samples  = input_tensor->dim[0];
    uint32_t num_features = input_tensor->dim[1];

    uint32_t *out_d;
    gpu_error_check(cudaMalloc((void**)&out_d, sizeof(uint32_t)));
    cudaMemset(out_d, 0, sizeof(uint32_t));

    uint32_t num_threads = min(num_samples, 1024);
    dim3 dimBlock(num_threads);
    dim3 dimGrid(ceil(num_samples * 1.0 / dimBlock.x));
    GetAccuratePredCountKernel<<<dimGrid, dimBlock>>>(
        out_d, input_tensor->values_d, y_d,
        num_samples, num_features
    );
    gpu_error_check(cudaGetLastError());

    uint32_t *out = (uint32_t *)malloc_check(sizeof(uint32_t));
    gpu_error_check(cudaMemcpy(out, out_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpu_error_check(cudaFree(out_d));

    return out;
}
