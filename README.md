# MNIST CNN with C and CUDA


This repo was directly copied from the [Programming Massively Parallel Processors](https://github.com/katsudon16/programming_massively_parallel_processors) repo, specifically from chapter 16: Deep Learning.

The repo aims to implement basic neural network layers using C and CUDA kernels from scratch.


## MNIST Data

The MNIST data can be found from multiple sources, including [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Download the dataset and store them in the `data` directory:

- Training files
```
- data/train-images-idx3-ubyte
- data/train-labels-idx1-ubyte
```

- Test files
```
- data/t10k-images-idx3-ubyte
- data/t10k-labels-idx1-ubyte
```

## Implementation

The main code can be found in `train_mnist.cu` file. 


### Image preprocessing

The initial 28x28 `uint8_t` images are:

1) converted into `float`
2) normalized (i.e., dividing by 255)
3) padded with 0s into 32x32 images


### Model architecture

The model architecture currently consists of the following layers:

- 2D convolutional layers: 16 filters, each `5 x 5`; produces output dimension of `[batch_size, 16, 28, 28]`
- Sigmoid layer; used *simply because* it's mentioned in the book
- Max pooling layer with kernel size `2 x 2`; produces output dimension of `[batch_size, 16, 14, 14]`
- Flatten layer; produces output dim of `[batch_size, 3136]`
- Linear layer with dimension `3136 x 10`; produces output dim of `[batch_size, 10]`
- Softmax layer

It then takes the logarithm of the softmax output and performs **negative log-likelihood** to compute the loss. Note that log softmax + NLL Loss is essentially **cross-entropy loss**.

### Tests

More info on the tests can be found in the test README file.


## Expected output

```
[INFO] # Samples in training set: 60000
The train dataset is splitted into training (n=36000) and validation (n=24000).
Epoch 0:
--> train loss: 610.436

Epoch 1:
--> train loss: 149.599
--> valid loss: 70.602 | accuracy: 76.425%

Epoch 2:
--> train loss: 86.212

Epoch 3:
--> train loss: 71.634
--> valid loss: 42.728 | accuracy: 87.442%

Epoch 4:
--> train loss: 63.749

Epoch 5:
--> train loss: 60.004
--> valid loss: 37.932 | accuracy: 88.367%

Epoch 6:
--> train loss: 56.884

Epoch 7:
--> train loss: 54.398
--> valid loss: 35.864 | accuracy: 88.371%

Epoch 8:
--> train loss: 52.558

Epoch 9:
--> train loss: 51.004
--> valid loss: 32.961 | accuracy: 89.721%

[INFO] # Samples in test set: 10000
Test accuracy: 90.520%
```

## Future improvements

The implementation of the CUDA kernels are still naive and may be slow. Further optimizations (or even moving the computation to CPU if using GPU is not necessary) will be done.

Optimizing the model architecture and training (e.g., adding ReLU, batch normalization, early stopping, using better evaluation metric, etc) is currently not the priority.