# Tests information


`run_training_tests.cu` runs one iteration of forward and backward pass of a smaller scale model architecture on a smaller number of samples.

To get the ground truth, the Xavier-initialized weight values are first generated and stored in `inputs` directory. A Python notebook `prepare_test_outputs.ipynb` is then run using PyTorch. The outputs and gradients of each layer are stored in `outputs` directory to be compared with the outputs of the CUDA/C implementations.