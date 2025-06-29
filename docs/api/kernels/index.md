# `kernels`: Custom Kernel Functions
This module contains advanced, non-stationary kernel functions for the Gaussian Process models.

* **`AttentiveKernel`:** A non-stationary kernel that uses a neural network to learn attention weights for a mixture of RBF kernels. This allows the model to adapt its assumptions about the data's correlation structure across the input space.

* **`NeuralSpectralKernel`:** Another non-stationary kernel that employs Multilayer perceptrons (MLPs) to learn the frequency, lengthscale, and variance of a spectral mixture. This provides a flexible way to model complex, non-stationary data.