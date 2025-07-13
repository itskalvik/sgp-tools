# `kernels`: Kernel Functions
This module contains kernel functions for the Gaussian Process models. This includes all kernel functions in gpflow, and advanced, non-stationary kernels. Use the `get_kernel` method to retrieve a kernel class by its string name.


* **`Attentive`:** A non-stationary kernel that uses a neural network to learn attention weights for a mixture of RBF kernels. This allows the model to adapt its assumptions about the data's correlation structure across the input space.

* **`NeuralSpectral`:** Another non-stationary kernel that employs Multilayer perceptrons (MLPs) to learn the frequency, lengthscale, and variance of a spectral mixture. This provides a flexible way to model complex, non-stationary data.

* **`gpflow.kernels`:** All available kernels in gpflow's kernels module.
   
::: sgptools.kernels.get_kernel
    options:
      show_root_heading: true
      show_source: true
