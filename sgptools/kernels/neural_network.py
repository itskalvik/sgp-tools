# Copyright 2024 The SGP-Tools Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi Layer Perceptron Model
"""

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.config import default_float

float_type = default_float()

from typing import List, Union, Callable


def xavier(dim_in: int, dim_out: int) -> np.ndarray:
    """
    Initializes weights using the Xavier (Glorot) uniform initialization method.
    This method aims to keep the variance of activations consistent across layers,
    helping to prevent vanishing/exploding gradients.

    Formula: $W \sim U(-\sqrt{6/(dim_{in} + dim_{out})}, \sqrt{6/(dim_{in} + dim_{out})})$

    Args:
        dim_in (int): The number of input units to the layer.
        dim_out (int): The number of output units from the layer.

    Returns:
        np.ndarray: A NumPy array of shape (dim_in, dim_out) containing
                    the initialized weights.
    """
    # Calculate the fan-in + fan-out for the scaling factor
    scale_factor = (2.0 / (dim_in + dim_out))**0.5
    # Generate random numbers from a normal (Gaussian) distribution
    # This is often used as an approximation for Xavier uniform in practice
    # or sometimes Xavier normal is explicitly implemented this way.
    return np.random.randn(dim_in, dim_out) * scale_factor


class NN(gpflow.base.Module):
    """
    A Multi-Layer Perceptron (MLP) model that is compatible with GPFlow,
    allowing its parameters (weights and biases) to be optimized as part of
    a GPflow model (e.g., within a custom kernel).

    The network consists of multiple fully connected (dense) layers with
    specified activation functions.

    Attributes:
        dims (List[int]): List of layer sizes, including input and output dimensions.
        activation_fn (Callable): Activation function for hidden layers.
        output_activation_fn (Callable): Activation function for the output layer.
        _weights (List[tf.Variable]): List of TensorFlow Variable for weights of each layer.
        _biases (List[tf.Variable]): List of TensorFlow Variable for biases of each layer.
    """

    def __init__(self,
                 dims: List[int],
                 activation_fn: Union[str, Callable] = 'selu',
                 output_activation_fn: Union[str, Callable] = 'softmax'):
        """
        Initializes the Multi-Layer Perceptron (MLP).

        Args:
            dims (List[int]): A list of integers specifying the size of each layer.
                              The first element is the input dimension, the last is
                              the output dimension, and intermediate elements are
                              hidden layer sizes.
                              Example: `[input_dim, hidden1_dim, hidden2_dim, output_dim]`
            activation_fn (Union[str, Callable]): The activation function to use for hidden layers.
                                                  Can be a string (e.g., 'relu', 'tanh', 'selu')
                                                  or a callable TensorFlow activation function.
                                                  Defaults to 'selu'.
            output_activation_fn (Union[str, Callable]): The activation function to use for the output layer.
                                                        Can be a string (e.g., 'softmax', 'sigmoid', 'softplus')
                                                        or a callable TensorFlow activation function.
                                                        Defaults to 'softplus'.

        Usage:
            ```python
            from sgptools.kernels.neural_network import NN
            import tensorflow as tf
            import numpy as np

            # Example: A simple MLP with one hidden layer
            mlp = NN(dims=[2, 10, 1], activation_fn='tanh', output_activation_fn='sigmoid')

            # Input data
            input_data = tf.constant(np.random.rand(5, 2), dtype=tf.float32)

            # Pass input through the network
            output = mlp(input_data)
            ```
        """
        super().__init__()
        self.dims = dims
        # Get TensorFlow activation functions from strings or use provided callables
        self.activation_fn = tf.keras.activations.get(
            activation_fn) if isinstance(activation_fn, str) else activation_fn
        self.output_activation_fn = tf.keras.activations.get(
            output_activation_fn) if isinstance(output_activation_fn,
                                                str) else output_activation_fn

        self._weights: List[tf.Variable] = []
        self._biases: List[tf.Variable] = []

        # Create weights and biases for each layer
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            # Use Xavier initialization for weights
            weight_init = xavier(dim_in, dim_out)
            self._weights.append(
                tf.Variable(weight_init, dtype=float_type, name=f'W_{i}'))

            # Initialize biases to zeros
            bias_init = np.zeros(dim_out, dtype=float_type)
            self._biases.append(
                tf.Variable(bias_init, dtype=float_type, name=f'b_{i}'))

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Performs a forward pass through the MLP.

        Args:
            X (tf.Tensor): (N, D_in); The input tensor to the MLP. `N` is the batch size,
                           `D_in` is the input dimension of the network.

        Returns:
            tf.Tensor: (N, D_out); The output tensor from the MLP. `D_out` is the output
                       dimension of the network.
        """
        # Process through hidden layers
        # The loop runs for (num_layers - 1) iterations, covering all hidden layers
        # and the input-to-first-hidden layer transition.
        for i in range(len(self.dims) -
                       2):  # Iterate up to second to last layer
            W = self._weights[i]
            b = self._biases[i]
            X = self.activation_fn(tf.matmul(X, W) + b)

        # Process through the last layer (output layer)
        W_last = self._weights[-1]  # Weights for the last layer
        b_last = self._biases[-1]  # Biases for the last layer
        X = self.output_activation_fn(tf.matmul(X, W_last) + b_last)

        return X
