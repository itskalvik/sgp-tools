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
"""Attentive Kernel function
"""

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.config import default_float

float_type = default_float()

from .neural_network import NN
from typing import List, Union, Optional


class Attentive(gpflow.kernels.Kernel):
    """
    Attentive Kernel function (non-stationary kernel function).

    This kernel uses a Multi-Layer Perceptron (MLP) to learn attention weights
    for a mixture of RBF kernel components, making it adapt to local data
    characteristics. It is based on the implementation from
    [Weizhe-Chen/attentive_kernels](https://github.com/Weizhe-Chen/attentive_kernels).

    Refer to the following paper for more details:
        - AK: Attentive Kernel for Information Gathering [Chen et al., 2022]

    Attributes:
        _free_amplitude (tf.Variable): The amplitude (variance) parameter of the kernel.
        lengthscales (tf.Variable): Fixed lengthscales for each RBF mixture component.
        num_lengthscales (int): Number of RBF mixture components.
        nn (NN): The Neural Network (MLP) used to generate attention representations.
    """

    def __init__(self,
                 lengthscales: Union[List[float], np.ndarray] = None,
                 hidden_sizes: List[int] = None,
                 amplitude: float = 1.0,
                 num_dim: int = 2):
        """
        Initializes the Attentive Kernel.

        Args:
            lengthscales (Union[List[float], np.ndarray]): A list or NumPy array of
                                                        lengthscale values to be used in the
                                                        RBF mixture components. These lengthscales
                                                        are not trained by the optimizer.
            hidden_sizes (List[int]): A list where each element specifies the number of hidden units
                                      in a layer of the MLPs. The length of this list determines
                                      the number of hidden layers. Defaults to [10, 10].
            amplitude (float): Initial amplitude (variance) of the kernel function.
                               This parameter is trainable. Defaults to 1.0.
            num_dim (int): The dimensionality of the input data points (e.g., 2 for 2D data).
                           Defaults to 2.

        Usage:
            ```python
            import gpflow
            import numpy as np
            from sgptools.kernels.attentive import Attentive

            # Example: 10 fixed lengthscales ranging from 0.01 to 2.0
            l_scales = np.linspace(0.01, 2.0, 10).astype(np.float32)
            
            # Initialize Attentive Kernel for 2D data
            kernel = Attentive(lengthscales=l_scales, hidden_sizes=[10, 10], num_dim=2)

            # You can then use this kernel in a GPflow model:
            # model = gpflow.models.GPR(data=(X_train, Y_train), kernel=kernel, noise_variance=0.1)
            # optimize_model(model)
            ```
        """
        super().__init__()
        if lengthscales is None:
            lengthscales = np.linspace(0.01, 2.0, 10)

        if hidden_sizes is None:
            hidden_sizes = [10, 10]  # Default if not provided
        else:
            hidden_sizes = list(hidden_sizes)

        with self.name_scope:
            self.num_lengthscales = len(lengthscales)
            self._free_amplitude = tf.Variable(amplitude,
                                               shape=[],
                                               trainable=True,
                                               dtype=float_type)
            # Lengthscales are treated as fixed parameters in this implementation
            self.lengthscales = tf.Variable(
                tf.cast(lengthscales, float_type),
                shape=[self.num_lengthscales],
                trainable=False,  # Not trainable
                dtype=float_type)

            # The neural network maps input dimensions to the number of lengthscales
            # to produce attention weights for each RBF component.
            # Structure: input_dim -> dim_hidden -> dim_hidden -> num_lengthscales
            self.nn = NN([num_dim] + hidden_sizes + [self.num_lengthscales],
                         output_activation_fn='softplus')

    @tf.autograph.experimental.do_not_convert
    def get_representations(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes normalized latent representations for input data points `X` using the MLP.
        These representations are used to calculate attention weights for the kernel mixture.

        Args:
            X (tf.Tensor): (N, D); Input data points. `N` is the number of points,
                           `D` is the input dimensionality (`num_dim`).

        Returns:
            tf.Tensor: (N, num_lengthscales); Normalized latent representations for each input point.
        """
        Z = self.nn(X)
        # Normalize the representations to have unit L2-norm along the last axis.
        # This is common in attention mechanisms.
        representations = Z / tf.norm(Z, axis=1, keepdims=True)
        return representations

    @tf.autograph.experimental.do_not_convert
    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the covariance matrix between input data points `X` and `X2`.
        If `X2` is None, it computes the covariance matrix `K(X, X)`.

        The covariance is calculated as a weighted sum of RBF kernels, where
        the weights are derived from the attention representations generated by the MLP.

        Formula (simplified):
        $K(X, X') = \text{amplitude} \times \text{attention}(X, X') \times \sum_{i=1}^{Q} \text{RBF}(||X-X'||, \text{lengthscale}_i) \times \text{attention_lengthscale}_i(X,X')$
        where $\text{attention}(X, X') = \text{representation}(X) \cdot \text{representation}(X')^T$.

        Args:
            X (tf.Tensor): (N1, D); Input data points. `N1` is the number of points,
                           `D` is the input dimensionality.
            X2 (Optional[tf.Tensor]): (N2, D); Optional second set of input data points.
                                     If None, `X` is used as `X2`. `N2` is the number of points.

        Returns:
            tf.Tensor: (N1, N2); The computed covariance matrix.
        """
        if X2 is None:
            X2_internal = X
        else:
            X2_internal = X2

        # Compute pairwise Euclidean distances between X and X2
        dist = cdist(X,
                     X2_internal)  # This returns (N1, N2) Euclidean distances

        # Get normalized latent representations for X and X2
        repre1 = self.get_representations(X)  # (N1, num_lengthscales)
        repre2 = self.get_representations(
            X2_internal)  # (N2, num_lengthscales)

        # Function to compute a single mixture component for the kernel
        # This function is mapped over each lengthscale index 'i'
        def get_mixture_component(i: tf.Tensor) -> tf.Tensor:
            """
            Computes a single RBF mixture component, incorporating attention
            based on the i-th dimension of the representations.
            """
            # attention_lengthscales: (N1, N2) matrix
            # This term scales the RBF based on similarity in the i-th latent dimension.
            attention_lengthscales = tf.tensordot(repre1[:, i],
                                                  repre2[:, i],
                                                  axes=0)

            # rbf(dist, self.lengthscales[i]) computes the RBF kernel for the current lengthscale
            # Element-wise multiplication with attention_lengthscales applies the attention.
            cov_mat_component = rbf(
                dist, self.lengthscales[i]) * attention_lengthscales
            return cov_mat_component

        # tf.map_fn applies `get_mixture_component` to each lengthscale index.
        # The result `cov_mat_per_ls` will be (num_lengthscales, N1, N2).
        cov_mat_per_ls = tf.map_fn(fn=get_mixture_component,
                                   elems=tf.range(self.num_lengthscales,
                                                  dtype=tf.int64),
                                   fn_output_signature=dist.dtype)

        # Sum all mixture components along the first axis to get (N1, N2)
        cov_mat_summed_components = tf.math.reduce_sum(cov_mat_per_ls, axis=0)

        # Overall attention term based on the dot product of representations
        # (N1, num_lengthscales) @ (num_lengthscales, N2) -> (N1, N2)
        attention_inputs = tf.matmul(repre1, repre2, transpose_b=True)

        # Final covariance: Apply the learned amplitude and the overall attention
        # Element-wise multiplication to scale the summed RBF components
        final_cov_mat = self._free_amplitude * attention_inputs * cov_mat_summed_components

        return final_cov_mat

    @tf.autograph.experimental.do_not_convert
    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the diagonal of the covariance matrix `K(X, X)`.

        Args:
            X (tf.Tensor): (N, D); Input data points. `N` is the number of points.

        Returns:
            tf.Tensor: (N,); A 1D tensor representing the diagonal elements of the
                        covariance matrix.
        """
        return self._free_amplitude * tf.ones((X.shape[0], ), dtype=X.dtype)


# --- Helper functions for kernel computations ---
@tf.autograph.experimental.do_not_convert
def rbf(dist: tf.Tensor, lengthscale: tf.Tensor) -> tf.Tensor:
    """
    Computes the Radial Basis Function (RBF) kernel component.

    Formula: $k(d, l) = \exp(-0.5 \times (d/l)^2)$

    Args:
        dist (tf.Tensor): Pairwise Euclidean distances (or other relevant distances).
                          Can be (N1, N2) or (N,).
        lengthscale (tf.Tensor): The lengthscale parameter, typically a scalar tensor.

    Returns:
        tf.Tensor: The RBF kernel values.
    """
    return tf.math.exp(-0.5 * tf.math.square(dist / lengthscale))


@tf.autograph.experimental.do_not_convert
def cdist(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Calculates the pairwise Euclidean distances between two sets of points.

    Args:
        x (tf.Tensor): (N1, D); First set of points.
        y (tf.Tensor): (N2, D); Second set of points.

    Returns:
        tf.Tensor: (N1, N2); A tensor where element (i, j) is the Euclidean distance
                   between `x[i, :]` and `y[j, :]`.
    """
    # Define a function to compute distances for a single row of `x` against all rows of `y`.
    # The `axis=1` ensures the norm is taken over the last dimension (the coordinates),
    # resulting in a scalar distance for each pair.
    per_x_dist = lambda i: tf.norm(x[i:(i + 1), :] - y, axis=1)

    # Use `tf.map_fn` to apply `per_x_dist` to each row of `x`.
    # `elems=tf.range(tf.shape(x)[0], dtype=tf.int64)` creates a sequence of indices (0, 1, ..., N1-1).
    distances = tf.map_fn(fn=per_x_dist,
                          elems=tf.range(tf.shape(x)[0], dtype=tf.int64),
                          fn_output_signature=x.dtype)

    return distances
