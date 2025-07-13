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
"""Provides a neural spectral kernel function along with an initialization function
"""

import tensorflow as tf
import numpy as np
import gc

import gpflow
from gpflow.config import default_jitter, default_float
from gpflow.models import SGPR
from gpflow.models.util import data_input_to_tensor

float_type = default_float()

from .neural_network import NN
from typing import List, Optional, Tuple, Union, Any


class NeuralSpectral(gpflow.kernels.Kernel):
    """
    Neural Spectral Kernel function (non-stationary kernel function).
    This kernel models non-stationarity by using multiple Multi-Layer Perceptrons (MLPs)
    to map input locations to frequency, lengthscale, and variance parameters for a
    mixture of spectral components.

    Based on the implementation from this [repo](https://github.com/sremes/nssm-gp/tree/master?tab=readme-ov-file).

    Refer to the following papers for more details:
        - Neural Non-Stationary Spectral Kernel [Remes et al., 2018]

    Attributes:
        input_dim (int): Dimensionality of the input data points.
        Q (int): Number of MLP mixture components used in the kernel function.
        num_hidden (int): Number of hidden layers in each MLP.
        freq (List[NN]): List of MLPs, one for each component, predicting frequencies.
        length (List[NN]): List of MLPs, one for each component, predicting lengthscales.
        var (List[NN]): List of MLPs, one for each component, predicting variances.
    """

    def __init__(self,
                 input_dim: int = 2,
                 active_dims: Optional[List[int]] = None,
                 Q: int = 1,
                 hidden_sizes: List[int] = None):
        """
        Initializes the Neural Spectral Kernel.

        Args:
            input_dim (int): Number of dimensions of the input data points (e.g., 2 for 2D data).
            active_dims (Optional[List[int]]): A list of indices specifying which input dimensions
                                                the kernel operates on. If None, all dimensions are active.
                                                Defaults to None.
            Q (int): The number of MLP mixture components used in the kernel function.
                     Each component has its own set of MLPs for frequency, lengthscale, and variance.
                     Defaults to 1.
            hidden_sizes (List[int]): A list where each element specifies the number of hidden units
                                      in a layer of the MLPs. The length of this list determines
                                      the number of hidden layers. Defaults to [32, 32].

        Usage:
            ```python
            import gpflow
            import numpy as np
            from sgptools.kernels.neural_spectral import NeuralSpectral

            # Initialize a Neural Spectral Kernel for 2D data with 3 mixture components
            # and MLPs with 2 hidden layers of 64 units each.
            kernel = NeuralSpectral(input_dim=2, Q=3, hidden_sizes=[64, 64])

            # You can then use this kernel in a GPflow model:
            # model = gpflow.models.SGPR(data=(X_train, Y_train), kernel=kernel, ...)
            ```
        """
        super().__init__(active_dims=active_dims)

        if hidden_sizes is None:
            hidden_sizes = [32, 32]  # Default if not provided
        else:
            hidden_sizes = list(hidden_sizes)

        self.input_dim = input_dim
        self.Q = Q
        self.num_hidden = len(hidden_sizes)

        # Initialize lists of MLPs for each component
        self.freq: List[NN] = []
        self.length: List[NN] = []
        self.var: List[NN] = []

        # Create Q sets of MLPs
        for q in range(self.Q):
            # MLP for frequency: maps input_dim -> hidden_sizes -> input_dim
            # Output activation 'softplus' ensures positive frequencies.
            freq_nn = NN([input_dim] + hidden_sizes + [input_dim],
                         output_activation_fn='softplus')

            # MLP for lengthscale: maps input_dim -> hidden_sizes -> input_dim
            # Output activation 'softplus' ensures positive lengthscales.
            length_nn = NN([input_dim] + hidden_sizes + [input_dim],
                           output_activation_fn='softplus')

            # MLP for variance: maps input_dim -> hidden_sizes -> 1 (scalar variance)
            # Output activation 'softplus' ensures positive variances.
            var_nn = NN([input_dim] + hidden_sizes + [1],
                        output_activation_fn='softplus')

            self.freq.append(freq_nn)
            self.length.append(length_nn)
            self.var.append(var_nn)

    @tf.autograph.experimental.do_not_convert
    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the covariance matrix between/amongst the input variables `X` and `X2`.
        If `X2` is None, the function computes `K(X, X)` (a symmetric covariance matrix).
        Otherwise, it computes `K(X, X2)` (a cross-covariance matrix).

        The kernel is a sum over `Q` mixture components, where each component's
        parameters (frequency, lengthscale, variance) are determined by MLPs
        based on the input locations.

        Args:
            X (tf.Tensor): (N1, D); First set of input variables to compute covariance from.
                           `N1` is the number of points, `D` is the dimensionality.
            X2 (Optional[tf.Tensor]): (N2, D); Optional second set of input variables.
                                     If provided, computes cross-covariance `K(X, X2)`.
                                     If None, computes auto-covariance `K(X, X)`.

        Returns:
            tf.Tensor: (N1, N2); The computed covariance matrix. If `X2` is None, the
                       diagonal of `K(X, X)` is jittered for numerical stability.
        """
        if X2 is None:
            X2_internal = X
            equal = True  # Flag to add jitter to diagonal for K(X,X)
        else:
            X2_internal = X2
            equal = False

        kern = tf.constant(0.0, dtype=float_type)  # Initialize kernel sum

        for q in range(self.Q):
            # Compute latent function values (frequencies, lengthscales, variances)
            # by passing input locations through the MLPs.
            freq_X, freq_X2 = self.freq[q](X), self.freq[q](
                X2_internal)  # (N, D) frequencies
            lens_X, lens_X2 = self.length[q](X), self.length[q](
                X2_internal)  # (N, D) lengthscales
            var_X, var_X2 = self.var[q](X), self.var[q](
                X2_internal)  # (N, 1) variances

            # Compute length-scale term (E) - based on inverse lengthscales and distances
            Xr = tf.expand_dims(X, 1)  # (N1, 1, D)
            X2r = tf.expand_dims(X2_internal, 0)  # (1, N2, D)
            l1 = tf.expand_dims(lens_X, 1)  # (N1, 1, D)
            l2 = tf.expand_dims(lens_X2, 0)  # (1, N2, D)

            L = tf.square(l1) + tf.square(
                l2)  # (N1, N2, D) - sum of squared lengthscales

            # D term: Squared difference scaled by L, summed over dimensions
            D_term = tf.square(Xr - X2r) / L  # (N1, N2, D)
            D_term = tf.reduce_sum(D_term, 2)  # (N1, N2) - sum over dimensions

            # Determinant term: Product over dimensions of (2 * l1 * l2 / L)^(1/2)
            det_term = tf.sqrt(2 * l1 * l2 / L)  # (N1, N2, D)
            det_term = tf.reduce_prod(det_term,
                                      2)  # (N1, N2) - product over dimensions

            # E term: Combine determinant and exponential of D_term
            E = det_term * tf.exp(-D_term)  # (N1, N2)

            # Compute cosine term (COS) - based on frequencies and dot products with X
            # (N1, D) * (N1, D) -> sum over D -> (N1, 1)
            muX = (tf.reduce_sum(freq_X * X, 1, keepdims=True) - tf.transpose(
                tf.reduce_sum(freq_X2 * X2_internal, 1, keepdims=True)))
            COS = tf.cos(2 * np.pi * muX)  # (N1, N2)

            # Compute kernel variance term (WW) - outer product of variance predictions
            WW = tf.matmul(var_X, var_X2,
                           transpose_b=True)  # (N1, 1) @ (1, N2) -> (N1, N2)

            # Compute the q'th kernel component and add to total kernel
            kern += WW * E * COS

        # Add jitter to the diagonal for K(X,X) matrices for numerical stability
        if equal:
            return robust_kernel(kern, tf.shape(X)[0])
        else:
            return kern

    @tf.autograph.experimental.do_not_convert
    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the diagonal of the covariance matrix `K(X, X)`.
        For the Neural Spectral Kernel, this is `sum_q(var_q(X)^2) + jitter`.

        Args:
            X (tf.Tensor): (N, D); Input data points. `N` is the number of points.

        Returns:
            tf.Tensor: (N,); A 1D tensor representing the diagonal elements of the
                        covariance matrix.
        """
        kd = default_jitter()  # Initialize with a small jitter
        for q in range(self.Q):
            # Sum of squared variance predictions from each MLP component
            kd += tf.square(self.var[q](X))
        return tf.squeeze(
            kd)  # Remove singleton dimension (e.g., (N, 1) -> (N,))


# --- Helper functions ---
@tf.autograph.experimental.do_not_convert
def robust_kernel(kern: tf.Tensor, shape_X_0: tf.Tensor) -> tf.Tensor:
    """
    Adds a small positive jitter to the diagonal of a covariance matrix
    to ensure numerical stability. This is particularly important for
    Cholesky decompositions or inverse calculations.

    Args:
        kern (tf.Tensor): The input covariance matrix.
        shape_X_0 (tf.Tensor): The size of the first dimension of the original input `X`
                               (i.e., the number of data points N). Used to create the identity matrix.

    Returns:
        tf.Tensor: The covariance matrix with jitter added to its diagonal.
    """
    jitter_val = 1e-3  # Fixed jitter value
    # Add jitter to the diagonal of the kernel matrix
    return kern + jitter_val * tf.eye(shape_X_0, dtype=float_type)


def init_neural_kernel(X_train: np.ndarray,
                       Y_train: np.ndarray,
                       inducing_variable: np.ndarray,
                       Q: int,
                       n_inits: int = 1,
                       hidden_sizes: Optional[List[int]] = None) -> SGPR:
    """
    Helper function to initialize a Sparse Gaussian Process Regression (SGPR) model
    with a Neural Spectral Kernel. This function can perform multiple random
    initializations and return the model with the best initial Evidence Lower Bound (ELBO).

    Refer to the original paper for more details:
        - Neural Non-Stationary Spectral Kernel [Remes et al., 2018]

    Args:
        X_train (np.ndarray): (n, d); Input training set points.
        Y_train (np.ndarray): (n, 1); Training set labels.
        inducing_variable (np.ndarray): (m, d); Initial inducing points. These are passed
                                        directly to the SGPR model.
        Q (int): The number of MLP mixture components for the Neural Spectral Kernel.
        n_inits (int): Number of times to randomly initialize the kernel's MLPs and
                       compute the initial ELBO. The model with the highest ELBO
                       among these initializations is returned. Defaults to 1.
        hidden_sizes (Optional[List[int]]): List of integers specifying the number of hidden
                                            units in each MLP layer. If None, [32, 32] is used.

    Returns:
        SGPR: The SGPR model instance initialized with the Neural Spectral Kernel
              that yielded the best initial ELBO.

    Usage:
        ```python
        import numpy as np
        import gpflow
        from sgptools.kernels.neural_spectral import init_neural_kernel
        from sgptools.utils.misc import get_inducing_pts # For initial inducing points

        # Dummy data
        X_train_data = np.random.rand(100, 2).astype(np.float32)
        Y_train_data = (np.sin(X_train_data[:, 0]) + np.cos(X_train_data[:, 1]))[:, None].astype(np.float32)
        
        # Initial inducing points (e.g., subset of training data or k-means centers)
        initial_inducing_points = get_inducing_pts(X_train_data, num_inducing=20)

        # Initialize the SGPR model with Neural Spectral Kernel
        # Try 3 random initializations for the MLPs.
        model_ns_kernel = init_neural_kernel(
            X_train=X_train_data,
            Y_train=Y_train_data,
            inducing_variable=initial_inducing_points,
            Q=5,              # 5 mixture components
            n_inits=3,        # 3 initializations
            hidden_sizes=[16, 16] # Custom hidden layer sizes
        )

        # You would typically optimize this model further using optimize_model:
        # from sgptools.utils.gpflow import optimize_model
        # optimize_model(model_ns_kernel)
        ```
    """
    # Convert NumPy arrays to TensorFlow tensors
    X_train_tf, Y_train_tf = data_input_to_tensor((X_train, Y_train))

    best_loglik = -np.inf  # Track the best ELBO found
    best_m: Optional[SGPR] = None  # Store the best model

    N, input_dim = X_train_tf.shape  # Get number of data points and input dimensionality

    for k_init_idx in range(n_inits):
        # Create a new NeuralSpectralKernel instance for each initialization
        current_kernel = NeuralSpectral(input_dim=input_dim,
                                        Q=Q,
                                        hidden_sizes=hidden_sizes)

        # Create an SGPR model with the current kernel initialization
        model = SGPR(data=(X_train_tf, Y_train_tf),
                     inducing_variable=inducing_variable,
                     kernel=current_kernel)

        # Compute the initial ELBO (Evidence Lower Bound)
        loglik = model.elbo()

        # Check if the current initialization is better than previous ones
        if loglik > best_loglik:
            best_loglik = loglik
            # Deepcopy the model to save its state, as it will be deleted/overwritten in next iteration
            # This requires gpflow.utilities.traversal.deepcopy or similar for GPflow models
            # For simplicity, we directly assign here, assuming shallow copy is sufficient
            # or that the user will optimize it later. For robust best model saving, a deepcopy is safer.
            best_m = model

        # Explicitly delete the model and run garbage collection to free memory
        # (important if n_inits is large and models are complex)
        del model
        gc.collect()

    return best_m
