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
"""Attentive kernel for non-stationary Gaussian processes."""

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.config import default_float

from .neural_network import NN
from typing import List, Union, Optional


class Attentive(gpflow.kernels.Kernel):
    """Attentive kernel (non-stationary kernel).

    This kernel uses a Multi-Layer Perceptron (MLP) to compute attention
    representations that weight a mixture of RBF components, producing a
    locally adaptive, non-stationary covariance function.

    Implementation based on [Weizhe-Chen/attentive_kernels](https://github.com/Weizhe-Chen/attentive_kernels).

    Refer to the following paper for more details:
        - AK: Attentive Kernel for Information Gathering [Chen et al., 2022]

    Attributes:
        _free_amplitude (tf.Variable):
            Trainable scalar amplitude parameter applied to the final covariance.
        lengthscales (tf.Variable):
            1D tensor of fixed lengthscale values for the RBF mixture components.
        num_lengthscales (int):
            Number of RBF mixture components.
        nn (NN):
            Neural network that maps input points to latent attention
            representations.
    """

    def __init__(
        self,
        lengthscales: Union[List[float], np.ndarray] = None,
        hidden_sizes: List[int] = None,
        amplitude: float = 1.0,
        num_dim: int = 2,
    ):
        """Initialize an Attentive kernel.

        Args:
            lengthscales (List[float] | np.ndarray | None):
                Positive lengthscale values used for the fixed RBF mixture
                components. These are treated as non-trainable parameters.
                If None, a default grid ``np.linspace(0.01, 2.0, 10)`` is used.
            hidden_sizes (List[int] | None):
                Hidden-layer widths of the MLP. The length of this list
                determines the number of hidden layers. If None, defaults to
                ``[10, 10]``.
            amplitude (float):
                Initial value for the trainable scalar amplitude parameter used
                to rescale the final covariance.
            num_dim (int):
                Dimensionality of each input data point (e.g. 2 for 2D inputs).

        Returns:
            None

        Usage:
            Basic usage with fixed lengthscales for 2D data::

                ```python
                import gpflow
                import numpy as np
                from sgptools.kernels.attentive import Attentive

                # Example: 10 fixed lengthscales ranging from 0.01 to 2.0
                l_scales = np.linspace(0.01, 2.0, 10).astype(np.float32)

                # Initialize Attentive kernel for 2D data
                kernel = Attentive(
                    lengthscales=l_scales,
                    hidden_sizes=[10, 10],
                    amplitude=1.0,
                    num_dim=2,
                )

                # Use this kernel in a GPflow model:
                # model = gpflow.models.GPR(
                #     data=(X_train, Y_train),
                #     kernel=kernel,
                #     noise_variance=0.1,
                # )
                # optimize_model(model)
                ```
        """
        super().__init__()
        if lengthscales is None:
            lengthscales = np.linspace(0.01, 2.0, 10)

        if hidden_sizes is None:
            hidden_sizes = [10, 10]
        else:
            hidden_sizes = list(hidden_sizes)

        with self.name_scope:
            self.num_lengthscales = len(lengthscales)
            self._free_amplitude = tf.Variable(
                amplitude,
                shape=[],
                trainable=True,
                dtype=default_float(),
            )

            # Lengthscales are fixed, not optimized.
            self.lengthscales = tf.Variable(
                tf.cast(lengthscales, default_float()),
                shape=[self.num_lengthscales],
                trainable=False,
                dtype=default_float(),
            )

            self.nn = NN(
                [num_dim] + hidden_sizes + [self.num_lengthscales],
                output_activation_fn="softplus",
            )

    @tf.autograph.experimental.do_not_convert
    def get_representations(self, X: tf.Tensor) -> tf.Tensor:
        """Compute normalized latent attention representations.

        Args:
            X (tf.Tensor):
                Tensor of shape (N, D). Input data points.

        Returns:
            tf.Tensor:
                Tensor of shape (N, num_lengthscales) containing unit-norm
                latent representation vectors used for generating attention
                weights.
        """
        Z = self.nn(X)
        representations = Z / tf.norm(Z, axis=1, keepdims=True)
        return representations

    @tf.autograph.experimental.do_not_convert
    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute full covariance matrix between X and X2.

        The covariance is a weighted sum of RBF mixture components modulated
        by attention representations in the learned latent space.

        Args:
            X (tf.Tensor):
                Tensor of shape (N1, D). First set of input points.
            X2 (tf.Tensor | None):
                Tensor of shape (N2, D). Optional second set of input points.
                If None, `X` is used for both arguments.

        Returns:
            tf.Tensor:
                Tensor of shape (N1, N2) containing the covariance matrix
                K(X, X2).
        """
        if X2 is None:
            X2_internal = X
        else:
            X2_internal = X2

        dist = cdist(X, X2_internal)
        repre1 = self.get_representations(X)
        repre2 = self.get_representations(X2_internal)

        def get_mixture_component(i: tf.Tensor) -> tf.Tensor:
            """Compute a single mixture RBF component.

            Args:
                i (tf.Tensor):
                    Scalar integer tensor representing a lengthscale index.

            Returns:
                tf.Tensor:
                    Tensor of shape (N1, N2) containing the i-th mixture
                    kernel component.
            """
            attention_lengthscales = tf.tensordot(
                repre1[:, i], repre2[:, i], axes=0
            )
            return rbf(dist, self.lengthscales[i]) * attention_lengthscales

        cov_mat_per_ls = tf.map_fn(
            fn=get_mixture_component,
            elems=tf.range(self.num_lengthscales, dtype=tf.int64),
            fn_output_signature=dist.dtype,
        )

        cov_mat_summed = tf.reduce_sum(cov_mat_per_ls, axis=0)
        attention_inputs = tf.matmul(repre1, repre2, transpose_b=True)

        return self._free_amplitude * attention_inputs * cov_mat_summed

    @tf.autograph.experimental.do_not_convert
    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """Compute the diagonal of K(X, X).

        Args:
            X (tf.Tensor):
                Tensor of shape (N, D). Input points.

        Returns:
            tf.Tensor:
                Tensor of shape (N,) containing the diagonal of the covariance
                matrix (constant when representations are unit norm).
        """
        return self._free_amplitude * tf.ones((X.shape[0],), dtype=X.dtype)

    def get_lengthscales(self, X: np.ndarray) -> np.ndarray:
        """Compute non-stationary effective lengthscales.

        Args:
            X (np.ndarray):
                Array of shape (N, D). Input points at which to estimate
                effective lengthscales.

        Returns:
            np.ndarray:
                Array of shape (N,) containing effective spatially varying
                lengthscale values at the given input locations.
        """
        lengthscales = self.lengthscales.numpy()
        preds = np.zeros(len(X))

        repre = self.get_representations(X)
        for i in range(len(lengthscales)):
            attention = tf.tensordot(
                repre[:, i], tf.transpose(repre[:, i]), axes=0
            )
            preds += np.diag(attention) * lengthscales[i]
        return preds


# ---- Helper functions --------------------------------------------------------


@tf.autograph.experimental.do_not_convert
def rbf(dist: tf.Tensor, lengthscale: tf.Tensor) -> tf.Tensor:
    """Compute RBF kernel values.

    The RBF kernel is defined as: $k(d, l) = \exp(-0.5 \times (d/l)^2)$

    Args:
        dist (tf.Tensor):
            Tensor of pairwise distances, shape (N1, N2) or (N,).
        lengthscale (tf.Tensor):
            Scalar tensor representing the lengthscale.

    Returns:
        tf.Tensor:
            Tensor of the same shape as `dist` containing RBF kernel values.
    """
    return tf.exp(-0.5 * tf.square(dist / lengthscale))


@tf.autograph.experimental.do_not_convert
def cdist(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Compute pairwise Euclidean distances.

    Args:
        x (tf.Tensor):
            Tensor of shape (N1, D). First input set.
        y (tf.Tensor):
            Tensor of shape (N2, D). Second input set.

    Returns:
        tf.Tensor:
            Tensor of shape (N1, N2) where entry (i, j) is the Euclidean
            distance between `x[i, :]` and `y[j, :]`.
    """
    per_x_dist = lambda i: tf.norm(x[i:(i + 1), :] - y, axis=1)

    distances = tf.map_fn(
        fn=per_x_dist,
        elems=tf.range(tf.shape(x)[0], dtype=tf.int64),
        fn_output_signature=x.dtype,
    )

    return distances
