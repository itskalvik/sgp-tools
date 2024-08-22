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

# Original SGP code from GPflow library (https://github.com/GPflow/GPflow)

"""Provides a sparse Gaussian process model with update, expand, and aggregate functions
"""

import numpy as np
import tensorflow as tf
from gpflow.models import SGPR
from gpflow.base import InputData, MeanAndVariance
from gpflow.config import default_float
from gpflow.utilities import add_noise_cov
from gpflow.base import InputData, MeanAndVariance
from gpflow.utilities import add_noise_cov, to_default_float
from gpflow.models.util import inducingpoint_wrapper
from .transformations import Transform


class AugmentedSGPR(SGPR):
    """SGPR model from the GPFlow library augmented to use a transform object's
    expand and aggregate functions on the inducing points where necessary. The object
    has an additional update function to update the kernel and noise variance parameters 
    (currently, the online updates part works only with RBF kernels).  


    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [Jakkala and Akella, 2023]
        - Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [Jakkala and Akella, 2024]

    Args:
        data (tuple): (X, y) ndarrays with inputs (n, d) and labels (n, 1)
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        noise_variance (float): data variance
        inducing_variable (ndarray): (m, d); Initial inducing points
        transform (Transform): Transform object
        inducing_variable_time (ndarray): (m, d); Temporal dimensions of the inducing points, 
                                            used when modeling spatio-temporal IPP
    """
    def __init__(
        self,
        *args,
        transform=None,
        inducing_variable_time=None,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        if transform is None:
            self.transform = Transform()
        else:
            self.transform = transform

        if inducing_variable_time is not None:
            self.inducing_variable_time = inducingpoint_wrapper(inducing_variable_time)
            self.transform.inducing_variable_time = self.inducing_variable_time
        else:
            self.inducing_variable_time = None

    def update(self, noise_variance, kernel):
        """Update SGP noise variance and kernel function parameters

        Args:
            noise_variance (float): data variance
            kernel (gpflow.kernels.Kernel): gpflow kernel function
        """
        self.likelihood.variance.assign(noise_variance)
        self.kernel.lengthscales.assign(kernel.lengthscales)
        self.kernel.variance.assign(kernel.variance)

    def _common_calculation(self) -> "SGPR.CommonTensors":
        """
        Matrices used in log-det calculation
        :return: A , B, LB, AAT with :math:`LLᵀ = Kᵤᵤ , A = L⁻¹K_{uf}/σ, AAT = AAᵀ,
            B = AAT+I, LBLBᵀ = B`
            A is M x N, B is M x M, LB is M x M, AAT is M x M
        """
        x, _ = self.data
        
        iv = self.inducing_variable.Z  # [M]
        iv = self.transform.expand(iv)

        kuf = self.kernel(iv, x)
        kuf = self.transform.aggregate(kuf)

        kuu = self.kernel(iv) + 1e-6 * tf.eye(tf.shape(iv)[0], dtype=iv.dtype)
        kuu = self.transform.aggregate(kuu)

        L = tf.linalg.cholesky(kuu)

        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)

        return self.CommonTensors(sigma_sq, sigma, A, B, LB, AAT, L)

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        common = self._common_calculation()
        output_shape = tf.shape(self.data[-1])
        num_data = to_default_float(output_shape[0])
        output_dim = to_default_float(output_shape[1])
        const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        logdet = self.logdet_term(common)
        quad = self.quad_term(common)
        constraints = self.transform.constraints(self.inducing_variable.Z)
        return const + logdet + quad + constraints

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        # could copy into posterior into a fused version
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        X_data, Y_data = self.data
        
        iv = self.inducing_variable.Z
        iv = self.transform.expand(iv)

        num_inducing = tf.shape(iv)[0]

        err = Y_data - self.mean_function(X_data)
        kuf = self.kernel(iv, X_data)
        kuu = self.kernel(iv) + 1e-6 * tf.eye(num_inducing, dtype=iv.dtype)
        Kus = self.kernel(iv, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )  # cache qinv
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var
