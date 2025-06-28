# Copyright 2024 The streaming_sparse_gp Contributors. All Rights Reserved.
# https://github.com/thangbui/streaming_sparse_gp/tree/master
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
"""Provides a streaming sparse Gaussian process model along with initialization function
"""

import tensorflow as tf
import numpy as np

import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow import covariances
from ..utils.data import get_inducing_pts
from typing import Union, Optional


class OSGPR_VFE(GPModel, InternalDataTrainingLossMixin):
    """Online Sparse Variational GP regression model from [streaming_sparse_gp](https://github.com/thangbui/streaming_sparse_gp/tree/master)

    Refer to the following paper for more details:
        - Streaming Gaussian process approximations [Bui et al., 2017]

    Args:
        data (tuple): (X, y) ndarrays with inputs (n, d) and labels (n, 1)
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        mu_old (ndarray): mean of old `q(u)`; here `u` are the latents corresponding to the inducing points `Z_old`
        Su_old (ndarray): posterior covariance of old `q(u)`
        Kaa_old (ndarray): prior covariance of old `q(u)`
        Z_old (ndarray): (m_old, d): Old initial inducing points
        Z (ndarray): (m_new, d): New initial inducing points
        mean_function (function): GP mean function
    """

    def __init__(self,
                 data,
                 kernel,
                 mu_old,
                 Su_old,
                 Kaa_old,
                 Z_old,
                 Z,
                 mean_function=None):
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(
            data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(
            data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = InducingPoints(Z)
        self.num_data = self.X.shape[0]

        self.mu_old = tf.Variable(mu_old,
                                  shape=tf.TensorShape(None),
                                  trainable=False)
        self.M_old = Z_old.shape[0]
        self.Su_old = tf.Variable(Su_old,
                                  shape=tf.TensorShape(None),
                                  trainable=False)
        self.Kaa_old = tf.Variable(Kaa_old,
                                   shape=tf.TensorShape(None),
                                   trainable=False)
        self.Z_old = tf.Variable(Z_old,
                                 shape=tf.TensorShape(None),
                                 trainable=False)

    def init_Z(self) -> np.ndarray:
        """
        Initializes the new set of inducing points (Z) for the OSGPR model.
        It combines a subset of the old inducing points (Z_old) with a subset
        of the current training data (X).

        Returns:
            np.ndarray: (M, d); A NumPy array of the newly initialized inducing points,
                        combining old and new data-based points.
        """
        M = self.inducing_variable.Z.shape[0]
        M_old = int(0.7 * M)  # Proportion of old inducing points to retain
        M_new = M - M_old  # Proportion of new data points to select

        # Randomly select M_old points from the old inducing points
        old_Z = self.Z_old.numpy()[np.random.permutation(M)[0:M_old], :]

        # Randomly select M_new points from the current training data
        new_Z = self.X.numpy()[
            np.random.permutation(self.X.shape[0])[0:M_new], :]

        # Vertically stack the selected old and new points to form the new Z
        Z = np.vstack((old_Z, new_Z))
        return Z

    def update(self, data, inducing_variable=None, update_inducing=True):
        """
        Configures the OSGPR model to adapt to a new batch of data.
        This method updates the model's data, its inducing points (optionally),
        and caches the posterior mean and covariance of the *old* inducing points
        to facilitate the streaming update equations.

        Note: After calling this update, the OSGPR model typically needs to be
        trained further using gradient-based optimization to fully incorporate
        the new data and optimize its parameters.

        Args:
            data (Tuple[np.ndarray, np.ndarray]): A tuple (X, y) representing the new batch
                                                  of input data `X` (n, d) and corresponding labels `y` (n, 1).
            inducing_variable (Optional[np.ndarray]): (m_new, d); Optional NumPy array for the new
                                                     set of inducing points. If None and `update_inducing`
                                                     is True, `init_Z` will be called to determine them.
                                                     Defaults to None.
            update_inducing (bool): If True, the inducing points will be updated. If False,
                                    they will remain as they were before the update call.
                                    Defaults to True.
        """
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(
            data)
        self.num_data = self.X.shape[0]

        # Store the current inducing points as 'old' for the next update step
        self.Z_old.assign(self.inducing_variable.Z)

        # Update the inducing points based on `update_inducing` flag
        if update_inducing:
            if inducing_variable is None:
                # If no explicit inducing_variable is provided, initialize new ones
                new_Z_init = self.init_Z()
            else:
                # Use the explicitly provided inducing_variable
                new_Z_init = inducing_variable
            self.inducing_variable.Z.assign(
                tf.constant(new_Z_init, dtype=self.inducing_variable.Z.dtype))
        # If update_inducing is False, inducing_variable.Z retains its current value.

        # Get posterior mean and covariance for the *old* inducing points using the current model state
        mu_old, Su_old = self.predict_f(self.Z_old, full_cov=True)
        self.mu_old.assign(mu_old)
        self.Su_old.assign(Su_old)

        # Get the prior covariance matrix for the *old* inducing points using the current kernel
        Kaa_old = self.kernel(self.Z_old)
        self.Kaa_old.assign(Kaa_old)

    def _common_terms(self):
        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.X)
        Kbb = covariances.Kuu(self.inducing_variable,
                              self.kernel,
                              jitter=jitter)
        Kba = covariances.Kuf(self.inducing_variable, self.kernel, self.Z_old)
        Kaa_cur = gpflow.utilities.add_noise_cov(self.kernel(self.Z_old),
                                                 jitter)
        Kaa = gpflow.utilities.add_noise_cov(self.Kaa_old, jitter)

        err = self.Y - self.mean_function(self.X)

        Sainv_ma = tf.linalg.solve(Saa, ma)
        Sinv_y = self.Y / sigma2
        c1 = tf.matmul(Kbf, Sinv_y)
        c2 = tf.matmul(Kba, Sainv_ma)
        c = c1 + c2

        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_c = tf.linalg.triangular_solve(Lb, c, lower=True)
        Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True) / sigma
        d1 = tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_b=True)

        LSa = tf.linalg.cholesky(Saa)
        Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.linalg.triangular_solve(LSa,
                                                      Kab_Lbinv,
                                                      lower=True)
        d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

        La = tf.linalg.cholesky(Kaa)
        Lainv_Kab_Lbinv = tf.linalg.triangular_solve(La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(Lainv_Kab_Lbinv, Lainv_Kab_Lbinv, transpose_a=True)

        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 - d3
        D = gpflow.utilities.add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        LDinv_Lbinv_c = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba,
                LDinv_Lbinv_c, err, d1)

    def maximum_log_likelihood_objective(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        jitter = gpflow.default_jitter()
        # jitter = gpflow.utilities.to_default_float(1e-4)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        Kfdiag = self.kernel(self.X, full_cov=False)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_Lbinv_c,
         err, Qff) = self._common_terms()

        LSa = tf.linalg.cholesky(Saa)
        Lainv_ma = tf.linalg.triangular_solve(LSa, ma, lower=True)

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
        # bound += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
        # log det term
        bound += -0.5 * N * tf.reduce_sum(tf.math.log(sigma2))
        bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # delta 1: trace term
        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))

        # delta 2: a and b difference
        bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))

        Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.linalg.diag_part(Sainv_Kaadiff) -
            tf.linalg.diag_part(Kainv_Kaadiff))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_Lbinv_c,
         err, Qff) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(
                tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = -tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs,
                             LDinv_Lbinv_Kbs,
                             transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var


def init_osgpr(X_train: np.ndarray,
               num_inducing: int = 10,
               lengthscales: Union[float, np.ndarray] = 1.0,
               variance: float = 1.0,
               noise_variance: float = 0.001,
               kernel: Optional[gpflow.kernels.Kernel] = None,
               ndim: int = 1) -> OSGPR_VFE:
    """
    Initializes an Online Sparse Variational Gaussian Process Regression (OSGPR_VFE) model.
    This function first fits a standard Sparse Gaussian Process Regression (SGPR) model
    to a dummy dataset (representing initial data/environment bounds) to obtain an
    initial set of optimized inducing points and their corresponding posterior.
    These are then used to set up the `OSGPR_VFE` model for streaming updates.

    Args:
        X_train (np.ndarray): (n, d); Unlabeled random sampled training points.
                              These points are primarily used to define the spatial bounds
                              and for initial selection of inducing points. Their labels are
                              set to zeros for the SGPR initialization.
        num_inducing (int): The number of inducing points to use for the OSGPR model. Defaults to 10.
        lengthscales (Union[float, np.ndarray]): Initial lengthscale(s) for the RBF kernel.
                                                 If a float, it's applied uniformly. If a NumPy array,
                                                 each element corresponds to a dimension. Defaults to 1.0.
        variance (float): Initial variance (amplitude) for the RBF kernel. Defaults to 1.0.
        noise_variance (float): Initial data noise variance for the Gaussian likelihood. Defaults to 0.001.
        kernel (Optional[gpflow.kernels.Kernel]): A pre-defined GPflow kernel function. If None,
                                     a `gpflow.kernels.SquaredExponential` (RBF) kernel is created
                                     with the provided `lengthscales` and `variance`. Defaults to None.
        ndim (int): Number of output dimensions for the dummy training labels `y_train`. Defaults to 1.

    Returns:
        OSGPR_VFE: An initialized `OSGPR_VFE` model instance, ready to accept
                   new data batches via its `update` method.

    Usage:
        ```python
        import numpy as np
        # from sgptools.core.osgpr import init_osgpr

        # Define some dummy training data to establish initial bounds
        X_initial_env = np.random.rand(100, 2) * 10
        
        # Initialize the OSGPR model
        online_gp_model = init_osgpr(
            X_initial_env,
            num_inducing=50,
            lengthscales=2.0,
            variance=1.5,
            noise_variance=0.01
        )

        # Example of updating the model with new data (typically in a loop)
        # new_X_batch = np.random.rand(10, 2) * 10
        # new_y_batch = np.sin(new_X_batch[:, 0:1]) + np.random.randn(10, 1) * 0.1
        # online_gp_model.update(data=(new_X_batch, new_y_batch))
        ```
    """
    if kernel is None:
        # If no kernel is provided, initialize a SquaredExponential (RBF) kernel.
        kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales,
                                                   variance=variance)

    # Create a dummy y_train: SGPR needs labels, but for initialization purposes here,
    # we use zeros as the actual labels will come in through online updates.
    y_train_dummy = np.zeros((len(X_train), ndim), dtype=X_train.dtype)

    # Select initial inducing points from X_train using get_inducing_pts utility
    Z_init = get_inducing_pts(X_train, num_inducing)

    # Initialize a standard SGPR model. This model helps in getting an initial
    # posterior (mu, Su) for the inducing points (Z_init) under the given kernel
    # and noise variance. This posterior then becomes the 'old' posterior for OSGPR_VFE.
    init_sgpr_model = gpflow.models.SGPR(data=(X_train, y_train_dummy),
                                         kernel=kernel,
                                         inducing_variable=Z_init,
                                         noise_variance=noise_variance)

    # Extract optimized (or initial) inducing points from the SGPR model
    Zopt_np = init_sgpr_model.inducing_variable.Z.numpy()

    # Predict the mean (mu) and full covariance (Su) of the latent function
    # at these initial inducing points (Zopt). This represents the 'old' posterior.
    mu_old_tf, Su_old_tf_full_cov = init_sgpr_model.predict_f(tf.constant(
        Zopt_np, dtype=X_train.dtype),
                                                              full_cov=True)

    # Kaa_old: Prior covariance matrix of the old inducing points
    Kaa_old_tf = init_sgpr_model.kernel(
        tf.constant(Zopt_np, dtype=X_train.dtype))

    # Prepare dummy initial data for OSGPR_VFE. This data will be overwritten
    # by the first actual `update` call.
    dummy_X_online = np.zeros([2, X_train.shape[-1]], dtype=X_train.dtype)
    dummy_y_online = np.zeros([2, ndim], dtype=X_train.dtype)

    # Initialize the OSGPR_VFE model with the extracted parameters.
    # The `Su_old_tf_full_cov` is expected to be a (1, M, M) tensor for single latent GP,
    # so we extract the (M, M) covariance matrix `Su_old_tf_full_cov[0]`.
    online_osgpr_model = OSGPR_VFE(
        data=(tf.constant(dummy_X_online), tf.constant(dummy_y_online)),
        kernel=init_sgpr_model.
        kernel,  # Pass the kernel (potentially optimized by SGPR init)
        mu_old=mu_old_tf,
        Su_old=Su_old_tf_full_cov[0],
        Kaa_old=Kaa_old_tf,
        Z_old=tf.constant(Zopt_np, dtype=X_train.dtype),
        Z=tf.constant(Zopt_np,
                      dtype=X_train.dtype))  # New Z is same as old Z initially

    # Assign the noise variance from the initial SGPR model to the OSGPR model's likelihood
    online_osgpr_model.likelihood.variance.assign(
        init_sgpr_model.likelihood.variance)

    return online_osgpr_model
