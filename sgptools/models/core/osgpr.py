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
from ...utils.data import get_inducing_pts


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
    def __init__(self, data, kernel, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=None):
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = InducingPoints(Z)
        self.num_data = self.X.shape[0]

        self.mu_old = tf.Variable(mu_old, shape=tf.TensorShape(None), trainable=False)
        self.M_old = Z_old.shape[0]
        self.Su_old = tf.Variable(Su_old, shape=tf.TensorShape(None), trainable=False)
        self.Kaa_old = tf.Variable(Kaa_old, shape=tf.TensorShape(None), trainable=False)
        self.Z_old = tf.Variable(Z_old, shape=tf.TensorShape(None), trainable=False)

    def update(self, data):
        """Configure the OSGPR to adapt to a new batch of data. 
        Note: The OSGPR needs to be trained using gradient-based approaches after update.

        Args:
            data (tuple): (X, y) ndarrays with new batch of inputs (n, d) and labels (n, 1)
        """
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        self.num_data = self.X.shape[0]

        self.Z_old = tf.Variable(self.inducing_variable.Z.numpy(), 
                                 shape=tf.TensorShape(None), 
                                 trainable=False)

        # Get posterior mean and covariance for the old inducing points
        mu_old, Su_old = self.predict_f(self.Z_old, full_cov=True)
        self.mu_old = tf.Variable(mu_old, shape=tf.TensorShape(None), trainable=False)
        self.Su_old = tf.Variable(Su_old, shape=tf.TensorShape(None), trainable=False)
        
        # Get the prior covariance matrix for the old inducing points
        Kaa_old = self.kernel(self.Z_old)
        self.Kaa_old = tf.Variable(Kaa_old, shape=tf.TensorShape(None), trainable=False)

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
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter)
        Kba = covariances.Kuf(self.inducing_variable, self.kernel, self.Z_old)
        Kaa_cur = gpflow.utilities.add_noise_cov(self.kernel(self.Z_old), jitter)
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
        LSainv_Kab_Lbinv = tf.linalg.triangular_solve(
            LSa, Kab_Lbinv, lower=True)
        d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

        La = tf.linalg.cholesky(Kaa)
        Lainv_Kab_Lbinv = tf.linalg.triangular_solve(
            La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(Lainv_Kab_Lbinv, Lainv_Kab_Lbinv, transpose_a=True)

        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 - d3
        D = gpflow.utilities.add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        LDinv_Lbinv_c = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_Lbinv_c, err, d1)

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
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()

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
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # delta 1: trace term
        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))

        # delta 2: a and b difference
        bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))

        Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.linalg.diag_part(Sainv_Kaadiff) - tf.linalg.diag_part(Kainv_Kaadiff))

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
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var


def init_osgpr(X_train, 
               num_inducing=10, 
               lengthscales=1.0, 
               variance=1.0,
               noise_variance=0.001,
               kernel=None):
    """Initialize a VFE OSGPR model with an RBF kernel with 
    unit variance and lengthcales, and 0.001 noise variance.
    Used in the Online Continuous SGP approach. 

    Args:
        X_train (ndarray): (n, d); Unlabeled random sampled training points. 
                        They only effect the initial inducing point locations, 
                        i.e., limits them to the bounds of the data
        num_inducing (int): Number of inducing points
        lengthscales (float or list): Kernel lengthscale(s), if passed as a list, 
                                each element corresponds to each data dimension
        variance (float): Kernel variance
        noise_variance (float): Data noise variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function

    Returns:
        online_param (OSGPR_VFE): Initialized online sparse Gaussian process model
    """

    if kernel is None:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales, 
                                                   variance=variance)
        
    y_train = np.zeros((len(X_train), 1), dtype=X_train.dtype)
    Z_init = get_inducing_pts(X_train, num_inducing)
    init_param = gpflow.models.SGPR((X_train, y_train),
                                    kernel, 
                                    inducing_variable=Z_init, 
                                    noise_variance=noise_variance)
    
    # Initialize the OSGPR model using the parameters from the SGPR model
    # The X_train and y_train here will be overwritten in the online phase 
    X_train = np.array([[0, 0], [0, 0]])
    y_train = np.array([0, 0]).reshape(-1, 1)
    Zopt = init_param.inducing_variable.Z.numpy()
    mu, Su = init_param.predict_f(Zopt, full_cov=True)
    Kaa = init_param.kernel(Zopt)
    online_param = OSGPR_VFE((X_train[:2], y_train[:2]),
                             init_param.kernel,
                             mu, Su[0], Kaa,
                             Zopt, Zopt)
    online_param.likelihood.variance.assign(init_param.likelihood.variance)

    return online_param