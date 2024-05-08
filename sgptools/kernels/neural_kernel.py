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

import numpy as np
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gpflow
from gpflow.config import default_jitter, default_float
from gpflow.models import SGPR
from gpflow.models.util import data_input_to_tensor

gpflow.config.set_default_float(np.float32)
float_type = default_float()


class NeuralSpectralKernel(gpflow.kernels.Kernel):
    """Neural Spectral Kernel function (non-stationary kernel function). 
    Based on the implementation from the following [repo](https://github.com/sremes/nssm-gp/tree/master?tab=readme-ov-file)

    Refer to the following papers for more details:
        - Neural Non-Stationary Spectral Kernel [Remes et al., 2018]

    Args:
        input_dim (int): Number of data dimensions
        active_dims (int): Number of data dimensions that are used for computing the covariances
        Q (int): Number of MLP mixture components used in the kernel function
        hidden_sizes (list): Number of hidden units in each MLP layer. Length of the list determines the number of layers.
    """
    def __init__(self, input_dim, active_dims=None, Q=1, hidden_sizes=[32, 32]):
        super().__init__(active_dims=active_dims)

        self.input_dim = input_dim
        self.Q = Q
        self.num_hidden = len(hidden_sizes)

        self.freq = []
        self.length = []
        self.var = []
        for q in range(self.Q):
            freq = keras.Sequential([layers.Dense(hidden_sizes[i], activation='selu') for i in range(self.num_hidden)] + 
                                    [layers.Dense(input_dim, activation='softplus')])
            length = keras.Sequential([layers.Dense(hidden_sizes[i], activation='selu') for i in range(self.num_hidden)] +
                                   [layers.Dense(input_dim, activation='softplus')])
            var = keras.Sequential([layers.Dense(hidden_sizes[i], activation='selu') for i in range(self.num_hidden)] +
                                   [layers.Dense(1, activation='softplus')])
            self.freq.append(freq)
            self.length.append(length)
            self.var.append(var)
        
    def K(self, X, X2=None):
        """Computes the covariances between/amongst the input variables

        Args:
            X (ndarray): Variables to compute the covariance matrix
            X2 (ndarray): If passed, the covariance between X and X2 is computed. Otherwise, 
                          the covariance between X and X is computed.

        Returns:
            cov (ndarray): covariance matrix
        """
        if X2 is None:
            X2 = X
            equal = True
        else:
            equal = False

        kern = 0.0
        for q in range(self.Q):
            # compute latent function values by the neural network
            freq, freq2 = self.freq[q](X), self.freq[q](X2)
            lens, lens2 = self.length[q](X), self.length[q](X2)
            var, var2 = self.var[q](X), self.var[q](X2)

            # compute length-scale term
            Xr = tf.expand_dims(X, 1)  # N1 1 D
            X2r = tf.expand_dims(X2, 0)  # 1 N2 D
            l1 = tf.expand_dims(lens, 1)  # N1 1 D
            l2 = tf.expand_dims(lens2, 0)  # 1 N2 D
            L = tf.square(l1) + tf.square(l2)  # N1 N2 D
            #D = tf.square((Xr - X2r) / L)  # N1 N2 D
            D = tf.square(Xr - X2r) / L  # N1 N2 D
            D = tf.reduce_sum(D, 2)  # N1 N2
            det = tf.sqrt(2 * l1 * l2 / L)  # N1 N2 D
            det = tf.reduce_prod(det, 2)  # N1 N2
            E = det * tf.exp(-D)  # N1 N2

            # compute cosine term
            muX = (tf.reduce_sum(freq * X, 1, keepdims=True)
                   - tf.transpose(tf.reduce_sum(freq2 * X2, 1, keepdims=True)))
            COS = tf.cos(2 * np.pi * muX)

            # compute kernel variance term
            WW = tf.matmul(var, var2, transpose_b=True)  # w*w'^T

            # compute the q'th kernel component
            kern += WW * E * COS
        if equal:
            return robust_kernel(kern, tf.shape(X)[0])
        else:
            return kern

    def K_diag(self, X):
        kd = default_jitter()
        for q in range(self.Q):
            kd += tf.square(self.var[q](X))
        return tf.squeeze(kd)

'''
Helper functions
'''
def robust_kernel(kern, shape_X):
    jitter = 1e-3
    return kern + jitter * tf.eye(shape_X, dtype=float_type)

def init_neural_kernel(x, y, inducing_variable, Q, n_inits=1, hidden_sizes=None):
    """Helper function to initialize a Neural Spectral Kernel function (non-stationary kernel function). 
    Based on the implementation from the following [repo](https://github.com/sremes/nssm-gp/tree/master?tab=readme-ov-file)

    Refer to the following papers for more details:
        - Neural Non-Stationary Spectral Kernel [Remes et al., 2018]

    Args:
        x (ndarray): (n, d); Input training set points
        y (ndarray): (n, 1); Training set labels
        inducing_variable (ndarray): (m, d); Initial inducing points
        Q (int): Number of MLP mixture components used in the kernel function
        n_inits (int): Number of times to initalize the kernel function (returns the best model)
        hidden_sizes (list): Number of hidden units in each MLP layer. Length of the list determines the number of layers.
    """
    x, y = data_input_to_tensor((x, y))

    print('Initializing neural spectral kernel...')
    best_loglik = -np.inf
    best_m = None
    N, input_dim = x.shape

    for k in range(n_inits):
        # gpflow.reset_default_graph_and_session()
        k = NeuralSpectralKernel(input_dim=input_dim, Q=Q, 
                                    hidden_sizes=hidden_sizes)
        model = SGPR((x, y), inducing_variable=inducing_variable, 
                        kernel=k)
        loglik = model.elbo()
        if loglik > best_loglik:
            best_loglik = loglik
            best_m = model
        del model
        gc.collect()
    print('Best init: %f' % best_loglik)

    return best_m