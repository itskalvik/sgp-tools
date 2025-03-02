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


class AttentiveKernel(gpflow.kernels.Kernel):
    """Attentive Kernel function (non-stationary kernel function). 
    Based on the implementation from this [repo](https://github.com/Weizhe-Chen/attentive_kernels)

    Refer to the following papers for more details:
        - AK: Attentive Kernel for Information Gathering [Chen et al., 2022]

    Args:
        lengthscales (List): List of lengthscales to use in the mixture components. The lengthscales are not trained.
        amplitude (int): Initial amplitude of the kernel function
        dim_hidden (int): Number of MLP hidden layer nodes (The NN will have two of these layers)
        num_dim (int): Number of dimensions of the data points
    """
    def __init__(self, 
                 lengthscales, 
                 dim_hidden=10,
                 amplitude=1.0,
                 num_dim=2): 
        super().__init__()
        with self.name_scope:
            self.num_lengthscales = len(lengthscales)
            self._free_amplitude = tf.Variable(amplitude, 
                                               shape=[],
                                               trainable=True,
                                               dtype=float_type)
            self.lengthscales = tf.Variable(lengthscales, 
                                            shape=[self.num_lengthscales], 
                                            trainable=False,
                                            dtype=float_type)
            
            self.nn = NN([num_dim, dim_hidden, dim_hidden, self.num_lengthscales])

    def get_representations(self, X):
        Z = self.nn(X)
        representations = Z / tf.norm(Z, axis=1, keepdims=True)
        return representations

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

        dist = cdist(X, X2)
        repre1 = self.get_representations(X)
        repre2 = self.get_representations(X2)

        def get_mixture_component(i):
            attention_lengthscales = tf.tensordot(repre1[:, i], repre2[:, i], axes=0)
            cov_mat = rbf(dist, self.lengthscales[i]) * attention_lengthscales   
            return cov_mat
        
        cov_mat = tf.map_fn(fn=get_mixture_component, 
                            elems=tf.range(self.num_lengthscales, dtype=tf.int64), 
                            fn_output_signature=dist.dtype)
        cov_mat = tf.math.reduce_sum(cov_mat, axis=0)
        attention_inputs = repre1 @ tf.transpose(repre2)
        cov_mat *= self._free_amplitude * attention_inputs

        return cov_mat
    
    def K_diag(self, X):
        return self._free_amplitude * tf.ones((X.shape[0]), dtype=X.dtype)
    
'''
Helper functions
'''
def rbf(dist, lengthscale):
    '''
    RBF kernel function
    '''
    return tf.math.exp(-0.5 * tf.math.square(dist / lengthscale))

def cdist(x, y):
    '''
    Calculate the pairwise euclidean distances
    '''
    # Calculate distance for a single row of x.
    per_x_dist = lambda i : tf.norm(x[i:(i+1),:] - y, axis=1)
    # Compute and stack distances for all rows of x.
    dist = tf.map_fn(fn=per_x_dist, 
                     elems=tf.range(tf.shape(x)[0], dtype=tf.int64), 
                     fn_output_signature=x.dtype)
    return dist