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

gpflow.config.set_default_float(np.float32)
float_type = default_float()

def xavier(dim_in, dim_out):
    return np.random.randn(dim_in, dim_out)*(2./(dim_in+dim_out))**0.5

class NN(gpflow.base.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            setattr(self, 'W_{}'.format(i), tf.Variable(xavier(dim_in, dim_out),
                                                        dtype=float_type))
            setattr(self, 'b_{}'.format(i), tf.Variable(np.zeros(dim_out),
                                                        dtype=float_type))

    def __call__(self, X):
        if X is not None:
            for i in range(len(self.dims) - 2):
                W = getattr(self, 'W_{}'.format(i))
                b = getattr(self, 'b_{}'.format(i))
                X = tf.nn.selu(tf.matmul(X, W) + b)
            W = getattr(self, 'W_{}'.format(i+1))
            b = getattr(self, 'b_{}'.format(i+1))
            X = tf.nn.softmax(tf.matmul(X, W) + b)
            return X