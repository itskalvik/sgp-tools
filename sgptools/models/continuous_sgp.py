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

import gpflow
import numpy as np

from ..utils.data import get_inducing_pts
from ..utils.gpflow import optimize_model

from .core.osgpr import OSGPR_VFE
from .core.augmented_sgpr import AugmentedSGPR


'''
Get sensor placement solution using the Continuous VFE-SGP method

Refer to the following papers for more details:
Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [Jakkala and Akella, 2023]
Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [Jakkala and Akella, 2024]

Args:
    num_inducing: Number of inducing points
    X_train: Numpy array (n ,d) with n d-dimensional data points
    noise_variance:
    kernel:
    num_steps: 
    lr: 
    Xu_init: 
    Xu_time:
    orientation:
    trace_fn:
'''
def continuous_sgp(num_inducing, X_train, noise_variance, kernel, 
                   transformer=None,
                   Xu_init=None, 
                   Xu_time=None, 
                   orientation=False,
                   **kwargs):

    # Generate init inducing points
    if Xu_init is None:
        Xu_init = get_inducing_pts(X_train, num_inducing, 
                                   orientation=orientation)

    # Fit spare GP
    sgpr = AugmentedSGPR((X_train, np.zeros((len(X_train), 1)).astype(X_train.dtype)),
                         noise_variance=noise_variance,
                         kernel=kernel, 
                         inducing_variable=Xu_init,
                         inducing_variable_time=Xu_time,
                         transformer=transformer)

    # Train the mode
    loss = optimize_model(sgpr,
                          kernel_grad=False, 
                          **kwargs)

    return sgpr, loss

'''
Initialize a VFE OSGPR model with an RBF kernel with 
unit variance and lengthcales, and 0.001 noise variance.
Used in the Online Continuous SGP approach. 

Args:
    X_train: Numpy array (n ,d) with n d-dimensional data points. 
             They only effect the initial inducing point locations, 
             i.e., limits them to the bounds of the data
    num_inducing: int, Number of inducing points
    lengthscales: float or list of floats for each dimension of the data. Kernel lengthscale
    variance: float, Kernel variance
    noise_variance: float, Data noise variance
'''
def init_osgpr(X_train, 
               num_inducing=10, 
               lengthscales=1.0, 
               variance=1.0,
               noise_variance=0.001):
    y_train = np.zeros((len(X_train), 1), dtype=X_train.dtype)
    Z_init = get_inducing_pts(X_train, num_inducing)
    init_param = gpflow.models.SGPR((X_train, y_train),
                                    gpflow.kernels.RBF(variance=variance, 
                                                       lengthscales=lengthscales), 
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