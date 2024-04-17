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

import numpy as np

from ..utils.data import get_inducing_pts
from ..utils.gpflow import optimize_model

from .core.augmented_sgpr import AugmentedSGPR


'''
Get sensor placement solution using the Continuous VFE-SGP method

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
