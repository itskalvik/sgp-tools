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


def continuous_sgp(num_inducing, X_train, noise_variance, kernel, 
                   transform=None,
                   Xu_init=None, 
                   Xu_time=None, 
                   orientation=False,
                   **kwargs):
    """Get sensor placement solutions using the Continuous-SGP method

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [[Jakkala and Akella, 2023](https://www.itskalvik.com/publication/sgp-sp/)]
        - Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [[Jakkala and Akella, 2024](https://www.itskalvik.com/publication/sgp-ipp/)]

    Args:
        num_inducing (int): Number of inducing points
        X_train (ndarray): (n, d); Unlabeled random sampled training points
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object
        Xu_init (ndarray): (m, d); Initial inducing points
        Xu_time (ndarray): (t, d); Temporal inducing points used in spatio-temporal models
        orientation (bool): If True, a additionl dimension is added to the 
                            inducing points to represent the FoV orientation

    Returns:
        sgpr (AugmentedSGPR): Optimized sparse Gaussian process model
        loss (ndarray): Loss values computed during training
    """
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
                         transform=transform)

    # Train the mode
    loss = optimize_model(sgpr,
                          kernel_grad=False, 
                          **kwargs)

    return sgpr, loss