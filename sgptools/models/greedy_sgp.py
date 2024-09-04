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

from .core.augmented_sgpr import AugmentedSGPR
from apricot import CustomSelection
import numpy as np


class GreedySGP:
    """Helper class to compute SGP's ELBO/optimization bound for a given set of sensor locations.
    Used by `get_greedy_sgp_sol` function to compute the solution sensor placements using the Greedy-SGP method.

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [[Jakkala and Akella, 2023](https://www.itskalvik.com/publication/sgp-sp/)]

    Args:
        num_inducing (int): Number of inducing points
        S (ndarray): (n, d); Candidate sensor placement locations
        V (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): Data noise variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object
    """
    def __init__(self, num_inducing, S, V, noise_variance, kernel, 
                 transform=None):
        self.sgp = AugmentedSGPR((V, np.zeros((len(V), 1))),
                                 noise_variance=noise_variance,
                                 kernel=kernel, 
                                 inducing_variable=S[:num_inducing],
                                 transform=transform)
        self.locs = S
        self.num_inducing = num_inducing
        self.inducing_dim = S.shape[1]

    def bound(self, x):
        """Computes the SGP's optimization bound using the inducing points `x` 

        Args:
            x (ndarray): (n); Indices of the solution placement locations

        Returns:
            elbo (float): Evidence lower bound/SGP's optimization bound value
        """
        x = np.array(x).reshape(-1).astype(int)
        Xu = np.ones((self.num_inducing, self.inducing_dim), dtype=np.float32)

        # Initialize all inducing points at the first solution placement location.
        # Ensures that the number of inducing points is always fixed and no additional
        # information is passed to the SGP
        Xu *= self.locs[x][0]

        # Copy all given solution placements to the inducing points set
        Xu[-len(x):] = self.locs[x]

        # Update the SGP inducing points
        self.sgp.inducing_variable.Z.assign(Xu)
        return self.sgp.elbo().numpy() # return the ELBO


def get_greedy_sgp_sol(num_sensors, candidates, X_train, noise_variance, kernel, 
                       transform=None):
    """Get sensor placement solutions using the Greedy-SGP method. Uses a greedy algorithm to 
    select sensor placements from a given discrete set of candidates locations.

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [[Jakkala and Akella, 2023](https://www.itskalvik.com/publication/sgp-sp/)]

    Args:
        num_sensors (int): Number of sensor locations to optimize
        candidates (ndarray): (n, d); Candidate sensor placement locations
        X_train (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object

    Returns:
        Xu (ndarray): (m, d); Solution sensor placement locations
    """
    sgp_model = GreedySGP(num_sensors, candidates, X_train, 
                          noise_variance, kernel, transform=transform)
    model = CustomSelection(num_sensors,
                            sgp_model.bound,
                            optimizer='naive',
                            verbose=False)
    sol = model.fit_transform(np.arange(len(candidates)).reshape(-1, 1))
    return candidates[sol.reshape(-1)]