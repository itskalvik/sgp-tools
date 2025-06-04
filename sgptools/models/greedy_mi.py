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

from ..utils.mutual_information import SLogMI
from apricot import CustomSelection
import numpy as np


class GreedyMI(SLogMI):
    """Helper class to compute mutual information using a Gaussian process for a given set of sensor locations.
    Used by `get_greedy_mi_sol` function to compute the solution sensor placements using the Greedy-MI method.

    Refer to the following papers for more details:
        - Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies [Krause et al., 2008]
        - Data-driven learning and planning for environmental sampling [Ma et al., 2018]

    Args:
        S (ndarray): (n, d); Candidate sensor placement locations
        V (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object
    """
    def __init__(self, S, V, noise_variance, kernel, 
                 transform=None):
        self.S = S
        self.kernel = kernel
        self.input_dim = S.shape[1]
        self.noise_variance = noise_variance
        self.transform = transform

        super().__init__(kernel, V, jitter=1e-6+noise_variance)

    def mutual_info(self, idx):
        # Map solution location indices to locations
        idx = np.array(idx).reshape(-1).astype(int)
        X = self.S[idx].reshape(-1, self.input_dim)

        # Apply transform if available
        if self.transform is not None:
            X = self.transform.expand(X)
        return self.get_mi(X).numpy()


def get_greedy_mi_sol(num_sensors, candidates, X_train, noise_variance, kernel, 
                      transform=None, optimizer='naive'):
    """Get sensor placement solutions using the GP-based mutual information approach (submodular objective function). 
    Uses a greedy algorithm to select sensor placements from a given discrete set of candidates locations.

    Refer to the following papers for more details:
        - Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies [Krause et al., 2008]
        - Data-driven learning and planning for environmental sampling [Ma et al., 2018]

    Args:
        num_sensors (int): Number of sensor locations to optimize
        candidates (ndarray): (n, d); Candidate sensor placement locations
        X_train (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object
        optimizer (str): Name of an optimizer available in the apricot library

    Returns:
        Xu (ndarray): (m, d); Solution sensor placement locations
    """
    mi_model = GreedyMI(candidates, X_train, noise_variance, kernel, transform)
    model = CustomSelection(num_sensors,
                            mi_model.mutual_info,
                            optimizer=optimizer,
                            verbose=False)
    sol = model.fit_transform(np.arange(len(candidates)).reshape(-1, 1))
    return candidates[sol.reshape(-1)]