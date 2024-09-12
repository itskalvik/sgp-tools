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

from apricot import CustomSelection
from gpflow.models.gpr import GPR
import numpy as np


class GreedyMI:
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
        self.V = V
        self.kernel = kernel
        self.input_dim = S.shape[1]
        self.noise_variance = noise_variance
        self.transform = transform

    def mutual_info(self, x):
        """Computes mutual information using the points `x` 

        Args:
            x (ndarray): (n); Indices of the solution placement locations

        Returns:
            MI (float): Mutual information between the placement x and candidate locations
        """
        x = np.array(x).reshape(-1).astype(int)
        A = self.S[x[:-1]].reshape(-1, self.input_dim)
        y = self.S[x[-1]].reshape(-1, self.input_dim)
    
        if len(A) == 0:
            sigma_a = 1.0
        else:
            if self.transform is not None:
                A = self.transform.expand(A)
            a_gp = GPR(data=(A, np.zeros((len(A), 1))),
                       kernel=self.kernel,
                       noise_variance=self.noise_variance)
            _, sigma_a = a_gp.predict_f(y)

        # Remove locations in A∪y from V to build A bar (Refer to Krause et al., 2008)
        V_ = self.V.copy()
        V_rows = V_.view([('', V_.dtype)] * V_.shape[1])

        if self.transform is not None:
            solution = self.S[x].reshape(-1, self.input_dim)
            A_ = self.transform.expand(solution)
        else:
            A_ = self.S[x]
        A_rows = A_.view([('', V_.dtype)] * A_.shape[1])

        V_ = np.setdiff1d(V_rows, A_rows).view(V_.dtype).reshape(-1, V_.shape[1])

        self.v_gp = GPR(data=(V_, np.zeros((len(V_), 1))), 
                        kernel=self.kernel,
                        noise_variance=self.noise_variance)
        _, sigma_v = self.v_gp.predict_f(y)

        return (sigma_a/sigma_v).numpy().squeeze()


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