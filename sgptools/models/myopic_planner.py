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
from ..utils.metrics import get_distance
from ..utils.mutual_information import SLogMI


class MyopicPlanner(SLogMI):
    """Class for optimizing sensor placements using the myopic planner

    Refer to the following paper for more details:
        - AK: Attentive Kernel for Information Gathering [Chen et al., 2022]

    Args:
        X_train (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object
    """
    def __init__(self, X_train, noise_variance, kernel, 
                 transform=None):
        self.transform = transform
        self.distance_budget = transform.distance_budget
        super().__init__(kernel, X_train, jitter=1e-6+noise_variance)

    def plan_waypoint(self, X):
        """Plan informative waypoint given the robot location.

        Args:
            X (ndarray): (n, d); Past robot waypoints
        """
        # Evaluate the candidates
        mi_values = []
        for X_sample in self.X_train:
            if len(X) > 0: 
                locs = np.vstack([X, X_sample])
                if self.transform is not None and len(X) > 1:
                    locs = self.transform.expand(locs).numpy().reshape(-1, 2)
                mi_values.append(self.get_mi(locs))
            else:
                mi_values.append(self.get_mi(np.array([X_sample])))
        mi_values = np.array(mi_values)       
        mi_values = ((mi_values - mi_values.min()) / mi_values.ptp())

        if len(X) > 0: 
            path_dist = get_distance(X)
            dists = np.linalg.norm(self.X_train - X[-1], axis=-1)
            # remove candidates that exceed the distance budget
            dists[path_dist+dists > self.distance_budget] = np.inf
            ptp = dists.ptp()
            if ptp == 0 or not np.isfinite(ptp):
                return X
            dists = ((dists - dists.min()) / ptp)
        else:
            dists = np.zeros(len(self.X_train))

        # Compute scores and select the candidate with the highest score
        scores = mi_values - dists
        waypoint = self.X_train[np.argmax(scores)]
        
        if len(X) > 0: 
            return np.vstack([X, waypoint])
        else:
            return np.array([waypoint])

    def optimize(self, 
                 num_sensors=10):
        """Optimizes the sensor placements using the myopic planner

        Args:
            num_sensors (int): Number of sensor locations to optimize.

        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        sol = []
        for i in range(num_sensors):
            sol = self.plan_waypoint(sol)
        return sol