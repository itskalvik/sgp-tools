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

import cma
import numpy as np
from shapely import geometry
from ..utils.metrics import get_mi
from ..utils.data import get_inducing_pts


class CMA_ES:
    """Class for optimizing sensor placements using CMA-ES (a genetic algorithm)

    Refer to the following paper for more details:
        - Adaptive Continuous-Space Informative Path Planning for Online Environmental Monitoring [Hitz et al., 2017]

    Args:
        X_train (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        distance_budget (float): Distance budget for when treating the inducing points 
                                 as waypoints of a path
        num_robots (int): Number of robots, used when modeling 
                          multi-robot IPP with a distance budget
        transform (Transform): Transform object
    """
    def __init__(self, X_train, noise_variance, kernel,
                 distance_budget=None,
                 num_robots=1,
                 transform=None):
        self.boundaries = geometry.MultiPoint([[p[0], p[1]] for p in X_train]).convex_hull
        self.X_train = X_train
        self.noise_variance = noise_variance
        self.kernel = kernel
        self.num_dim = X_train.shape[-1]
        self.distance_budget = distance_budget
        self.num_robots = num_robots
        self.transform = transform

    def update(self, noise_variance, kernel):
        """Update GP noise variance and kernel function parameters

        Args:
            noise_variance (float): data variance
            kernel (gpflow.kernels.Kernel): gpflow kernel function
        """
        self.noise_variance = noise_variance
        self.kernel = kernel
    
    def objective(self, X):
        """Objective function (GP-based Mutual Information)

        Args:
            X (ndarray): (n, d); Current solution sensor placement locations
        """
        # MI does not depend on waypoint order (reshape to -1, num_dim)
        X = np.array(X).reshape(-1, self.num_dim)
        constraints_loss = 0.0
        if self.transform is not None:
            X = self.transform.expand(X)
            constraints_loss = self.transform.constraints(X)

        try:
            mi = -get_mi(X, self.X_train, self.noise_variance, self.kernel)
            mi -= constraints_loss
            mi = mi.numpy()
        except:
            mi = 0.0 # if the cholskey decomposition fails
        return mi
    
    def optimize(self, 
                 num_sensors=10, 
                 max_steps=5000, 
                 tol=1e-6, 
                 X_init=None,
                 verbose=0,
                 seed=1234):
        """Optimizes the sensor placements using CMA-ES without any constraints

        Args:
            num_sensors (int): Number of sensor locations to optimize
            max_steps (int): Maximum number of optimization steps
            tol (float): Convergence tolerance to decide when to stop optimization
            X_init (ndarray): (m, d); Initial inducing points
            verbose (int): The level of verbosity.
            seed (int): The algorithm will use it to seed the randomnumber generator, ensuring replicability.
            
        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        sigma0 = 1.0
        
        if X_init is None:
            X_init = get_inducing_pts(self.X_train, num_sensors, random=True)
        X_init = X_init.reshape(-1)

        xopt, _ = cma.fmin2(self.objective, X_init, sigma0, 
                            options={'maxfevals': max_steps,
                                     'verb_disp': verbose,
                                     'tolfun': tol,
                                     'seed': seed},
                            restarts=5)
        
        xopt = np.array(xopt).reshape(-1, self.num_dim)
        if self.transform is not None:
            xopt = self.transform.expand(xopt, 
                                         expand_sensor_model=False)
            if not isinstance(xopt, np.ndarray):
                xopt = xopt.numpy()
        return xopt.reshape(-1, self.num_dim)