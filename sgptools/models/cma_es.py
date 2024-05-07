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
    """
    def __init__(self, X_train, noise_variance, kernel,
                 distance_budget=None,
                 num_robots=1):
        self.boundaries = geometry.MultiPoint([[p[0], p[1]] for p in X_train]).convex_hull
        self.X_train = X_train
        self.noise_variance = noise_variance
        self.kernel = kernel
        self.num_dim = X_train.shape[-1]
        self.distance_budget = distance_budget
        self.num_robots = num_robots

    def update(self, noise_variance, kernel):
        """Update GP noise variance and kernel function parameters

        Args:
            noise_variance (float): data variance
            kernel (gpflow.kernels.Kernel): gpflow kernel function
        """
        self.noise_variance = noise_variance
        self.kernel = kernel

    def constraint(self, X):
        """Constraint function for the optimization problem (constraint to limit the boundary of the region)
        Does not work well with CMA-ES as it is a step function and is not continuous

        Args:
            X (ndarray): (n, d); Current sensor placement locations
        """
        X = np.array(X).reshape(-1, self.num_dim)
        lagrangian = [self.boundaries.contains(geometry.Point(x[0], x[1])) for x in X]
        lagrangian = np.logical_not(lagrangian).astype(float)
        return lagrangian
    
    def distance_constraint(self, X):
        """Constraint function for the optimization problem (constraint to limit the total travel distance)
        Does not work well with CMA-ES as it is a step function and is not continuous

        Args:
            X (ndarray): (n, d); Current sensor placement locations
        """
        X = np.array(X).reshape(self.num_robots, -1, self.num_dim)
        dists = np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1)
        lagrangian = dists - self.distance_budget
        lagrangian_mask = np.logical_not(lagrangian <= 0)
        lagrangian[lagrangian_mask] = 0
        lagrangian = np.sum(lagrangian)
        return lagrangian
    
    def objective(self, X, X_fixed=None):
        """Objective function (GP-based Mutual Information)

        Args:
            X (ndarray): (n, d); Initial sensor placement locations
            X_fixed (ndarray): (m, d); Inducing points that are not optimized and are always 
                                added to the inducing points set during loss function computation
        """
        # Append fixed path to current solution path
        if X_fixed is not None:
            X = np.array(X).reshape(self.num_robots, -1, self.num_dim)
            X = np.concatenate([X_fixed, X], axis=1)

        # MI does not depend on waypoint order (reshape to -1, num_dim)
        X = np.array(X).reshape(-1, self.num_dim)
        try:
            mi = -get_mi(X, self.noise_variance, self.kernel, self.X_train)
        except:
            mi = 0.0 # if the cholskey decomposition fails
        return mi
    
    def optimize(self, num_sensors=10, max_steps=100, tol=1e-2, 
                 X_init=None, X_fixed=None):
        """Optimizes the SP objective function using CMA-ES without any constraints

        Args:
            num_sensors (int): Number of sensor locations to optimize
            max_steps (int): Maximum number of optimization steps
            tol (float): Convergence tolerance to decide when to stop optimization
            X_init (ndarray): (m, d); Initial inducing points
            X_fixed (ndarray): (m, d); Inducing points that are not optimized and are always 
                                added to the inducing points set during loss function computation

        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        sigma0 = 1.0
        
        if X_init is None:
            X_init = get_inducing_pts(self.X_train, num_sensors, random=True)
        X_init = X_init.reshape(-1)

        if X_fixed is not None:
            X_fixed = np.array(X_fixed).reshape(self.num_robots, -1, self.num_dim)

        xopt, _ = cma.fmin2(self.objective, X_init, sigma0, 
                            options={'maxfevals': max_steps,
                                     'verb_disp': 0,
                                     'tolfun': tol},
                            args=(X_fixed,))
        
        if X_fixed is not None:
            xopt = np.array(xopt).reshape(self.num_robots, -1, self.num_dim)
            xopt = np.concatenate([X_fixed, xopt], axis=1)

        return xopt.reshape(-1, self.num_dim)
    
    def doptimize(self, num_sensors=10, max_steps=100, tol=1e-2):
        """Optimizes the SP objective function using CMA-ES with a distance budget constraint

        Args:
            num_sensors (int): Number of sensor locations to optimize
            max_steps (int): Maximum number of optimization steps
            tol (float): Convergence tolerance to decide when to stop optimization

        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        sigma0 = 1.0
        idx = np.random.randint(len(self.X_train), size=num_sensors)
        x_init = self.X_train[idx].reshape(-1)
        cfun = cma.ConstrainedFitnessAL(self.objective, self.distance_constraint)
        xopt, _ = cma.fmin2(cfun, x_init, sigma0, 
                            options={'maxfevals': max_steps, 
                                     'verb_disp': 0,
                                     'tolfun': tol},
                            callback=cfun.update)
        return xopt.reshape(-1, self.num_dim)

    def coptimize(self, num_sensors=10, max_steps=100, tol=1e-2):
        """Optimizes the SP objective function using CMA-ES with the constraints
        to ensure that the sensors are placed within the boundaries of the region

        Args:
            num_sensors (int): Number of sensor locations to optimize
            max_steps (int): Maximum number of optimization steps
            tol (float): Convergence tolerance to decide when to stop optimization

        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        sigma0 = 1.0
        idx = np.random.randint(len(self.X_train), size=num_sensors*self.num_robots)
        x_init = self.X_train[idx].reshape(-1)
        cfun = cma.ConstrainedFitnessAL(self.objective, self.constraint)
        xopt, _ = cma.fmin2(cfun, x_init, sigma0, 
                            options={'maxfevals': max_steps, 
                                     'verb_disp': 0,
                                     'tolfun': tol},
                            callback=cfun.update)
        return xopt.reshape(-1, self.num_dim)
