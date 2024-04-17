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


'''
Class for optimizing sensor placements with CMA-ES
'''
class CMA_ES:
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
        self.noise_variance = noise_variance
        self.kernel = kernel

    '''
    Constraint function for the problem (Boundary of the region)
    Does not work well with CMA-ES as it is step function and not continuous
    '''
    def constraint(self, X):
        X = np.array(X).reshape(-1, self.num_dim)
        lagrangian = [self.boundaries.contains(geometry.Point(x[0], x[1])) for x in X]
        lagrangian = np.logical_not(lagrangian).astype(float)
        return lagrangian
    
    def distance_constraint(self, X):
        X = np.array(X).reshape(self.num_robots, -1, self.num_dim)
        dists = np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1)
        lagrangian = dists - self.distance_budget
        lagrangian_mask = np.logical_not(lagrangian <= 0)
        lagrangian[lagrangian_mask] = 0
        lagrangian = np.sum(lagrangian)
        return lagrangian
    
    '''
    Objective function (GP-based Mutual Information)

    Args:
        X       : Solution sensing locations
        X_fixed : Sensing locations that are not optimized, but included when
                  computing the objective function. Used for online IPP (optional)
    '''
    def objective(self, X, X_fixed=None):
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
    
    '''
    Optimizes the SP objective function with CMA-ES without any constraints

    Args:
        num_sensors: Number of sensing locations to optimize
                     (ignored when X_init is not None)
        max_steps  : Maximum number of optimization steps
        tol        : Convergence tolerence threshold (i.e., when to stop optimizing)
        X_init     : Initial sensing locations (optional)
        X_fixed    : Sensing locations that are not optimized, but included when
                     computing the objective function. Used for online IPP (optional)
    '''
    def optimize(self, num_sensors=10, max_steps=100, tol=1e-2, 
                 X_init=None, X_fixed=None):
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
    
    '''
    Optimizes the objective function with CMA-ES along with the constraints
    to ensure that path length is within the distance budget
    '''
    def coptimize(self, num_sensors=10, num_steps=100, tol=1e-2):
        sigma0 = 1.0
        idx = np.random.randint(len(self.X_train), size=num_sensors)
        x_init = self.X_train[idx].reshape(-1)
        cfun = cma.ConstrainedFitnessAL(self.objective, self.distance_constraint)
        xopt, _ = cma.fmin2(cfun, x_init, sigma0, 
                            options={'maxfevals': num_steps, 
                                     'verb_disp': 0,
                                     'tolfun': tol},
                            callback=cfun.update)
        return xopt.reshape(-1, self.num_dim)

    '''
    Optimizes the objective function with CMA-ES along with the constraints
    to ensure that the sensors are placed within the boundaries of the region
    '''
    def doptimize(self, num_sensors=10, num_steps=100, tol=1e-2):
        sigma0 = 1.0
        idx = np.random.randint(len(self.X_train), size=num_sensors*self.num_robots)
        x_init = self.X_train[idx].reshape(-1)
        cfun = cma.ConstrainedFitnessAL(self.objective, self.constraint)
        xopt, _ = cma.fmin2(cfun, x_init, sigma0, 
                            options={'maxfevals': num_steps, 
                                     'verb_disp': 0,
                                     'tolfun': tol},
                            callback=cfun.update)
        return xopt.reshape(-1, self.num_dim)
