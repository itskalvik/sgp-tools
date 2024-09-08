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
from ..utils.metrics import get_mi
from ..utils.data import get_inducing_pts
from bayes_opt import BayesianOptimization


class BayesianOpt:
    """Class for optimizing sensor placements using Bayesian Optimization

    Refer to the following papers for more details:
        - UAV route planning for active disease classification [Vivaldini et al., 2019]
        - Occupancy map building through Bayesian exploration [Francis et al., 2019]

    Args:
        X_train (ndarray): (n, d); Locations in the environment used to approximate the monitoring regions
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        transform (Transform): Transform object
    """
    def __init__(self, X_train, noise_variance, kernel,
                 transform=None):
        self.X_train = X_train
        self.noise_variance = noise_variance
        self.kernel = kernel
        self.num_dim = X_train.shape[-1]
        self.transform = transform

        # use the boundaries of the region as the search space
        self.pbounds_dim = []
        for i in range(self.num_dim):
            self.pbounds_dim.append((np.min(X_train[:, i]), np.max(X_train[:, i])))
  
    def objective(self, **kwargs):
        """Objective function (GP-based Mutual Information)

        Args:
            x<i> (ndarray): (1, d); Current solution sensor placement location i
        """
        # MI does not depend on waypoint order (reshape to -1, num_dim)
        X = []
        for i in range(len(kwargs)):
            X.append(kwargs['x{}'.format(i)])
        X = np.array(X).reshape(-1, self.num_dim)
        if self.transform is not None:
            X = self.transform.expand(X)
            constraints_loss = self.transform.constraints(X)

        try:
            mi = get_mi(X, self.X_train, self.noise_variance, self.kernel)
            mi += constraints_loss
            mi = mi.numpy()
        except:
            mi = -1e4 # if the cholskey decomposition fails
        return mi

    def optimize(self, 
                 num_sensors=10, 
                 max_steps=100,  
                 X_init=None,
                 init_points=10,
                 verbose=0, 
                 seed=1234):
        """Optimizes the sensor placements using Bayesian Optimization without any constraints

        Args:
            num_sensors (int): Number of sensor locations to optimize.
            max_steps (int): Maximum number of optimization steps. 
            X_init (ndarray): (m, d); Initial inducing points.
            init_points (int): Number of random solutions used for initial exploration. 
                               Random exploration can help by diversifying the exploration space. 
            verbose (int): The level of verbosity.
            seed (int): The algorithm will use it to seed the randomnumber generator, ensuring replicability.

        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        if X_init is None:
            X_init = get_inducing_pts(self.X_train, num_sensors, random=True)
        else:
            num_sensors = len(X_init.reshape(-1, self.num_dim))
        X_init = X_init.reshape(-1)

        pbounds = {}
        for i in range(self.num_dim*num_sensors):
            pbounds['x{}'.format(i)] = self.pbounds_dim[i%self.num_dim]

        optimizer = BayesianOptimization(f=self.objective,
                                         pbounds=pbounds,
                                         verbose=verbose,
                                         random_state=seed,
                                         allow_duplicate_points=True)
        optimizer.maximize(init_points=init_points,
                           n_iter=max_steps)

        sol = []
        for i in range(self.num_dim*num_sensors):
            sol.append(optimizer.max['params']['x{}'.format(i)])
        sol = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            sol = self.transform.expand(sol,
                                        expand_sensor_model=False)
            if not isinstance(sol, np.ndarray):
                sol = sol.numpy()
        return sol.reshape(-1, self.num_dim)