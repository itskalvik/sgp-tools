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
    """
    def __init__(self, X_train, noise_variance, kernel):
        self.X_train = X_train
        self.noise_variance = noise_variance
        self.kernel = kernel
        self.num_dim = X_train.shape[-1]
        
        # use the boundaries of the region as the search space
        self.pbounds_dim = []
        for i in range(self.num_dim):
            self.pbounds_dim.append((np.min(X_train[:, i]), np.max(X_train[:, i])))

    def objective(self, **kwargs):
        """Computes the objective function (mutual information) for the sensor placement problem
        """
        X = []
        for i in range(len(kwargs)):
            X.append(kwargs['x{}'.format(i)])
        X = np.array(X).reshape(-1, self.num_dim)
        return -get_mi(X, self.noise_variance, self.kernel, self.X_train)
    
    def optimize(self, 
                 num_sensors=10, 
                 max_steps=100,  
                 X_init=None,
                 init_points=10):
        """Optimizes the sensor placements using Bayesian Optimization without any constraints

        Args:
            num_sensors (int): Number of sensor locations to optimize
            max_steps (int): Maximum number of optimization steps 
            X_init (ndarray): (m, d); Initial inducing points
            init_points (int): How many steps of random exploration you want to perform. 
                               Random exploration can help by diversifying the exploration space. 

        Returns:
            Xu (ndarray): (m, d); Solution sensor placement locations
        """
        if X_init is None:
            X_init = get_inducing_pts(self.X_train, num_sensors, random=True)
        X_init = X_init.reshape(-1)

        pbounds = {}
        for i in range(self.num_dim*num_sensors):
            pbounds['x{}'.format(i)] = self.pbounds_dim[i%self.num_dim]

        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=pbounds,
            verbose=0,
            random_state=1,
            allow_duplicate_points=True
        )

        optimizer.maximize(
            init_points=init_points,
            n_iter=max_steps,
        )

        sol = []
        for i in range(self.num_dim*num_sensors):
            sol.append(optimizer.max['params']['x{}'.format(i)])
        return np.array(sol).reshape(-1, self.num_dim)

