from .utils.misc import cont2disc
from .objectives import get_objective
from .utils.data import get_inducing_pts
from .utils.gpflow import optimize_model
from .core.augmented_sgpr import AugmentedSGPR

import numpy as np

import cma
from copy import deepcopy
from shapely import geometry
from apricot import CustomSelection
from bayes_opt import BayesianOptimization


class Base:
    def  __init__(self,
                  num_sensing,
                  X_objective,
                  kernel,
                  noise_variance,
                  transform=None,
                  num_robots=1,
                  X_candidates=None,
                  **kwargs):
        self.num_sensing = num_sensing
        self.num_dim = X_objective.shape[-1]
        self.num_robots = num_robots
        self.X_candidates = X_candidates

    def optimize(self):
        raise NotImplementedError

    def update(kernel, noise_variance):
        raise NotImplementedError
    
    def get_hyperparameters(self):
        raise NotImplementedError


class BayesianOpt(Base):
    def  __init__(self,
                  num_sensing,
                  X_objective,
                  kernel,
                  noise_variance,
                  transform=None,
                  num_robots=1,
                  X_candidates=None,
                  objective='SLogMI',
                  **kwargs):
        super().__init__(num_sensing,
                         X_objective,
                         kernel,
                         noise_variance,
                         transform,
                         num_robots,
                         X_candidates)
        self.transform = transform

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective,
                                                      kernel,
                                                      noise_variance,
                                                      **kwargs)
        else:
            self.objective = objective

        # Use the boundaries of the X_objective area as the search space limits
        pbounds_dims = []
        for i in range(self.num_dim):
            pbounds_dims.append((np.min(X_objective[:, i]), 
                                 np.max(X_objective[:, i])))
        self.pbounds = {}
        for i in range(self.num_dim*self.num_sensing*self.num_robots):
            self.pbounds[f'x{i}'] = pbounds_dims[i%self.num_dim]

    def update(self, kernel, noise_variance):
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self):
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance
    
    def optimize(self, 
                 max_steps=50,  
                 init_points=10,
                 verbose=0,
                 seed=None,
                 **kwargs):
        # Maximize the objective
        optimizer = BayesianOptimization(f=self._objective,
                                         pbounds=self.pbounds,
                                         verbose=verbose,
                                         random_state=seed,
                                         allow_duplicate_points=True)
        optimizer.maximize(init_points=init_points,
                           n_iter=max_steps)

        sol = []
        for i in range(self.num_dim*self.num_sensing*self.num_robots):
            sol.append(optimizer.max['params']['x{}'.format(i)])
        sol = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            sol = self.transform.expand(sol,
                                        expand_sensor_model=False)
            if not isinstance(sol, np.ndarray):
                sol = sol.numpy()

        # Map solution locations to candidates set locations
        if self.X_candidates is not None:
            sol = cont2disc(sol, self.X_candidates)

        sol = sol.reshape(self.num_robots, -1, self.num_dim)
        return sol
    
    def _objective(self, **kwargs):
        """Objective function

        Args:
            x<i> (ndarray): (1, d); Current solution sensor placement location i
        """
        X = []
        for i in range(len(kwargs)):
            X.append(kwargs['x{}'.format(i)])
        X = np.array(X).reshape(-1, self.num_dim)

        constraint_penality = 0.0
        if self.transform is not None:
            X = self.transform.expand(X)
            constraint_penality = self.transform.constraints(X)
        reward = self.objective(X) # maximize
        reward += constraint_penality # minimize (large negative value when constraint is unsatisfied)
        return reward.numpy()


class CMA(Base):
    def  __init__(self,
                  num_sensing,
                  X_objective,
                  kernel,
                  noise_variance,
                  transform=None,
                  num_robots=1,
                  X_candidates=None,
                  objective='SLogMI',
                  X_init=None,
                  **kwargs):
        super().__init__(num_sensing,
                         X_objective,
                         kernel,
                         noise_variance,
                         transform,
                         num_robots,
                         X_candidates)
        self.transform = transform
        if X_init is None:
            X_init = get_inducing_pts(X_objective, 
                                      num_sensing*self.num_robots)
        self.X_init = X_init.reshape(-1)

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective,
                                                      kernel,
                                                      noise_variance,
                                                      **kwargs)
        else:
            self.objective = objective

        # Use the boundaries of the X_objective area as the search space limits
        self.pbounds = geometry.MultiPoint([[p[0], p[1]] for p in X_objective]).convex_hull

    def update(self, kernel, noise_variance):
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self):
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance
    
    def optimize(self, 
                 max_steps=500,  
                 tol=1e-6,
                 verbose=0,
                 seed=None,
                 restarts=5,
                 **kwargs):
        sigma0 = 1.0
        # Minimize the objective
        sol, _ = cma.fmin2(self._objective, self.X_init, sigma0, 
                           options={'maxfevals': max_steps,
                                    'verb_disp': verbose,
                                    'tolfun': tol,
                                    'seed': seed},
                           restarts=restarts)
        
        sol = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            sol = self.transform.expand(sol,
                                        expand_sensor_model=False)
            if not isinstance(sol, np.ndarray):
                sol = sol.numpy()

        # Map solution locations to candidates set locations
        if self.X_candidates is not None:
            sol = cont2disc(sol, self.X_candidates)

        sol = sol.reshape(self.num_robots, -1, self.num_dim)
        return sol
    
    def _objective(self, X):
        """Objective function

        Args:
            X (ndarray): (n, d); Current solution sensor placement locations
        """
        X = np.array(X).reshape(-1, self.num_dim)
        constraint_penality = 0.0
        if self.transform is not None:
            X = self.transform.expand(X)
            constraint_penality = self.transform.constraints(X)
        reward = self.objective(X) # maximize
        reward += constraint_penality # minimize (large negative value when constraint is unsatisfied)
        return -reward.numpy()
    
    def update_transform(self, transform):
        self.transform = transform

    def get_transform(self):
        return deepcopy(self.transform)
    

class ContinuousSGP(Base):
    def  __init__(self,
                  num_sensing,
                  X_objective,
                  kernel,
                  noise_variance,
                  transform=None,
                  num_robots=1,
                  X_candidates=None,
                  X_init=None,
                  X_time=None, 
                  orientation=False,
                  **kwargs):
        super().__init__(num_sensing,
                         X_objective,
                         kernel,
                         noise_variance,
                         transform,
                         num_robots,
                         X_candidates)
        if X_init is None:
            X_init = get_inducing_pts(X_objective, 
                                      num_sensing*self.num_robots,
                                      orientation=orientation)

        # Fit the SGP
        train_set = (X_objective, 
                     np.zeros((len(X_objective), 1)).astype(X_objective.dtype))
        self.sgpr = AugmentedSGPR(train_set,
                                  noise_variance=noise_variance,
                                  kernel=kernel, 
                                  inducing_variable=X_init,
                                  inducing_variable_time=X_time,
                                  transform=transform)

    def update(self, kernel, noise_variance):
        self.sgpr.update(kernel, noise_variance)

    def get_hyperparameters(self):
        return deepcopy(self.sgpr.kernel), \
               self.sgpr.likelihood.variance.numpy()
    
    def optimize(self, 
                 max_steps=500, 
                 lr=1e-2, 
                 optimizer='scipy.L-BFGS-B', 
                 verbose=0,
                 **kwargs):
        verbose = True if verbose > 0 else False
        loss = optimize_model(self.sgpr,
                              max_steps=max_steps,
                              kernel_grad=False, 
                              lr=lr, 
                              optimizer=optimizer, 
                              verbose=verbose,
                              **kwargs)
        sol = self.sgpr.inducing_variable.Z
        sol = self.sgpr.transform.expand(sol,
                                         expand_sensor_model=False)
        if not isinstance(sol, np.ndarray):
            sol = sol.numpy()

        # Map solution locations to candidates set locations
        if self.X_candidates is not None:
            sol = cont2disc(sol, self.X_candidates)

        sol = sol.reshape(self.num_robots, -1, self.num_dim)
        return sol

    @property
    def transform(self):
        return self.sgpr.transform
    

class GreedyObjective(Base):
    def  __init__(self,
                  num_sensing,
                  X_objective,
                  kernel,
                  noise_variance,
                  transform=None,
                  num_robots=1,
                  X_candidates=None,
                  objective='SLogMI',
                  **kwargs):
        super().__init__(num_sensing,
                         X_objective,
                         kernel,
                         noise_variance,
                         transform,
                         num_robots,
                         X_candidates)
        if X_candidates is None:
            self.X_candidates = X_objective

        if transform is not None:
            try:
                num_robots = transform.num_robots 
            except:
                num_robots = 1
            error = f"num_robots is not equal in transform:{num_robots} and GreedySGP:{self.num_robots}"
            assert self.num_robots == num_robots, error

        error = f"num_robots={self.num_robots}; GreedySGP only supports num_robots=1"
        assert self.num_robots == 1, error

        self.transform = transform
        self.X_objective = X_objective

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective,
                                                      kernel,
                                                      noise_variance,
                                                      **kwargs)
        else:
            self.objective = objective

    def update(self, kernel, noise_variance):
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self):
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance
    
    def optimize(self, 
                 optimizer='naive',
                 verbose=0,
                 **kwargs):
        verbose = True if verbose > 0 else False
        model = CustomSelection(self.num_sensing,
                                self._objective,
                                optimizer=optimizer,
                                verbose=False)
        sol = model.fit_transform(np.arange(len(self.X_candidates)).reshape(-1, 1))
        sol = np.array(sol).reshape(-1).astype(int)
        sol = self.X_candidates[sol]
        sol = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            sol = self.transform.expand(sol,
                                        expand_sensor_model=False)
            if not isinstance(sol, np.ndarray):
                sol = sol.numpy()
        sol = sol.reshape(self.num_robots, -1, self.num_dim)
        return sol

    def _objective(self, X):
        """Objective function

        Args:
            X (ndarray): (n, d); Current solution sensor placement locations
        """
        # Map solution location indices to locations
        X = np.array(X).reshape(-1).astype(int)
        X = self.X_objective[X].reshape(-1, self.num_dim)
        constraint_penality = 0.0
        if self.transform is not None:
            X = self.transform.expand(X)
            constraint_penality = self.transform.constraints(X)
        reward = self.objective(X) # maximize
        reward -= constraint_penality # minimize
        return reward.numpy()
        

class GreedySGP(Base):
    def  __init__(self,
                  num_sensing,
                  X_objective,
                  kernel,
                  noise_variance,
                  transform=None,
                  num_robots=1,
                  X_candidates=None,
                  **kwargs):
        super().__init__(num_sensing,
                         X_objective,
                         kernel,
                         noise_variance,
                         transform,
                         num_robots,
                         X_candidates)
        if X_candidates is None:
            self.X_candidates = X_objective
            
        if transform is not None:
            try:
                num_robots = transform.num_robots 
            except:
                num_robots = 1
            error = f"num_robots is not equal in transform:{num_robots} and GreedySGP:{self.num_robots}"
            assert self.num_robots == num_robots, error

        error = f"num_robots={self.num_robots}; GreedySGP only supports num_robots=1"
        assert self.num_robots == 1, error

        self.X_objective = X_objective

        # Fit the SGP
        train_set = (X_objective, 
                     np.zeros((len(X_objective), 1)).astype(X_objective.dtype))
        X_init = get_inducing_pts(X_objective, 
                                  num_sensing)
        self.sgpr = AugmentedSGPR(train_set,
                                  noise_variance=noise_variance,
                                  kernel=kernel, 
                                  inducing_variable=X_init,
                                  transform=transform)

    def update(self, kernel, noise_variance):
        self.sgpr.update(kernel, noise_variance)

    def get_hyperparameters(self):
        return deepcopy(self.sgpr.kernel), \
               self.sgpr.likelihood.variance.numpy()

    def optimize(self, 
                 optimizer='naive',
                 verbose=0,
                 **kwargs):
        verbose = True if verbose > 0 else False
        model = CustomSelection(self.num_sensing,
                                self._objective,
                                optimizer=optimizer,
                                verbose=False)
        sol = model.fit_transform(np.arange(len(self.X_candidates)).reshape(-1, 1))
        sol = np.array(sol).reshape(-1).astype(int)
        sol = self.X_candidates[sol]
        sol = np.array(sol).reshape(-1, self.num_dim)
        sol = self.sgpr.transform.expand(sol,
                                         expand_sensor_model=False)
        if not isinstance(sol, np.ndarray):
            sol = sol.numpy()
        sol = sol.reshape(self.num_robots, -1, self.num_dim)
        return sol

    def _objective(self, X):
        """Objective function

        Args:
            X (ndarray): (n, d); Current solution sensor placement locations
        """
        # Map solution location indices to locations
        # Since SGP requires num_sensing points,
        # pad the current greedy solution with the 
        # first location in the solution
        X = np.array(X).reshape(-1).astype(int)
        num_pad = self.num_sensing - len(X)
        X_pad = np.zeros(num_pad, dtype=int)
        X = np.concatenate([X, X_pad])
        X = self.X_objective[X].reshape(-1, self.num_dim)

        # Update the SGP inducing points
        self.sgpr.inducing_variable.Z.assign(X)
        return self.sgpr.elbo().numpy()

    @property
    def transform(self):
        return self.sgpr.transform


METHODS = {
    'BayesianOpt': BayesianOpt,
    'CMA': CMA,
    'ContinuousSGP': ContinuousSGP,
    'GreedyObjective': GreedyObjective,
    'GreedySGP': GreedySGP,
}

def get_method(method):
    return METHODS[method]
