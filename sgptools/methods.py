import numpy as np
import cma
from copy import deepcopy
from shapely import geometry
from apricot import CustomSelection
from bayes_opt import BayesianOptimization
import gpflow
import tensorflow as tf
from typing import Optional, List, Tuple, Union, Dict, Any, Type

from sgptools.utils.misc import cont2disc, get_inducing_pts
from sgptools.objectives import get_objective, Objective
from sgptools.utils.gpflow import optimize_model
from sgptools.core.augmented_sgpr import AugmentedSGPR
from sgptools.core.transformations import Transform  # Import Transform for type hinting


class Method:
    """
    Method class for optimization methods.

    Attributes:
        num_sensing (int): Number of sensing locations to optimize.
        num_dim (int): Dimensionality of the data points.
        num_robots (int): Number of robots/agents.
        X_objective (np.ndarray): (n, d); Data points used to define the objective function.
        kernel (gpflow.kernels.Kernel): GPflow kernel function.
        noise_variance (float): Data noise variance.
        transform (Optional[Transform]): Transform object to apply to inducing points.
        X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
        num_dim (int): Dimensionality of the sensing locations.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 **kwargs: Any):
        """
        Initializes the Method class.

        Args:
            num_sensing (int): Number of sensing locations to optimize.
            X_objective (np.ndarray): (n, d); Data points used to define the objective function.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 Defaults to None.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            **kwargs: Additional keyword arguments.
        """
        self.num_sensing = num_sensing
        self.num_robots = num_robots
        self.X_candidates = X_candidates
        if num_dim is None:
            self.num_dim = X_objective.shape[-1]
        else:
            self.num_dim = num_dim

    def optimize(self) -> np.ndarray:
        """
        Optimizes the sensor placements/path(s).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized sensing locations.
        """
        raise NotImplementedError

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the underlying model/objective.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing the kernel and noise variance.
        """
        raise NotImplementedError


class BayesianOpt(Method):
    """
    Implements informative sensor placement/path optimization using Bayesian Optimization.

    This method optimizes a given objective function (e.g., Mutual Information)
    by sampling and evaluating points in the search space, building a surrogate
    model, and using an acquisition function to guide further sampling.

    Refer to the following papers for more details:
        - UAV route planning for active disease classification [Vivaldini et al., 2019]
        - Occupancy map building through Bayesian exploration [Francis et al., 2019]

    Attributes:
        objective (Objective): The objective function to be optimized.
        transform (Optional[Transform]): Transform object applied to inducing points.
        pbounds (Dict[str, Tuple[float, float]]): Dictionary defining the search space bounds.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 objective: Union[str, Objective] = 'SLogMI',
                 **kwargs: Any):
        """
        Initializes the BayesianOpt optimizer.

        Args:
            num_sensing (int): Number of sensing locations to optimize.
            X_objective (np.ndarray): (n, d); Data points used to define the objective function.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 Defaults to None.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            objective (Union[str, Objective]): The objective function to use. Can be a string (e.g., 'SLogMI', 'MI')
                                         or an instance of an objective class. Defaults to 'SLogMI'.
            **kwargs: Additional keyword arguments passed to the objective function.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.transform = transform

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective, kernel,
                                                      noise_variance, **kwargs)
        else:
            self.objective = objective

        # Use the boundaries of the X_objective area as the search space limits
        pbounds_dims: List[Tuple[float, float]] = []
        for i in range(self.num_dim):
            pbounds_dims.append(
                (np.min(X_objective[:, i]), np.max(X_objective[:, i])))
        self.pbounds: Dict[str, Tuple[float, float]] = {}
        for i in range(self.num_dim * self.num_sensing * self.num_robots):
            self.pbounds[f'x{i}'] = pbounds_dims[i % self.num_dim]

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the objective function.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters from the objective.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing a deep copy of the kernel and the noise variance.
        """
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance

    def optimize(self,
                 max_steps: int = 50,
                 init_points: int = 10,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimizes the sensor placement/path using Bayesian Optimization.

        Args:
            max_steps (int): Maximum number of optimization steps (iterations). Defaults to 50.
            init_points (int): Number of random exploration steps before Bayesian Optimization starts. Defaults to 10.
            verbose (bool): Verbosity, if True additional details will by reported. Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized sensing locations.

        Usage:
            ```python
            # Assuming X_train, candidates, kernel_opt, noise_variance_opt are defined
            bo_method = BayesianOpt(
                num_sensing=10,
                X_objective=X_train,
                kernel=kernel_opt,
                noise_variance=noise_variance_opt,
                transform=IPPTransform(num_robots=1), # Example transform
                X_candidates=candidates
            )
            optimized_solution = bo_method.optimize(max_steps=50, init_points=10)
            ```
        """
        verbose = 1 if verbose else 0
        optimizer = BayesianOptimization(f=self._objective,
                                         pbounds=self.pbounds,
                                         verbose=verbose,
                                         random_state=seed,
                                         allow_duplicate_points=True)
        optimizer.maximize(init_points=init_points, n_iter=max_steps)

        sol: List[float] = []
        for i in range(self.num_dim * self.num_sensing * self.num_robots):
            sol.append(optimizer.max['params'][f'x{i}'])

        sol_np = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            try:
                sol_np = self.transform.expand(sol_np,
                                               expand_sensor_model=False)
            except TypeError:
                pass

            if not isinstance(sol_np, np.ndarray):
                sol_np = sol_np.numpy()

        # Map solution locations to candidates set locations if X_candidates is provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self, **kwargs: float) -> float:
        """
        Internal objective function to be maximized by the Bayesian Optimization.

        This function reshapes the input parameters from the optimizer, applies
        any specified transformations, calculates the objective value, and
        applies a penalty for constraint violations.

        Args:
            **kwargs: Keyword arguments where keys are 'x0', 'x1', ..., representing
                      the flattened sensor placement coordinates.

        Returns:
            float: The objective value (reward - constraint penalty) to be maximized.
        """
        X_list: List[float] = []
        for i in range(len(kwargs)):
            X_list.append(kwargs[f'x{i}'])
        X = np.array(X_list).reshape(-1, self.num_dim)

        constraint_penality: float = 0.0
        if self.transform is not None:
            X_expanded = self.transform.expand(X)
            constraint_penality = self.transform.constraints(X)
            reward = self.objective(X_expanded)  # maximize
        else:
            reward = self.objective(X)  # maximize

        reward += constraint_penality  # minimize (large negative value when constraint is unsatisfied)
        return reward.numpy()


class CMA(Method):
    """
    Implements informative sensor placement/path optimization using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    CMA-ES is a powerful black-box optimization algorithm for non-convex problems.

    Refer to the following paper for more details:
        - Adaptive Continuous-Space Informative Path Planning for Online Environmental Monitoring [Hitz et al., 2017]

    Attributes:
        objective (Objective): The objective function to be minimized/maximized.
        transform (Optional[Transform]): Transform object applied to inducing points.
        X_init (np.ndarray): Initial solution guess for the optimization.
        pbounds (geometry.MultiPoint): The convex hull of the objective area, used implicitly for bounds.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 objective: Union[str, Objective] = 'SLogMI',
                 X_init: Optional[np.ndarray] = None,
                 **kwargs: Any):
        """
        Initializes the CMA-ES optimizer.

        Args:
            num_sensing (int): Number of sensing locations to optimize.
            X_objective (np.ndarray): (n, d); Data points used to define the objective function.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 Defaults to None.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            objective (Union[str, Objective]): The objective function to use. Can be a string (e.g., 'SLogMI', 'MI')
                                         or an instance of an objective class. Defaults to 'SLogMI'.
            X_init (Optional[np.ndarray]): (num_sensing * num_robots, num_dim); Initial guess for sensing locations.
                                            If None, initial points are randomly selected from X_objective.
            **kwargs: Additional keyword arguments passed to the objective function.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.transform = transform
        if X_init is None:
            X_init = get_inducing_pts(X_objective,
                                      num_sensing * self.num_robots)
        else:
            # override num_dim with initial inducing points dim, in case it differes from X_objective dim
            self.num_dim = X_init.shape[-1]

        self.X_init: np.ndarray = X_init.reshape(-1)  # Flattened initial guess

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective, kernel,
                                                      noise_variance, **kwargs)
        else:
            self.objective = objective

        # Use the boundaries of the X_objective area as the search space limits
        self.pbounds = geometry.MultiPoint([[p[0], p[1]]
                                            for p in X_objective]).convex_hull

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the objective function.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters from the objective.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing a deep copy of the kernel and the noise variance.
        """
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance

    def optimize(self,
                 max_steps: int = 500,
                 tol: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 restarts: int = 5,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimizes the sensor placement/path using CMA-ES.

        Args:
            max_steps (int): Maximum number of optimization steps (function evaluations). Defaults to 500.
            tol (float): Tolerance for termination. Defaults to 1e-6.
            verbose (bool): Verbosity, if True additional details will by reported. Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            restarts (int): Number of restarts for CMA-ES. Defaults to 5.
            **kwargs: Additional keyword arguments for CMA-ES.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized sensing locations.

        Usage:
            ```python
            # Assuming X_train, candidates, kernel_opt, noise_variance_opt are defined
            cma_method = CMA(
                num_sensing=10,
                X_objective=X_train,
                kernel=kernel_opt,
                noise_variance=noise_variance_opt,
                transform=IPPTransform(num_robots=1), # Example transform
                X_candidates=candidates
            )
            optimized_solution = cma_method.optimize(max_steps=1000)
            ```
        """
        sigma0 = 1.0
        verbose = 1 if verbose else 0
        sol, _ = cma.fmin2(self._objective,
                           self.X_init,
                           sigma0,
                           options={
                               'maxfevals': max_steps,
                               'verb_disp': verbose,
                               'tolfun': tol,
                               'seed': seed
                           },
                           restarts=restarts)

        sol_np = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            try:
                sol_np = self.transform.expand(sol_np,
                                               expand_sensor_model=False)
            except TypeError:
                pass
            if not isinstance(sol_np, np.ndarray):
                sol_np = sol_np.numpy()

        # Map solution locations to candidates set locations if X_candidates is provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self, X: np.ndarray) -> float:
        """
        Internal objective function to be minimized by CMA-ES.

        This function reshapes the input array, applies any specified transformations,
        calculates the objective value, and applies a penalty for constraint violations.
        Note: CMA-ES minimizes, so the reward (which is to be maximized) is returned as negative.

        Args:
            X (np.ndarray): (num_sensing * num_robots * num_dim); Flattened array of
                            current solution sensor placement locations.

        Returns:
            float: The negative objective value (-reward + constraint penalty) to be minimized.
        """
        X_reshaped = np.array(X).reshape(-1, self.num_dim)
        constraint_penality: float = 0.0
        if self.transform is not None:
            X_expanded = self.transform.expand(X_reshaped)
            constraint_penality = self.transform.constraints(X_reshaped)
            reward = self.objective(X_expanded)  # maximize
        else:
            reward = self.objective(X_reshaped)  # maximize
        if not np.isfinite(reward):
            reward = -1e6 # CMA does not like inf values
        reward += constraint_penality  # minimize (large negative value when constraint is unsatisfied)
        return -reward.numpy()  # Return negative as CMA-ES minimizes

    def update_transform(self, transform: Transform) -> None:
        """
        Updates the transform object used by the CMA-ES optimizer.

        Args:
            transform (Transform): The new transform object.
        """
        self.transform = transform

    def get_transform(self) -> Transform:
        """
        Retrieves a deep copy of the transform object.

        Returns:
            Transform: A deep copy of the transform object.
        """
        return deepcopy(self.transform)


class ContinuousSGP(Method):
    """
    Implements informative sensor placement/path optimization using a Sparse Gaussian Process (SGP).

    This method optimizes the inducing points of an SGP model to maximize the ELBO or other SGP-related objectives.

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [[Jakkala and Akella, 2023](https://www.itskalvik.com/publication/sgp-sp/)]
        - Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes [[Jakkala and Akella, 2024](https://www.itskalvik.com/publication/sgp-ipp/)]

    Attributes:
        sgpr (AugmentedSGPR): The Augmented Sparse Gaussian Process Regression model.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 X_init: Optional[np.ndarray] = None,
                 X_time: Optional[np.ndarray] = None,
                 orientation: bool = False,
                 **kwargs: Any):
        """
        Initializes the ContinuousSGP optimizer.

        Args:
            num_sensing (int): Number of sensing locations (inducing points) to optimize.
            X_objective (np.ndarray): (n, d); Data points used to approximate the bounds of the environment.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 Defaults to None.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            X_init (Optional[np.ndarray]): (num_sensing * num_robots, d); Initial inducing points.
                                            If None, initial points are randomly selected from X_objective.
            X_time (Optional[np.ndarray]): (m, d); Temporal dimensions of the inducing points, used when
                                            modeling spatio-temporal IPP. Defaults to None.
            orientation (bool): If True, adds an additional dimension to model sensor FoV rotation angle
                                when selecting initial inducing points. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        if X_init is None:
            X_init = get_inducing_pts(X_objective,
                                      num_sensing * self.num_robots,
                                      orientation=orientation)
        else:
            # override num_dim with initial inducing points dim, in case it differes from X_objective dim
            self.num_dim = X_init.shape[-1]

        # Initialize the SGP
        dtype = X_objective.dtype
        train_set: Tuple[tf.Tensor, tf.Tensor] = (tf.constant(X_objective,
                                                              dtype=dtype),
                                                  tf.zeros(
                                                      (len(X_objective), 1),
                                                      dtype=dtype))
        self.sgpr = AugmentedSGPR(train_set,
                                  noise_variance=noise_variance,
                                  kernel=kernel,
                                  inducing_variable=X_init,
                                  inducing_variable_time=X_time,
                                  transform=transform)

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the SGP model.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.
        """
        self.sgpr.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters from the SGP model.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing a deep copy of the kernel and the noise variance.
        """
        return deepcopy(self.sgpr.kernel), \
               self.sgpr.likelihood.variance.numpy()

    def optimize(self,
                 max_steps: int = 500,
                 optimizer: str = 'scipy.L-BFGS-B',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimizes the inducing points of the SGP model.

        Args:
            max_steps (int): Maximum number of optimization steps. Defaults to 500.
            optimizer (str): Optimizer "<backend>.<method>" to use for training (e.g., 'scipy.L-BFGS-B', 'tf.adam').
                             Defaults to 'scipy.L-BFGS-B'.
            verbose (bool): Verbosity, if True additional details will by reported. Defaults to False.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized inducing points (sensing locations).

        Usage:
            ```python
            # Assuming X_train, candidates, kernel_opt, noise_variance_opt are defined
            csgp_method = ContinuousSGP(
                num_sensing=10,
                X_objective=dataset.X_train,
                kernel=kernel_opt,
                noise_variance=noise_variance_opt,
                transform=IPPTransform(num_robots=1), # Example transform
                X_candidates=candidates # Only if the solution needs to be mapped to candidates
            )
            optimized_solution = csgp_method.optimize(max_steps=500, optimizer='scipy.L-BFGS-B')
            ```
        """
        _ = optimize_model(
            self.sgpr,
            max_steps=max_steps,
            optimize_hparams=
            False,  # Inducing points are optimized, not kernel hyperparameters
            optimizer=optimizer,
            verbose=verbose,
            **kwargs)

        sol: tf.Tensor = self.sgpr.inducing_variable.Z
        try:
            sol_expanded = self.transform.expand(sol,
                                                 expand_sensor_model=False)
        except TypeError:
            sol_expanded = sol
        if not isinstance(sol_expanded, np.ndarray):
            sol_np = sol_expanded.numpy()
        else:
            sol_np = sol_expanded

        # Map solution locations to candidates set locations if X_candidates is provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    @property
    def transform(self) -> Transform:
        """
        Gets the transform object associated with the SGP model.

        Returns:
            Transform: The transform object.
        """
        return self.sgpr.transform


class GreedyObjective(Method):
    """
    Implements informative sensor placement/path optimization using a greedy approach based on a specified objective function.

    This method iteratively selects the best sensing location from a set of candidates
    that maximizes the objective function. It currently supports only single-robot scenarios.

    Refer to the following papers for more details:
        - Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies [Krause et al., 2008]
        - Data-driven learning and planning for environmental sampling [Ma et al., 2018]

    Attributes:
        objective (Objective): The objective function to be maximized (e.g., Mutual Information).
        transform (Optional[Transform]): Transform object applied to selected locations.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 objective: Union[str, Objective] = 'SLogMI',
                 **kwargs: Any):
        """
        Initializes the GreedyObjective optimizer.

        Args:
            num_sensing (int): Number of sensing locations to select.
            X_objective (np.ndarray): (n, d); Data points used to define the objective function.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 If None, X_objective is used as candidates.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            objective (Union[str, Objective]): The objective function to use. Can be a string (e.g., 'SLogMI', 'MI')
                                         or an instance of an objective class. Defaults to 'SLogMI'.
            **kwargs: Additional keyword arguments passed to the objective function.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.X_objective = X_objective
        if X_candidates is None:
            self.X_candidates = X_objective  # Default candidates to objective points

        if transform is not None:
            try:
                num_robots_transform = transform.num_robots
            except AttributeError:
                num_robots_transform = 1  # Assume single robot if num_robots not defined in transform
            error = f"num_robots is not equal in transform: {num_robots_transform} and GreedyObjective: {self.num_robots}"
            assert self.num_robots == num_robots_transform, error

        error = f"num_robots={self.num_robots}; GreedyObjective only supports num_robots=1"
        assert self.num_robots == 1, error

        self.transform = transform

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective, kernel,
                                                      noise_variance, **kwargs)
        else:
            self.objective = objective

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the objective function.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters from the objective.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing a deep copy of the kernel and the noise variance.
        """
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance

    def optimize(self,
                 optimizer: str = 'naive',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimizes sensor placement using a greedy approach.

        Args:
            optimizer (str): The greedy optimizer strategy (e.g., 'naive', 'lazy'). Defaults to 'naive'.
            verbose (bool): Verbosity, if True additional details will by reported. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized sensing locations.

        Usage:
            ```python
            # Assuming X_train, candidates, kernel_opt, noise_variance_opt are defined
            greedy_obj_method = GreedyObjective(
                num_sensing=5,
                X_objective=X_train,
                kernel=kernel_opt,
                noise_variance=noise_variance_opt,
                X_candidates=candidates
            )
            optimized_solution = greedy_obj_method.optimize(optimizer='naive')
            ```
        """
        model = CustomSelection(self.num_sensing,
                                self._objective,
                                optimizer=optimizer,
                                verbose=verbose)

        # apricot's CustomSelection expects indices, so pass a dummy array of indices
        sol_indices = model.fit_transform(
            np.arange(len(self.X_candidates)).reshape(-1, 1))
        sol_indices = np.array(sol_indices).reshape(-1).astype(int)
        sol_locations = self.X_candidates[sol_indices]

        sol_locations = np.array(sol_locations).reshape(-1, self.num_dim)
        if self.transform is not None:
            try:
                sol_locations = self.transform.expand(
                    sol_locations, expand_sensor_model=False)
            except TypeError:
                pass
            if not isinstance(sol_locations, np.ndarray):
                sol_locations = sol_locations.numpy()
        sol_locations = sol_locations.reshape(self.num_robots, -1,
                                              self.num_dim)
        return sol_locations

    def _objective(self, X_indices: np.ndarray) -> float:
        """
        Internal objective function for the greedy selection.

        This function maps the input indices to actual locations, applies any
        transformations, calculates the objective value, and applies a penalty
        for constraint violations.

        Args:
            X_indices (np.ndarray): (n, 1); Array of indices corresponding to candidate locations.

        Returns:
            float: The objective value (reward - constraint penalty) for the given selection.
        """
        # Map solution location indices to locations
        X_indices_flat = np.array(X_indices).reshape(-1).astype(int)
        X_locations = self.X_objective[X_indices_flat].reshape(
            -1, self.num_dim)

        constraint_penality: float = 0.0
        if self.transform is not None:
            X_expanded = self.transform.expand(X_locations)
            constraint_penality = self.transform.constraints(X_locations)
            reward = self.objective(X_expanded)  # maximize
        else:
            reward = self.objective(X_locations)  # maximize

        reward += constraint_penality  # minimize (large negative value when constraint is unsatisfied)
        return reward.numpy()


class GreedySGP(Method):
    """
    Implements informative sensor placement/path optimization using a greedy approach combined with a Sparse Gaussian Process (SGP) ELBO objective.

    This method iteratively selects inducing points to maximize the SGP's ELBO.
    It currently supports only single-robot scenarios.

    Refer to the following papers for more details:
        - Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [[Jakkala and Akella, 2023](https://www.itskalvik.com/publication/sgp-sp/)]

    Attributes:
        sgpr (AugmentedSGPR): The Augmented Sparse Gaussian Process Regression model.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 **kwargs: Any):
        """
        Initializes the GreedySGP optimizer.

        Args:
            num_sensing (int): Number of sensing locations (inducing points) to select.
            X_objective (np.ndarray): (n, d); Data points used to train the SGP model.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 If None, X_objective is used as candidates.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.X_objective = X_objective
        if X_candidates is None:
            self.X_candidates = X_objective  # Default candidates to objective points

        if transform is not None:
            try:
                num_robots_transform = transform.num_robots
            except AttributeError:
                num_robots_transform = 1  # Assume single robot if num_robots not defined in transform
            error = f"num_robots is not equal in transform: {num_robots_transform} and GreedySGP: {self.num_robots}"
            assert self.num_robots == num_robots_transform, error

        error = f"num_robots={self.num_robots}; GreedySGP only supports num_robots=1"
        assert self.num_robots == 1, error

        # Initialize the SGP
        dtype = X_objective.dtype
        train_set: Tuple[tf.Tensor, tf.Tensor] = (tf.constant(X_objective,
                                                              dtype=dtype),
                                                  tf.zeros(
                                                      (len(X_objective), 1),
                                                      dtype=dtype))

        X_init = get_inducing_pts(X_objective, num_sensing)
        self.sgpr = AugmentedSGPR(train_set,
                                  noise_variance=noise_variance,
                                  kernel=kernel,
                                  inducing_variable=X_init,
                                  transform=transform)

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the SGP model.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.
        """
        self.sgpr.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters from the SGP model.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing a deep copy of the kernel and the noise variance.
        """
        return deepcopy(self.sgpr.kernel), \
               self.sgpr.likelihood.variance.numpy()

    def optimize(self,
                 optimizer: str = 'naive',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimizes sensor placement using a greedy SGP approach.

        Args:
            optimizer (str): The greedy optimizer strategy (e.g., 'naive', 'lazy'). Defaults to 'naive'.
            verbose (bool): Verbosity, if True additional details will by reported. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized sensing locations.

        Usage:
            ```python
            # Assuming X_train, candidates, kernel_opt, noise_variance_opt are defined
            greedy_sgp_method = GreedySGP(
                num_sensing=5,
                X_objective=X_train,
                kernel=kernel_opt,
                noise_variance=noise_variance_opt,
                X_candidates=candidates
            )
            optimized_solution = greedy_sgp_method.optimize(optimizer='naive')
            ```
        """
        model = CustomSelection(self.num_sensing,
                                self._objective,
                                optimizer=optimizer,
                                verbose=verbose)

        # apricot's CustomSelection expects indices, so pass a dummy array of indices
        sol_indices = model.fit_transform(
            np.arange(len(self.X_candidates)).reshape(-1, 1))
        sol_indices = np.array(sol_indices).reshape(-1).astype(int)
        sol_locations = self.X_candidates[sol_indices]

        sol_locations = np.array(sol_locations).reshape(-1, self.num_dim)
        try:
            sol_expanded = self.transform.expand(sol_locations,
                                                 expand_sensor_model=False)
        except AttributeError:
            sol_expanded = sol_locations
        if not isinstance(sol_expanded, np.ndarray):
            sol_np = sol_expanded.numpy()
        else:
            sol_np = sol_expanded

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self, X_indices: np.ndarray) -> float:
        """
        Internal objective function for the greedy SGP selection.

        This function maps the input indices to actual locations and updates
        the SGP model's inducing points to calculate the ELBO. The ELBO is
        then used as the objective for greedy maximization.

        Args:
            X_indices (np.ndarray): (n, 1); Array of indices corresponding to candidate locations.

        Returns:
            float: The ELBO of the SGP model for the given inducing points.
        """
        # Map solution location indices to locations
        # Since SGP requires num_sensing points,
        # pad the current greedy solution with the
        # first location in the solution (or zeros if no points selected yet)
        X_indices_flat = np.array(X_indices).reshape(-1).astype(int)
        num_pad = self.num_sensing - len(X_indices_flat)

        # Ensure that if X_indices_flat is empty, we still create a valid padding array
        if len(X_indices_flat) == 0 and num_pad > 0:
            X_pad = np.zeros(num_pad, dtype=int)
        elif len(X_indices_flat) > 0 and num_pad > 0:
            X_pad = np.full(num_pad, X_indices_flat[0], dtype=int)
        else:  # num_pad is 0 or negative
            X_pad = np.array([], dtype=int)

        X_combined_indices = np.concatenate([X_indices_flat, X_pad])
        X_locations = self.X_objective[X_combined_indices].reshape(
            -1, self.num_dim)

        # Update the SGP inducing points
        self.sgpr.inducing_variable.Z.assign(X_locations)
        return self.sgpr.elbo().numpy()

    @property
    def transform(self) -> Transform:
        """
        Gets the transform object associated with the SGP model.

        Returns:
            Transform: The transform object.
        """
        return self.sgpr.transform


class DifferentiableObjective(Method):
    """
    Implements informative sensor placement/path planning optimization by directly
    differentiating through the objective function.

    This method leverages TensorFlow's automatic differentiation capabilities to
    optimize the sensing locations (or path waypoints) by treating them as
    trainable variables and minimizing a given objective function (e.g., Mutual
    Information). This approach can be more efficient than black-box methods like
    Bayesian Optimization or CMA-ES, especially when the objective function is
    smooth. However, the method is also more prone to getting stuck in local minima.

    Attributes:
        transform (Optional[Transform]): Transform object to apply to the solution.
        X_sol (tf.Variable): The solution (e.g., sensor locations) being optimized.
        objective (Objective): The objective function to be optimized.
    """
    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 objective: Union[str, Objective] = 'SLogMI',
                 X_init: Optional[np.ndarray] = None,
                 X_time: Optional[np.ndarray] = None,
                 orientation: bool = False,
                 **kwargs: Any):
        """
        Initializes the DifferentiableObjective optimizer.

        Args:
            num_sensing (int): Number of sensing locations to optimize.
            X_objective (np.ndarray): (n, d); Data points used to define the objective function.
            kernel (gpflow.kernels.Kernel): GPflow kernel function.
            noise_variance (float): Data noise variance.
            transform (Optional[Transform]): Transform object to apply to inducing points. Defaults to None.
            num_robots (int): Number of robots/agents. Defaults to 1.
            X_candidates (Optional[np.ndarray]): (c, d); Discrete set of candidate locations for sensor placement.
                                                 Defaults to None.
            num_dim (Optional[int]): Dimensionality of the sensing locations. Defaults to dimensonality of X_objective.
            objective (Union[str, Objective]): The objective function to use. Can be a string (e.g., 'SLogMI', 'MI')
                                         or an instance of an objective class. Defaults to 'SLogMI'.
            X_init (Optional[np.ndarray]): (num_sensing * num_robots, d); Initial solution.
                                            If None, initial points are randomly selected from X_objective.
            X_time (Optional[np.ndarray]): (m, d); Temporal dimensions of the inducing points, used when
                                            modeling spatio-temporal IPP. Defaults to None.
            orientation (bool): If True, adds an additional dimension to model sensor FoV rotation angle
                                when selecting initial inducing points. Defaults to False.
            **kwargs: Additional keyword arguments passed to the objective function.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.transform = transform
        if X_candidates is None:
            self.X_candidates = X_objective  # Default candidates to objective points

        if X_init is None:
            X_init = get_inducing_pts(X_objective,
                                      num_sensing * self.num_robots,
                                      orientation=orientation)
        else:
            # override num_dim with initial inducing points dim, in case it differes from X_objective dim
            self.num_dim = X_init.shape[-1]
        self.X_sol = tf.Variable(X_init, dtype=X_init.dtype)

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective, kernel,
                                                      noise_variance, **kwargs)
        else:
            self.objective = objective

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance parameters of the objective function.

        Args:
            kernel (gpflow.kernels.Kernel): Updated GPflow kernel function.
            noise_variance (float): Updated data noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Retrieves the current kernel and noise variance hyperparameters from the objective.

        Returns:
            Tuple[gpflow.kernels.Kernel, float]: A tuple containing a deep copy of the kernel and the noise variance.
        """
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance
    
    def optimize(self,
                 max_steps: int = 500,
                 optimizer: str = 'scipy.L-BFGS-B',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimizes the sensor placement/path by differentiating through the objective function.

        Args:
            max_steps (int): Maximum number of optimization steps. Defaults to 500.
            optimizer (str): Optimizer "<backend>.<method>" to use for training (e.g., 'scipy.L-BFGS-B', 'tf.adam').
                             Defaults to 'scipy.L-BFGS-B'.
            verbose (bool): Verbosity, if True additional details will by reported. Defaults to False.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            np.ndarray: (num_robots, num_sensing, num_dim); Optimized sensing locations.

        Usage:
            ```python
            # Assuming X_train, candidates, kernel_opt, noise_variance_opt are defined
            diff_obj_method = DifferentiableObjective(
                num_sensing=10,
                X_objective=X_train,
                kernel=kernel_opt,
                noise_variance=noise_variance_opt,
                transform=IPPTransform(num_robots=1), # Example transform
                X_candidates=candidates
            )
            optimized_solution = diff_obj_method.optimize(max_steps=500, optimizer='scipy.L-BFGS-B')
            ```
        """
        _ = optimize_model(
            training_loss = self._objective,
            max_steps=max_steps,
            trainable_variables=[self.X_sol],
            optimizer=optimizer,
            verbose=verbose,
            **kwargs)

        sol: tf.Tensor = self.X_sol
        try:
            sol_expanded = self.transform.expand(sol,
                                                 expand_sensor_model=False)
        except TypeError:
            sol_expanded = sol
        if not isinstance(sol_expanded, np.ndarray):
            sol_np = sol_expanded.numpy()
        else:
            sol_np = sol_expanded

        # Map solution locations to candidates set locations if X_candidates is provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np
    
    def _objective(self) -> float:
        """
        Internal objective function to be minimized by the optimizer.

        This function applies any specified transformations to the current solution,
        calculates the objective value, and applies a penalty for constraint
        violations.

        Returns:
            float: The objective value (reward + constraint penalty).
        """
        constraint_penality: float = 0.0
        if self.transform is not None:
            X_expanded = self.transform.expand(self.X_sol)
            constraint_penality = self.transform.constraints(self.X_sol)
            reward = self.objective(X_expanded)  # maximize
        else:
            reward = self.objective(self.X_sol)  # maximize

        reward += constraint_penality  # minimize (large negative value when constraint is unsatisfied)
        return reward


METHODS: Dict[str, Type[Method]] = {
    'BayesianOpt': BayesianOpt,
    'CMA': CMA,
    'ContinuousSGP': ContinuousSGP,
    'GreedyObjective': GreedyObjective,
    'GreedySGP': GreedySGP,
    'DifferentiableObjective': DifferentiableObjective
}


def get_method(method: str) -> Type[Method]:
    """
    Retrieves an optimization method class by its string name.

    Args:
        method (str): The name of the optimization method (e.g., 'ContinuousSGP', 'CMA').

    Returns:
        Type[Method]: The class of the requested optimization method.

    Raises:
        KeyError: If the method name is not found.

    Usage:
        ```python
        # To get the ContinuousSGP class
        ContinuousSGPClass = get_method('ContinuousSGP')
        # You can then instantiate it:
        # CSGP_instance = ContinuousSGPClass(...)
        ```
    """
    if method not in METHODS:
        raise KeyError(f"Method '{method}' not found. Available methods: {', '.join(METHODS.keys())}")
    return METHODS[method]
