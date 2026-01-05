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
 
from copy import deepcopy
from typing import Optional, List, Tuple, Union, Dict, Any, Type

import numpy as np
import tensorflow as tf
import gpflow
import cma
import shapely
from shapely import geometry
from apricot import CustomSelection
from bayes_opt import BayesianOptimization
from numba import njit

from sgptools.utils.tsp import run_tsp
from sgptools.utils.metrics import get_distance
from sgptools.utils.misc import cont2disc, get_inducing_pts
from sgptools.objectives import get_objective, Objective
from sgptools.utils.gpflow import optimize_model
from sgptools.core.augmented_sgpr import AugmentedSGPR
from sgptools.core.transformations import Transform  # for type hinting


class Method:
    """
    Base class for informative sensing / path-planning optimization methods.

    All methods optimize a set of sensing locations or waypoints,
    typically under a task-specific objective defined over a Gaussian process
    model (e.g., mutual information, ELBO).

    Attributes:
        num_sensing: Number of sensing locations (or waypoints) to optimize
            per robot.
        num_dim: Dimensionality of each sensing location (e.g., 2 for (x, y),
            3 for (x, y, θ)).
        num_robots: Number of robots / agents whose paths or sensing locations
            are being optimized.
        X_candidates: Optional discrete set of candidate sensing locations
            with shape `(c, num_dim)`. If provided, continuous solutions may be
            snapped to the closest candidates via `cont2disc`.
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
        Base initializer for an optimization method.

        Parameters
        ----------
        num_sensing:
            Number of sensing locations (inducing points / waypoints) to be
            optimized per robot.
        X_objective:
            Array of shape `(n, d)` containing the inputs used to define the
            objective (e.g., training inputs or a spatial grid). The last
            dimension `d` is used as the default `num_dim` if `num_dim` is not
            provided explicitly.
        kernel:
            GPflow kernel used by the objective. Stored only indirectly through
            subclasses (via their objective models).
        noise_variance:
            Observation noise variance used in the objective.
        transform:
            Optional `Transform` object that maps inducing points to an
            expanded representation (e.g., IPP path expansion, sensor FoV).
            Also used for constraint evaluation. Not stored here, but passed
            through to subclasses as needed.
        num_robots:
            Number of robots / agents. The total number of optimized points
            is `num_sensing * num_robots`. Defaults to 1.
        X_candidates:
            Optional array of shape `(c, d)` representing a discrete set of
            feasible sensing locations. When provided, many methods map their
            continuous solution to this candidate set using `cont2disc`.
        num_dim:
            Dimensionality of each sensing location. If `None`, defaults to
            `X_objective.shape[-1]`.
        **kwargs:
            Additional keyword arguments are accepted for forward compatibility,
            but ignored by the base class.
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
        Run the optimization procedure and return the optimized sensing
        locations / waypoints.

        Returns
        -------
        np.ndarray
            Array with shape `(num_robots, num_sensing, num_dim)` containing
            the optimized sensing locations.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Update the kernel and noise-variance hyperparameters used by the
        underlying objective or SGP model.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise-variance hyperparameters used by
        the underlying objective or SGP model.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A tuple `(kernel, noise_variance)` containing copies of the current
            hyperparameters.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError


class BayesianOpt(Method):
    """
    Informative sensor placement / path optimization using Bayesian
    Optimization over a continuous search space.

    A Bayesian optimization loop is run over a flattened vector containing all
    sensing locations for all robots. At each iteration, the candidate
    locations are reshaped, optionally transformed (for IPP / FoV modeling),
    evaluated under a GP-based objective (e.g. mutual information), and
    penalized by any constraints provided by the `Transform`.

    References
    ----------
    - Vivaldini et al., 2019. *UAV route planning for active disease
      classification.*
    - Francis et al., 2019. *Occupancy map building through Bayesian
      exploration.*

    Attributes
    ----------
    objective:
        Objective object encapsulating the GP-based information measure to
        maximize.
    transform:
        Optional transform applied to candidate sensing locations before
        evaluating the objective.
    pbounds:
        Dictionary mapping parameter names `'x0', 'x1', ...` to their search
        bounds `(lower, upper)`, as required by `bayes_opt.BayesianOptimization`.
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
        Initialize a Bayesian optimization-based method.

        Parameters
        ----------
        num_sensing:
            Number of sensing locations per robot to optimize.
        X_objective:
            Array of shape `(n, d)` used to define the underlying objective.
            The bounds of this set are used to define the BO search space.
        kernel:
            GPflow kernel used inside the objective.
        noise_variance:
            Observation noise variance used inside the objective.
        transform:
            Optional transform applied to the candidate solution before
            evaluating the objective (and constraints). For example, an
            `IPPTransform`.
        num_robots:
            Number of robots / agents. Defaults to 1.
        X_candidates:
            Optional discrete candidate set of locations with shape `(c, d)`.
            If provided, the final continuous solution is snapped to the
            nearest candidate locations.
        num_dim:
            Dimensionality of the sensing locations. If `None`, defaults to
            `X_objective.shape[-1]`.
        objective:
            Objective specification. Either a string key understood by
            `get_objective` (e.g. `'SLogMI'`, `'MI'`) or an already-instantiated
            `Objective` object.
        **kwargs:
            Additional keyword arguments forwarded to the objective constructor
            when `objective` is a string.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.transform = transform

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective, kernel,
                                                      noise_variance, **kwargs)
        else:
            self.objective = objective

        # Use the coordinate-wise min/max of X_objective as BO bounds
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
        Update the kernel and noise variance used by the underlying objective.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise variance used by the objective.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A deep copy of the kernel and the current noise variance.
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
        Run Bayesian optimization to obtain informative sensing locations.

        Parameters
        ----------
        max_steps:
            Number of Bayesian optimization iterations after the initial random
            exploration. Defaults to 50.
        init_points:
            Number of purely random evaluations before BO starts. Defaults to 10.
        verbose:
            If `True`, print progress messages from `BayesianOptimization`.
        seed:
            Optional random seed to make BO reproducible.
        **kwargs:
            Extra keyword arguments forwarded to `BayesianOptimization`
            (currently unused in this wrapper, but accepted for flexibility).

        Returns
        -------
        np.ndarray
            Array of shape `(num_robots, num_sensing, num_dim)` containing the
            optimized sensing locations in the original coordinate space.
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

        # Reshape BO solution to (total_points, num_dim)
        sol_np = np.array(sol).reshape(-1, self.num_dim)
        if self.transform is not None:
            # Use the transform for constraints and internal path logic,
            # but disable sensor model expansion (e.g., FoV) when returning
            # waypoint locations.
            sol_np = self.transform.expand(sol_np,
                                           expand_sensor_model=False)

            if not isinstance(sol_np, np.ndarray):
                sol_np = sol_np.numpy()

        # Optionally snap to candidate set
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self, **kwargs: float) -> float:
        """
        Objective function passed to `BayesianOptimization`.

        Parameters are expected as a flattened dictionary `{ 'x0': ..., 'x1': ... }`,
        which is reshaped into `(num_sensing * num_robots, num_dim)` to form
        continuous sensing locations. The method:

        1. Reshapes the flattened vector into locations.
        2. Optionally applies the `transform` (including constraints).
        3. Evaluates the GP-based objective.
        4. Adds the constraint penalty returned by the transform.
        5. Returns the scalar objective as a Python float.

        The underlying objective is *maximized*. Transform constraints are
        expected to return non-positive values, so larger violations produce
        more negative penalties.

        Parameters
        ----------
        **kwargs:
            Flattened coordinates keyed by `'x0', 'x1', ...`.

        Returns
        -------
        float
            Objective value to be maximized by `BayesianOptimization`.
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

        # Transform constraints are typically <= 0; adding them penalizes violations.
        reward += constraint_penality
        return reward.numpy()


class CMA(Method):
    """
    Informative sensor placement / path optimization using CMA-ES
    (Covariance Matrix Adaptation Evolution Strategy).

    CMA-ES is a derivative-free, population-based genetic optimizer well-suited for
    non-convex, non-smooth objectives. Here, it searches over the flattened
    vector of sensing locations / waypoints.

    Reference
    ---------
    - Hitz et al., 2017. *Adaptive Continuous-Space Informative Path Planning
      for Online Environmental Monitoring.*

    Attributes
    ----------
    objective:
        Objective object to evaluate information gain.
    transform:
        Optional transform applied to candidate solutions (e.g., for IPP / FoV).
    X_init:
        Flattened initial guess of the sensing locations.
    pbounds:
        Convex hull of the `X_objective` points, used as an implicit geometric
        bound (not enforced directly by CMA-ES).
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
        Initialize a CMA-ES-based optimization method.

        Parameters
        ----------
        num_sensing:
            Number of sensing locations per robot.
        X_objective:
            Array of shape `(n, d)` used to define the GP objective and
            to build the convex hull bounds.
        kernel:
            GPflow kernel used inside the objective.
        noise_variance:
            Observation noise variance used inside the objective.
        transform:
            Optional transform applied to candidate solutions before objective
            evaluation and constraints.
        num_robots:
            Number of robots / agents. Defaults to 1.
        X_candidates:
            Optional discrete candidate set of locations with shape `(c, d)`.
            If provided, continuous solutions are snapped to candidates.
        num_dim:
            Dimensionality of sensing locations. If `None`, defaults to
            `X_objective.shape[-1]`, or to `X_init.shape[-1]` if `X_init`
            is provided.
        objective:
            Objective specification, either a string key for
            `get_objective` or a pre-instantiated `Objective`.
        X_init:
            Initial guess for the sensing locations, with shape
            `(num_sensing * num_robots, num_dim)`. If `None`, an initial set
            is selected via `get_inducing_pts`.
        **kwargs:
            Extra keyword arguments forwarded to the objective constructor
            when `objective` is a string.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        self.transform = transform
        if X_init is None:
            X_init = get_inducing_pts(X_objective,
                                      num_sensing * self.num_robots)
        else:
            # Override num_dim with the dimensionality of the initial solution
            self.num_dim = X_init.shape[-1]

        self.X_init: np.ndarray = X_init.reshape(-1)  # Flattened initial guess

        if isinstance(objective, str):
            self.objective = get_objective(objective)(X_objective, kernel,
                                                      noise_variance, **kwargs)
        else:
            self.objective = objective

        # Use the convex hull of the objective inputs as a geometric bound
        self.pbounds = geometry.MultiPoint([[p[0], p[1]]
                                            for p in X_objective]).convex_hull

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Update the kernel and noise variance used by the objective.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise variance used by the objective.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A deep copy of the kernel and the current noise variance.
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
        Run CMA-ES to obtain informative sensing locations.

        Parameters
        ----------
        max_steps:
            Maximum number of function evaluations (CMA-ES iterations). Defaults
            to 500.
        tol:
            Function-value tolerance for termination (stopping criterion
            passed to CMA). Defaults to `1e-6`.
        verbose:
            If `True`, CMA-ES prints progress messages.
        seed:
            Optional random seed for reproducibility.
        restarts:
            Number of CMA-ES restarts allowed. Defaults to 5.
        **kwargs:
            Additional keyword arguments forwarded to `cma.fmin2` (currently
            unused in this wrapper but accepted for flexibility).

        Returns
        -------
        np.ndarray
            Array of shape `(num_robots, num_sensing, num_dim)` containing the
            optimized sensing locations.
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
            sol_np = self.transform.expand(sol_np,
                                           expand_sensor_model=False)
            if not isinstance(sol_np, np.ndarray):
                sol_np = sol_np.numpy()

        # Snap to candidate set if provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self, X: np.ndarray) -> float:
        """
        Objective function passed to CMA-ES (to be *minimized*).

        The internal objective (e.g., mutual information) is naturally
        maximized. CMA-ES, however, minimizes. To reconcile this, the method
        returns `-reward` (plus any constraint penalty), where `reward`
        is the value returned by the underlying objective.

        Steps:
        1. Reshape the flattened input `X` to `(num_points, num_dim)`.
        2. Optionally apply the transform (including constraints).
        3. Evaluate the GP-based objective (reward).
        4. Add the constraint penalty.
        5. Return the negative of this value as a Python float.

        Parameters
        ----------
        X:
            Flattened array of length `num_sensing * num_robots * num_dim`
            containing the current candidate solution.

        Returns
        -------
        float
            Negative objective value to be minimized by CMA-ES. Large positive
            returns correspond to poor solutions; large negative returns
            correspond to good solutions.
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
            reward = -1e6  # CMA does not handle inf values

        # Transform constraints are typically <= 0; adding them penalizes violations.
        reward += constraint_penality
        return -reward.numpy()  # CMA-ES minimizes

    def update_transform(self, transform: Transform) -> None:
        """
        Replace the transform used by the CMA-ES method.

        Parameters
        ----------
        transform:
            New `Transform` instance to use for expansion and constraints.
        """
        self.transform = transform

    def get_transform(self) -> Transform:
        """
        Return a deep copy of the transform used by this method.

        Returns
        -------
        Transform
            Deep copy of the current transform.
        """
        return deepcopy(self.transform)


class ContinuousSGP(Method):
    """
    Informative sensing / path optimization via direct optimization of
    Sparse Gaussian Process (SGP) inducing points.

    This method treats the inducing locations of an `AugmentedSGPR` model as
    the decision variables and optimizes them with respect to the SGP's ELBO
    (or another internal objective implemented by `AugmentedSGPR`).

    References
    ----------
    - Jakkala & Akella, 2024. *Multi-Robot Informative Path Planning from
      Regression with Sparse Gaussian Processes.*
    - Jakkala & Akella, 2025. *Fully differentiable sensor placement and 
      informative path planning.*

    Attributes
    ----------
    sgpr:
        `AugmentedSGPR` model whose inducing points are being optimized.
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
        Initialize a continuous SGP-based optimization method.

        Parameters
        ----------
        num_sensing:
            Number of inducing points (sensing locations) per robot.
        X_objective:
            Array of shape `(n, d)` used to define the spatial domain and
            training inputs for the SGP.
        kernel:
            GPflow kernel for the SGP model.
        noise_variance:
            Observation noise variance for the SGP model.
        transform:
            Optional `Transform` to apply to inducing points for IPP or FoV
            modeling. Passed directly into `AugmentedSGPR`.
        num_robots:
            Number of robots / agents. The total number of inducing points is
            `num_sensing * num_robots`. Defaults to 1.
        X_candidates:
            Optional candidate set `(c, d)` used to snap the final continuous
            inducing locations to discrete locations.
        num_dim:
            Dimensionality of sensing locations. If `None`, defaults to
            `X_objective.shape[-1]`, or to `X_init.shape[-1]` if an initial
            solution is provided.
        X_init:
            Initial inducing points with shape `(num_sensing * num_robots, d)`.
            If `None`, points are chosen via `get_inducing_pts`. If given,
            its dimensionality overrides `num_dim`.
        X_time:
            Optional temporal coordinates (e.g. for spatio-temporal models),
            passed as `inducing_variable_time` to `AugmentedSGPR`.
        orientation:
            If `True` and `X_init` is not provided, `get_inducing_pts` is
            allowed to include an orientation dimension for the inducing points.
        **kwargs:
            Additional keyword arguments forwarded to `AugmentedSGPR` if needed
            (currently unused here but accepted for flexibility).
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)
        if X_init is None:
            X_init = get_inducing_pts(X_objective,
                                      num_sensing * self.num_robots,
                                      orientation=orientation)
        else:
            # Override num_dim with the dimensionality of the initial inducing points
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
        Update the kernel and noise variance used by the SGP model.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.
        """
        self.sgpr.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise variance of the SGP model.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A deep copy of the kernel and the current likelihood variance.
        """
        return deepcopy(self.sgpr.kernel), \
               self.sgpr.likelihood.variance.numpy()

    def optimize(self,
                 max_steps: int = 500,
                 optimizer: str = 'scipy.L-BFGS-B',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimize the inducing points of the SGP model.

        The ELBO (or equivalent objective defined within `AugmentedSGPR`) is
        optimized w.r.t. the inducing locations only; kernel hyperparameters
        are kept fixed.

        Parameters
        ----------
        max_steps:
            Maximum number of optimization steps. Defaults to 500.
        optimizer:
            Optimizer specification in the form `"backend.method"` (e.g.
            `'scipy.L-BFGS-B'`, `'tf.adam'`), as expected by `optimize_model`.
        verbose:
            If `True`, print progress information during optimization.
        **kwargs:
            Extra keyword arguments forwarded to `optimize_model`.

        Returns
        -------
        np.ndarray
            Array of shape `(num_robots, num_sensing, num_dim)` containing the
            optimized inducing locations.
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
        sol_expanded = self.transform.expand(sol,
                                             expand_sensor_model=False)
        if not isinstance(sol_expanded, np.ndarray):
            sol_np = sol_expanded.numpy()
        else:
            sol_np = sol_expanded

        # Snap to candidate set if provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    @property
    def transform(self) -> Transform:
        """
        Transform associated with the underlying SGP model.

        Returns
        -------
        Transform
            The `Transform` instance used by `AugmentedSGPR`.
        """
        return self.sgpr.transform


class GreedyObjective(Method):
    """
    Informative sensor placement / path optimization using a greedy selection
    based on a generic objective function.

    The method iteratively adds sensing locations from a discrete candidate
    set to maximize a user-specified objective (e.g., mutual information),
    using `apricot.CustomSelection` as the selection engine. Only single-robot
    scenarios are supported.

    References
    ----------
    - Krause et al., 2008. *Near-Optimal Sensor Placements in Gaussian
      Processes: Theory, Efficient Algorithms and Empirical Studies.*
    - Ma et al., 2018. *Data-driven learning and planning for environmental
      sampling.*

    Attributes
    ----------
    objective:
        Objective object to maximize over the chosen locations.
    transform:
        Optional transform applied to selected locations.
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
        Initialize a greedy objective-based method.

        Parameters
        ----------
        num_sensing:
            Number of sensing locations to select.
        X_objective:
            Array of shape `(n, d)` used to define the objective (e.g. GP
            training inputs).
        kernel:
            GPflow kernel used inside the objective.
        noise_variance:
            Observation noise variance used inside the objective.
        transform:
            Optional transform applied to selected locations before evaluating
            the objective and constraints.
        num_robots:
            Number of robots / agents. `GreedyObjective` currently supports
            only `num_robots = 1` and will assert otherwise.
        X_candidates:
            Discrete candidate locations with shape `(c, d)`. If `None`,
            defaults to `X_objective`.
        num_dim:
            Dimensionality of the sensing locations. If `None`, defaults to
            `X_objective.shape[-1]`.
        objective:
            Objective specification (string key or `Objective` instance) used
            by `get_objective` when a string is given.
        **kwargs:
            Additional keyword arguments forwarded to the objective constructor
            when `objective` is a string.
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
        Update the kernel and noise variance used by the objective.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise variance used by the objective.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A deep copy of the kernel and the current noise variance.
        """
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance

    def optimize(self,
                 optimizer: str = 'naive',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Run greedy selection over the candidate set.

        Parameters
        ----------
        optimizer:
            Greedy strategy identifier passed to `apricot.CustomSelection`
            (e.g., `'naive'`, `'lazy'`).
        verbose:
            If `True`, print progress information from apricot.
        **kwargs:
            Additional keyword arguments forwarded to `CustomSelection`
            (currently unused in this wrapper but accepted for flexibility).

        Returns
        -------
        np.ndarray
            Array of shape `(num_robots, num_sensing, num_dim)` containing the
            selected sensing locations.
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
            sol_locations = self.transform.expand(
                sol_locations, expand_sensor_model=False)
            if not isinstance(sol_locations, np.ndarray):
                sol_locations = sol_locations.numpy()
        sol_locations = sol_locations.reshape(self.num_robots, -1,
                                              self.num_dim)
        return sol_locations

    def _objective(self, X_indices: np.ndarray) -> float:
        """
        Objective callback used by `apricot.CustomSelection`.

        The input is an array of candidate indices. The method:
        1. Maps indices to candidate locations.
        2. Optionally applies the transform (and constraints).
        3. Evaluates the underlying objective.
        4. Adds the transform constraint penalty.
        5. Returns the resulting scalar as a Python float.

        Parameters
        ----------
        X_indices:
            Array of shape `(n, 1)` containing indices into `self.X_objective`
            / `self.X_candidates`.

        Returns
        -------
        float
            Objective value to be maximized by apricot's greedy selection
            routine.
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

        reward += constraint_penality
        return reward.numpy()


class GreedySGP(Method):
    """
    Greedy sensing / placement using a Sparse GP (SGP) ELBO objective.

    At each greedy step, candidate inducing points are selected and used to
    update the inducing variables of an `AugmentedSGPR` model, and the ELBO
    is evaluated. Only single-robot settings are currently supported.

    Reference
    ---------
    - Jakkala & Akella, 2025. *Fully differentiable sensor placement and 
      informative path planning.*

    Attributes
    ----------
    sgpr:
        `AugmentedSGPR` model whose ELBO is used as greedy objective.
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
        Initialize a greedy SGP-based method.

        Parameters
        ----------
        num_sensing:
            Number of inducing points to select.
        X_objective:
            Array of shape `(n, d)` used as training inputs for the SGP model.
        kernel:
            GPflow kernel for the SGP model.
        noise_variance:
            Observation noise variance for the SGP model.
        transform:
            Optional `Transform` applied to inducing points inside the SGP
            model (e.g., IPP transforms).
        num_robots:
            Number of robots / agents. `GreedySGP` currently supports only
            `num_robots = 1` and will assert otherwise.
        X_candidates:
            Discrete candidate set `(c, d)`. If `None`, defaults to
            `X_objective`.
        num_dim:
            Dimensionality of sensing locations. If `None`, defaults to
            `X_objective.shape[-1]`.
        **kwargs:
            Additional keyword arguments accepted for forward compatibility
            (unused here).
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
        Update the kernel and noise variance used by the SGP model.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.
        """
        self.sgpr.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise variance of the SGP model.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A deep copy of the kernel and the current likelihood variance.
        """
        return deepcopy(self.sgpr.kernel), \
               self.sgpr.likelihood.variance.numpy()

    def optimize(self,
                 optimizer: str = 'naive',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Run greedy selection using the SGP's ELBO as objective.

        Parameters
        ----------
        optimizer:
            Greedy strategy identifier passed to `apricot.CustomSelection`
            (e.g., `'naive'`, `'lazy'`).
        verbose:
            If `True`, print progress information from apricot.
        **kwargs:
            Additional keyword arguments forwarded to `CustomSelection`
            (currently unused here but accepted for flexibility).

        Returns
        -------
        np.ndarray
            Array of shape `(num_robots, num_sensing, num_dim)` containing the
            selected sensing locations.
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
        sol_expanded = self.transform.expand(sol_locations,
                                             expand_sensor_model=False)
        if not isinstance(sol_expanded, np.ndarray):
            sol_np = sol_expanded.numpy()
        else:
            sol_np = sol_expanded

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self, X_indices: np.ndarray) -> float:
        """
        Objective callback used by `apricot.CustomSelection` for greedy SGP.

        Given a (possibly partial) set of indices, this method:
        1. Maps indices to candidate locations.
        2. Pads the selection to `num_sensing` points (so the SGP remains well-defined).
        3. Updates the SGP's inducing variables.
        4. Returns the SGP ELBO for the resulting inducing set.

        Parameters
        ----------
        X_indices:
            Array of shape `(n, 1)` containing indices into `self.X_objective`
            / `self.X_candidates`.

        Returns
        -------
        float
            ELBO value of the SGP model for this inducing set, as a Python
            float. Larger values correspond to better selections.
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
        Transform associated with the underlying SGP model.

        Returns
        -------
        Transform
            The `Transform` instance used by `AugmentedSGPR`.
        """
        return self.sgpr.transform


class DifferentiableObjective(Method):
    """
    Informative sensor placement / path planning by directly differentiating
    through the objective function.

    The sensing locations (or waypoints) are represented as TensorFlow
    variables, and a first-order optimizer (e.g., L-BFGS, Adam) is used to
    minimize a scalar loss built from the objective and constraints. This can
    be more sample-efficient than black-box methods, but is more sensitive to
    local minima.

    Attributes
    ----------
    transform:
        Optional transform applied to the current solution.
    X_sol:
        TensorFlow variable representing the current solution locations.
    objective:
        Objective object that maps (transformed) sensing locations to a scalar
        value.
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
        Initialize a differentiable-objective method.

        Parameters
        ----------
        num_sensing:
            Number of sensing locations per robot.
        X_objective:
            Array of shape `(n, d)` used to define the objective (e.g., GP
            training inputs).
        kernel:
            GPflow kernel used inside the objective.
        noise_variance:
            Observation noise variance used inside the objective.
        transform:
            Optional transform applied to the solution before evaluating the
            objective and constraints.
        num_robots:
            Number of robots / agents. The total number of optimized points is
            `num_sensing * num_robots`.
        X_candidates:
            Optional candidate set `(c, d)` to which the final continuous
            solution can be snapped.
        num_dim:
            Dimensionality of sensing locations. If `None`, defaults to
            `X_objective.shape[-1]`, or to `X_init.shape[-1]` if given.
        objective:
            Objective specification (string key or `Objective` instance) used to
            construct the reward function.
        X_init:
            Initial sensing locations with shape `(num_sensing * num_robots, d)`.
            If `None`, points are selected via `get_inducing_pts`. If given,
            its dimensionality overrides `num_dim`.
        X_time:
            (Reserved for future use with spatio-temporal models; not used
            directly here.)
        orientation:
            If `True` and `X_init` is not provided, `get_inducing_pts` may add
            an orientation dimension to the initial points.
        **kwargs:
            Additional keyword arguments forwarded to the objective constructor
            when `objective` is a string.
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
            # Override num_dim with the dimensionality of the initial solution
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
        Update the kernel and noise variance used by the objective.

        Parameters
        ----------
        kernel:
            New GPflow kernel instance.
        noise_variance:
            New observation noise variance.
        """
        self.objective.update(kernel, noise_variance)

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return the current kernel and noise variance used by the objective.

        Returns
        -------
        (gpflow.kernels.Kernel, float)
            A deep copy of the kernel and the current noise variance.
        """
        return deepcopy(self.objective.kernel), \
               self.objective.noise_variance

    def optimize(self,
                 max_steps: int = 500,
                 optimizer: str = 'scipy.L-BFGS-B',
                 verbose: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Optimize sensing locations by differentiating through the objective.

        `self.X_sol` is treated as a trainable variable and optimized using the
        specified optimizer and the internal `_objective` as the scalar loss.

        Parameters
        ----------
        max_steps:
            Maximum number of optimization steps. Defaults to 500.
        optimizer:
            Optimizer specification `"backend.method"` (e.g., `'scipy.L-BFGS-B'`,
            `'tf.adam'`) passed to `optimize_model`.
        verbose:
            If `True`, print progress information during optimization.
        **kwargs:
            Extra keyword arguments forwarded to `optimize_model`.

        Returns
        -------
        np.ndarray
            Array of shape `(num_robots, num_sensing, num_dim)` containing the
            optimized sensing locations.
        """
        _ = optimize_model(
            training_loss=self._objective,
            max_steps=max_steps,
            trainable_variables=[self.X_sol],
            optimizer=optimizer,
            verbose=verbose,
            **kwargs)

        sol: tf.Tensor = self.X_sol
        if self.transform is not None:
            sol = self.transform.expand(sol,
                                        expand_sensor_model=False)
        if not isinstance(sol, np.ndarray):
            sol_np = sol.numpy()
        else:
            sol_np = sol

        # Snap to candidate set if provided
        if self.X_candidates is not None:
            sol_np = cont2disc(sol_np, self.X_candidates)

        sol_np = sol_np.reshape(self.num_robots, -1, self.num_dim)
        return sol_np

    def _objective(self) -> float:
        """
        Scalar loss function used by `optimize_model`.

        The objective is built as:

        .. code-block:: text

            loss = objective(X_expanded) + constraint_penalty

        where both terms are produced by the `Transform`. Depending on the
        sign conventions of `objective` and `constraints`, this loss can be
        interpreted as either a negative reward or a penalized reward. The
        optimizer *minimizes* this loss.

        Returns
        -------
        tf.Tensor
            Scalar TensorFlow value representing the loss to minimize.
        """
        constraint_penality: float = 0.0
        if self.transform is not None:
            X_expanded = self.transform.expand(self.X_sol)
            constraint_penality = self.transform.constraints(self.X_sol)
            reward = self.objective(X_expanded)  # maximize (before sign handling)
        else:
            reward = self.objective(self.X_sol)  # maximize (before sign handling)

        # Transform constraints are typically <= 0; adding them penalizes violations.
        reward += constraint_penality
        return reward

# -----------------------------------------------------------------------------

class HexCover(Method):
    """
    Hexagonal lattice coverage based on GP kernel hyperparameters.

    This method constructs a deterministic hexagonal tiling over a rectangular
    2D environment such that the GP posterior variance at every point in the
    environment is bounded by a user-specified threshold (under the same
    sufficient condition used in the minimal HexCover implementation).
        
    Refer to the following paper for more details:
        - Approximation Algorithms for Robot Tours in Random Fields with 
        Guaranteed Estimation Accuracy [Dutta et al., 2023]

    Implementation based on Dr. Shamak Dutta's original code.

    Notes
    -----
    - Only supports 2D spatial domains (first two coordinates).
    - Multi-robot settings are not supported (`num_robots` must be 1).
    - The total number of points is determined by the tiling; it may be
      different from `num_sensing`. As with `GreedyCover`, returning fewer
      than `num_sensing` points is allowed.
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
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 pbounds: Optional[np.ndarray] = None,
                 **kwargs: Any):
        """
        Initialize a HexCover method.

        Parameters
        ----------
        num_sensing : int
            Maximum number of sensing locations (not strictly enforced; the
            tiling determines the actual number of points).
        X_objective : ndarray, shape (n, d)
            Environment points. Used only to infer the bounding rectangle
            (min/max in the first two dimensions) when `height`/`width` are
            not provided.
        kernel : gpflow.kernels.Kernel
            GP kernel (assumed to have a `variance` and `lengthscales`
            attribute, e.g., SquaredExponential).
        noise_variance : float
            Observation noise variance.
        transform : Transform or None
            Reserved for compatibility with other methods.
        num_robots : int
            Must be 1. Multi-robot tilings are not supported.
        X_candidates : ndarray or None
            Ignored. Present for API compatibility with other methods.
        num_dim : int or None
            Dimensionality of points. Defaults to `X_objective.shape[-1]`.
        height : float or None
            Environment height in the y-direction. If None, inferred from
            `X_objective` as `y_max - y_min`.
        width : float or None
            Environment width in the x-direction. If None, inferred from
            `X_objective` as `x_max - x_min`.
        pbounds:
            Coordinates of the environment boundry polygon, used to ensure all 
            sensing locations are inside the environment.
        kwargs : dict
            Ignored. Accepted for forward compatibility.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)

        assert num_robots == 1, "HexCover only supports num_robots = 1."

        self.kernel = kernel
        self.noise_variance = float(noise_variance)

        # Store environment points for dtype and potential debugging
        self.X_objective = np.asarray(X_objective)

        if self.X_objective.ndim != 2 or self.X_objective.shape[1] < 2:
            raise ValueError(
                "HexCover requires X_objective with at least 2 spatial dimensions."
            )

        # Bounding box of the environment in (x, y) from X_objective
        mins = self.X_objective[:, :2].min(axis=0)
        maxs = self.X_objective[:, :2].max(axis=0)
        default_extent = maxs - mins

        self.origin = mins  # shift from [0, W] x [0, H] to actual coords
        self.width = float(width) if width is not None else float(default_extent[0])
        self.height = float(height) if height is not None else float(default_extent[1])

        if pbounds is not None:
            self.pbounds = geometry.Polygon(pbounds)
        else:
            self.pbounds = None

        # Extract scalar lengthscale and prior variance
        self._extract_kernel_scalars()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_kernel_scalars(self) -> float:
        """
        Extract scalar kernel hyperparameters and store them on the instance.

        This method computes and stores:
        - `self.lengthscale`: a scalar lengthscale (minimum across dimensions
          or locations if needed).
        - `self.prior_variance`: a scalar prior variance.

        The implementation handles both stationary and certain non-stationary
        kernels (with a `get_lengthscales` method).
        """
        # Non-stationary kernel
        if hasattr(self.kernel, 'get_lengthscales'):
            lengthscale = self.kernel.get_lengthscales(self.X_objective)
            lengthscale = np.min(lengthscale)
            prior_variance = self.kernel(self.X_objective,
                                         self.X_objective).numpy().max()
        elif hasattr(self.kernel, 'lengthscales'):  # Stationary kernel
            lengthscale = float(self.kernel.lengthscales.numpy())
            prior_variance = float(self.kernel.variance.numpy())
        else:  # Neural Spectral kernel
            lengthscale = 0.5
            prior_variance = self.kernel(self.X_objective,
                                         self.X_objective).numpy().max()
        self.lengthscale = lengthscale
        self.prior_variance = prior_variance

    def _compute_rmin(self, post_var_threshold: Optional[float] = None) -> float:
        """
        Compute the sufficient radius $r_{\\min}$ for the hexagonal tiling.

        The radius is computed using the same sufficient condition as in the
        minimal HexCover implementation:

        $r_{\\min} = L \\sqrt{-\\log\\left(
            \\frac{(\\sigma_0^2 - \\Delta)(\\sigma_0^2 + \\sigma^2)}
                 {\\sigma_0^4}
        \\right)}$,

        where
        - $L$ is the kernel lengthscale,
        - $\\sigma_0^2$ is the prior variance,
        - $\\sigma^2$ is the noise variance, and
        - $\\Delta$ is the allowed posterior variance threshold.

        Parameters
        ----------
        post_var_threshold : float or None
            Posterior variance threshold :math:`\\Delta`. If None, uses the
            current value stored in `self.post_var_threshold`.

        Returns
        -------
        float
            The sufficient radius :math:`r_{\\min}` for the hexagonal tiling.

        Raises
        ------
        ValueError
            If the computed term inside the logarithm is not in (0, 1),
            which indicates incompatible hyperparameters and/or threshold.
        """
        if post_var_threshold is None:
            post_var_threshold = self.post_var_threshold

        sigma0_sq = self.prior_variance
        sigma_sq = float(self.noise_variance)
        delta = float(post_var_threshold)
        term = (sigma0_sq - delta) * (sigma0_sq + sigma_sq) / (sigma0_sq ** 2)

        if term <= 0.0 or term >= 1.0:
            raise ValueError(
                f"Invalid term inside log when computing r_min: {term}. "
                "Check kernel hyperparameters and post_var_threshold."
            )

        return self.lengthscale * np.sqrt(-np.log(term))

    @staticmethod
    def _hexagonal_tiling(height: float,
                          width: float,
                          radius: float,
                          fill_edge: bool = True) -> np.ndarray:
        """
        Generate a hexagonal tiling over a rectangular region.

        Parameters
        ----------
        height : float
            Height of the environment in the y-direction.
        width : float
            Width of the environment in the x-direction.
        radius : float
            Hexagon circumradius :math:`r_{\\min}`.
        fill_edge : bool, optional
            If True, add additional centers near the environment boundary
            to reduce uncovered gaps. Default is True.

        Returns
        -------
        ndarray of shape (k, 2)
            Array of 2D points representing hexagon centers in local
            `[0, width] × [0, height]` coordinates.
        """
        hs = 3.0 * radius
        vs = np.sqrt(3.0) * radius

        # first set of centers
        nc = int(np.floor(width / hs) + 1)
        nr = int(np.floor(height / vs) + 1)
        x = list(np.linspace(0.0, (nc - 1) * hs, nc))
        y = list(np.linspace(0.0, (nr - 1) * vs, nr))

        if fill_edge:
            if (nc - 1) * hs + radius < width:
                x.append(width)
            if (nr - 1) * vs + radius < height:
                y.append(height)

        X, Y = np.meshgrid(x, y)
        first_centers = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # second set of centers (offset grid)
        nc = int(np.floor((width / hs) + 0.5))
        nr = int(np.floor((height / vs) + 0.5))
        x = list(np.linspace(hs / 2.0, (nc - 1) * hs + hs / 2.0, nc))
        y = list(np.linspace(vs / 2.0, (nr - 1) * vs + vs / 2.0, nr))

        if fill_edge:
            if (nc - 1) * hs + hs / 2.0 + radius < width:
                x.append(width)
            if (nr - 1) * vs + vs / 2.0 + radius < height:
                y.append(height)

        X, Y = np.meshgrid(x, y)
        second_centers = np.stack([X.ravel(), Y.ravel()], axis=-1)

        return np.concatenate([first_centers, second_centers], axis=0)

    # ------------------------------------------------------------------
    # Method API
    # ------------------------------------------------------------------
    def update(self,
               kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Update kernel and noise variance hyperparameters.
        """
        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        self._extract_kernel_scalars()

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return current kernel and noise variance as (kernel, noise_variance).
        """
        return deepcopy(self.kernel), float(self.noise_variance)

    def optimize(self,
                 post_var_threshold: Optional[float] = None,
                 return_fovs: bool = False,
                 tsp: bool = True,
                 **kwargs: Any) -> np.ndarray:
        """
        Construct the hexagonal coverage pattern.

        Parameters
        ----------
        post_var_threshold : float or None, optional
            Target posterior variance threshold :math:`\\Delta`. If None,
            defaults to `0.2 * prior_variance` (following the minimal
            implementation where `delta = 0.2 * sigma0**2`).
        return_fovs : bool, optional
            If True, also returns a list of polygonal fields of view (FoVs)
            corresponding to regular hexagons centered at the sensing
            locations. Default is False.
        tsp : bool, optional
            If True, runs a TSP heuristic (`run_tsp`) to order the sensing
            locations. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to `run_tsp` when `tsp` is True.

        Returns
        -------
        X_sol : ndarray of shape (1, k, d)
            Selected sensing locations. `k` is determined by the tiling and
            may differ from `num_sensing`. The last dimension `d` matches
            `self.num_dim`; only the first two coordinates are used for the
            spatial layout, the remaining coordinates are zero.

        If `return_fovs` is True, the return value is:

        (X_sol, fovs) : (ndarray, list of shapely.geometry.Polygon)
            `X_sol` as above, together with a list of regular hexagonal FoVs
            centered at each selected sensing location.
        """
        # Posterior variance threshold Δ
        if post_var_threshold is None:
            self.post_var_threshold = 0.2 * self.prior_variance
        else:
            self.post_var_threshold = float(post_var_threshold)

        if self.post_var_threshold >= self.prior_variance:
            raise ValueError(
                f"post_var_threshold must be smaller than the kernel variance: {self.prior_variance:.2f}."
            )

        # Compute r_min for the current kernel / noise / threshold
        rmin = self._compute_rmin(post_var_threshold)

        # Tiling in local [0, width] x [0, height] coordinates
        centers_2d = self._hexagonal_tiling(self.height, self.width, rmin)

        # Shift to actual environment coordinates using the inferred origin
        X_sol = centers_2d + self.origin[None, :]

        # Remove sensing locations outside the boundaries
        if self.pbounds is not None:
            points = shapely.points(X_sol)
            inside_idx = shapely.contains(self.pbounds, points)
            X_sol = X_sol[inside_idx]

        if tsp:
            X_sol, _ = run_tsp(X_sol, **kwargs)
        X_sol = np.array(X_sol).reshape(self.num_robots, -1, self.num_dim)

        if return_fovs:
            return X_sol, self._get_fovs(X_sol, rmin)
        else:
            return X_sol

    def _get_fovs(self, X_sol, radius):
        """
        Construct polygonal fields of view (FoVs) for the sensing locations.

        For each selected sensing location, this method creates a regular
        hexagon centered at the sensing point with the given radius. The
        resulting polygons approximate the spatial footprint of each
        sensing location.

        Parameters
        ----------
        X_sol : ndarray of shape (1, k, d)
            Sensing locations returned by :meth:`optimize`. Only the first
            two coordinates of each point are used.
        radius : float
            Hexagon side length (or circumradius) used to construct each FoV.

        Returns
        -------
        fovs : list of shapely.geometry.Polygon
            List of regular hexagonal polygons, one per sensing location.
        """
        fovs = []
        for pt in X_sol[0]:
            fov = HexCover._create_regular_hexagon(pt[0], pt[1], radius)
            fovs.append(fov)
        return fovs
    
    @staticmethod
    def _create_regular_hexagon(center_x, center_y, side_length):
        """
        Create a regular hexagon polygon centered at a given point.

        Parameters
        ----------
        center_x : float
            x-coordinate of the hexagon center.
        center_y : float
            y-coordinate of the hexagon center.
        side_length : float
            Side length (and effective radius) of the hexagon.

        Returns
        -------
        shapely.geometry.Polygon
            Polygon representing the regular hexagon.
        """
        coords = []
        for i in range(6):
            # Start at 0 degrees for the first vertex and move counter-clockwise
            angle_deg = 60 * i
            angle_rad = np.radians(angle_deg)
            x = center_x + side_length * np.cos(angle_rad)
            y = center_y + side_length * np.sin(angle_rad)
            coords.append((x, y))
        return geometry.Polygon(coords)

#-----------------------------------------------------------------------------

@njit
def _compute_gains_numba(remaining_idxs, coverages, current_coverage):
    """
    Compute marginal gains for remaining candidates (Numba-accelerated).

    Parameters
    ----------
    remaining_idxs : 1D ndarray[int]
        Indices of still-available candidates.
    coverages : 2D ndarray[bool], shape (n_candidates, v)
        Binary coverage mask for each candidate.
    current_coverage : 1D ndarray[bool], shape (v,)
        Binary mask of currently covered environment points.

    Returns
    -------
    ndarray[int]
        Marginal gain (number of newly covered points) for each candidate.
    """
    m = remaining_idxs.shape[0]
    v = current_coverage.shape[0]
    gains = np.empty(m, dtype=np.int64)

    for k in range(m):
        idx = remaining_idxs[k]
        cov_i = coverages[idx]
        gain = 0
        for j in range(v):
            if cov_i[j] and (not current_coverage[j]):
                gain += 1
        gains[k] = gain

    return gains

class GreedyCover(HexCover):
    """
    Greedy sensing-location selection via GP posterior-variance “coverage”.

    The method selects points from a discrete candidate set to cover as many
    objective/environment points as possible under a *single-measurement*
    Gaussian Process (GP) variance reduction criterion.

    Coverage criterion
    ------------------
    Let:
      - x_i be a candidate sensing location
      - x_j be an objective/environment point
      - k(·,·) be the prior GP kernel covariance
      - σ_n^2 be the observation noise variance (self.noise_variance)

    The prior variances are:
        v_cand[i] = k(x_i, x_i)
        v_obj[j]  = k(x_j, x_j)

    After observing y at x_i (single observation), the GP posterior variance at
    x_j is:
        v_post(j | i) = v_obj[j] - k(x_i, x_j)^2 / (v_cand[i] + σ_n^2)

    Candidate i is said to *cover* objective point j if:
        v_post(j | i) <= post_var_threshold

    This is equivalent to the implemented inequality:
        k(x_i, x_j)^2 >= (v_obj[j] - post_var_threshold) * (v_cand[i] + σ_n^2)

    Algorithm
    ---------
    1) Build a boolean coverage matrix coverages[i, j].
    2) Greedily select the candidate with the largest number of newly covered
       objective points.
    3) Stop when the target coverage fraction is reached or the sensing budget
       is exhausted.
    4) Optionally order the selected points via a TSP solver.

    Notes
    -----
    - Current implementation assumes a single robot (num_robots == 1).
    - May return fewer than num_sensing points if the target is reached early.
    - Raises ValueError if the target coverage is not achievable from the
      candidate set.
    """

    def optimize(self,
                 post_var_threshold: float = 0.7,
                 target_fraction: int = 100,
                 return_fovs: bool = False,
                 slack_var: float = 0.15,
                 **kwargs) -> np.ndarray:
        """
        Run greedy GP-coverage selection.

        This method constructs a coverage mask using the GP posterior variance test:

            v_post(j | i) = v_obj[j] - k(x_i, x_j)^2 / (v_cand[i] + σ_n^2)
            covered(i, j) := v_post(j | i) <= post_var_threshold

        and then greedily selects candidates that maximize the number of *newly*
        covered objective points until either:
        - target_fraction percent of objective points are covered, or
        - num_sensing points have been selected.

        Parameters
        ----------
        post_var_threshold:
            Posterior variance upper bound at objective points (same units as the
            kernel variance). Lower values demand stronger information gain.
        target_fraction:
            Desired percent coverage in [0, 100]. (Using float allows e.g., 95.0.)
        return_fovs:
            If True, also return polygons summarizing each selected candidate’s
            covered region (convex hull of covered objective points, buffered).
        slack_var:
            Non-negative slack used to lower the post_var_threshold when generating 
            the candidate set via HexCover (i.e., ``post_var_threshold - slack_var``), 
            potentially generating extra candidates and improving the chance of 
            reaching the target coverage.
        **kwargs:
            Extra arguments forwarded to the TSP ordering routine (and currently
            also forwarded to HexCover.optimize — see code improvement notes).

        Returns
        -------
        X_sol:
            Array shaped (num_robots, k, d) with k <= num_sensing selected points.
        (X_sol, fovs):
            If return_fovs is True, also returns a list of shapely Polygons.
        """
        if not hasattr(self, "coverages"):
            # Increase slack variance until we can reach the target fraction,
            # or until the effective threshold would become non-positive.
            slack = float(slack_var)
            max_fraction = float("-inf")

            while (post_var_threshold - slack) > 0.0:
                max_fraction = self._compute_coverage_maps(
                    post_var_threshold,
                    target_fraction,
                    slack,  # use the current (possibly increased) slack
                    **kwargs,
                )

                if max_fraction >= target_fraction:
                    break

                slack += float(slack_var)
                print("Failed to achieve target coverage. Retrying with increased slack variance...")

            if max_fraction < target_fraction:
                raise ValueError(
                    f"Target coverage {target_fraction:.2f}% is not achievable; "
                    f"maximum possible {max_fraction:.2f}% with "
                    f"post_var_threshold {post_var_threshold:.2f} and "
                    f"slack_var {slack:.2f}."
                )

        # ---------------- Greedy loop ----------------
        n = len(self.X_candidates)
        selected_mask = np.zeros(n, dtype=bool)
        selected = []

        current_coverage = np.zeros(self.X_objective.shape[0], 
                                    dtype=bool)
        current_sum = 0

        while current_sum < self.target_sum and len(selected) < self.num_sensing:
            remaining = np.where(~selected_mask)[0]
            if remaining.size == 0:
                break

            gains = _compute_gains_numba(
                remaining.astype(np.int64),
                self.coverages,
                current_coverage
            )

            best_pos = int(np.argmax(gains))
            best_gain = int(gains[best_pos])
            best_idx = int(remaining[best_pos])

            if best_gain <= 0:
                break

            current_coverage |= self.coverages[best_idx]
            current_sum = int(current_coverage.sum())

            selected_mask[best_idx] = True
            selected.append(best_idx)

            if current_sum >= self.target_sum:
                break

        # ---------------- Prepare output ----------------
        if len(selected) == 0:
            return np.zeros((1, 0, self.num_dim), dtype=self.X_objective.dtype)

        X_sol = self.X_candidates[selected]
        X_sol, _ = run_tsp(X_sol, **kwargs)
        X_sol = np.array(X_sol).reshape(self.num_robots, -1, self.num_dim)

        if return_fovs:
            return X_sol, self._get_fovs(self.coverages[selected])
        else:
            return X_sol

    def _compute_coverage_maps(self, post_var_threshold, 
                               target_fraction, 
                               slack_var, 
                               **kwargs):
        """
        Build the candidate set and the boolean coverage matrix.

        Steps
        -----
        1) Generate candidate points using HexCover with a tighter threshold
        (var_threshold - slack_var).
        2) Compute:
            v_cand[i] = k(x_i, x_i)
            v_obj[j]  = k(x_j, x_j)
            K[i, j]   = k(x_i, x_j)
        3) Mark covered(i, j) true when:
            K[i, j]^2 >= (v_obj[j] - var_threshold) * (v_cand[i] + σ_n^2)
        4) Compute the integer number of objective points required to meet
        target_fraction, and verify achievability.

        Raises
        ------
        ValueError:
            If the candidate set cannot achieve the requested target_fraction.
        """
        # ---------------- Candidate & environment sets ----------------
        X_objective = self.X_objective

        # Get candidates using HexCover
        self.X_candidates = super().optimize(post_var_threshold=post_var_threshold - slack_var,
                                             tsp=False,
                                             **kwargs)[0]

        X_objective = np.asarray(X_objective)
        X_candidates = np.asarray(self.X_candidates, 
                                  dtype=X_objective.dtype)

        # ---------------- Compute coverage maps ----------------
        candidate_vars = self.kernel.K_diag(X_candidates).numpy()
        objective_vars = self.kernel.K_diag(X_objective).numpy()
        fact_1 = np.maximum(objective_vars - post_var_threshold, 0.0)
        fact_2 = candidate_vars + self.noise_variance
        var_condition = np.outer(fact_2, fact_1)
        prior_covs = self.kernel(X_candidates, X_objective).numpy()
        self.coverages = (np.square(prior_covs) > var_condition).astype(bool)
        del var_condition, prior_covs, fact_1, fact_2, candidate_vars, objective_vars

        self.target_sum = int(np.ceil(X_objective.shape[0] * target_fraction / 100.0))

        # Sanity check to ensure target fraction coverage can be achieved from candidate locations
        num_covered = len(np.where(np.sum(self.coverages, axis=0) > 0)[0])
        max_fraction = (100.0 * num_covered) / float(X_objective.shape[0])
        return max_fraction
    
    def _get_fovs(self, coverages, buffer: float = 0.5):
        """
        Convert coverage masks into polygonal “fields of view” (FoVs).

        For each selected candidate row in `coverages`, collect the objective points
        that are covered, compute their convex hull, and apply a geometric buffer
        (dilation). Candidates covering fewer than 3 points are skipped.

        Parameters
        ----------
        coverages:
            Boolean array of shape (k, n_obj) where each row indicates which
            objective points are covered by one selected candidate.
        buffer:
            Buffer radius passed to Shapely's .buffer(...).

        Returns
        -------
        list[shapely.geometry.Polygon]
            Buffered convex hull polygon per candidate (when enough points exist).
        """
        fovs = []
        for cover in coverages:
            mask = self.X_objective[cover]
            if len(mask) > 2:
                fov = geometry.MultiPoint(mask).convex_hull
                fov = fov.buffer(buffer)
                fovs.append(fov)
        return fovs

# -----------------------------------------------------------------------------

@njit
def _path_length_numba(points):
    """
    Total Euclidean length of a polyline. Numba version.
    """
    n = points.shape[0]
    if n < 2:
        return 0.0

    d = points.shape[1]
    total = 0.0
    for i in range(n - 1):
        s = 0.0
        for j in range(d):
            diff = points[i + 1, j] - points[i, j]
            s += diff * diff
        total += s ** 0.5
    return total

@njit
def _approx_dist_numba(p_nodes, x):
    """
    Approximate path length after inserting point x into an existing route.
    Numba version of approx_dist.
    """
    n = p_nodes.shape[0]
    d = p_nodes.shape[1]

    if n == 0:
        return 0.0

    if n == 1:
        seq = np.empty((2, d), dtype=p_nodes.dtype)
        for j in range(d):
            seq[0, j] = p_nodes[0, j]
            seq[1, j] = x[j]
        return _path_length_numba(seq)

    # Distances from x to each existing node
    best_idx = 0
    best_dist = 1e30
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = p_nodes[i, j] - x[j]
            s += diff * diff
        dist = s ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    idx = best_idx

    # Build two candidate sequences, as in the Python version
    if idx == n - 1:  # nearest is last
        # seq1: insert after last
        seq1 = np.empty((n + 1, d), dtype=p_nodes.dtype)
        for i in range(n):
            for j in range(d):
                seq1[i, j] = p_nodes[i, j]
        for j in range(d):
            seq1[n, j] = x[j]

        # seq2: insert before last
        seq2 = np.empty((n + 1, d), dtype=p_nodes.dtype)
        for i in range(n - 1):
            for j in range(d):
                seq2[i, j] = p_nodes[i, j]
        for j in range(d):
            seq2[n - 1, j] = x[j]
        for j in range(d):
            seq2[n, j] = p_nodes[n - 1, j]

    elif idx == 0:  # nearest is first
        # seq1: insert after first
        seq1 = np.empty((n + 1, d), dtype=p_nodes.dtype)
        for j in range(d):
            seq1[0, j] = p_nodes[0, j]
            seq1[1, j] = x[j]
        for i in range(1, n):
            for j in range(d):
                seq1[i + 1, j] = p_nodes[i, j]

        # seq2: insert before first
        seq2 = np.empty((n + 1, d), dtype=p_nodes.dtype)
        for j in range(d):
            seq2[0, j] = x[j]
        for i in range(n):
            for j in range(d):
                seq2[i + 1, j] = p_nodes[i, j]

    else:  # nearest is in the middle
        # seq1: insert after idx
        seq1 = np.empty((n + 1, d), dtype=p_nodes.dtype)
        # up to idx
        for i in range(idx + 1):
            for j in range(d):
                seq1[i, j] = p_nodes[i, j]
        # x
        for j in range(d):
            seq1[idx + 1, j] = x[j]
        # remaining
        for i in range(idx + 1, n):
            for j in range(d):
                seq1[i + 1, j] = p_nodes[i, j]

        # seq2: insert before idx
        seq2 = np.empty((n + 1, d), dtype=p_nodes.dtype)
        # up to idx-1
        for i in range(idx):
            for j in range(d):
                seq2[i, j] = p_nodes[i, j]
        # x
        for j in range(d):
            seq2[idx, j] = x[j]
        # from idx
        for i in range(idx, n):
            for j in range(d):
                seq2[i + 1, j] = p_nodes[i, j]

    dist1 = _path_length_numba(seq1)
    dist2 = _path_length_numba(seq2)

    if dist1 < dist2:
        return dist1
    else:
        return dist2

@njit
def _compute_deltas_all_numba(remaining_idxs, selected_idxs, X,
                              coverage_arr, current_cover, distance):
    """
    Numba-accelerated computation of distance_deltas and area_deltas
    for all remaining candidates.

    Parameters
    ----------
    remaining_idxs : 1D ndarray[int]
        Indices of candidates that have not yet been selected.
    selected_idxs : 1D ndarray[int]
        Indices of currently selected candidates.
    X : 2D ndarray[float], shape (m, d)
        Candidate locations.
    coverage_arr : 2D ndarray[bool], shape (m, v)
        Binary coverage mask for each candidate, where coverage_arr[i, j]
        indicates whether candidate i covers environment point j.
    current_cover : 1D ndarray[bool], shape (v,)
        Binary mask of currently covered environment points.
    distance : float
        Current path length.

    Returns
    -------
    distance_deltas : 1D ndarray[float]
        Increase in path length if each remaining candidate were added.
    area_deltas : 1D ndarray[int]
        Number of newly covered environment points for each candidate.
    """
    m_rem = remaining_idxs.shape[0]
    v = current_cover.shape[0]
    d = X.shape[1]

    distance_deltas = np.empty(m_rem, dtype=np.float64)
    area_deltas = np.empty(m_rem, dtype=np.int64)

    # Build current route locations from selected_idxs
    n_sel = selected_idxs.shape[0]
    locs_current = np.empty((n_sel, d), dtype=X.dtype)
    for i in range(n_sel):
        idx_sel = selected_idxs[i]
        for j in range(d):
            locs_current[i, j] = X[idx_sel, j]

    for k in range(m_rem):
        idx = remaining_idxs[k]

        # Distance delta
        x = X[idx]
        new_distance = _approx_dist_numba(locs_current, x)
        distance_deltas[k] = new_distance - distance

        # Area delta = number of newly covered env points
        cov_i = coverage_arr[idx]
        gain = 0
        for j in range(v):
            if cov_i[j] and (not current_cover[j]):
                gain += 1
        area_deltas[k] = gain

    return distance_deltas, area_deltas


class GCBCover(GreedyCover):
    """
    Greedy coverage selection with a path-length budget.

    This class extends `GreedyCover` by adding a travel constraint: it
    greedily selects sensing locations that improve GP-coverage, but only keeps
    additions whose resulting path (computed by a TSP solver) stays within a
    user-specified `distance_budget`.

    Coverage model
    --------------
    Coverage is inherited from `GreedyCover` and is based on a single-
    measurement GP posterior-variance test. For candidate x_i and objective x_j:

        v_post(j | i) = v_obj[j] - k(x_i, x_j)^2 / (v_cand[i] + σ_n^2)

    Candidate i covers objective j if:

        v_post(j | i) <= post_var_threshold

    which is equivalent to:

        k(x_i, x_j)^2 >= (v_obj[j] - post_var_threshold) * (v_cand[i] + σ_n^2)

    Algorithm summary
    -----------------
    1) Precompute boolean coverages[i, j] for all candidate/objective pairs
       (via `_compute_coverage_maps`, inherited from GreedyCover).
    2) Initialize with the single candidate covering the most objective points.
    3) Iteratively propose candidates using a generalized cost/benefit score:
         score = (newly-covered points) / (additional distance)
       where distance/area deltas are computed by a fast helper (Numba).
    4) For each proposal, re-solve a TSP over the selected points and accept
       the new point only if the path length is <= `distance_budget`.
    5) Stop when target coverage is met, the sensing budget is met, or no
       feasible improvements remain.

    Notes
    -----
    - Current implementation assumes `num_robots == 1`.
    - The method may return fewer than `num_sensing` points due to the distance
      budget or early attainment of the coverage target.
    - If `start_nodes` are provided (kwargs), they are prepended to the output
      route; ensure the distance budget logic accounts for them (see code notes).
    """

    def optimize(self,
                 post_var_threshold: float = 0.7,
                 target_fraction: int = 100,
                 distance_budget: float = float("inf"),
                 return_fovs: bool = False,
                 slack_var: float = 0.15,
                 **kwargs) -> np.ndarray:
        """
        Run the GCB selection with a path-length constraint.

        Parameters
        ----------
        post_var_threshold:
            Posterior variance upper bound used to binarize coverage (same meaning
            as in :meth:`GreedyCover.optimize`).
        target_fraction:
            Desired percent coverage in [0, 100]. The method stops once the number
            of covered objective points reaches `ceil(n_obj * target_fraction/100)`.
        distance_budget:
            Maximum allowed path length for the selected sensing locations (after
            TSP re-ordering). Use `inf` to disable the budget.
        return_fovs:
            If True, also return polygon FoVs derived from covered objective points.
        slack_var:
            Non-negative slack used to lower the post_var_threshold when generating 
            the candidate set via HexCover (i.e., ``post_var_threshold - slack_var``), 
            potentially generating extra candidates and improving the chance of 
            reaching the target coverage.
        **kwargs:
            Extra arguments forwarded to the TSP solver. If using special route
            constraints (e.g., `start_nodes`), ensure those constraints are also
            reflected in the distance-budget checks.

        Returns
        -------
        X_sol:
            Array shaped (num_robots, k, d) with k <= num_sensing selected points.
        (X_sol, fovs):
            If return_fovs is True, also returns a list of shapely Polygons.
        """
        if not hasattr(self, "coverages"):
            # Increase slack variance until we can reach the target fraction,
            # or until the effective threshold would become non-positive.
            slack = float(slack_var)
            max_fraction = float("-inf")

            while (post_var_threshold - slack) > 0.0:
                max_fraction = self._compute_coverage_maps(
                    post_var_threshold,
                    target_fraction,
                    slack,  # use the current (possibly increased) slack
                    **kwargs,
                )

                if max_fraction >= target_fraction:
                    break

                slack += float(slack_var)
                print("Failed to achieve target coverage. Retrying with increased slack variance...")

            if max_fraction < target_fraction:
                raise ValueError(
                    f"Target coverage {target_fraction:.2f}% is not achievable; "
                    f"maximum possible {max_fraction:.2f}% with "
                    f"post_var_threshold {post_var_threshold:.2f} and "
                    f"slack_var {slack:.2f}."
                )
        
        if kwargs.get('start_nodes', None) is not None:
            offset = 1
        else:
            offset = 0
        
        # ----- Initial location: best single coverage -----
        single_areas = self.coverages.sum(axis=1)
        first_idx = int(np.argmax(single_areas))

        selected_idxs = [first_idx]
        distance = 0.0
        current_cover = self.coverages[first_idx].copy()
        current_sum = current_cover.sum()

        remaining = np.array([i for i in range(len(self.X_candidates)) if i != first_idx],
                             dtype=np.int64)

        # ----- GCB loop -----
        while (remaining.size > 0 and
               current_sum < self.target_sum and
               len(selected_idxs) < self.num_sensing):
            remaining_arr = remaining.astype(np.int64)
            selected_arr = np.array(selected_idxs, dtype=np.int64)

            distance_deltas, area_deltas = _compute_deltas_all_numba(
                remaining_arr, 
                selected_arr, 
                self.X_candidates, 
                self.coverages,
                current_cover, distance
            )

            # area gain per extra distance
            safe_dist = np.where(distance_deltas <= 0.0, 1e-9, distance_deltas)
            ratios = area_deltas / safe_dist

            # inner loop: find a feasible candidate under distance_budget
            while remaining.size > 0:
                pos = int(np.argmax(ratios))
                best_idx = int(remaining[pos])
                best_ratio = ratios[pos]

                # Remove from remaining pools
                remaining = np.delete(remaining, pos)
                ratios = np.delete(ratios, pos)
                distance_deltas = np.delete(distance_deltas, pos)
                area_deltas = np.delete(area_deltas, pos)

                if best_ratio <= 0:
                    # No positive area/distance candidates left
                    remaining = np.array([], dtype=np.int64)
                    break

                # Recompute TSP path including this candidate
                idx_list = selected_idxs + [best_idx]
                locs = self.X_candidates[idx_list]

                # run_tsp must be available in the module scope
                _, dist_list, indices_list = run_tsp(
                    locs,
                    initial_route=[list(range(1, len(locs) + 1))],
                    return_indices=True,
                    **kwargs
                )

                new_distance = dist_list[0]
                order = indices_list[0][offset:] - offset
                new_selected_idxs = [idx_list[i] for i in order]

                if new_distance <= distance_budget:
                    selected_idxs = new_selected_idxs
                    distance = new_distance

                    # Recompute coverage and area with new selection
                    current_cover = np.any(self.coverages[selected_idxs], axis=0)
                    current_sum = int(current_cover.sum())
                    break  # back to outer while

        # ----- Prepare outputs -----
        X_sol = self.X_candidates[selected_idxs]
        start_nodes = kwargs.get('start_nodes', None)
        if start_nodes is not None:
            X_sol = np.vstack([start_nodes, X_sol])
        X_sol = np.array(X_sol).reshape(self.num_robots, -1, self.num_dim)
    
        # Greedy solution
        X_sol_greedy = super(GCBCover, self).optimize(post_var_threshold=post_var_threshold,
                                        target_fraction=target_fraction,
                                        return_fovs=return_fovs,
                                        **kwargs)

        if return_fovs:
            fovs_greedy = X_sol_greedy[1]
            X_sol_greedy = X_sol_greedy[0]

        if get_distance(X_sol_greedy[0]) < get_distance(X_sol[0]):
            X_sol = X_sol_greedy
            if return_fovs:
                fovs = fovs_greedy
        elif return_fovs:
            fovs = self._get_fovs(self.coverages[selected_idxs])

        if return_fovs:
            return X_sol, fovs
        else:
            return X_sol


METHODS: Dict[str, Type[Method]] = {
    'BayesianOpt': BayesianOpt,
    'CMA': CMA,
    'ContinuousSGP': ContinuousSGP,
    'GreedyObjective': GreedyObjective,
    'GreedySGP': GreedySGP,
    'DifferentiableObjective': DifferentiableObjective,
    'HexCover': HexCover,
    'GreedyCover': GreedyCover,
    'GCBCover': GCBCover,
}


def get_method(method: str) -> Type[Method]:
    """
    Retrieve an optimization method class by name.

    Parameters
    ----------
    method:
        Name of the optimization method. Must be one of the keys in
        :data:`METHODS`, e.g. `'ContinuousSGP'`, `'CMA'`, `'BayesianOpt'`,
        `'GreedyObjective'`, etc.

    Returns
    -------
    Type[Method]
        The corresponding method class.

    Raises
    ------
    KeyError
        If `method` is not a valid key in :data:`METHODS`.

    Usage
    --------
    ```python
    ContinuousSGPClass = get_method('ContinuousSGP')
    csgp = ContinuousSGPClass(
        num_sensing=10,
        X_objective=X_train,
        kernel=kernel_opt,
        noise_variance=noise_var_opt,
    )
    ```
    """
    if method not in METHODS:
        raise KeyError(f"Method '{method}' not found. Available methods: {', '.join(METHODS.keys())}")
    return METHODS[method]
