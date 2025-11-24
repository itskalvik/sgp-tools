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
