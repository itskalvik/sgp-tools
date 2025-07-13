import tensorflow as tf
import numpy as np
import gpflow
from typing import Type, Any, Dict


def jitter_fn(cov: tf.Tensor, jitter: float = 1e-6) -> tf.Tensor:
    """
    Adds a small positive value (jitter) to the diagonal of a covariance matrix
    for numerical stability. This prevents issues with ill-conditioned matrices
    during computations like Cholesky decomposition or determinant calculation.

    Args:
        cov (tf.Tensor): The input covariance matrix. Expected to be a square matrix.
        jitter (float): The small positive value to add to the diagonal. Defaults to 1e-6.

    Returns:
        tf.Tensor: The covariance matrix with jitter added to its diagonal.

    Usage:
        ```python
        # Example covariance matrix
        cov_matrix = tf.constant([[1.0, 0.5], [0.5, 1.0]], dtype=tf.float64)
        # Add default jitter
        jittered_cov = jitter_fn(cov_matrix)
        # Add custom jitter
        custom_jittered_cov = jitter_fn(cov_matrix, jitter=1e-5)
        ```
    """
    cov = tf.linalg.set_diag(cov, tf.linalg.diag_part(cov) + jitter)
    return cov


class Objective:
    """
    Base class for objective functions used in optimization.
    Subclasses must implement the `__call__` method to define the objective.
    """

    def __init__(self,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 jitter: float = 1e-6,
                 **kwargs: Any):
        """
        Initializes the base objective. This constructor primarily serves to define
        the expected parameters for all objective subclasses.

        Args:
            X_objective (np.ndarray): The fixed set of data points (e.g., candidate locations
                                      or training data points) against which MI is computed.
                                      Shape: (N, D).
            kernel (gpflow.kernels.Kernel): The GPflow kernel function to compute covariances.
            noise_variance (float): The observed data noise variance, which is added to the jitter.
            jitter (float): A small positive value to add for numerical stability to covariance
                            matrix diagonals. Defaults to 1e-6.
            **kwargs: Arbitrary keyword arguments.
        """
        self.X_objective = tf.constant(X_objective, dtype=tf.float64)
        self.kernel = kernel
        self.noise_variance = noise_variance
        # Total jitter includes the noise variance
        self._base_jitter = jitter
        self.jitter_fn = lambda cov: jitter_fn(
            cov, jitter=self._base_jitter + self.noise_variance)

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the objective value for a given set of input points `X`.
        This method must be implemented by subclasses.

        Args:
            X (tf.Tensor): The input points for which the objective is to be computed.
                           Shape: (M, D) where M is number of points, D is dimension.

        Returns:
            tf.Tensor: The computed objective value.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance for the MI objective.
        This method is crucial for optimizing the GP hyperparameters externally
        and having the objective function reflect those changes.

        Args:
            kernel (gpflow.kernels.Kernel): The updated GPflow kernel function.
            noise_variance (float): The updated data noise variance.
        """
        # Update kernel's trainable variables (e.g., lengthscales, variance)
        for self_var, var in zip(self.kernel.trainable_variables,
                                 kernel.trainable_variables):
            self_var.assign(var)

        self.noise_variance = noise_variance
        # Update the jitter function to reflect the new noise variance
        self.jitter_fn = lambda cov: jitter_fn(
            cov, jitter=self._base_jitter + self.noise_variance)


class MI(Objective):
    """
    Computes the Mutual Information (MI) between a fixed set of objective points
    (`X_objective`) and a variable set of input points (`X`).

    MI is calculated as:
    $MI(X; X_{objective}) = log|K(X,X)| + log|K(X_{objective},X_{objective})| - log|K(X \cup X_{objective}, X \cup X_{objective})|$

    Jitter is added to the diagonal of the covariance matrices to ensure numerical stability.
    """

    def __init__(self,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 jitter: float = 1e-6,
                 cache: bool = True,
                 **kwargs: Any):
        """
        Initializes the Mutual Information (MI) objective.

        Args:
            X_objective (np.ndarray): The fixed set of data points (e.g., candidate locations
                                      or training data points) against which MI is computed.
                                      Shape: (N, D).
            kernel (gpflow.kernels.Kernel): The GPflow kernel function to compute covariances.
            noise_variance (float): The observed data noise variance, which is added to the jitter.
            jitter (float): A small positive value to add for numerical stability to covariance
                            matrix diagonals. Defaults to 1e-6.
            cache (bool): If `True`, $K(X_{objective}, X_{objective})$ will be computed in the `_init__`
                          and reused in the `__call__` for faster computation time. Defaults to True.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(X_objective, kernel, noise_variance, jitter, **kwargs)
        self.cache = cache
        if self.cache:
            # K(X_objective, X_objective)
            self.K_obj_obj = self.kernel(self.X_objective)
            # Compute log determinants
            self.logdet_K_obj_obj = tf.math.log(tf.linalg.det(
                self.jitter_fn(self.K_obj_obj)))
            
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the Mutual Information for the given input points `X`.

        Args:
            X (tf.Tensor): The input points (e.g., sensing locations) for which
                           MI is to be computed. Shape: (M, D).

        Returns:
            tf.Tensor: The computed Mutual Information value.

        Usage:
            ```python
            import gpflow
            import numpy as np
            # Assume X_objective and kernel are defined
            # X_objective = np.random.rand(100, 2)
            # kernel = gpflow.kernels.SquaredExponential()
            # noise_variance = 0.1

            mi_objective = MI(X_objective=X_objective, kernel=kernel, noise_variance=noise_variance)
            X_sensing = tf.constant(np.random.rand(10, 2), dtype=tf.float64)
            mi_value = mi_objective(X_sensing)
            ```
        """
        # K(X_objective, X_objective)
        if self.cache:
            K_obj_obj = self.K_obj_obj
        else:
            K_obj_obj = self.kernel(self.X_objective)
        # K(X, X)
        K_X_X = self.kernel(X)
        # K(X_objective U X, X_objective U X)
        K_combined = self.kernel(tf.concat([self.X_objective, X], axis=0))

        # Compute log determinants
        if self.cache:
            logdet_K_obj_obj = self.logdet_K_obj_obj
        else:
            logdet_K_obj_obj = tf.math.log(tf.linalg.det(
                self.jitter_fn(K_obj_obj)))
        logdet_K_X_X = tf.math.log(tf.linalg.det(self.jitter_fn(K_X_X)))
        logdet_K_combined = tf.math.log(
            tf.linalg.det(self.jitter_fn(K_combined)))

        # MI formula
        mi = logdet_K_obj_obj + logdet_K_X_X - logdet_K_combined

        return mi

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance for the MI objective.
        This method is crucial for optimizing the GP hyperparameters externally
        and having the objective function reflect those changes.

        Args:
            kernel (gpflow.kernels.Kernel): The updated GPflow kernel function.
            noise_variance (float): The updated data noise variance.
        """
        super().update(kernel, noise_variance)
        if self.cache:
            # K(X_objective, X_objective)
            self.K_obj_obj = self.kernel(self.X_objective)
            # Compute log determinants
            self.logdet_K_obj_obj = tf.math.log(tf.linalg.det(
                self.jitter_fn(self.K_obj_obj)))

class SLogMI(MI):
    """
    Computes the Mutual Information (MI) using `tf.linalg.slogdet` for numerical stability,
    especially for large or ill-conditioned covariance matrices.

    The slogdet (sign and log determinant) method computes the sign and the natural
    logarithm of the absolute value of the determinant of a square matrix.
    This is more numerically stable than computing the determinant directly and then
    taking the logarithm, as `tf.linalg.det` can return very small or very large
    numbers that lead to underflow/overflow when `tf.math.log` is applied.

    Jitter is also added to the diagonal for additional numerical stability.
    """

    def __init__(self,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 jitter: float = 1e-6,
                 cache: bool = True,
                 **kwargs: Any):
        """
        Initializes the Mutual Information (MI) objective based on `tf.linalg.slogdet`.

        Args:
            X_objective (np.ndarray): The fixed set of data points (e.g., candidate locations
                                      or training data points) against which MI is computed.
                                      Shape: (N, D).
            kernel (gpflow.kernels.Kernel): The GPflow kernel function to compute covariances.
            noise_variance (float): The observed data noise variance, which is added to the jitter.
            jitter (float): A small positive value to add for numerical stability to covariance
                            matrix diagonals. Defaults to 1e-6.
            cache (bool): If `True`, $K(X_{objective}, X_{objective})$ will be computed in the `_init__`
                          and reused in the `__call__` for faster computation time. Defaults to True.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(X_objective, kernel, noise_variance, jitter, cache=False, **kwargs)
        self.cache = cache
        if self.cache:
            # K(X_objective, X_objective)
            self.K_obj_obj = self.kernel(self.X_objective)
            # Compute log determinants
            _, self.logdet_K_obj_obj = tf.linalg.slogdet(self.jitter_fn(self.K_obj_obj))

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the Mutual Information for the given input points `X` using `tf.linalg.slogdet`.

        Args:
            X (tf.Tensor): The input points (e.g., sensing locations) for which
                           MI is to be computed. Shape: (M, D).

        Returns:
            tf.Tensor: The computed Mutual Information value.

        Usage:
            ```python
            import gpflow
            import numpy as np
            # Assume X_objective and kernel are defined
            # X_objective = np.random.rand(100, 2)
            # kernel = gpflow.kernels.SquaredExponential()
            # noise_variance = 0.1

            slogmi_objective = SLogMI(X_objective=X_objective, kernel=kernel, noise_variance=noise_variance)
            X_sensing = tf.constant(np.random.rand(10, 2), dtype=tf.float64)
            mi_value = slogmi_objective(X_sensing)
            ```
        """
        # K(X_objective, X_objective)
        if self.cache:
            K_obj_obj = self.K_obj_obj
        else:
            K_obj_obj = self.kernel(self.X_objective)
        # K(X, X)
        K_X_X = self.kernel(X)
        # K(X_objective U X, X_objective U X)
        K_combined = self.kernel(tf.concat([self.X_objective, X], axis=0))

        # Compute log determinants using slogdet for numerical stability
        if self.cache:
            logdet_K_obj_obj = self.logdet_K_obj_obj
        else:
            _, logdet_K_obj_obj = tf.linalg.slogdet(self.jitter_fn(K_obj_obj))

        _, logdet_K_X_X = tf.linalg.slogdet(self.jitter_fn(K_X_X))
        _, logdet_K_combined = tf.linalg.slogdet(self.jitter_fn(K_combined))

        # MI formula
        mi = logdet_K_obj_obj + logdet_K_X_X - logdet_K_combined

        return mi


class SchurMI(SLogMI):
    """
    Computes Mutual Information (MI) using the Schur complement for improved
    numerical stability and computational efficiency.

    This method leverages the properties of block matrix determinants to reformulate
    the MI calculation. The standard MI formula is:
    $MI = \\log|K_{XX}| + \\log|K_{oo}| - \\log|K_{combined}|$
    where $K_{XX} = K(X, X)$, $K_{oo} = K(X_{objective}, X_{objective})$, and
    $K_{combined}$ is the kernel of the union of the points.

    Using the Schur complement identity for determinants,
    $\\log|K_{combined}| = \\log|K_{oo}| + \\log|K_{XX} - K_{Xo} K_{oo}^{-1} K_{oX}|$,
    the MI calculation simplifies to:
    $MI = \\log|K_{XX}| - \\log|SchurComplement|$
    where the Schur Complement is $K_{XX} - K_{Xo} K_{oo}^{-1} K_{oX}$.

    This approach is particularly efficient when the objective is evaluated
    multiple times for different sensing locations $X$ but with a fixed set of
    $X_{objective}$ points. By caching the inverse of $K_{oo}$, we avoid costly
    recomputations. Like `SLogMI`, this class uses `tf.linalg.slogdet` and
    adds jitter for robust computation.
    """
    def __init__(self,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 jitter: float = 1e-6,
                 cache: bool = True,
                 **kwargs: Any):
        """
        Initializes the SchurMI objective.

        Args:
            X_objective (np.ndarray): The fixed set of data points against which
                                      MI is computed. Shape: (N, D).
            kernel (gpflow.kernels.Kernel): The GPflow kernel to compute covariances.
            noise_variance (float): The observed data noise variance.
            jitter (float): A small value added to the diagonal of covariance
                            matrices for numerical stability. Defaults to 1e-6.
            cache (bool): If `True`, the inverse of $K(X_{objective}, X_{objective})$
                          is pre-computed and cached to accelerate subsequent MI
                          calculations. Defaults to True.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(X_objective, kernel, noise_variance, jitter, cache=False, **kwargs)
        self.cache = cache
        if self.cache:
            # K(X_objective, X_objective)
            self.K_obj_obj = self.kernel(self.X_objective)
            # Compute the inverse
            self.inv_K_obj_obj = tf.linalg.inv(self.jitter_fn(self.K_obj_obj))

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the Mutual Information for the input points `X` using the
        Schur complement method.

        Args:
            X (tf.Tensor): The input points (e.g., sensing locations) for which
                           MI is to be computed. Shape: (M, D).

        Returns:
            tf.Tensor: The computed Mutual Information value.

        Usage:
            ```python
            import gpflow
            import numpy as np
            # Assume X_objective and kernel are defined
            # X_objective = np.random.rand(100, 2)
            # kernel = gpflow.kernels.SquaredExponential()
            # noise_variance = 0.1

            schur_mi_objective = SchurMI(
                X_objective=X_objective,
                kernel=kernel,
                noise_variance=noise_variance
            )
            X_sensing = tf.constant(np.random.rand(10, 2), dtype=tf.float64)
            mi_value = schur_mi_objective(X_sensing)
            ```
        """
        if self.cache:
            inv_K_obj_obj = self.inv_K_obj_obj
        else:
            # K(X_objective, X_objective)
            K_obj_obj = self.kernel(self.X_objective)
            # Compute the inverse
            inv_K_obj_obj = tf.linalg.inv(self.jitter_fn(K_obj_obj))

        K_X_X = self.kernel(X)
        _, logdet_K_X_X = tf.linalg.slogdet(self.jitter_fn(K_X_X))
        K_X_obj = self.kernel(X, self.X_objective)
        transpose_K_X_obj = tf.transpose(K_X_obj)
        schur = K_X_X - K_X_obj @ inv_K_obj_obj @ transpose_K_X_obj
        _, schur_det = tf.linalg.slogdet(self.jitter_fn(schur))
        mi = logdet_K_X_X - schur_det
        return mi

    def update(self, kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Updates the kernel and noise variance for the MI objective.
        This method is crucial for optimizing the GP hyperparameters externally
        and having the objective function reflect those changes.

        Args:
            kernel (gpflow.kernels.Kernel): The updated GPflow kernel function.
            noise_variance (float): The updated data noise variance.
        """
        super().update(kernel, noise_variance)
        if self.cache:
            # K(X_objective, X_objective)
            self.K_obj_obj = self.kernel(self.X_objective)
            # Compute the inverse
            self.inv_K_obj_obj = tf.linalg.inv(self.jitter_fn(self.K_obj_obj))
            

class  AOptimal(Objective):          
    """
    Computes the A-optimal design metric.

    A-optimality aims to minimize the trace of the
    covariance matrix $Tr(K(X, X))$. Since optimization algorithms typically
    minimize a function, this objective returns the negative trace, which
    is then maximized.
    """  
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the negative trace of the covariance matrix $-Tr(K(X, X))$.

        Args:
            X (tf.Tensor): The input points (e.g., sensing locations) for which
                           the objective is to be computed. Shape: (M, D).

        Returns:
            tf.Tensor: The computed A-optimal metric value.
            
        Usage:
            ```python
            import gpflow
            import numpy as np
            # Assume kernel is defined
            # X_objective = np.random.rand(100, 2) # Not used by A-Optimal but required by base class
            # kernel = gpflow.kernels.SquaredExponential()
            # noise_variance = 0.1

            a_optimal_objective = AOptimal(
                X_objective=X_objective,
                kernel=kernel,
                noise_variance=noise_variance
            )
            X_sensing = tf.constant(np.random.rand(10, 2), dtype=tf.float64)
            a_optimal_value = a_optimal_objective(X_sensing)
            ```
        """
        # K(X, X)
        K_X_X = self.kernel(X)
        trace_K_X_X = tf.linalg.trace(self.jitter_fn(K_X_X))
        return -trace_K_X_X


class  BOptimal(Objective):     
    """
    Computes the B-optimal design metric.

    Refer to the following paper for more details:
        - Approximate Sequential Optimization for Informative Path Planning [Ott et al., 2024]

    B-optimality minimizes the trace of the inverse of the covariance matrix 
    $-Tr(K(X, X)^{-1})$. Since optimization 
    algorithms typically minimize a function, this objective returns 
    $Tr(K(X, X)^{-1})$, which is then maximized.
    """       
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the trace of the inverse of the covariance matrix $Tr(K(X, X)^{-1})$.

        Args:
            X (tf.Tensor): The input points (e.g., sensing locations) for which
                           the objective is to be computed. Shape: (M, D).

        Returns:
            tf.Tensor: The computed B-optimal metric value.
            
        Usage:
            ```python
            import gpflow
            import numpy as np
            # Assume kernel is defined
            # X_objective = np.random.rand(100, 2) # Not used by B-Optimal but required by base class
            # kernel = gpflow.kernels.SquaredExponential()
            # noise_variance = 0.1

            b_optimal_objective = BOptimal(
                X_objective=X_objective,
                kernel=kernel,
                noise_variance=noise_variance
            )
            X_sensing = tf.constant(np.random.rand(10, 2), dtype=tf.float64)
            b_optimal_value = b_optimal_objective(X_sensing)
            ```
        """
        # K(X, X)
        K_X_X = self.kernel(X)
        inv_K_X_X = tf.linalg.inv(self.jitter_fn(K_X_X))
        trace_inv_K_X_X = tf.linalg.trace(inv_K_X_X)
        return trace_inv_K_X_X
    

class  DOptimal(Objective):            
    """
    Computes the D-optimal design metric.

    D-optimality seeks to minimize the determinant of the posterior
    covariance matrix $|K(X, X)|$. The objective returns
    the negative log-determinant of $K(X, X)$, which is maximized during
    optimization. `tf.linalg.slogdet` is used for numerical stability.
    """
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Computes the negative log-determinant of the covariance matrix $-log|K(X, X)|$.

        Args:
            X (tf.Tensor): The input points (e.g., sensing locations) for which
                           the objective is to be computed. Shape: (M, D).

        Returns:
            tf.Tensor: The computed D-optimal metric value.
            
        Usage:
            ```python
            import gpflow
            import numpy as np
            # Assume kernel is defined
            # X_objective = np.random.rand(100, 2) # Not used by D-Optimal but required by base class
            # kernel = gpflow.kernels.SquaredExponential()
            # noise_variance = 0.1

            d_optimal_objective = DOptimal(
                X_objective=X_objective,
                kernel=kernel,
                noise_variance=noise_variance
            )
            X_sensing = tf.constant(np.random.rand(10, 2), dtype=tf.float64)
            d_optimal_value = d_optimal_objective(X_sensing)
            ```
        """
        # K(X, X)
        K_X_X = self.kernel(X)
        _, logdet_K_X_X = tf.linalg.slogdet(self.jitter_fn(K_X_X))
        return -logdet_K_X_X
    

OBJECTIVES: Dict[str, Type[Objective]] = {
    'MI': MI,
    'SLogMI': SLogMI,
    'SchurMI': SchurMI,
    'AOptimal': AOptimal,
    'BOptimal': BOptimal,
    'DOptimal': DOptimal,
}


def get_objective(objective_name: str) -> Type[Objective]:
    """
    Retrieves an objective function class by its string name.

    Args:
        objective_name (str): The name of the objective function (e.g., 'MI', 'SLogMI').

    Returns:
        Type[Objective]: The class of the requested objective function.

    Raises:
        KeyError: If the objective name is not found in the registered OBJECTIVES.

    Usage:
        ```python
        # Get the Mutual Information objective class
        MIObjectiveClass = get_objective('MI')
        # You can then instantiate it:
        # mi_instance = MIObjectiveClass(X_objective=..., kernel=..., noise_variance=...)
        ```
    """
    if objective_name not in OBJECTIVES:
        raise KeyError(f"Objective '{objective_name}' not found. Available options: {list(OBJECTIVES.keys())}")
    return OBJECTIVES[objective_name]
