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

from scipy.stats import multivariate_normal
import tensorflow as tf
import numpy as np
import gpflow
from typing import Tuple
from sgptools.objectives import SLogMI


def gaussian_entropy(K: np.ndarray) -> float:
    """
    Computes the entropy of a multivariate Gaussian distribution defined by its
    covariance matrix `K`. This is often used to quantify the uncertainty or
    information content of a Gaussian Process.

    Args:
        K (np.ndarray): (n, n); A square NumPy array representing the covariance matrix.
                        Must be positive semi-definite.

    Returns:
        float: The entropy of the multivariate Gaussian distribution.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.metrics import gaussian_entropy

        # Example covariance matrix for 2 variables
        covariance_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        entropy_value = gaussian_entropy(covariance_matrix)
        ```
    """
    # Using scipy's multivariate_normal for entropy calculation
    # allow_singular=True to handle cases where the covariance matrix might be singular
    # or near-singular due to numerical issues, preventing errors.
    return float(
        multivariate_normal(mean=None, cov=K, allow_singular=True).entropy())


def get_mi(Xu: np.ndarray, X_objective: np.ndarray, noise_variance: float,
           kernel: gpflow.kernels.Kernel) -> float:
    """
    Computes the Mutual Information (MI) between a set of sensing locations (`Xu`)
    and a set of objective/candidate locations (`X_objective`) using a Gaussian Process model.
    MI quantifies the reduction in uncertainty about `X_objective` given `Xu`.
    Internally, it uses the `SLogMI` objective from `sgptools.objectives` for numerical stability.

    Args:
        Xu (np.ndarray): (m, d); NumPy array of sensing locations. `m` is the number of
                         sensing points, `d` is the dimensionality.
        X_objective (np.ndarray): (n, d); NumPy array of candidate or objective locations. `n` is the number of
                                  objective points, `d` is the dimensionality.
        noise_variance (float): The noise variance of the Gaussian Process likelihood.
        kernel (gpflow.kernels.Kernel): A GPflow kernel object used to compute covariances.

    Returns:
        float: The computed Mutual Information value.

    Usage:
        ```python
        import numpy as np
        import gpflow
        from sgptools.utils.metrics import get_mi

        # Dummy data
        X_sensing_locs = np.array([[0.1, 0.1], [0.5, 0.5]], dtype=np.float64)
        X_candidate_locs = np.array([[0.2, 0.2], [0.6, 0.6], [0.9, 0.9]], dtype=np.float64)
        noise_var = 0.1
        rbf_kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0, variance=1.0)

        mi_value = get_mi(X_sensing_locs, X_candidate_locs, noise_var, rbf_kernel)
        ```
    """
    # Ensure inputs are TensorFlow tensors for compatibility with SLogMI
    # SLogMI expects tf.Tensor, not np.ndarray for X_objective
    # Assuming SLogMI's init takes np.ndarray for X_objective and converts it
    # If not, convert X_objective here: tf.constant(X_objective, dtype=tf.float64)
    mi_model = SLogMI(
        X_objective=X_objective,
        kernel=kernel,
        noise_variance=noise_variance,
        jitter=1e-6)  # jitter is added to noise_variance in SLogMI
    # SLogMI's __call__ method expects a tf.Tensor for X (Xu in this context)
    return float(mi_model(tf.constant(Xu, dtype=tf.float64)).numpy())


def get_elbo(Xu: np.ndarray,
             X_env: np.ndarray,
             noise_variance: float,
             kernel: gpflow.kernels.Kernel,
             baseline: bool = False) -> float:
    """
    Computes the Evidence Lower Bound (ELBO) of a Sparse Gaussian Process (SGP) model.
    The ELBO is a lower bound on the marginal likelihood and is commonly used as
    an optimization objective for sparse GPs. Optionally, a baseline can be
    subtracted to ensure the ELBO is positive or to compare against a trivial model.

    Args:
        Xu (np.ndarray): (m, d); NumPy array of inducing points. `m` is the number of
                         inducing points, `d` is the dimensionality.
        X_env (np.ndarray): (n, d); NumPy array of data points representing the environment
                            or training data. `n` is the number of data points, `d` is the dimensionality.
        noise_variance (float): The noise variance of the Gaussian Process likelihood.
        kernel (gpflow.kernels.Kernel): A GPflow kernel object.
        baseline (bool): If True, a baseline ELBO (computed with a single inducing point at [0,0])
                         is subtracted from the main ELBO. This can normalize the ELBO value.
                         Defaults to False.

    Returns:
        float: The computed ELBO value.

    Usage:
        ```python
        import numpy as np
        import gpflow
        from sgptools.utils.metrics import get_elbo

        # Dummy data
        X_environment = np.random.rand(100, 2) * 10 # Environment data
        inducing_points = np.array([[2.0, 2.0], [8.0, 8.0]], dtype=np.float64) # Inducing points
        noise_var = 0.1
        rbf_kernel = gpflow.kernels.SquaredExponential(lengthscales=2.0, variance=1.0)

        # Compute ELBO without baseline
        elbo_no_baseline = get_elbo(inducing_points, X_environment, noise_var, rbf_kernel)

        # Compute ELBO with baseline
        elbo_with_baseline = get_elbo(inducing_points, X_environment, noise_var, rbf_kernel, baseline=True)
        ```
    """
    # Convert Xu to TensorFlow tensor for SGPR
    tf_Xu = tf.constant(Xu, dtype=tf.float64)
    # X_env is expected as (X, Y) tuple, but for ELBO calculation without Y, pass (X_env, zeros)
    tf_X_env = tf.constant(X_env, dtype=tf.float64)
    y_dummy = tf.zeros((tf_X_env.shape[0], 1), dtype=tf.float64)

    baseline_value = 0.0
    if baseline:
        # Create a temporary SGPR model with a single dummy inducing point for baseline
        sgpr_baseline = gpflow.models.SGPR(data=(tf_X_env, y_dummy),
                                           noise_variance=noise_variance,
                                           kernel=kernel,
                                           inducing_variable=tf.constant(
                                               [[0.0, 0.0]], dtype=tf.float64))
        baseline_value = float(sgpr_baseline.elbo().numpy())

    # Create the main SGPR model with the provided inducing points
    sgpr_model = gpflow.models.SGPR(data=(tf_X_env, y_dummy),
                                    noise_variance=noise_variance,
                                    kernel=kernel,
                                    inducing_variable=tf_Xu)

    return float((sgpr_model.elbo() - baseline_value).numpy())


def get_kl(Xu: np.ndarray, X_env: np.ndarray, noise_variance: float,
           kernel: gpflow.kernels.Kernel) -> float:
    """
    Computes the Kullback-Leibler (KL) divergence between a full Gaussian Process (GP)
    and a Sparse Gaussian Process (SGP) approximation. This KL divergence term is
    part of the ELBO objective in sparse GPs.

    Args:
        Xu (np.ndarray): (m, d); NumPy array of inducing points for the SGP.
        X_env (np.ndarray): (n, d); NumPy array of data points representing the environment
                            or training data.
        noise_variance (float): The noise variance of the Gaussian Process likelihood.
        kernel (gpflow.kernels.Kernel): A GPflow kernel object.

    Returns:
        float: The computed KL divergence value (specifically, the trace term
               from the KL divergence in the ELBO formulation, $0.5 \text{Tr}(K_{ff} - Q_{ff}) / \sigma^2$).

    Usage:
        ```python
        import numpy as np
        import gpflow
        from sgptools.utils.metrics import get_kl

        # Dummy data
        X_environment = np.random.rand(100, 2) * 10
        inducing_points = np.array([[2.0, 2.0], [8.0, 8.0]], dtype=np.float64)
        noise_var = 0.1
        rbf_kernel = gpflow.kernels.SquaredExponential(lengthscales=2.0, variance=1.0)

        kl_value = get_kl(inducing_points, X_environment, noise_var, rbf_kernel)
        ```
    """
    tf_Xu = tf.constant(Xu, dtype=tf.float64)
    tf_X_env = tf.constant(X_env, dtype=tf.float64)
    y_dummy = tf.zeros((tf_X_env.shape[0], 1), dtype=tf.float64)

    sgpr_model = gpflow.models.SGPR(data=(tf_X_env, y_dummy),
                                    noise_variance=noise_variance,
                                    kernel=kernel,
                                    inducing_variable=tf_Xu)

    # Accessing common terms used in ELBO calculation from GPflow's internal methods
    # This involves private methods (_common_calculation), so be aware of potential
    # breaking changes in future GPflow versions.
    common = sgpr_model._common_calculation()
    sigma_sq = common.sigma_sq
    AAT = common.AAT  # AAT = A @ A.T, where A = L⁻¹Kuf/σ

    # kdiag: diagonal of Kff (prior covariance for all data points)
    kdiag = sgpr_model.kernel(tf_X_env, full_cov=False)

    # trace_k: Tr(Kff) / σ²
    trace_k = tf.reduce_sum(kdiag / sigma_sq)
    # trace_q: Tr(Qff) / σ² = Tr(Kuf.T @ Kuu^-1 @ Kuf) / σ²
    # From the ELBO derivation, Tr(Q_N N) / sigma^2 is Tr(AAT)
    trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))

    # KL divergence trace term: 0.5 * Tr(Kff - Qff) / σ²
    trace_term = 0.5 * (trace_k - trace_q)

    return float(trace_term.numpy())


def get_rmse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between predicted and ground truth values.

    Args:
        y_pred (np.ndarray): (n, 1); NumPy array of predicted values.
        y_test (np.ndarray): (n, 1); NumPy array of ground truth values.

    Returns:
        float: The computed RMSE.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.metrics import get_rmse

        # Dummy data
        predictions = np.array([[1.1], [2.2], [3.3]])
        ground_truth = np.array([[1.0], [2.0], [3.0]])

        rmse_value = get_rmse(predictions, ground_truth)
        ```
    """
    error = y_pred - y_test
    return float(np.sqrt(np.mean(np.square(error))))


def get_reconstruction(
        sensor_data: Tuple[np.ndarray, np.ndarray], X_test: np.ndarray,
        noise_variance: float,
        kernel: gpflow.kernels.Kernel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Gaussian Process (GP)-based reconstruction (mean prediction and variance)
    of a data field. The provided `sensor_data` serves as the training set for the GP model,
    and predictions are made over `X_test`.

    Args:
        sensor_data (Tuple[np.ndarray, np.ndarray]): A tuple containing:
                                                    - Xu_X (np.ndarray): (m, d); Input locations from sensor measurements.
                                                    - Xu_y (np.ndarray): (m, 1); Corresponding labels (measurements) from sensors.
        X_test (np.ndarray): (n, d); NumPy array of testing input locations
                             (points where the data field needs to be estimated).
        noise_variance (float): The noise variance of the Gaussian Process likelihood.
        kernel (gpflow.kernels.Kernel): A GPflow kernel object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
                                       - y_pred (np.ndarray): (n, 1); Predicted mean estimates of the data field at `X_test`.
                                       - y_var (np.ndarray): (n, 1); Predicted variance of the data field at `X_test`.

    Usage:
        ```python
        import numpy as np
        import gpflow
        from sgptools.utils.metrics import get_reconstruction

        # Dummy sensor data (training data for GP)
        sensor_locs = np.array([[0.1, 0.1], [0.3, 0.3], [0.7, 0.7]], dtype=np.float64)
        sensor_vals = np.array([[0.5], [1.5], [2.5]], dtype=np.float64)
        
        # Dummy test locations (where we want predictions)
        test_locs = np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]], dtype=np.float64)

        noise_var = 0.05
        rbf_kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0, variance=1.0)

        predicted_means, predicted_vars = get_reconstruction(
            (sensor_locs, sensor_vals), test_locs, noise_var, rbf_kernel
        )
        ```
    """
    Xu_X, Xu_y = sensor_data

    # Initialize and train a GP Regression (GPR) model
    gpr = gpflow.models.GPR(data=(Xu_X, Xu_y),
                            noise_variance=noise_variance,
                            kernel=kernel)

    # Predict the mean and variance at the test locations
    y_pred_tf, y_var_tf = gpr.predict_f(X_test)

    # Convert TensorFlow tensors to NumPy arrays and reshape
    y_pred = y_pred_tf.numpy().reshape(-1, 1)
    y_var = y_var_tf.numpy().reshape(-1, 1)

    return y_pred, y_var


def get_distance(X: np.ndarray) -> float:
    """
    Computes the total length of a path defined by a sequence of waypoints.
    The length is calculated as the sum of Euclidean distances between consecutive waypoints.

    Args:
        X (np.ndarray): (m, d); NumPy array where each row represents a waypoint
                        and columns represent its coordinates (e.g., (x, y) or (x, y, z)).
                        `m` is the number of waypoints, `d` is the dimensionality.

    Returns:
        float: The total length of the path.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.metrics import get_distance

        # Example 2D path with 3 waypoints
        path_waypoints_2d = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 7.0]])
        distance_2d = get_distance(path_waypoints_2d)

        # Example 3D path
        path_waypoints_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        distance_3d = get_distance(path_waypoints_3d)
        ```
    """
    if X.shape[0] < 2:
        return 0.0  # A path needs at least two points to have a length

    # Compute Euclidean distance (L2-norm) between consecutive points
    # `X[1:] - X[:-1]` calculates the vector differences between adjacent waypoints
    dist_segments = np.linalg.norm(X[1:] - X[:-1], axis=-1)

    # Sum the lengths of all segments to get the total path length
    total_distance = np.sum(dist_segments)
    return float(total_distance)


def get_smse(y_pred: np.ndarray, y_test: np.ndarray, var: np.ndarray) -> float:
    """
    Computes the Standardized Mean Square Error (SMSE).
    SMSE is a variant of MSE where each squared error term is divided by
    the predicted variance. It's particularly useful in Bayesian contexts
    as it accounts for the model's uncertainty in its predictions.

    Args:
        y_pred (np.ndarray): (n, 1); NumPy array of predicted values.
        y_test (np.ndarray): (n, 1); NumPy array of ground truth values.
        var (np.ndarray): (n, 1); NumPy array of predicted variances for each prediction.

    Returns:
        float: The computed SMSE value.

    Raises:
        ValueError: If `var` contains zero or negative values, which would lead to division by zero or invalid results.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.metrics import get_smse

        # Dummy data
        predictions = np.array([[1.1], [2.2], [3.3]])
        ground_truth = np.array([[1.0], [2.0], [3.0]])
        # Predicted variances (must be positive)
        variances = np.array([[0.01], [0.04], [0.09]]) 

        smse_value = get_smse(predictions, ground_truth, variances)
        ```
    """
    if np.any(var <= 0):
        raise ValueError(
            "Predicted variance (var) must be strictly positive for SMSE calculation."
        )

    error = y_pred - y_test
    # Element-wise division by variance
    smse_val = np.mean(np.square(error) / var)
    return float(smse_val)


def get_nlpd(y_pred: np.ndarray, y_test: np.ndarray, var: np.ndarray) -> float:
    """
    Computes the Negative Log Predictive Density (NLPD).
    NLPD is a measure of how well a probabilistic model predicts new data.
    A lower NLPD indicates a better fit. For a Gaussian predictive distribution,
    it is derived from the log-likelihood of the true observations under the
    predicted Gaussian.

    Args:
        y_pred (np.ndarray): (n, 1); NumPy array of predicted mean values.
        y_test (np.ndarray): (n, 1); NumPy array of ground truth values.
        var (np.ndarray): (n, 1); NumPy array of predicted variances for each prediction.

    Returns:
        float: The computed NLPD value.

    Raises:
        ValueError: If `var` contains zero or negative values, which would lead to invalid log or division.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.metrics import get_nlpd

        # Dummy data
        predictions = np.array([[1.1], [2.2], [3.3]])
        ground_truth = np.array([[1.0], [2.0], [3.0]])
        # Predicted variances (must be positive)
        variances = np.array([[0.01], [0.04], [0.09]]) 

        nlpd_value = get_nlpd(predictions, ground_truth, variances)
        ```
    """
    if np.any(var <= 0):
        raise ValueError(
            "Predicted variance (var) must be strictly positive for NLPD calculation."
        )

    error = y_pred - y_test
    # Calculate NLPD terms for each point
    nlpd_terms = 0.5 * np.log(
        2 * np.pi) + 0.5 * np.log(var) + 0.5 * np.square(error) / var

    # Return the mean NLPD across all points
    return float(np.mean(nlpd_terms))
