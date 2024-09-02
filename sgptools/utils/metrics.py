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


def gaussian_entropy(K):
    """Computes GP-based entropy from a kernel matrix

    Args:
        K (ndarray): (n, n); kernel matrix

    Returns:
        entropy (float): Entropy computed from the kernel matrix
    """
    return multivariate_normal(mean=None, cov=K, allow_singular=True).entropy()

def get_mi(Xu, candidate_locs, noise_variance, kernel):
    """Computes mutual information between the sensing locations and the candidate locations

    Args:
        Xu (ndarray): (m, d); Sensing locations
        candidate_locs (ndarray): (n, d); Candidate sensing locations 
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        
    Returns:
        mi (float): Mutual information computed using a GP
    """
    Xu = np.array(Xu)
    candidate_locs = np.array(candidate_locs)

    gp = gpflow.models.GPR(data=(Xu, np.zeros((len(Xu), 1))),
                           kernel=kernel,
                           noise_variance=noise_variance)
    _, sigma_a = gp.predict_f(candidate_locs, full_cov=True)
    sigma_a = sigma_a.numpy()[0]
    cond_entropy = gaussian_entropy(sigma_a)

    K = kernel(candidate_locs, full_cov=True).numpy()
    K += noise_variance * np.eye(len(candidate_locs))
    entropy = gaussian_entropy(K)
    
    return float(entropy - cond_entropy)

def get_elbo(Xu, X_env, noise_variance, kernel, baseline=False):
    """Computes the ELBO of the SGP, corrected to be positive

    Args:
        Xu (ndarray): (m, d); Sensing locations
        X_env (ndarray): (n, d); Data points used to approximate the bounds of the environment
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function
        baseline (bool): If True, the ELBO is adjusted to be positive

    Returns:
        elbo (float): ELBO of the SGP
    """
    if baseline:
        sgpr = gpflow.models.SGPR(X_env,
                                  noise_variance=noise_variance,
                                  kernel=kernel,
                                  inducing_variable=[[0, 0]])
        baseline = sgpr.elbo().numpy()
    else:
        baseline = 0.0

    sgpr = gpflow.models.SGPR(X_env,
                              noise_variance=noise_variance,
                              kernel=kernel, 
                              inducing_variable=Xu)
    return (sgpr.elbo() - baseline).numpy()

def get_kl(Xu, X_env, noise_variance, kernel):
    """Computes the KL divergence between the SGP and the GP

    Args:
        Xu (ndarray): (m, d); Sensing locations
        X_env (ndarray): (n, d); Data points used to approximate the bounds of the environment
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function

    Returns:
        kl (float): KL divergence between the SGP and the GP
    """
    sgpr = gpflow.models.SGPR(X_env,
                              noise_variance=noise_variance,
                              kernel=kernel,
                              inducing_variable=Xu)

    common = sgpr._common_calculation()
    sigma_sq = common.sigma_sq
    AAT = common.AAT

    x, _ = sgpr.data
    kdiag = sgpr.kernel(x, full_cov=False)

    # tr(K) / σ²
    trace_k = tf.reduce_sum(kdiag / sigma_sq)
    # tr(Q) / σ²
    trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))
    # tr(K - Q) / σ²
    trace = trace_k - trace_q
    trace = 0.5 * trace
    
    return float(trace.numpy())

def get_rmse(y_pred, y_test):
    """Computes the root-mean-square error between `y_pred` and `y_test`

    Args:
        y_pred (ndarray): (n, 1); Predicted data field estimate
        y_test (ndarray): (n, 1); Ground truth data field 

    Returns:
        rmse (float): Computed RMSE
    """
    return np.sqrt(np.mean(np.square(y_pred - y_test)))

def get_reconstruction(sensor_data, X_test, noise_variance, kernel):
    """Computes the GP-based data field estimates with the solution placements as the training set

    Args:
        sensor_data (ndarray tuple): ((m, d), (m, 1)); Sensing locations' input 
                             and corresponding ground truth labels
        X_test (ndarray): (n, d); Testing data input locations
        noise_variance (float): data variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function

    Returns:
        y_pred (ndarray): (n, 1); Predicted data field estimates
        y_var (ndarray): (n, 1); Prediction variance at each location in the data field
    """
    Xu_X, Xu_y = sensor_data

    # Get the GP predictions
    gpr = gpflow.models.GPR((Xu_X, Xu_y),
                            noise_variance=noise_variance,
                            kernel=kernel)
    y_pred, y_var = gpr.predict_f(X_test)
    y_pred = y_pred.numpy().reshape(-1, 1)

    return y_pred, y_var

def get_distance(X):
    """Compute the length of a path (L2-norm)

    Args:
        X (ndarray): (m, d); Waypoints of a path

    Returns:
        dist (float): Total path length
    """
    dist = np.linalg.norm(X[1:] - X[:-1], axis=-1)
    dist = np.sum(dist)
    return dist


if __name__=='__main__':
   pass