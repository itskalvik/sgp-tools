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


'''
Get the GP entropy from a kernel matrix
'''
def gaussian_entropy(K):
    return multivariate_normal(mean=None, cov=K, allow_singular=True).entropy()

'''
Get the mutual information between the data and the candidate locations
'''
def get_mi(Xu, data_variance, rbf, candidate_locs):
    Xu = np.array(Xu)
    candidate_locs = np.array(candidate_locs)

    gp = gpflow.models.GPR(data=(Xu, np.zeros((len(Xu), 1))),
                           kernel=rbf,
                           noise_variance=data_variance)
    _, sigma_a = gp.predict_f(candidate_locs, full_cov=True)
    sigma_a = sigma_a.numpy()[0]
    cond_entropy = gaussian_entropy(sigma_a)

    K = rbf(candidate_locs, full_cov=True).numpy()
    K += data_variance * np.eye(len(candidate_locs))
    entropy = gaussian_entropy(K)
    
    return float(entropy - cond_entropy)

'''
Get the ELBO of the SGP, corrected to be positive
'''
def get_elbo(Xu, test_data, data_variance, rbf, baseline=False):
    if baseline:
        sgpr = gpflow.models.SGPR(test_data,
                                noise_variance=data_variance,
                                kernel=rbf,
                                inducing_variable=[[0, 0]])
        baseline = sgpr.elbo().numpy()
    else:
        baseline = 0.0

    sgpr = gpflow.models.SGPR(test_data,
                              noise_variance=data_variance,
                              kernel=rbf, 
                              inducing_variable=Xu)
    return (sgpr.elbo() - baseline).numpy()

'''
Get the KL divergence between the SGP and the GP
'''
def get_kl(Xu, test_data, data_variance, rbf):
    sgpr = gpflow.models.SGPR(test_data,
                              noise_variance=data_variance,
                              kernel=rbf,
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
    return np.sqrt(np.mean(np.square(y_pred - y_test)))

'''
Get the GP predictions with the solution placements as the training set
'''
def get_reconstruction(Xu, X_test, noise_variance, kernel):
    Xu_X, Xu_y = Xu

    # Get the GP predictions
    gpr = gpflow.models.GPR((Xu_X, Xu_y),
                            noise_variance=noise_variance,
                            kernel=kernel)
    y_pred, y_var = gpr.predict_f(X_test)
    y_pred = y_pred.numpy().reshape(-1, 1)

    return y_pred, y_var

'''
Compute the length of a path (L2-norm)
'''
def get_distance(X):
    dist = np.linalg.norm(X[1:] - X[:-1], axis=-1)
    dist = np.sum(dist)
    return dist


if __name__=='__main__':
   pass