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

from .core.augmented_sgpr import AugmentedSGPR
from apricot import CustomSelection
import numpy as np


'''
SGP-based sensor placement approach. Uses a greedy algorithm to 
select sensor placements from a given discrete set of candidates.

Refer to the following paper for more details:
Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces [Jakkala and Akella, 2023]

Args:
    num_inducing: number of inducing points
    S: Candidate sensor locations
    V: Environmental data locations
    noise_variance: data variance
    kernel: kernel function
    Xu_fixed: fixed inducing points
    transform: (optional) Transform object
'''
class GreedySGP:
    def __init__(self, num_inducing, S, V, noise_variance, kernel, 
                 Xu_fixed=None, 
                 transform=None):
        self.gp = AugmentedSGPR((V, np.zeros((len(V), 1))),
                                noise_variance=noise_variance,
                                kernel=kernel, 
                                inducing_variable=S[:num_inducing],
                                transform=transform)
        self.locs = S
        self.Xu_fixed = Xu_fixed
        self.num_inducing = num_inducing
        self.inducing_dim = S.shape[1]

    def bound(self, x):
        x = np.array(x).reshape(-1).astype(int)
        Xu = np.ones((self.num_inducing, self.inducing_dim), dtype=np.float32)
        Xu *= self.locs[x][0]
        Xu[-len(x):] = self.locs[x]

        if self.Xu_fixed is not None:
            Xu[:len(self.Xu_fixed)] = self.Xu_fixed

        self.gp.inducing_variable.Z.assign(Xu)
        return self.gp.elbo().numpy()

'''
Get sensor placement solution using the Greedy VFE-SGP method
'''
def get_greedy_sgp_sol(num_sensors, candidates, X_train, noise_variance, kernel, 
                       transform=None):
    sgp_model = GreedySGP(num_sensors, candidates, X_train, 
                          noise_variance, kernel, transform=transform)
    model = CustomSelection(num_sensors,
                            sgp_model.bound,
                            optimizer='naive',
                            verbose=False)
    sol = model.fit_transform(np.arange(len(candidates)).reshape(-1, 1))
    return candidates[sol.reshape(-1)]