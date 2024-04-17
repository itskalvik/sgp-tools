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

from .core.augmented_gpr import AugmentedGPR
from apricot import CustomSelection
import numpy as np


'''
GP-based mutual information for sensor placement approach 
(submodular objective function)

Args:
    S: Candidate sensor locations
    V: Environmental data locations
    noise_variance: data variance
    kernel: kernel function
'''
class GreedyMI:
    def __init__(self, S, V, noise_variance, kernel, transformer=None):
        self.S = S
        self.V = V
        self.kernel = kernel
        self.input_dim = S.shape[1]
        self.noise_variance = noise_variance
        self.transformer = transformer
                                 
    def mutual_info(self, x):
        x = np.array(x).reshape(-1).astype(int)
        A = self.S[x[:-1]].reshape(-1, self.input_dim)
        y = self.S[x[-1]].reshape(-1, self.input_dim)
    
        if len(A) == 0:
            sigma_a = 1.0
        else:
            if self.transformer is not None:
                A = self.transformer.expand(A)
            a_gp = AugmentedGPR(data=(A, np.zeros((len(A), 1))),
                                kernel=self.kernel,
                                noise_variance=self.noise_variance,
                                transformer=self.transformer)
            _, sigma_a = a_gp.predict_f(y, aggregate_train=True)

        # Remove locations in A to build A bar
        V_ = self.V.copy()
        V_rows = V_.view([('', V_.dtype)] * V_.shape[1])
        if self.transformer is not None:
            A_ = self.transformer.expand(self.S[x]).numpy()
        else:
            A_ = self.S[x]
        A_rows = A_.view([('', V_.dtype)] * A_.shape[1])
        V_ = np.setdiff1d(V_rows, A_rows).view(V_.dtype).reshape(-1, V_.shape[1])

        self.v_gp = AugmentedGPR(data=(V_, np.zeros((len(V_), 1))), 
                                 kernel=self.kernel,
                                 noise_variance=self.noise_variance,
                                 transformer=self.transformer)
        _, sigma_v = self.v_gp.predict_f(y)

        return (sigma_a/sigma_v).numpy().squeeze()

'''
Get sensor placement solution using the Mutual information method
'''
def get_greedy_mi_sol(num_sensors, candidates, X_train, noise_variance, kernel, 
               transformer=None, 
               optimizer='naive'):
    mi_model = GreedyMI(candidates, X_train, noise_variance, kernel, transformer)
    model = CustomSelection(num_sensors,
                            mi_model.mutual_info,
                            optimizer=optimizer,
                            verbose=False)
    sol = model.fit_transform(np.arange(len(candidates)).reshape(-1, 1))
    return candidates[sol.reshape(-1)]