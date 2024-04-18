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

# Original GP code from GPflow library (https://github.com/GPflow/GPflow)

import gpflow
from gpflow.models import GPR
import tensorflow as tf

from gpflow.base import InputData, MeanAndVariance
from gpflow.utilities import add_likelihood_noise_cov, assert_params_false
from .transformations import Transformer


'''
GPR model from the GPFlow library augmented to use a Transformer object's
expand and aggregate functions on the data points where necessary.  

Args:
    transformer: Transformer object from sgp-tools
'''
class AugmentedGPR(GPR):
    def __init__(
        self,
        *args,
        transformer,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        if transformer is None:
            self.transformer = Transformer()
        else:
            self.transformer = transformer

    def predict_f(
        self, Xnew: InputData, 
        full_cov: bool = True, 
        full_output_cov: bool = False,
        aggregate_train: bool = False,
    ) -> MeanAndVariance:
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)
        if self.transformer is not None:
            Xnew = self.transformer.expand(Xnew)

        X, Y = self.data
        err = Y - self.mean_function(X)

        kmm = self.kernel(X)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X, Xnew)
        kmm_plus_s = add_likelihood_noise_cov(kmm, self.likelihood, X)

        if self.transformer is not None:
            kmn = self.transformer.aggregate(tf.transpose(kmn))
            kmn = tf.transpose(kmn)
            knn = self.transformer.aggregate(knn)

        if aggregate_train:
            kmm_plus_s = self.transformer.aggregate(kmm_plus_s)
            err = self.transformer.aggregate(err)
            # reduce kmn only if it was not reduced before
            # which can when train and test data are the same size
            if kmn.shape[0] != kmn.shape[1]:
                kmn = self.transformer.aggregate(kmn)
        
        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var