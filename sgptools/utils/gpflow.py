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

import gpflow
from gpflow.utilities.traversal import print_summary

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt


'''
Helper function to plot the training loss
'''
def plot_loss(losses, save_file=None):
    plt.plot(losses)
    plt.title('Log Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

'''
Train a GP to get the kernel parameters
'''
def get_model_params(X_train, y_train, 
                     max_steps=1500, 
                     lr=1e-2, 
                     print_params=True, 
                     lengthscales=1.0, 
                     variance=1.0, 
                     noise_variance=0.1,
                     kernel=None,
                     **kwargs):
    if kernel is None:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales, 
                                                   variance=variance)

    gpr_gt = gpflow.models.GPR(data=(X_train, y_train), 
                               kernel=kernel,
                               noise_variance=noise_variance)

    if max_steps > 0:
        loss = optimize_model(gpr_gt, max_steps=max_steps, lr=lr, **kwargs)
    else:
        loss = 0

    if print_params:
        print_summary(gpr_gt)
    
    return loss, gpr_gt.likelihood.variance, kernel

'''
Trains a GP/SGP model for given number of steps.

Args:
    model: GPflow GP/SGP model to train
    max_steps: Maximum number of training steps
    kernel_grad: whether to train the kernel parameters or not
    lr: learning rate
    optimizer: optimizer to use ('scipy' or 'tf')
    verbose: whether to print training progress
    trace_fn: function to trace metrics during training
              - if None, trace the loss
              - if 'traceXu', trace the inducing points
    convergence_criterion: It True, enables early stopping on loss plateau
    trainable_variables: 
'''
def optimize_model(model, 
                   max_steps=2000, 
                   kernel_grad=True, 
                   lr=1e-2, 
                   optimizer='tf', 
                   verbose=False, 
                   trace_fn=None,
                   convergence_criterion=True,
                   trainable_variables=None,
                   tol=None):
    # Train all variables if trainable_variables are not provided
    # If kernel_gradient is False, disable the kernel parameter gradient updates
    if trainable_variables is None and kernel_grad:
        trainable_variables=model.trainable_variables
    elif trainable_variables is None and not kernel_grad:
        trainable_variables=model.trainable_variables[:1]
        
    if optimizer == 'scipy':
        opt = gpflow.optimizers.Scipy()
        losses = opt.minimize(model.training_loss,
                              trainable_variables,
                              options=dict(disp=verbose, maxiter=max_steps),
                              tol=tol)
        losses = losses.fun
    else:
        if trace_fn is None:
            trace_fn = lambda x: x.loss
        elif trace_fn == 'traceXu':
            def trace_fn(traceable_quantities):
                return model.inducing_variable.Z.numpy()

        opt = tf.keras.optimizers.Adam(lr)
        loss_fn = model.training_loss
        if convergence_criterion:
            convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
                                            atol=1e-5, 
                                            window_size=50,
                                            min_num_steps=int(max_steps*0.1))
        else:
            convergence_criterion = None
        losses = tfp.math.minimize(loss_fn,
                                   trainable_variables=trainable_variables,
                                   num_steps=max_steps,
                                   optimizer=opt,
                                   convergence_criterion=convergence_criterion,
                                   trace_fn=trace_fn)
        losses = losses.numpy()
    
    return losses


if __name__ == "__main__":
    pass