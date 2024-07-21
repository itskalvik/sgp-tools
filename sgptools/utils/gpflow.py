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


def plot_loss(losses, save_file=None):
    """Helper function to plot the training loss

    Args:
        losses (list): list of loss values
        save_file (str): If passed, the loss plot will be saved to the `save_file`
    """
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

def get_model_params(X_train, y_train, 
                     max_steps=1500, 
                     lr=1e-2, 
                     print_params=True, 
                     lengthscales=1.0, 
                     variance=1.0, 
                     noise_variance=0.1,
                     kernel=None,
                     **kwargs):
    """Train a GP on the given training set

    Args:
        X_train (ndarray): (n, d); Training set inputs
        y_train (ndarray): (n, 1); Training set labels
        max_steps (int): Maximum number of optimization steps
        lr (float): Optimization learning rate
        print_params (bool): If True, prints the optimized GP parameters
        lengthscales (float or list): Kernel lengthscale(s), if passed as a list, 
                                each element corresponds to each data dimension
        variance (float): Kernel variance
        noise_variance (float): Data noise variance
        kernel (gpflow.kernels.Kernel): gpflow kernel function

    Returns:
        loss (list): Loss values obtained during training
        variance (float): Optimized data noise variance
        kernel (gpflow.kernels.Kernel): Optimized gpflow kernel function
    """
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

def optimize_model(model, 
                   max_steps=2000, 
                   kernel_grad=True, 
                   lr=1e-2, 
                   optimizer='tf', 
                   method=None,
                   verbose=False, 
                   trace_fn=None,
                   convergence_criterion=True,
                   trainable_variables=None,
                   tol=None):
    """
    Trains a GP/SGP model

    Args:
        model (gpflow.models): GPflow GP/SGP model to train
        max_steps (int): Maximum number of training steps
        kernel_grad (bool): If False, the kernel parameters will not be optimized
        lr (float): Optimization learning rate
        optimizer (str): Optimizer to use for training (`scipy` or `tf`)
        method (str): Optimization method refer to scipy minimize and tf optimizers for full list
        verbose (bool): If true, the training progress will be printed
        trace_fn (str): Function to trace metrics during training. 
                        If `None`, the loss values are traced;
                        if `traceXu`, it the inducing points states at each optimization step are traced
        convergence_criterion (bool): It True, enables early stopping when the loss plateaus
        trainable_variables (list): List of model variables to train 
                                    (can be used to limit training to a subset of variables)
        tol (float): Convergence tolerance to decide when to stop optimization
    """
    # Train all variables if trainable_variables are not provided
    # If kernel_gradient is False, disable the kernel parameter gradient updates
    if trainable_variables is None and kernel_grad:
        trainable_variables=model.trainable_variables
    elif trainable_variables is None and not kernel_grad:
        trainable_variables=model.trainable_variables[:1]
        
    if optimizer == 'scipy':
        if method is None:
            method = 'L-BFGS-B'
        opt = gpflow.optimizers.Scipy()
        losses = opt.minimize(model.training_loss,
                              trainable_variables,
                              method=method,
                              options=dict(disp=verbose, maxiter=max_steps),
                              tol=tol)
        losses = losses.fun
    else:
        if trace_fn is None:
            trace_fn = lambda x: x.loss
        elif trace_fn == 'traceXu':
            def trace_fn(traceable_quantities):
                return model.inducing_variable.Z.numpy()

        if method is None:
            method = 'adam'
        opt = tf.keras.optimizers.get(method)
        opt.learning_rate = lr
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