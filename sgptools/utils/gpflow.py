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
from gpflow import set_trainable
from gpflow.utilities.traversal import print_summary
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .misc import get_inducing_pts
from typing import Union, List, Optional, Tuple, Any, Callable
from tensorflow.keras import optimizers


def get_model_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_steps: int = 1500,
    verbose: bool = True,
    lengthscales: Union[float, List[float]] = 1.0,
    variance: float = 1.0,
    noise_variance: float = 0.1,
    kernel: Optional[gpflow.kernels.Kernel] = None,
    return_model: bool = False,
    train_inducing_pts: bool = False,
    num_inducing_pts: int = 500,
    **kwargs: Any
) -> Union[Tuple[np.ndarray, float, gpflow.kernels.Kernel], Tuple[
        np.ndarray, float, gpflow.kernels.Kernel, Union[gpflow.models.GPR,
                                                        gpflow.models.SGPR]]]:
    """
    Trains a Gaussian Process (GP) or Sparse Gaussian Process (SGP) model on the given training set.
    A Sparse GP is used if the training set size exceeds 1500 samples.

    Args:
        X_train (np.ndarray): (n, d); Training set input features. `n` is the number of samples,
                              `d` is the number of input dimensions.
        y_train (np.ndarray): (n, 1); Training set labels. `n` is the number of samples.
        max_steps (int): Maximum number of optimization steps. Defaults to 1500.
        verbose (bool): If True, prints a summary of the optimized GP parameters. Defaults to True.
        lengthscales (Union[float, List[float]]): Initial kernel lengthscale(s). If a float, it's
                                  applied uniformly to all dimensions. If a list, each element
                                  corresponds to a data dimension. Defaults to 1.0.
        variance (float): Initial kernel variance. Defaults to 1.0.
        noise_variance (float): Initial data noise variance. Defaults to 0.1.
        kernel (Optional[gpflow.kernels.Kernel]): A pre-defined GPflow kernel function. If None,
                                     a `gpflow.kernels.SquaredExponential` kernel is created
                                     with the provided `lengthscales` and `variance`. Defaults to None.
        return_model (bool): If True, the trained GP/SGP model object is returned along with
                             loss, variance, and kernel. Defaults to False.
        train_inducing_pts (bool): If True and using a Sparse GP model, the inducing points
                                   are optimized along with other model parameters. If False,
                                   inducing points remain fixed (default for SGP). Defaults to False.
        num_inducing_pts (int): Number of inducing points to use when training a Sparse GP model.
                                Ignored if `len(X_train)` is less than or equal to 1500. Defaults to 500.
        **kwargs: Additional keyword arguments passed to the `optimize_model` function.

    Returns:
        Union[Tuple[np.ndarray, float, gpflow.kernels.Kernel], Tuple[np.ndarray, float, gpflow.kernels.Kernel, Union[gpflow.models.GPR, gpflow.models.SGPR]]]:
        - If `return_model` is False:
            Tuple: (loss (np.ndarray), variance (float), kernel (gpflow.kernels.Kernel)).
            `loss` is an array of loss values obtained during training.
            `variance` is the optimized data noise variance.
            `kernel` is the optimized GPflow kernel function.
        - If `return_model` is True:
            Tuple: (loss (np.ndarray), variance (float), kernel (gpflow.kernels.Kernel), gp (Union[gpflow.models.GPR, gpflow.models.SGPR])).
            `gp` is the optimized GPflow GPR or SGPR model object.

    Usage:
        ```python
        import numpy as np
        # Generate some dummy data
        X = np.random.rand(1000, 2) * 10
        y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2]) + np.random.randn(1000, 1) * 0.1

        # Train a GPR model (since 1000 samples <= 1500)
        losses, noise_var, trained_kernel = get_model_params(X, y, max_steps=500, verbose=True)

        # Train an SGPR model (more than 1500 samples)
        X_large = np.random.rand(2000, 2) * 10
        y_large = np.sin(X_large[:, 0:1]) + np.cos(X_large[:, 1:2]) + np.random.randn(2000, 1) * 0.1
        losses_sgpr, noise_var_sgpr, trained_kernel_sgpr, sgpr_model = \
            get_model_params(X_large, y_large, max_steps=500, num_inducing_pts=100, return_model=True)
        ```
    """
    if kernel is None:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales,
                                                   variance=variance)

    model: Union[gpflow.models.GPR, gpflow.models.SGPR]
    trainable_variables_list: List[tf.Variable]

    if len(X_train) <= 1500:
        model = gpflow.models.GPR(data=(X_train, y_train),
                                  kernel=kernel,
                                  noise_variance=noise_variance)
        trainable_variables_list = model.trainable_variables
    else:
        inducing_pts = get_inducing_pts(X_train, num_inducing_pts)
        model = gpflow.models.SGPR(data=(X_train, y_train),
                                   kernel=kernel,
                                   inducing_variable=inducing_pts,
                                   noise_variance=noise_variance)
        if train_inducing_pts:
            trainable_variables_list = model.trainable_variables
        else:
            # Exclude inducing points from trainable variables if not specified to be trained
            # Assuming inducing_variable is the first parameter in SGPR's trainable_variables
            trainable_variables_list = model.trainable_variables[1:]

    loss_values: np.ndarray
    if max_steps > 0:
        loss_values = optimize_model(
            model,
            max_steps=max_steps,
            trainable_variables=trainable_variables_list,
            verbose=verbose,
            **kwargs)
    else:
        # If no optimization steps, return an array with a single '0' loss
        loss_values = np.array([0.0])

    if verbose:
        print_summary(model)

    if return_model:
        return loss_values, model.likelihood.variance.numpy(), kernel, model
    else:
        return loss_values, model.likelihood.variance.numpy(), kernel


class TraceInducingPts(gpflow.monitor.MonitorTask):
    """
    A GPflow monitoring task designed to trace the state of inducing points
    at every step during optimization of a Sparse Gaussian Process (SGP) model.
    This is particularly useful for visualizing the movement of inducing points
    during training.

    Attributes:
        trace (List[np.ndarray]): A list to store the numpy arrays of inducing points
                                  at each optimization step.
        model (Union[gpflow.models.GPR, gpflow.models.SGPR]): The GPflow model being monitored.
    """

    def __init__(self, model: Union[gpflow.models.GPR, gpflow.models.SGPR]):
        """
        Initializes the TraceInducingPts monitor task.

        Args:
            model (Union[gpflow.models.GPR, gpflow.models.SGPR]): The GPflow GP or SGP model instance
                                                   to monitor. It is expected to have an
                                                   `inducing_variable.Z` attribute and potentially
                                                   a `transform` attribute.
        """
        super().__init__()
        self.trace: List[np.ndarray] = []
        self.model = model

    def run(self, **kwargs: Any) -> None:
        """
        Executes the monitoring task. This method is called by the GPflow `Monitor`
        at specified intervals. It extracts the current inducing points, applies
        any associated transformations (e.g., `IPPTransform`'s fixed points expansion),
        and appends them to the internal trace list.

        Args:
            **kwargs: Additional keyword arguments (e.g., `step`, `loss_value`)
                      passed by the `gpflow.monitor.Monitor` framework.

        Usage:
            This method is called internally by `gpflow.monitor.Monitor` and typically
            not invoked directly by the user.
        """
        Xu = self.model.inducing_variable.Z
        Xu_exp: np.ndarray
        # Apply IPP fixed points transform if available, without expanding sensor model
        try:
            Xu_exp = self.model.transform.expand(
                Xu, expand_sensor_model=False).numpy()
        except AttributeError:
            Xu_exp = Xu
        self.trace.append(Xu_exp)

    def run(self, **kwargs):
        '''
        Method used to extract the inducing points and 
        apply IPP fixed points transform if available
        '''
        Xu = self.model.inducing_variable.Z
        Xu_exp = self.model.transform.expand(
            Xu, expand_sensor_model=False).numpy()
        self.trace.append(Xu_exp)

    def get_trace(self) -> np.ndarray:
        """
        Returns the collected inducing points at each optimization step.

        Returns:
            np.ndarray: (num_steps, num_inducing_points, num_dimensions);
                        An array where:
                        - `num_steps` is the number of optimization steps monitored.
                        - `num_inducing_points` is the number of inducing points.
                        - `num_dimensions` is the dimensionality of the inducing points.

        Usage:
            ```python
            # Assuming `model` is an SGPR and `opt_losses` was called with `trace_fn='traceXu'`
            # trace_task = TraceInducingPts(model)
            # Then retrieve trace after optimization
            # inducing_points_history = trace_task.get_trace()
            ```
        """
        return np.array(self.trace)


def optimize_model(model: Optional[Union[gpflow.models.GPR, gpflow.models.SGPR]]=None,
                   max_steps: int = 2000,
                   optimize_hparams: bool = True,
                   optimizer: str = 'scipy.L-BFGS-B',
                   verbose: bool = False,
                   trace_fn: Optional[Union[str, Callable[[Any], Any]]] = None,
                   convergence_criterion: bool = True,
                   trainable_variables: Optional[List[tf.Variable]] = None,
                   training_loss: Optional[Callable[[Any], Any]] = None,
                   **kwargs: Any) -> np.ndarray:
    """
    Trains a GPflow GP or SGP model using either SciPy's optimizers or TensorFlow's optimizers.

    Args:
        model (Union[gpflow.models.GPR, gpflow.models.SGPR]): The GPflow model (GPR or SGPR) to be trained. 
                            Optionally, you can instead pass a loss function with the `training_loss` argument. 
        max_steps (int): Maximum number of training steps (iterations). Defaults to 2000.
        optimize_hparams (bool): If `False`, the model hyperparameters (kernel parameters and data likelihood) 
                            will not be optimized. This is ignored if `trainable_variables` is explicitly passed.
                            Defaults to True.
        optimizer (str): Specifies the optimizer to use in "<backend>.<method>" format.
                         Supported backends: `scipy` and `tf` (TensorFlow).
                         - For `scipy` backend: Refer to `scipy.optimize.minimize` documentation for available
                           methods (e.g., 'L-BFGS-B', 'CG'). Only first-order and quasi-Newton methods
                           that do not require the Hessian are supported.
                         - For `tf` backend: Refer to `tf.keras.optimizers` for available methods
                           (e.g., 'Adam', 'SGD').
                         Defaults to 'scipy.L-BFGS-B'.
        verbose (bool): If `True`, the training progress will be printed. For SciPy optimizers,
                        this controls `disp` option. Defaults to False.
        trace_fn (Optional[Union[str, Callable[[Any], Any]]]): Specifies what to trace during training:
                            - `None`: Returns the loss values.
                            - `'traceXu'`: Traces the inducing points' states at each optimization step.
                                           This increases computation time.
                            - `Callable`: A custom function that takes the traceable quantities from the optimizer
                                          and returns the desired output.
                                          - For `scipy` backend: Refer to `gpflow.monitor.MonitorTask`
                                          - For `tf` backend: Refer to `trace_fn` argument of `tfp.math.minimize`
                            Defaults to None.
        convergence_criterion (bool): If `True` and using a TensorFlow optimizer, it enables early
                                      stopping when the loss plateaus (using `tfp.optimizer.convergence_criteria.LossNotDecreasing`).
                                      Defaults to True.
        trainable_variables (Optional[List[tf.Variable]]): A list of specific model variables to train.
                                    If None, variables are determined based on `kernel_grad`. Defaults to None.
        training_loss (Optional[Callable[[Any], Any]]): A custom training loss function.
        **kwargs: Additional keyword arguments passed to the backend optimizers.

    Returns:
        np.ndarray: An array of loss values (or traced quantities if `trace_fn` is specified)
                    recorded during the optimization process. The shape depends on `trace_fn`.

    Raises:
        ValueError: If an invalid optimizer format or an unsupported backend is specified.

    Usage:
        ```python
        import gpflow
        import numpy as np

        # Create a dummy model (e.g., GPR for simplicity)
        X = np.random.rand(100, 1)
        y = X + np.random.randn(100, 1) * 0.1
        kernel = gpflow.kernels.SquaredExponential()
        model = gpflow.models.GPR((X, y), kernel=kernel, noise_variance=0.1)

        # 1. Optimize using SciPy's L-BFGS-B (default)
        losses_scipy = optimize_model(model, max_steps=500, verbose=True)

        # 2. Optimize using TensorFlow's Adam optimizer
        # Re-initialize model to reset parameters for new optimization
        model_tf = gpflow.models.GPR((X, y), kernel=gpflow.kernels.SquaredExponential(), noise_variance=0.1)
        losses_tf = optimize_model(model_tf, max_steps=1000, learning_rate=0.01, optimizer='tf.Adam', verbose=False)

        # 3. Optimize SGPR and trace inducing points
        X_sgpr = np.random.rand(2000, 2)
        y_sgpr = np.sin(X_sgpr[:, 0:1]) + np.random.randn(2000, 1) * 0.1
        inducing_points_init = get_inducing_pts(X_sgpr, 100)
        sgpr_model = gpflow.models.SGPR((X_sgpr, y_sgpr), kernel=gpflow.kernels.SquaredExponential(),
                                        inducing_variable=inducing_points_init, noise_variance=0.1)
        traced_ips = optimize_model(sgpr_model, max_steps=100, optimizer='tf.Adam', trace_fn='traceXu', verbose=False)
        ```
    """
    reset_trainable = False
    # Determine which variables to train
    if trainable_variables is None:
        # Disable hyperparameter gradients (kernel and likelihood parameters)
        if not optimize_hparams:
            set_trainable(model.kernel, False)
            set_trainable(model.likelihood, False)
            reset_trainable = True
        trainable_variables = model.trainable_variables

    # Determine the training loss function
    if training_loss is None:
        training_loss = model.training_loss

    # Parse optimizer string
    optimizer_parts = optimizer.split('.')
    if len(optimizer_parts) != 2:
        raise ValueError(
            f"Invalid optimizer format! Expected <backend>.<method>; got {optimizer}"
        )
    backend, method = optimizer_parts

    losses_output: Any  # Will hold the final loss values or traced data

    if backend == 'scipy':
        # Configure SciPy monitor if tracing is requested
        scipy_monitor: Optional[gpflow.monitor.Monitor] = None
        trace_task_instance: Optional[TraceInducingPts] = None
        if trace_fn == 'traceXu':
            trace_task_instance = TraceInducingPts(model)
            # Period=1 means run task at every step
            task_group = gpflow.monitor.MonitorTaskGroup(trace_task_instance,
                                                         period=1)
            scipy_monitor = gpflow.monitor.Monitor(task_group)

        opt = gpflow.optimizers.Scipy()
        # SciPy optimize method returns a `ScipyOptimizerResults` object
        # which has `fun` attribute for the final loss. `step_callback` is used for tracing.
        results = opt.minimize(
            training_loss,
            trainable_variables,
            method=method,
            options=dict(disp=verbose, maxiter=max_steps),
            step_callback=scipy_monitor,  # Pass the monitor as step_callback
            **kwargs)

        if trace_fn == 'traceXu' and trace_task_instance is not None:
            losses_output = trace_task_instance.task_groups[0].tasks[
                0].get_trace()
        else:
            # If no tracing or non-Xu tracing, the `results.fun` contains the final loss
            losses_output = np.array([results.fun
                                      ])  # Return as an array for consistency
            # Note: For SciPy, `losses.fun` is typically just the final loss, not a history.
            # To get history, a custom callback capturing loss at each step would be needed.

    elif backend == 'tf':
        tf_trace_fn: Optional[Callable[[Any], Any]] = None
        if trace_fn is None:
            # Default TF trace function to capture loss history
            tf_trace_fn = lambda traceable_quantities: traceable_quantities.loss
        elif trace_fn == 'traceXu':

            def tf_trace_fn(traceable_quantities):
                return model.inducing_variable.Z.numpy()
        elif callable(trace_fn):
            tf_trace_fn = trace_fn
        else:
            raise ValueError(
                f"Invalid trace_fn for TensorFlow backend: {trace_fn}")

        # Get Keras optimizer instance
        opt = getattr(optimizers, method)(**kwargs)

        # Configure convergence criterion
        tf_convergence_criterion: Optional[
            tfp.optimizer.convergence_criteria.ConvergenceCriterion] = None
        if convergence_criterion:
            tf_convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
                atol=1e-5,  # Absolute tolerance for checking decrease
                window_size=
                50,  # Number of steps to consider for plateau detection
                min_num_steps=int(
                    max_steps *
                    0.1)  # Minimum steps before early stopping is considered
            )

        # Run TensorFlow optimization
        results_tf = tfp.math.minimize(
            training_loss,
            trainable_variables=trainable_variables,
            num_steps=max_steps,
            optimizer=opt,
            convergence_criterion=tf_convergence_criterion,
            trace_fn=tf_trace_fn)

        # Fallback to just final loss if no proper trace captured
        losses_output = np.array(results_tf.numpy())

    else:
        raise ValueError(
            f"Invalid backend! Expected `scipy` or `tf`; got {backend}")

    # Reset trainable variables
    if not optimize_hparams and reset_trainable:
        set_trainable(model.kernel, True)
        set_trainable(model.likelihood, True)

    return losses_output
