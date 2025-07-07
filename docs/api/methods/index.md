# `methods`: Optimization Algorithms

This module provides various algorithms to optimize the sensor placements or paths. Use the `get_methods` method to retrieve an optimization method class by its string name.

* **`ContinuousSGP`:** This method directly optimizes the inducing points of the `AugmentedSGPR` model to maximize the Evidence Lower Bound (ELBO). This is the main SGP-based optimization approach proposed in the papers associated with this library.

* **`GreedySGP` and `GreedyObjective`:** These implement greedy algorithms for sensor placement. `GreedySGP` iteratively selects inducing points to maximize the SGP's ELBO, while `GreedyObjective` uses a more general objective function like Mutual Information.

* **`BayesianOpt`:** This method uses Bayesian Optimization, a powerful black-box optimization algorithm, to find the best sensor locations by maximizing a general objective function.

* **`CMA`:** This method uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES), a powerful black-box optimization algorithm, to find the best sensor locations by maximizing a general objective function.

* **`DifferentiableObjective`:** This method leverages TensorFlow's automatic differentiation to directly optimize the objective function with respect to the sensor locations. This can be more efficient than black-box methods for smooth objective functions. However, the method is also more prone to getting stuck in local minima.


::: sgptools.methods.get_method
    options:
      show_root_heading: true
      show_source: true
