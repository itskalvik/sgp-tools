# `methods`: Optimization Algorithms

This module provides various algorithms to optimize the sensor placements or paths.

* **`ContinuousSGP`:** This method directly optimizes the inducing points of the `AugmentedSGPR` model to maximize the Evidence Lower Bound (ELBO). This is the main SGP-based optimization approach proposed in the papers associated with this library.

* **`GreedySGP` and `GreedyObjective`:** These implement greedy algorithms for sensor placement. `GreedySGP` iteratively selects inducing points to maximize the SGP's ELBO, while `GreedyObjective` uses a more general objective function like Mutual Information.

* **`BayesianOpt`:** This method uses Bayesian Optimization, a powerful black-box optimization algorithm, to find the best sensor locations by maximizing a general objective function.

* **`CMA`:** This method uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES), a powerful black-box optimization algorithm, to find the best sensor locations by maximizing a general objective function.