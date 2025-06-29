# `utils`: Utility Functions

This module provides a collection of helper functions for data processing, model training, and other tasks.

* **`data`:** Contains the `Dataset` class, which handles loading, preprocessing, and sampling of data from various sources, including GeoTIFF files and synthetic data generation.

* **`gpflow`:** Provides utility functions for training GPflow models, such as `get_model_params` for finding optimal kernel hyperparameters and `optimize_model` for running the optimization. It also includes a `TraceInducingPts` class to monitor the movement of inducing points during training.

* **`misc`:** A collection of miscellaneous helper functions, including `get_inducing_pts` for selecting initial inducing points (via k-means or random sampling), `cont2disc` for mapping continuous solutions to a discrete set of candidates, and `polygon2candidates` for sampling points within a polygon.

* **`tsp`:** Provides functionality for solving the Traveling Salesperson Problem (TSP) using Google's OR-Tools. The `run_tsp` function can find effient paths for single or multiple vehicles, with optional start and end nodes, and can resample the resulting paths to have a fixed number of points.