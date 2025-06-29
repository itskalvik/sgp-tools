# `core`: Core Gaussian Process Models and Transformations

This module contains the fundamental building blocks for modeling and transforming sensor data.

* **`AugmentedSGPR` and `AugmentedGPR`:** These are extensions of GPflow's `SGPR` and `GPR` models. They are "augmented" to incorporate custom `Transformations` on the inducing points, which is a key feature of this library for modeling complex sensor setups.

* **`Transformations`:** This is a crucial part of the library, defining how inducing points in the SGP are manipulated to represent different physical sensing scenarios.

    * **`Transform`:** The base class for all transformations.

    * **`IPPTransform`:** A versatile transform for Informative Path Planning (IPP). It can model continuous sensing paths (by interpolating points between waypoints), handle multi-robot scenarios, and enforce distance constraints on the paths. It also supports online IPP where some waypoints are fixed.

    * **`SquareTransform` and `SquareHeightTransform`:** These transforms model non-point, 2D fields of view (FoV). `SquareTransform` creates a square FoV with a fixed size and optimizable orientation, while `SquareHeightTransform` models a FoV whose size depends on the sensor's height from the ground (the z-dimension).

* **`osgpr`:** This module provides an implementation of an Online Sparse Variational GP regression model (`OSGPR_VFE`), which is designed for streaming data scenarios where the model is updated sequentially with new data batches; used for adaptive IPP. The `init_osgpr` function helps in setting up this model.