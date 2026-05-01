# How It All Works Together: A Conceptual Workflow

A typical use case of the `sgptools` library follows a clean, four-step pipeline:

**1. Ingest Data (`Dataset`)** Start by instantiating a `Dataset` object. This class handles the ingestion and optional standardization of your environment data. You can load static maps (such as `.tif` images or `NumPy` arrays) or stream raw, real-time sensor data directly from an autonomous system.

**2. Define Constraints & Sensor Models (`Transform`) [Optional]** Tailor the problem geometry by applying a `Transform` object. This step maps your physical and operational reality to the optimizer. For instance, use an `IPPTransform` to enforce a travel distance budget for multi-robot path planning, or apply a specific footprint (like a `SquareTransform`) to accurately model the field of view.

**3. Select an Optimization Strategy (`methods`)** Choose the mathematical solver best suited to your objective from the `methods` module:

* **Coverage with Guarantees:** Use `GreedyCover`, `GCBCover`, or `HexCover` for IPP with uncertainty guarantees, ensuring the maximum posterior variance remains below a specified threshold.
* **SGP-Based IPP:** Use `ContinuousSGP` for fast, continuous-space IPP powered by Sparse Gaussian Processes.
* **Black-Box IPP:** Use `BayesianOpt` or `CMA` for derivative-free, continuous path optimization.
* **Greedy IPP:** Use `GreedyObjective` for discrete-space optimization via a greedy algorithm. This seamlessly integrates with any metric from the `objectives` module.
* **Differentiable IPP:** Use `DifferentiableObjective` for continuous-space optimization using gradient descent. Like the greedy approach, this is fully compatible with any metric from the `objectives` module.

**4. Execute the Solver (`optimize()`)** Finally, call the `.optimize()` method on your selected optimizer. This triggers the underlying algorithm—such as maximizing the ELBO—and returns the final optimized waypoints or sensing locations.