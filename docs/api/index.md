# How It All Works Together: A Conceptual Workflow

A typical use case of the `sgptools` library would follow these steps:

1. **Load Data:** A user would start by creating a `Dataset` object from their data, which could be a `.tif` file or a `NumPy` array. The `Dataset` class handles the necessary preprocessing and standardization. Alternatively, the user can use real-time data from a robot. 

2. **Define a Transformation:** Based on the problem, the user would instantiate a `Transform` object. For example, for a multi-robot path planning problem with a distance budget, they would use `IPPTransform`. For a single sensor with a square field of view, they might use `SquareTransform`.

3. **Choose an Optimization Method:** The user would then select an optimization method from the `methods` module. For the novel SGP-based approach, they would choose `ContinuousSGP`. For comparison with other methods, they could use `BayesianOpt`, `CMA`, or the greedy methods.

4. **Run Optimization:** The `optimize()` method of the chosen optimizer is called. This will run the optimization algorithm (e.g., maximizing the ELBO in the case of `ContinuousSGP`) and return the optimized sensor locations or paths.

5. **Post-processing:** The solution might be post-processed, for example, by mapping the continuous locations to a set of discrete candidates using `cont2disc`.