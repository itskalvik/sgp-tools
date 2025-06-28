# Methods

This section provides details on the various optimization methods implemented in SGP-Tools.

Sensor placement and informative path planning methods in this package:

- `BayesianOpt`: Provides a Bayesian optimization based approach that maximizes an objective to get sensor placements
- `CMA`: Provides a genetic algorithm (CMA-ES) based approach that maximizes an objective to get sensor placements
- `ContinuousSGP`: Provides an SGP-based sensor placement approach that is optimized using gradient descent
- `GreedyObjective`: Provides a greedy algorithm based approach that maximizes an objective to get sensor placements
- `GreedySGP`: Provides an SGP-based sensor placement approach that is optimized using a greedy algorithm