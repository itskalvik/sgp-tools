# `objectives`: Information-Theoretic Objectives

This module defines the objective functions that the optimization methods aim to maximize.  Use the `get_objective` method to retrieve an objective function class by its string name.

* **`MI`, `SLogMI`, and `SchurMI`:** These classes compute the Mutual Information (MI) between a set of sensing locations $X$ and a set of objective locations $X_{objective}$, using the kernel fuunction $K$:

    * **`MI`:** A naive implementation of MI.

    * **`SLogMI`:** Uses a numerically stable implementation of MI based on the log-determinant of the covariance matrix.

    * **`SchurMI`:** Computes MI using the Schur complement for improved numerical stability and computational efficiency.

* **`AOptimal`:** Computes the A-optimal design metric, which minimizes $Tr(K(X, X))$.

* **`BOptimal`:** Computes the B-optimal design metric, which minimizes $-Tr(K(X, X)^{-1})$.

* **`DOptimal`:** Computes the D-optimal design metric, which minimizes $|K(X, X)|$.


::: sgptools.methods.get_objective
    options:
      show_root_heading: true
      show_source: true
