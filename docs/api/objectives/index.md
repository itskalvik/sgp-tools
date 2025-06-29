# `objectives`: Information-Theoretic Objectives

This module defines the objective functions that the optimization methods aim to maximize.

* **`MI` and `SLogMI`:** These classes compute the Mutual Information between a set of sensing locations and a set of objective locations. 

    * `MI` quantifies the expected information gain from making measurements at the chosen locations. 
    * `SLogMI` uses a more numerically stable implementation of MI based on the log-determinant of the covariance matrix.