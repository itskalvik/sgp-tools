"""

SGP-Tools: SGP-based Optimization Tools

Software Suite for Sensor Placement and Informative Path Planning.

The library includes python code for the following:
- Greedy algorithm-based approaches
- Bayesian optimization-based approaches
- Genetic algorithm-based approaches
- Sparse Gaussian process (SGP)-based approaches

"""

__version__ = "1.1.6"
__author__ = 'Kalvik'

from .models.core import *
from .models import *
from .utils import *