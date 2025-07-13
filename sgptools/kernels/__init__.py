# sgptools/kernels/__init__.py
from .attentive import Attentive
from .neural_spectral import NeuralSpectral
import gpflow

from typing import Dict, Type
import inspect


KERNELS: Dict[str, Type[gpflow.kernels.Kernel]] = {
    'NeuralSpectral': NeuralSpectral,
    'Attentive': Attentive,
}
KERNELS.update(dict(inspect.getmembers(gpflow.kernels, inspect.isclass)))


def get_kernel(kernel: str) -> Type[gpflow.kernels.Kernel]:
    """
    Retrieves a Kernel class from the `KERNELS` dictionary based on its string name.

    Args:
        kernel (str): The name of the kernel to retrieve. The name must be a key
                      in the `KERNELS` dictionary. Includes all available kernels in gpflow.
                      e.g., 'NeuralSpectralKernel', 'AttentiveKernel', 'RBF'.

    Returns:
        Type[Kernel]: The kernel class corresponding to the provided kernel name.
                      This class can then be instantiated to create a kernel object.

    Raises:
        KeyError: If the provided `kernel` name does not exist in the `KERNELS` dictionary.

    Usage:
        ```python
        from sgptools.kernels import get_kernel
        import gpflow
        import numpy as np

        # --- Select and instantiate a kernel ---
        # 1. Get the RBF kernel class
        RBFKernelClass = get_kernel('RBF')
        # 2. Instantiate the kernel with specific parameters
        rbf_kernel = RBFKernelClass(lengthscales=1.2)

        # --- Or for a more complex custom kernel ---
        NeuralKernelClass = get_kernel('NeuralSpectral')
        neural_kernel = NeuralKernelClass(input_dim=2, Q=3, hidden_sizes=[32, 32])

        # --- Example of using the kernel in a GPR model ---
        # Dummy data
        X = np.random.rand(10, 1)
        Y = np.sin(X) + np.random.randn(*X.shape)*0.1
        # Create a model with the selected RBF kernel
        model = gpflow.models.GPR(data=(X, Y), kernel=rbf_kernel)
        ```
    """
    if kernel not in KERNELS:
        raise KeyError(f"Kernel '{kernel}' not found. Available options: {list(KERNELS.keys())}")
    return KERNELS[kernel]