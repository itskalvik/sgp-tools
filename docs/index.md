<p align="center">
  <img src="assets/SGP-Tools.png#only-light" alt="SGP-Tools Logo" width="600"/>
  <img src="assets/logo_dark.png#only-dark" alt="SGP-Tools Logo" width="600"/>
</p>

<p align="center">
  <em>A Python library for efficient sensor placement and informative path planning</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/sgptools/"><img alt="PyPI" src="https://img.shields.io/pypi/v/sgptools.svg"></a>
  <a href="https://github.com/itskalvik/sgptools/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/sgptools.svg"></a>
</p>

---

**SGP-Tools** is a powerful and flexible Python library designed for optimizing sensor placements and planning informative paths for robotic systems, enabling efficient and scalable solutions for environment monitoring.

<p align="center">
  <img src="assets/point_sensing.gif" width="49%">
  <img src="assets/non-point_sensing.gif" width="49%">
  <img src="assets/AIPP-4R.gif" width="49%">
  <img src="assets/AIPP-non-point_sensing.gif" width="49%">
</p>

---

## Why SGP-Tools?

-   **State-of-the-Art Algorithms**: Includes a variety of optimization methods including greedy algorithms, Bayesian optimization, CMA-ES, and SGP-based optimization.
-   **Advanced Modeling Capabilities**: Go beyond simple point sensing with tools for informative path planning for multi-robot systems and complex sensor field-of-view (FoV) models.
-   **Non-Stationary Kernels**: Capture complex, real-world phenomena with specialized non-stationary kernels like the Neural Spectral Kernel and the Attentive Kernel.
-   **Flexible and Extensible**: Built on GPflow and TensorFlow, the library is designed to be modular and easy to extend with your own custom methods, kernels, and objectives.

---

## Installation
The library is available as a ```pip``` package. To install the package, run the following command:

```
python3 -m pip install sgptools
```

Installation from source:

```
git clone https://github.com/itskalvik/sgp-tools.git
cd sgp-tools/
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Note: The requirements.txt file contains packages and their latest versions that were last verified to be working without any issues.

## Quick Start
Please refer to the [example Jupyter notebooks](examples/IPP.html) demonstrating the methods included in the library 😄

## SGP-based Informative Path Planning
<p align="center"><div class="video-con"><iframe width="560" height="315" src="https://www.youtube.com/embed/G-RKFa1vNHM?si=PLmrmkCwXRj7mc4A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div></p>

## Datasets
* High-resolution topography and bathymetry data can be downloaded from [NOAA Digital Coast](https://coast.noaa.gov/digitalcoast/)

## About
Please consider citing the following papers if you use SGP-Tools in your academic work 😄

```
@misc{JakkalaA23SP,
AUTHOR={Kalvik Jakkala and Srinivas Akella},
TITLE={Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces},
NOTE= {Preprint},
YEAR={2023},
URL={https://www.itskalvik.com/research/publication/sgp-sp/},
}

@inproceedings{JakkalaA24IPP,
AUTHOR={Kalvik Jakkala and Srinivas Akella},
TITLE={Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes},
booktitle={IEEE International Conference on Robotics and Automation, {ICRA}},
YEAR={2024},
PUBLISHER = {{IEEE}},
URL={https://www.itskalvik.com/research/publication/sgp-ipp/}
}

@inproceedings{JakkalaA25AIPP,
AUTHOR={Kalvik Jakkala and Srinivas Akella},
TITLE={Fully Differentiable Adaptive Informative Path Planning},
booktitle={IEEE International Conference on Robotics and Automation, {ICRA}},
YEAR={2025},
PUBLISHER = {{IEEE}},
URL={https://www.itskalvik.com/research/publication/sgp-aipp/}
}
``` 

## Acknowledgements
This work was funded in part by the UNC Charlotte Office of Research and Economic Development and by NSF under Award Number IIP-1919233.