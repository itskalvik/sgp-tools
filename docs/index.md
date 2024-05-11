![Image title](assets/SGP-Tools.png#only-light)
![Image title](assets/logo_dark.png#only-dark)

SGP-Tools is a software suite for [Sensor Placement](https://itskalvik.com/publication/sgp-sp) and [Informative Path Planning](https://itskalvik.com/publication/sgp-ipp).

The library includes python code for the following:

- Greedy algorithm-based approaches
- Bayesian optimization-based approaches
- Genetic algorithm-based approaches
- Sparse Gaussian process (SGP)-based approaches

<p align="center"><img src="assets/point_sensing.gif"><img src="assets/non-point_sensing.gif"></p>

## Installation
Run the following commands to install the package:

```
sudo apt-get install libhdf5-dev python3-pip -y
python3 -m pip install sgptools
```

## Quick Start
Please refer to the [demos](https://github.com/itskalvik/sgp-tools/tree/master/demos) folder for Jupyter notebooks demonstrating all the methods included in the library 😄

## Method Summary
<p align="center"><div class="video-con"><iframe width="560" height="315" src="https://www.youtube.com/embed/G-RKFa1vNHM?si=PLmrmkCwXRj7mc4A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div></p>

## About
Please consider citing the following papers if you use SGP-Tools in your academic work 😄

```
@misc{JakkalaA23SP,
AUTHOR={Kalvik Jakkala and Srinivas Akella},
TITLE={Efficient Sensor Placement from Regression with Sparse Gaussian Processes in Continuous and Discrete Spaces},
NOTE= {Preprint},
YEAR={2023},
URL={https://itskalvik.github.io/publication/sgp-sp},
}

@inproceedings{JakkalaA24IPP,
AUTHOR={Kalvik Jakkala and Srinivas Akella},
TITLE={Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes},
booktitle={IEEE International Conference on Robotics and Automation, {ICRA}},
YEAR={2024},
PUBLISHER = {{IEEE}},
URL={https://itskalvik.github.io/publication/sgp-ipp}
}
``` 

## Acknowledgements
This work was funded in part by the UNC Charlotte Office of Research and Economic Development and by NSF under Award Number IIP-1919233.

## License
The SGP-Tools software suite is licensed under the terms of the Apache License 2.0.
See LICENSE for more information.