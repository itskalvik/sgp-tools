from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from scipy.cluster.vq import kmeans2
from shapely import geometry
import geopandas as gpd

import numpy as np
from typing import Tuple, Optional, Union


def get_inducing_pts(data: np.ndarray,
                     num_inducing: int,
                     orientation: bool = False,
                     random: bool = False) -> np.ndarray:
    """
    Selects a subset of data points to be used as inducing points.
    By default, it uses k-means clustering to select representative points.
    Alternatively, it can select points randomly.
    If `orientation` is True, an additional dimension representing a rotation angle
    is appended to each inducing point.

    Args:
        data (np.ndarray): (n, d_in); Input data points from which to select inducing points.
                           `n` is the number of data points, `d_in` is the input dimensionality.
        num_inducing (int): The desired number of inducing points to select.
        orientation (bool): If True, a random orientation angle (in radians, from 0 to 2*pi)
                            is added as an additional dimension to each inducing point.
                            Defaults to False.
        random (bool): If True, inducing points are selected randomly from `data`.
                       If False, k-means clustering (`kmeans2`) is used for selection.
                       Defaults to False.

    Returns:
        np.ndarray: (m, d_out); Inducing points. `m` is `num_inducing`.
                    `d_out` is `d_in` if `orientation` is False, or `d_in + 1` if `orientation` is True.
                    If `orientation` is True, the last dimension contains angles in radians.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.misc import get_inducing_pts

        # Example data (1000 2D points)
        data_points = np.random.rand(1000, 2) * 100

        # 1. Select 50 inducing points using k-means (default)
        inducing_points_kmeans = get_inducing_pts(data_points, 50)

        # 2. Select 20 inducing points randomly with orientation
        inducing_points_random_oriented = get_inducing_pts(data_points, 20, orientation=True, random=True)
        ```
    """
    if random:
        # Randomly select `num_inducing` indices from the data
        idx = np.random.choice(len(data), size=num_inducing, replace=False)
        Xu = data[idx]
    else:
        # Use k-means clustering to find `num_inducing` cluster centers
        # `minit="points"` initializes centroids by picking random data points
        Xu = kmeans2(data, num_inducing, minit="points")[0]

    if orientation:
        # Generate random angles between 0 and 2*pi (radians)
        thetas = np.random.uniform(0, 2 * np.pi, size=(Xu.shape[0], 1))
        # Concatenate the points with their corresponding angles
        Xu = np.concatenate([Xu, thetas], axis=1)

    return Xu


def cont2disc(
    Xu: np.ndarray,
    candidates: np.ndarray,
    candidate_labels: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Maps continuous space locations (`Xu`) to the closest points in a discrete
    set of candidate locations (`candidates`) using a Hungarian algorithm
    (linear sum assignment) for optimal matching. This ensures each `Xu` point
    is matched to a unique candidate.

    Args:
        Xu (np.ndarray): (m, d); Continuous space points (e.g., optimized sensor locations).
                         `m` is the number of points, `d` is the dimensionality.
        candidates (np.ndarray): (n, d); Discrete set of candidate locations.
                                 `n` is the number of candidates, `d` is the dimensionality.
        candidate_labels (Optional[np.ndarray]): (n, 1); Optional labels corresponding to
                                                the discrete set of candidate locations.
                                                If provided, the matched labels are also returned.
                                                Defaults to None.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        - If `candidate_labels` is None:
            np.ndarray: (m, d); Discrete space points' locations (`Xu_X`),
                        where each point in `Xu` is mapped to its closest
                        unique point in `candidates`.
        - If `candidate_labels` is provided:
            Tuple[np.ndarray, np.ndarray]: (`Xu_X`, `Xu_y`).
            `Xu_X` (np.ndarray): (m, d); The matched discrete locations.
            `Xu_y` (np.ndarray): (m, 1); The labels corresponding to `Xu_X`.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.misc import cont2disc

        # Example continuous points
        continuous_points = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
        # Example discrete candidates
        discrete_candidates = np.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.6]])
        # Example candidate labels (optional)
        discrete_labels = np.array([[10.0], [20.0], [15.0]])

        # 1. Map without labels
        mapped_points = cont2disc(continuous_points, discrete_candidates)

        # 2. Map with labels
        mapped_points_X, mapped_points_y = cont2disc(continuous_points, discrete_candidates, discrete_labels)
        ```
    """
    # Sanity check to handle empty inputs gracefully
    if len(candidates) == 0 or len(Xu) == 0:
        if candidate_labels is not None:
            return np.empty((0, Xu.shape[1])), np.empty((0, 1))
        else:
            return np.empty((0, Xu.shape[1]))

    # Compute pairwise Euclidean distances between candidates and Xu
    dists = pairwise_distances(candidates, Y=Xu, metric='euclidean')

    # Use the Hungarian algorithm (linear_sum_assignment) to find the optimal
    # assignment of rows (candidates) to columns (Xu points) that minimizes
    # the total cost (distances). `row_ind` gives the indices of the rows
    # (candidates) chosen, `col_ind` gives the corresponding indices of `Xu`.
    row_ind, col_ind = linear_sum_assignment(dists)

    # Select the candidate locations that were matched to Xu points
    Xu_X = candidates[row_ind].copy()

    if candidate_labels is not None:
        # If labels are provided, select the corresponding labels as well
        Xu_y = candidate_labels[row_ind].copy()
        return Xu_X, Xu_y
    else:
        return Xu_X


def polygon2candidates(vertices: np.ndarray,
                       num_samples: int = 5000,
                       random_seed: Optional[int] = None) -> np.ndarray:
    """
    Samples a specified number of candidate points randomly within a polygon defined by its vertices.
    This function leverages `geopandas` for geometric operations.

    Args:
        vertices (np.ndarray): (v, 2); A NumPy array where each row represents the (x, y)
                               coordinates of a vertex defining the polygon. `v` is the
                               number of vertices. The polygon is closed automatically if
                               the first and last vertices are not identical.
        num_samples (int): The desired number of candidate points to sample within the polygon.
                           Defaults to 5000.
        random_seed (Optional[int]): Seed for reproducibility of the random point sampling.
                                     Defaults to None.

    Returns:
       np.ndarray: (n, 2); A NumPy array where each row represents the (x, y) coordinates
                   of a sampled candidate sensor placement location. `n` is `num_samples`.

    Usage:
        ```python
        import numpy as np
        # from sgptools.utils.misc import polygon2candidates

        # Define vertices for a square polygon
        square_vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

        # Sample 100 candidate points within the square
        sampled_candidates = polygon2candidates(square_vertices, num_samples=100, random_seed=42)
        ```
    """
    # Create a shapely Polygon object from the provided vertices
    poly = geometry.Polygon(vertices)

    # Create a GeoSeries containing the polygon, which enables sampling points
    sampler = gpd.GeoSeries([poly])

    # Sample random points within the polygon
    candidates_geoseries = sampler.sample_points(
        size=num_samples,
        rng=random_seed)  # `rng` is for random number generator seed

    # Extract coordinates from the GeoSeries of points and convert to a NumPy array
    candidates_array = candidates_geoseries.get_coordinates().to_numpy()

    return candidates_array


def project_waypoints(waypoints: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    This function maps a given path (a sequence of waypoints) to a new path
    consisting of points from a discrete set of candidate locations. It ensures
    that the original visitation order of the waypoints is preserved in the
    final projected path.

    Args:
        waypoints (np.ndarray): (m, d); The continuous waypoints of the robot's path,
                                where `m` is the number of waypoints and `d` is the
                                dimensionality.
        candidates (np.ndarray): (n, d); The discrete set of candidate locations,
                                 where `n` is the number of candidates.

    Returns:
        np.ndarray: (m, d); The projected waypoints on the discrete candidate set,
                    ordered to match the original path sequence.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.misc import project_waypoints

        # A path with 3 waypoints in a 2D space
        path_waypoints = np.array([[0.1, 0.1], [0.8, 0.8], [0.1, 0.9]])

        # A set of 4 possible discrete locations
        candidate_locations = np.array([[0, 0], [1, 1], [0, 1], [0.5, 0.5]])

        # Project the path onto the candidate locations
        projected_path = project_waypoints(path_waypoints, candidate_locations)

        # The output will be a new path of shape (3, 2) composed of points from
        # candidate_locations, ordered to best match the original path.
        # e.g., [[0, 0], [1, 1], [0, 1]]
        ```
    """
    waypoints_disc = cont2disc(waypoints, candidates)
    dists = pairwise_distances(waypoints, Y=waypoints_disc, metric='euclidean')
    _, col_ind = linear_sum_assignment(dists)
    waypoints_valid = waypoints_disc[col_ind].copy()
    return waypoints_valid