from .metrics import get_distance

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from scipy.cluster.vq import kmeans2
from shapely import geometry
import geopandas as gpd

import matplotlib.pyplot as plt
import numpy as np


def get_inducing_pts(data, num_inducing, orientation=False, random=False):
    """Selects a subset of the data points to be used as inducing points. 
    The default approach uses kmeans to select the subset. 

    Args:
        data (ndarray): (n, 2); Data points to select the inducing points from 
        num_inducing (int): Number of inducing points
        orientation (bool): If True, add an additional dimension to model the sensor 
                            FoV rotation angle
        random (bool): If True, the subset of inducing points are selected randomly 
                       instead of using kmeans

    Returns:
        Xu (ndarray): (m, d); Inducing points in the position and orientation space.
                        `m` is the number of inducing points, 
                        `d` is the dimension of the space (x, y, optional - angle in radians)
    """
    if random:
        idx = np.random.randint(len(data), size=num_inducing)
        Xu = data[idx]
    else:
        Xu = kmeans2(data, num_inducing, minit="points")[0]
    if orientation:
        thetas = np.random.uniform(0, 2 * np.pi, size=(Xu.shape[0], 1))
        Xu = np.concatenate([Xu, thetas], axis=1)
    return Xu

def cont2disc(Xu, candidates, candidate_labels=None):
    """Map continuous space locations to a discrete set of candidate location

    Args:
        Xu (ndarray): (m, 2); Continuous space points
        candidates (ndarray): (n, 2); Discrete set of candidate locations
        candidate_labels (ndarray): (n, 1); Labels corresponding to the discrete set of candidate locations

    Returns:
        Xu_x (ndarray): Discrete space points' locations 
        Xu_y (ndarray): Labels of the discrete space points. Returned only if `candidate_labels`
                        was passed to the function

    """
    # Sanity check to ensure that there are sensing locations and candidates to match
    if len(candidates)==0 or len(Xu)==0:
        if candidate_labels is not None:
            return [], []
        else:
            return []
    
    dists = pairwise_distances(candidates, Y=Xu, metric='euclidean')
    row_ind, _ = linear_sum_assignment(dists)
    Xu_X = candidates[row_ind].copy()
    if candidate_labels is not None:
        Xu_y = candidate_labels[row_ind].copy()
        return Xu_X, Xu_y
    else:
        return Xu_X

def plot_paths(paths, candidates=None, title=None):
    """Function to plot the IPP solution paths

    Args:
        paths (ndarray): (r, m, 2); `r` paths with `m` waypoints each
        candidates (ndarray): (n, 2); Candidate unlabeled locations used in the SGP-based sensor placement approach
        title (str): Title of the plot
    """
    plt.figure()
    for i, path in enumerate(paths):
        plt.plot(path[:, 0], path[:, 1], 
                    c='r', label='Path', zorder=1, marker='o')
        plt.scatter(path[0, 0], path[0, 1], 
                    c='g', label='Start', zorder=2, marker='o')
        if candidates is not None:
            plt.scatter(candidates[:, 0], candidates[:, 1], 
                        c='k', s=1, label='Unlabeled Train-Set Points', zorder=0)
        if i==0:
            plt.legend(bbox_to_anchor=(1.0, 1.02))
    if title is not None:
        plt.title(title)
    plt.gca().set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')

def interpolate_path(waypoints, sampling_rate=0.05):
    """Interpolate additional points between the given waypoints to simulate continuous sensing robots

    Args:
        waypoints (n, d): Waypoints of the robot's path
        sampling_rate (float): Distance between each pair of interpolated points

    Returns:
        path (ndarray): (p, d) Interpolated path, `p` depends on the sampling_rate rate
    """
    interpolated_path = []
    for i in range(2, len(waypoints)+1):
        dist = get_distance(waypoints[i-2:i])
        num_samples = int(dist / sampling_rate)
        points = np.linspace(waypoints[i-1], waypoints[i-2], num_samples)
        interpolated_path.extend(points)
    return np.array(interpolated_path)

def _reoder_path(path, waypoints):
    """Reorder the waypoints to match the order of the points in the path.
    The waypoints are mathched to the closest points in the path. Used by project_waypoints.

    Args:
        path (n, d): Robot path, i.e., waypoints in the path traversal order
        waypoints (n, d): Waypoints that need to be reordered to match the target path

    Returns:
        waypoints (n, d): Reordered waypoints of the robot's path
    """
    dists = pairwise_distances(path, Y=waypoints, metric='euclidean')
    _, col_ind = linear_sum_assignment(dists)
    Xu = waypoints[col_ind].copy()
    return Xu

def project_waypoints(waypoints, candidates):
    """Project the waypoints back to the candidate set while retaining the 
    waypoint visitation order.

    Args:
        waypoints (n, d): Waypoints of the robot's path
        candidates (ndarray): (n, 2); Discrete set of candidate locations

    Returns:
        waypoints (n, d): Projected waypoints of the robot's path
    """
    waypoints_disc = cont2disc(waypoints, candidates)
    waypoints_valid = _reoder_path(waypoints, waypoints_disc)
    return waypoints_valid

def ploygon2candidats(vertices, 
                      num_samples=5000, 
                      random_seed=2024):
    """Sample unlabeled candidates within a polygon

    Args:
        vertices (ndarray): (v, 2) of vertices that define the polygon
        num_samples (int): Number of samples to generate
        random_seed (int): Random seed for reproducibility

    Returns:
       candidates (ndarray): (n, 2); Candidate sensor placement locations
    """
    poly = geometry.Polygon(vertices)
    sampler = gpd.GeoSeries([poly])
    candidates = sampler.sample_points(size=num_samples,
                                       rng=random_seed)
    candidates = candidates.get_coordinates().to_numpy()
    return candidates
