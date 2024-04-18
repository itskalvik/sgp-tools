from .metrics import get_distance

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from scipy.cluster.vq import kmeans2

import matplotlib.pyplot as plt
import numpy as np


'''
Selects a subset of the data points to be used as inducing points.

Args:
    data: [n, 2] - data points to select inducing points from 
    num_inducing: [m] - number of inducing points

Returns:
    Xu: [m, 3] - Inducing points in the position and orientation space.
                 m is the number of inducing points,
                 3 is the dimension of the space (x, y, angle in radians)
'''
def get_inducing_pts(data, num_inducing, orientation=False, random=False):
    if random:
        idx = np.random.randint(len(data), size=num_inducing)
        Xu = data[idx]
    else:
        Xu = kmeans2(data, num_inducing, minit="points")[0]
    if orientation:
        thetas = np.random.uniform(0, 2 * np.pi, size=(Xu.shape[0], 1))
        Xu = np.concatenate([Xu, thetas], axis=1)
    return Xu

'''
Convert SGP continuous solution to discrete solution
'''
def cont2disc(Xu, candidates, candidate_labels=None):
    # Sanity check to ensure that there are candidates to match
    if len(candidates)==0:
        return []
    dists = pairwise_distances(candidates, Y=Xu, metric='euclidean')
    row_ind, _ = linear_sum_assignment(dists)
    Xu_X = candidates[row_ind].copy()
    if candidate_labels is not None:
        Xu_y = candidate_labels[row_ind].copy()
        return Xu_X, Xu_y
    else:
        return Xu_X

'''
Function to plot IPP solution paths
'''
def plot_paths(paths, candidates=None, title=None):
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
    plt.xlabel('X')
    plt.ylabel('Y')

'''
Interpolate additional points between the given waypoints

Args:
    waypoints: locations between which 
    sampling_rate: distance between a pair of interpolated points
'''
def interpolate_path(waypoints, sampling_rate=0.05):
    interpolated_path = []
    for i in range(2, len(waypoints)+1):
        dist = get_distance(waypoints[i-2:i])
        num_samples = int(dist / sampling_rate)
        points = np.linspace(waypoints[i-1], waypoints[i-2], num_samples)
        interpolated_path.extend(points)
    return np.array(interpolated_path)