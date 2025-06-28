# Copyright 2024 The SGP-Tools Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.metrics import pairwise_distances
from shapely.geometry import LineString
import numpy as np


def run_tsp(
    nodes: np.ndarray,
    num_vehicles: int = 1,
    max_dist: float = 25.0,
    depth: int = 1,
    resample: Optional[int] = None,
    start_nodes: Optional[np.ndarray] = None,
    end_nodes: Optional[np.ndarray] = None,
    time_limit: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Method to run TSP/VRP with arbitrary start and end nodes,
    and without any distance constraint.

    Args:
        nodes (np.ndarray): (# nodes, ndim); Nodes to visit.
        num_vehicles (int): Number of robots/vehicles.
        max_dist (float): Maximum distance allowed for each path when handling multi-robot case.
        depth (int): Internal parameter used to track re-try recursion depth.
        resample (Optional[int]): Each solution path will be resampled to have
                                   `resample` number of points.
        start_nodes (Optional[np.ndarray]): (# num_vehicles, ndim); Optional array of start nodes from which
                                             to start each vehicle's solution path.
        end_nodes (Optional[np.ndarray]): (# num_vehicles, ndim); Optional array of end nodes at which
                                           to end each vehicle's solution path.
        time_limit (int): TSP runtime time limit in seconds.

    Returns:
        Tuple[Optional[np.ndarray], Optional[List[float]]]:
            - paths (np.ndarray): Solution paths if found, otherwise None.
            - distances (List[float]): List of path lengths if paths are found, otherwise None.
    """
    if depth > 5:
        print('Warning: Max depth reached')
        return None, None

    # Backup original nodes
    original_nodes = np.copy(nodes)

    # Add the start and end nodes to the node list
    if end_nodes is not None:
        assert end_nodes.shape == (num_vehicles, nodes.shape[-1]), \
            "Incorrect end_nodes shape, should be (num_vehicles, ndim)!"
        nodes = np.concatenate([end_nodes, nodes])
    if start_nodes is not None:
        assert start_nodes.shape == (num_vehicles, nodes.shape[-1]), \
            "Incorrect start_nodes shape, should be (num_vehicles, ndim)!"
        nodes = np.concatenate([start_nodes, nodes])

    # Add dummy 0 location to get arbitrary start and end node sols
    if start_nodes is None or end_nodes is None:
        distance_mat = np.zeros((len(nodes) + 1, len(nodes) + 1))
        distance_mat[1:, 1:] = pairwise_distances(nodes, nodes) * 1e4
        trim_paths = True  #shift to account for dummy node
    else:
        distance_mat = pairwise_distances(nodes, nodes) * 1e4
        trim_paths = False
    distance_mat = distance_mat.astype(int)
    max_dist = int(max_dist * 1e4)

    # Get start and end node indices for ortools
    if start_nodes is None:
        start_idx = np.zeros(num_vehicles, dtype=int)
        num_start_nodes = 0
    else:
        start_idx = np.arange(num_vehicles) + int(trim_paths)
        num_start_nodes = len(start_nodes)

    if end_nodes is None:
        end_idx = np.zeros(num_vehicles, dtype=int)
    else:
        end_idx = np.arange(num_vehicles) + num_start_nodes + int(trim_paths)

    # used by ortools
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_mat[from_node][to_node]

    # num_locations, num vehicles, start, end
    manager = pywrapcp.RoutingIndexManager(len(distance_mat), num_vehicles,
                                           start_idx.tolist(),
                                           end_idx.tolist())
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    if num_vehicles > 1:
        # Dummy distaance constraint to ensure all paths have similar length
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            max_dist,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit
    solution = routing.SolveWithParameters(search_parameters)

    paths: Optional[List[np.ndarray]] = None
    distances: Optional[List[float]] = None

    if solution is not None:
        paths_indices, distances_raw = _get_routes(manager, routing, solution,
                                                   num_vehicles, start_idx,
                                                   end_idx, trim_paths)

        # Check for empty paths and retry with increased max_dist if necessary
        for path in paths_indices:
            if len(path) < 2:
                print(
                    "TSP Warning: Empty path detected, retrying with increased max_dist."
                )
                # Recalculate max_dist based on the current average distance
                mean_dist = np.mean(
                    distances_raw) / 1e4 if distances_raw else max_dist
                return run_tsp(
                    original_nodes,
                    num_vehicles,
                    mean_dist * (1.5 / depth),
                    depth + 1,
                    resample,
                    start_nodes,
                    end_nodes,
                    time_limit,
                )
        paths = [nodes[path] for path in paths_indices]
        distances = [d / 1e4 for d in distances_raw]

    else:
        print(
            "TSP Warning: No solution found, retrying with increased max_dist."
        )
        return run_tsp(
            original_nodes,
            num_vehicles,
            max_dist * 1.5,
            depth + 1,
            resample,
            start_nodes,
            end_nodes,
            time_limit,
        )

    # Resample each solution path to have resample number of points
    if resample is not None and paths is not None:
        paths = np.array([resample_path(path, resample) for path in paths])

    return paths, distances


def _get_routes(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    num_vehicles: int,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    trim_paths: bool,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
    """
    Solves the Traveling Salesperson Problem (TSP) or Vehicle Routing Problem (VRP)
    using Google OR-Tools. This method supports multiple vehicles/robots, optional
    start and end nodes for each vehicle, and an optional maximum distance constraint
    per path. It also includes a retry mechanism with increased maximum distance
    if no solution is found.

    Args:
        nodes (np.ndarray): (# nodes, ndim); A NumPy array of coordinates for all nodes (locations)
                            that need to be visited. `ndim` is the dimensionality (e.g., 2 for 2D, 3 for 3D).
        num_vehicles (int): The number of vehicles (robots) available to visit the nodes. Defaults to 1.
        max_dist (float): The maximum allowed travel distance for each vehicle's path.
                          This constraint is particularly relevant for multi-vehicle scenarios. Defaults to 25.0.
        depth (int): Internal parameter used to track the recursion depth for retries when
                     no solution is found. Users should typically not modify this. Defaults to 1.
        resample (Optional[int]): If provided, each solution path will be resampled to have
                                  exactly `resample` number of points (waypoints). This is useful
                                  for standardizing path representations. Defaults to None.
        start_nodes (Optional[np.ndarray]): (# num_vehicles, ndim); Optional NumPy array specifying
                                            the starting coordinates for each vehicle. If None,
                                            OR-Tools will find arbitrary start points. Defaults to None.
        end_nodes (Optional[np.ndarray]): (# num_vehicles, ndim); Optional NumPy array specifying
                                          the ending coordinates for each vehicle. If None,
                                          OR-Tools will find arbitrary end points. Defaults to None.
        time_limit (int): The maximum time (in seconds) that OR-Tools will spend searching for a solution.
                          Defaults to 10.

    Returns:
        Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
        - If a solution is found:
            Tuple[List[np.ndarray], List[float]]: A tuple containing:
                - paths (List[np.ndarray]): A list of NumPy arrays, where each array
                                            represents a vehicle's path (sequence of visited nodes).
                                            Shape of each array: (num_waypoints, ndim).
                - distances (List[float]): A list of floats, where each float is the
                                           total length of the corresponding vehicle's path.
        - If no solution is found after retries:
            Tuple[None, None]: Returns `(None, None)`.

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.tsp import run_tsp

        # Example 1: Single TSP, find best path through 5 points
        nodes_single = np.array([[0,0], [1,1], [0,2], [2,2], [1,0]], dtype=np.float64)
        paths_single, dists_single = run_tsp(nodes_single, num_vehicles=1, time_limit=5)

        # Example 2: Multi-robot VRP with start/end nodes and resampling
        nodes_multi = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]], dtype=np.float64)
        start_points = np.array([[0,0], [7,7]], dtype=np.float64)
        end_points = np.array([[0,7], [7,0]], dtype=np.float64)

        paths_multi, dists_multi = run_tsp(
            nodes_multi,
            num_vehicles=2,
            max_dist=10.0, # Max distance for each robot
            resample=10,   # Resample each path to 10 points
            start_nodes=start_points,
            end_nodes=end_points,
            time_limit=15
        )
        ```
    """
    paths: List[np.ndarray] = []
    distances: List[int] = []
    for vehicle_id in range(num_vehicles):
        path: List[int] = []
        route_distance = 0
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        path.append(manager.IndexToNode(index))
        distances.append(route_distance)

        # Remove dummy start/end points if they were added
        if trim_paths:
            path_array = np.array(path)
            # Adjust indices if a dummy node was added at the beginning
            if (start_idx[vehicle_id] == 0 and path_array[0] == 0
                    and path_array.shape[0] > 1):
                path_array = path_array[1:]
            # Adjust indices if a dummy node was added at the end
            if (end_idx[vehicle_id] == 0 and path_array[-1] == 0
                    and path_array.shape[0] > 0):
                path_array = path_array[:-1]

            # Shift all indices down by 1 if a dummy node was prepended to the overall distance matrix
            if np.any(start_idx == 0) or np.any(end_idx == 0):
                path_array = path_array - 1
                path_array = path_array[
                    path_array
                    >= 0]  # Ensure no negative indices from the shift

            paths.append(path_array)
        else:
            paths.append(np.array(path))
    return paths, distances


def resample_path(waypoints: np.ndarray, num_inducing: int = 10) -> np.ndarray:
    """Resamples a given path (sequence of waypoints) to have a fixed number of
    `num_inducing` points. This is useful for standardizing path representations
    or for converting a path with an arbitrary number of waypoints into a
    fixed-size representation for models. The resampling maintains the path's
    shape and geometric integrity.

    Args:
        waypoints (np.ndarray): (num_waypoints, ndim); A NumPy array representing the
                                waypoints of a path. `num_waypoints` is the original
                                number of points, `ndim` is the dimensionality.
        num_inducing (int): The desired number of points in the resampled path. Defaults to 10.

    Returns:
        np.ndarray: (num_inducing, ndim); The resampled path with `num_inducing` points.

    Raises:
        Exception: If the input `ndim` is not 2 or 3 (as `shapely.geometry.LineString`
                   primarily supports 2D/3D geometries).

    Usage:
        ```python
        import numpy as np
        from sgptools.utils.tsp import resample_path

        # Example 2D path
        original_path_2d = np.array([[0,0], [1,5], [3,0], [5,5]], dtype=np.float64)
        resampled_path_2d = resample_path(original_path_2d, num_inducing=5)

        # Example 3D path
        original_path_3d = np.array([[0,0,0], [1,1,1], [2,0,2]], dtype=np.float64)
        resampled_path_3d = resample_path(original_path_3d, num_inducing=7)
        ```
    """
    ndim = np.shape(waypoints)[-1]
    if not (ndim == 2 or ndim == 3):
        raise Exception(f"ndim={ndim} is not supported for path resampling!")
    line = LineString(waypoints)
    distances = np.linspace(0, line.length, num_inducing)
    points = [line.interpolate(distance) for distance in distances]
    if ndim == 2:
        resampled_points = np.array([[p.x, p.y] for p in points])
    elif ndim == 3:
        resampled_points = np.array([[p.x, p.y, p.z] for p in points])
    return resampled_points
