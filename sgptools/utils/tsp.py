# Copyright 2024 The SGP-Tools Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from shapely.geometry import LineString
from sklearn.metrics import pairwise_distances

_SCALE = 10_000  # factor to convert floating distances to integer costs
_MAX_RECURSION_DEPTH = 5


def run_tsp(
    nodes: np.ndarray,
    num_vehicles: int = 1,
    max_dist: Optional[float] = None,
    depth: int = 1,
    resample: Optional[int] = None,
    start_nodes: Optional[np.ndarray] = None,
    end_nodes: Optional[np.ndarray] = None,
    time_limit: int = 10,
    solution_limit: Optional[int] = None,
    initial_route: Optional[List[List[int]]] = None,
    return_indices: bool = False,
) -> Tuple:
    """Solve a TSP/VRP using OR-Tools with optional start/end nodes and max distance.

    Supports:
      * Single-vehicle TSP.
      * Multi-vehicle VRP with optional per-vehicle start and end locations.
      * Optional maximum distance per vehicle.
      * Optional resampling of each resulting path to a fixed number of points.
      * Optional warm-start with an initial route.
      * Optional return of route indices in addition to coordinates.

    Args:
        nodes: Array of shape (n_nodes, ndim). Nodes to visit.
        num_vehicles: Number of vehicles/robots.
        max_dist: Maximum allowed travel distance per vehicle (same units as
            `nodes`). If None, no distance constraint is enforced.
        depth: Recursion depth used internally for retry logic.
        resample: If provided, each solution path is resampled to this many
            points.
        start_nodes: Array of shape (num_vehicles, ndim). Optional start node
            for each vehicle.
        end_nodes: Array of shape (num_vehicles, ndim). Optional end node for
            each vehicle.
        time_limit: Solver time limit in seconds.
        solution_limit: Optional limit on the number of solutions OR-Tools will
            search.
        initial_route: Optional initial routes for warm starting, in OR-Tools
            format: a list of lists, one list per vehicle, each containing node
            indices in the OR-Tools node space (not coordinates).
        return_indices: If True, the function returns the path indices along
            with the coordinates and distances.

    Returns:
        If `return_indices` is False:
            (paths, distances)

        If `return_indices` is True:
            (paths, distances, paths_indices)

        Where:
          * paths: list of np.ndarray, one per vehicle, each of shape
            (num_waypoints, ndim). None if no solution was found.
          * distances: list of floats, one per vehicle, with total path length
            for each vehicle. None if no solution was found.
          * paths_indices: list of 1D np.ndarrays of node indices (into the
            input `nodes` array) for each vehicle, only when
            `return_indices=True`.

    Usage:
        Basic single-vehicle TSP without a distance constraint:

        ```python
        import numpy as np
        nodes = np.array([[0, 0], [1, 1], [0, 2], [2, 2]], dtype=np.float64)
        paths, dists = run_tsp(nodes, num_vehicles=1, max_dist=None, time_limit=5)
        ```

        Single-vehicle TSP **with** a maximum distance constraint:

        ```python
        paths, dists = run_tsp(nodes, num_vehicles=1, max_dist=10.0, time_limit=5)
        ```

        Multi-vehicle VRP with start/end nodes, max distance, and resampling:

        ```python
        nodes_multi = np.array([[1,1], [2,2], [3,3], [4,4]], dtype=np.float64)
        start_points = np.array([[0,0], [5,5]], dtype=np.float64)
        end_points = np.array([[0,5], [5,0]], dtype=np.float64)
        paths_multi, dists_multi = run_tsp(
            nodes_multi,
            num_vehicles=2,
            max_dist=10.0,
            resample=10,
            start_nodes=start_points,
            end_nodes=end_points,
            time_limit=10,
        )
        ```
    """
    if depth > _MAX_RECURSION_DEPTH:
        print("TSP Warning: Max recursion depth reached; giving up.")
        if return_indices:
            return None, None, None
        return None, None

    original_nodes = np.copy(nodes)

    # Append end and start nodes (if provided) to the node list
    if end_nodes is not None:
        assert end_nodes.shape == (num_vehicles, nodes.shape[-1]), (
            "Incorrect end_nodes shape, should be (num_vehicles, ndim)!"
        )
        nodes = np.concatenate([end_nodes, nodes], axis=0)

    if start_nodes is not None:
        assert start_nodes.shape == (num_vehicles, nodes.shape[-1]), (
            "Incorrect start_nodes shape, should be (num_vehicles, ndim)!"
        )
        nodes = np.concatenate([start_nodes, nodes], axis=0)

    # Build distance matrix, possibly with a dummy node at index 0 to allow
    # arbitrary start/end locations when they are not provided.
    if start_nodes is None or end_nodes is None:
        distance_mat = np.zeros((len(nodes) + 1, len(nodes) + 1), dtype=float)
        distance_mat[1:, 1:] = pairwise_distances(nodes, nodes) * _SCALE
        trim_paths = True  # need to post-process paths to remove dummy node
    else:
        distance_mat = pairwise_distances(nodes, nodes) * _SCALE
        trim_paths = False

    distance_mat = distance_mat.astype(int)

    # Start indices for each vehicle
    if start_nodes is None:
        start_idx = np.zeros(num_vehicles, dtype=int)
        num_start_nodes = 0
    else:
        start_idx = np.arange(num_vehicles, dtype=int) + int(trim_paths)
        num_start_nodes = len(start_nodes)

    # End indices for each vehicle
    if end_nodes is None:
        end_idx = np.zeros(num_vehicles, dtype=int)
    else:
        end_idx = np.arange(num_vehicles, dtype=int) + num_start_nodes + int(trim_paths)

    # Distance callback used by OR-Tools
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_mat[from_node, to_node]

    # num_locations, num_vehicles, start, end
    manager = pywrapcp.RoutingIndexManager(
        len(distance_mat),
        num_vehicles,
        start_idx.tolist(),
        end_idx.tolist(),
    )
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Optional distance constraint, now used for single-vehicle as well.
    if max_dist is not None:
        max_dist_scaled = int(max_dist * _SCALE)
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,               # no slack
            max_dist_scaled, # max route distance (scaled)
            True,            # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit

    if solution_limit is not None:
        search_parameters.solution_limit = solution_limit

    # Solve, with optional warm start.
    if initial_route is not None:
        routing.CloseModelWithParameters(search_parameters)
        initial_solution = routing.ReadAssignmentFromRoutes(initial_route, True)
        solution = routing.SolveFromAssignmentWithParameters(
            initial_solution, search_parameters
        )
    else:
        solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        if max_dist is not None:
            print("TSP Warning: No solution found, retrying with increased max_dist.")
            new_max_dist = max_dist * 1.5
            return run_tsp(
                original_nodes,
                num_vehicles=num_vehicles,
                max_dist=new_max_dist,
                depth=depth + 1,
                resample=resample,
                start_nodes=start_nodes,
                end_nodes=end_nodes,
                time_limit=time_limit,
                solution_limit=solution_limit,
                initial_route=initial_route,
                return_indices=return_indices,
            )
        else:
            print("TSP Warning: No solution found (no max_dist constraint to relax).")
            if return_indices:
                return None, None, None
            return None, None

    paths_indices, distances_raw = _get_routes(
        manager=manager,
        routing=routing,
        solution=solution,
        num_vehicles=num_vehicles,
        start_idx=start_idx,
        end_idx=end_idx,
        trim_paths=trim_paths,
    )

    # Check for empty paths and retry with a more relaxed max_dist if needed.
    if max_dist is not None:
        for path in paths_indices:
            if len(path) < 2:
                print(
                    "TSP Warning: Empty path detected, retrying with increased max_dist."
                )
                if distances_raw:
                    mean_dist = float(np.mean(distances_raw)) / _SCALE
                else:
                    mean_dist = max_dist

                new_max_dist = mean_dist * (1.5 / max(depth, 1))
                return run_tsp(
                    original_nodes,
                    num_vehicles=num_vehicles,
                    max_dist=new_max_dist,
                    depth=depth + 1,
                    resample=resample,
                    start_nodes=start_nodes,
                    end_nodes=end_nodes,
                    time_limit=time_limit,
                    solution_limit=solution_limit,
                    initial_route=initial_route,
                    return_indices=return_indices,
                )

    # Convert indices to coordinates
    paths = [nodes[path] for path in paths_indices]
    distances = [d / _SCALE for d in distances_raw]

    # Optional resampling of each path
    if resample is not None:
        paths = [resample_path(path, resample) for path in paths]

    if return_indices:
        return paths, distances, paths_indices
    return paths, distances


def _get_routes(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    num_vehicles: int,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    trim_paths: bool,
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract per-vehicle routes and distances from an OR-Tools solution.

    Args:
        manager: OR-Tools RoutingIndexManager instance.
        routing: OR-Tools RoutingModel instance.
        solution: OR-Tools Assignment representing the solved routes.
        num_vehicles: Number of vehicles.
        start_idx: Array of start indices (in OR-Tools node space) for each vehicle.
        end_idx: Array of end indices (in OR-Tools node space) for each vehicle.
        trim_paths: If True, remove dummy node(s) and shift indices back to the
            original `nodes` indexing.

    Returns:
        paths_indices: List of 1D np.ndarrays of node indices for each vehicle.
        distances_raw: List of integer route costs (still in scaled units).
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
                previous_index,
                index,
                vehicle_id,
            )

        path.append(manager.IndexToNode(index))
        distances.append(route_distance)

        # Remove dummy start/end points and shift indices if required.
        path_array = np.array(path, dtype=int)

        if trim_paths:
            # Remove dummy start node at the beginning if present.
            if (
                start_idx[vehicle_id] == 0
                and path_array.shape[0] > 1
                and path_array[0] == 0
            ):
                path_array = path_array[1:]

            # Remove dummy end node at the end if present.
            if (
                end_idx[vehicle_id] == 0
                and path_array.shape[0] > 0
                and path_array[-1] == 0
            ):
                path_array = path_array[:-1]

            # Shift all indices down by 1 if a dummy node was prepended to the
            # overall distance matrix at index 0.
            if np.any(start_idx == 0) or np.any(end_idx == 0):
                path_array = path_array - 1
                path_array = path_array[path_array >= 0]

        paths.append(path_array)

    return paths, distances


def resample_path(waypoints: np.ndarray, num_inducing: int = 10) -> np.ndarray:
    """Resample a path (sequence of waypoints) to a fixed number of points.

    Uses shapely's LineString to interpolate along the path, preserving its
    geometric shape as much as possible.

    Args:
        waypoints: Array of shape (num_waypoints, ndim) representing the path.
            ndim must be 2 or 3.
        num_inducing: Desired number of points in the resampled path.

    Returns:
        Array of shape (num_inducing, ndim) with the resampled path.

    Raises:
        ValueError: If `ndim` is not 2 or 3.

    Usage:
        Resampling a 2D path:

        ```python
        import numpy as np
        original_path = np.array([[0, 0], [1, 5], [3, 0], [5, 5]], dtype=np.float64)
        resampled = resample_path(original_path, num_inducing=5)
        ```

        Resampling a 3D path:

        ```python
        original_path_3d = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2]], dtype=np.float64)
        resampled_3d = resample_path(original_path_3d, num_inducing=7)
        ```
    """
    ndim = int(np.shape(waypoints)[-1])
    if ndim not in (2, 3):
        raise ValueError(f"ndim={ndim} is not supported for path resampling!")

    line = LineString(waypoints)
    distances = np.linspace(0, line.length, num_inducing)
    points = [line.interpolate(distance) for distance in distances]

    if ndim == 2:
        resampled_points = np.array([[p.x, p.y] for p in points], dtype=float)
    else:  # ndim == 3
        resampled_points = np.array([[p.x, p.y, p.z] for p in points], dtype=float)

    return resampled_points
