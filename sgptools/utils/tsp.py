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

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.metrics import pairwise_distances
from shapely.geometry import LineString
import numpy as np


def run_tsp(nodes, 
            num_vehicles=1, 
            max_dist=25, 
            depth=1, 
            resample=None, 
            start_idx=None,
            end_idx=None):
    """Method to run TSP/VRP with arbitrary start and end nodes, 
    and without any distance constraint
    
    Args:
        nodes (ndarray): (# nodes, n_dim); Nodes to visit 
        num_vehicles (int): Number of robots/vehicles
        max_dist (float): Maximum distance allowed for each path when handling mutli-robot case
        depth (int): Internal parameter used to track re-try recursion depth
        resample (int): Each solution path will be resampled to have
                        `resample` number of points
        start_idx (list): Optionl list of start node indices from which to start the solution path 
        end_idx (list): Optionl list of end node indices from which to start the solution path 

    Returns:
        paths (ndarray): Solution paths
        distances (list): List of path lengths
    """
    if depth > 5:
        print('Warning: Max depth reached')
        return None, None
        
    # Add dummy 0 location to get arbitrary start and end node sols
    if start_idx is None or end_idx is None:
        distance_mat = np.zeros((len(nodes)+1, len(nodes)+1))
        distance_mat[1:, 1:] = pairwise_distances(nodes, nodes)*1e4
        trim_paths = True
    else:
        distance_mat = pairwise_distances(nodes, nodes)*1e4
        trim_paths = False
    distance_mat = distance_mat.astype(int)
    max_dist = int(max_dist*1e4)

    if start_idx is None:
        start_idx = [0]*num_vehicles
    elif trim_paths:
        start_idx = [i+1 for i in start_idx]

    if end_idx is None:
        end_idx = [0]*num_vehicles
    elif trim_paths:
        end_idx = [i+1 for i in end_idx]

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_mat[from_node][to_node]

    # num_locations, num vehicles, start, end
    manager = pywrapcp.RoutingIndexManager(len(distance_mat), 
                                           num_vehicles, 
                                           start_idx,
                                           end_idx)
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
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 10
    solution = routing.SolveWithParameters(search_parameters)
       
    paths = None
    if solution is not None:
        paths, distances = get_routes(manager, routing, 
                                      solution, num_vehicles, 
                                      start_idx, end_idx, trim_paths)
        for path in paths:
            if len(path) < 2:
                print('TSP Warning: Empty path detected')
                return run_tsp(nodes, num_vehicles, int(np.mean(distances)*(1.5/depth)), depth+1)
    else:
        print('TSP Warning: No solution found')
        return run_tsp(nodes, num_vehicles, int(max_dist*1.5), depth+1)

    # Map paths from node indices to node locations
    paths = [nodes[path] for path in paths]

    # Resample each solution path to have resample number of points
    if resample is not None:
        paths = np.array([resample_path(path, resample) for path in paths])

    # Convert distances back to floats in the original scale of the nodes
    distances = np.array(distances)/1e4
    return paths, distances
    
'''
Method to extract route from or-tools solution
'''
def get_routes(manager, routing, solution, num_vehicles, start_idx, end_idx, trim_paths):
    paths = []
    distances = []
    for vehicle_id in range(num_vehicles):
        path = []
        route_distance = 0
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        path.append(manager.IndexToNode(index))
        distances.append(route_distance)
        # remove dummy start/end point
        if trim_paths:
            path = np.array(path)-1
            if start_idx[vehicle_id] == 0:
                path = path[1:]
            if end_idx[vehicle_id] == 0:
                path = path[:-1]
        paths.append(path)
    return paths, distances

def resample_path(waypoints, num_inducing=10):
    """Function to map path with arbitrary number of waypoints to 
    inducing points path with fixed number of waypoints

    Args:
        waypoints (ndarray): (num_waypoints, n_dim); waypoints of path from vrp solver
        num_inducing (int): Number of inducing points (waypoints) in the returned path
        
    Returns:
        points (ndarray): (num_inducing, n_dim); Resampled path
    """
    line = LineString(waypoints)
    distances = np.linspace(0, line.length, num_inducing)
    points = [line.interpolate(distance) for distance in distances]
    points = np.array([[p.x, p.y] for p in points])
    return points