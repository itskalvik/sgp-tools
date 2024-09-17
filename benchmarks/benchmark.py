import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import json
import argparse
import numpy as np
from time import time
from copy import deepcopy
from collections import defaultdict
import tensorflow_probability as tfp

from sgptools.utils.data import *
from sgptools.utils.misc import *
from sgptools.utils.metrics import *
from sgptools.utils.tsp import run_tsp
from sgptools.utils.gpflow import get_model_params

from sgptools.models.bo import *
from sgptools.models.cma_es import *
from sgptools.models.greedy_mi import *
from sgptools.models.greedy_sgp import *
from sgptools.models.core.osgpr import *
from sgptools.models.continuous_sgp import *
from sgptools.models.core.augmented_gpr import *
from sgptools.models.core.transformations import *

gpflow.config.set_default_likelihood_positive_minimum(1e-4)

np.random.seed(1234)
tf.random.set_seed(1234)


'''
Adaptive IPP (data collection phase is excluded from the total runtime)
The hyperparameters GP is initilized with the hyperparameters of the kernel and variance in ipp_model

Args:
    X_train: numpy array (n, 2), Inputs X data used to initilize the OSGPR inducing points
    ipp_model:  CMA_ES or SGPR model from sgp-tools
    Xu_init: numpy array (num_robots, num_waypoints, 2) Current/Initial Solution
    path2data: Function that takes the path and returns the data from the path
    continuous_ipp: bool, If True, model continuous sensing robots
    ipp_method: str, 'SGP' or 'CMA', method used for IPP
    param_method: str, 'GP' or 'SSGP', method used for hyperparameter updates
    plot: bool, If True, all intermediate IPP solutions are plotted and saved to disk

Returns:
    sol_data_X: Locations X where the robot traversed
    sol_data_y: Ground truth label data from the dataset corresponding to sol_data_X
    total_time_param : Total runtime of the hyperparameter update phase of the online IPP approach 
                       excluding the time taken to get the 
                       data collected along the solution paths.
    total_time_ipp : Total runtime of the IPP update phase of the online IPP approach 
                     excluding the time taken to get the 
                     data collected along the solution paths.
'''
def run_aipp(X_train, ipp_model, Xu_init, path2data, 
             continuous_ipp=False, 
             ipp_method='SGP', 
             param_method='GP',
             plot=False):
    total_time_param = 0
    total_time_ipp = 0
    num_robots = Xu_init.shape[0]
    num_waypoints = Xu_init.shape[1]
    curr_sol = Xu_init

    if continuous_ipp:
        offset = 2
    else:
        offset = 1

    init_kernel = deepcopy(ipp_model.kernel)
    try: # SGP
        init_noise_variance = ipp_model.likelihood.variance
    except: # CMA
        init_noise_variance = ipp_model.noise_variance

    if param_method=='SSGP':
        param_model = init_osgpr(X_train, 
                                 num_inducing=40, 
                                 kernel=init_kernel,
                                 noise_variance=init_noise_variance)
        
    sol_data_X = []
    sol_data_y = []
    for time_step in range(offset, num_waypoints+1):
        # Get the new data
        last_visited = curr_sol.copy()[:, time_step-offset:time_step]
        data_X_batch = []
        data_y_batch = []
        for r in range(num_robots):
            X_new, y_new = path2data(last_visited[r])
            data_X_batch.extend(X_new)
            data_y_batch.extend(y_new)            
            sol_data_X.extend(X_new)
            sol_data_y.extend(y_new)

        # Skip param and path update if no data was collected
        if len(data_X_batch) == 0:
            continue

        # Skip param and path update for the last waypoint
        if time_step == num_waypoints:
            break

        # Init/update hyperparameters model
        start_time = time()
        if param_method=='GP':
            # Starting from initial params ensures recovery from bad params
            _, noise_variance, kernel = get_model_params(np.array(sol_data_X), 
                                                         np.array(sol_data_y),
                                                         kernel=deepcopy(init_kernel),
                                                         noise_variance=init_noise_variance,
                                                         print_params=False,
                                                         optimizer='scipy',
                                                         method='CG')
            # Clip to avoid floats being interpreted as NANs
            noise_variance = np.clip(noise_variance.numpy(), 1e-4, 5.0)
        elif param_method=='SSGP':
            param_model.update((np.array(data_X_batch), 
                                np.array(data_y_batch)))
            optimize_model(param_model, 
                           trainable_variables=param_model.trainable_variables[1:], 
                           optimizer='scipy',
                           method='CG')
            noise_variance = param_model.likelihood.variance
            kernel = param_model.kernel
        end_time = time()
        total_time_param += end_time - start_time

        if plot:
            plt.figure()
            plt.scatter(X_train[:, 0], X_train[:, 1], s=1)
            for i in range(num_robots):
                plt.scatter(np.array(data_X_batch)[:, 0], 
                            np.array(data_X_batch)[:, 1])
                plt.plot(curr_sol[i, :, 0], curr_sol[i, :, 1])
            plt.savefig(f'{ipp_method}-{time_step}.png')
            plt.close()

        # SGP-IPP update
        Xu_visited = curr_sol.copy()[:, :time_step]
        ipp_model.transform.update_Xu_fixed(Xu_visited)
        ipp_model.update(noise_variance, kernel)
        start_time = time()
        if ipp_method == 'SGP':
            _ = optimize_model(ipp_model,
                               kernel_grad=False, 
                               optimizer='scipy',
                               method='CG', 
                               max_steps=200)
            curr_sol = ipp_model.inducing_variable.Z
            curr_sol = ipp_model.transform.expand(curr_sol, 
                                                  expand_sensor_model=False).numpy()
        elif ipp_method == 'CMA':
            curr_sol = ipp_model.optimize(X_init=curr_sol,
                                          max_steps=500)
        curr_sol = curr_sol.reshape(num_robots, num_waypoints, 2)
        end_time = time()
        total_time_ipp += end_time - start_time

    budget_constraint = ipp_model.transform.constraints(curr_sol.reshape(-1, 2))
    slack = -10 if num_robots==1 else -300 # Use larger slack for multi-robot case
    budget_satisfied = budget_constraint > slack

    return np.array(sol_data_X), np.array(sol_data_y), total_time_param, total_time_ipp, budget_satisfied


def main(dataset_path, 
         num_mc, 
         num_robots, 
         max_dist, 
         sampling_rate, 
         xrange,
         methods,
         distance_budget):
    dataset = dataset_path.split('/')[-1][:-4]
    print(f'Dataset: {dataset}')
    print(f'Num MC: {num_mc}')
    print(f'Num Robots: {num_robots}')
    print(f'Sampling Rate: {sampling_rate}')
    print(f'Dataset Path: {dataset_path}')
    print(f'Range: {xrange}')
    print(f'Distance Budget: {distance_budget}')
    print('Methods:')
    for method in methods:
        print(f'-{method}')

    fname = f'{dataset}_{num_robots}R_{sampling_rate}S'
    if distance_budget:
        fname += '_B'

    if 'Greedy' in ''.join(methods):
        fname += '_D'

    # Configure discrete/continuous sensing robot model
    if sampling_rate > 2:
        continuous_ipp = True
        path2data = lambda x : cont2disc(interpolate_path(x, sampling_rate=0.2), X, y)
    else:
        continuous_ipp = False
        path2data = lambda x : cont2disc(x, X, y)

    # Get the data
    X_train, y_train, X_test, y_test, candidates, X, y = get_dataset(dataset_path)
    
    # Get oracle hyperparameters to benchmark rmse
    start_time = time()
    _, noise_variance_opt, kernel_opt = get_model_params(X_train, y_train, 
                                                         print_params=True,
                                                         optimizer='scipy')
    end_time = time()
    gp_time = end_time - start_time
    print(f'GP Training Time: {gp_time:.2f}')

    results = dict()
    for num_waypoints in xrange:
        results[num_waypoints] = {m:defaultdict(list) for m in methods}

    for _ in range(num_mc):
        for num_waypoints in xrange:
            print(f'\nNum Waypoints: {num_waypoints}', flush=True)

            # Get random hyperparameters
            _, noise_variance, kernel = get_model_params(X_train, y_train, 
                                                         max_steps=0,
                                                         print_params=False)
            
            # Sample random hyperparameters and
            # set lower and upper limits on the hyperparameters
            kernel.variance = gpflow.Parameter(
                np.random.normal(1.0, 0.25),
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(0.1),
                    gpflow.utilities.to_default_float(20.0),
                ),
            )
            kernel.lengthscales = gpflow.Parameter(
                np.random.normal(1.0, 0.25),
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(0.1),
                    gpflow.utilities.to_default_float(20.0),
                ),
            )

            # Generate initial paths
            Xu_init = get_inducing_pts(X_train, num_waypoints*num_robots)
            Xu_init, _ = run_tsp(Xu_init, 
                                 num_vehicles=num_robots, 
                                 max_dist=max_dist, 
                                 resample=num_waypoints,
                                 time_limit=30)

            # Setup the IPP Transform
            transform = IPPTransform(num_robots=num_robots,
                                     sampling_rate=sampling_rate)
            distances = transform.distance(Xu_init.reshape(-1, 2))
            print(f"Path length(s): {distances}")
    
            if distance_budget:
                # Set the distance budget to the length of the shortest path minus 5.0
                budget = np.min(distances)-5.0
                print(f'Distance Budget: {budget:.4f}')
                transform.distance_budget = budget
                transform.constraint_weight = 250.
            else:
                budget_satisfied=True

            # ---------------------------------------------------------------------------------

            for method in methods:
                if method=='Adaptive-SGP':
                    ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                                 X_train, 
                                                 noise_variance, 
                                                 kernel,
                                                 deepcopy(transform),
                                                 Xu_init=Xu_init.reshape(-1, 2), 
                                                 max_steps=0)
                    solution_X, solution_y, param_time, ipp_time, budget_satisfied = run_aipp(X_train, 
                                                                                ipp_sgpr, 
                                                                                Xu_init,
                                                                                path2data,
                                                                                continuous_ipp,
                                                                                'SGP',
                                                                                'SSGP' if continuous_ipp else 'GP')

                # ---------------------------------------------------------------------------------

                if method=='Adaptive-CMA-ES':
                    cma_es = CMA_ES(candidates, 
                                    noise_variance, 
                                    kernel,
                                    num_robots=num_robots,
                                    transform=deepcopy(transform))
                    solution_X, solution_y, param_time, ipp_time, budget_satisfied = run_aipp(X_train, 
                                                                            cma_es, 
                                                                            Xu_init,
                                                                            path2data,
                                                                            continuous_ipp,
                                                                            'CMA',
                                                                            'SSGP' if continuous_ipp else 'GP')
                    
                # ---------------------------------------------------------------------------------

                if method=='SGP':
                    start_time = time()
                    ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                                 X_train, 
                                                 noise_variance_opt, 
                                                 kernel_opt,
                                                 transform,
                                                 Xu_init=Xu_init.reshape(-1, 2), 
                                                 optimizer='scipy')
                    solution = ipp_sgpr.inducing_variable.Z.numpy()
                    solution = solution.reshape(num_robots, num_waypoints, 2)
                    end_time = time()
                    ipp_time = end_time-start_time

                    budget_constraint = ipp_sgpr.transform.constraints(ipp_sgpr.inducing_variable.Z)
                    budget_satisfied = budget_constraint > -10.

                    solution_X, solution_y = [], []
                    for r in range(num_robots):
                        X_new, y_new = path2data(solution[r])
                        solution_X.extend(X_new)
                        solution_y.extend(y_new)
                    solution_X = np.array(solution_X)
                    solution_y = np.array(solution_y)

                # ---------------------------------------------------------------------------------

                if method=='Discrete-SGP':
                    start_time = time()
                    ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                                 X_train, 
                                                 noise_variance_opt, 
                                                 kernel_opt,
                                                 transform,
                                                 Xu_init=Xu_init.reshape(-1, 2), 
                                                 optimizer='scipy')
                    solution = ipp_sgpr.inducing_variable.Z.numpy()
                    solution = cont2disc(solution, candidates)
                    solution = solution.reshape(num_robots, num_waypoints, 2)
                    end_time = time()
                    ipp_time = end_time-start_time

                    budget_constraint = ipp_sgpr.transform.constraints(ipp_sgpr.inducing_variable.Z)
                    budget_satisfied = budget_constraint > -10.

                    solution_X, solution_y = [], []
                    for r in range(num_robots):
                        X_new, y_new = path2data(solution[r])
                        solution_X.extend(X_new)
                        solution_y.extend(y_new)
                    solution_X = np.array(solution_X)
                    solution_y = np.array(solution_y)

                # ---------------------------------------------------------------------------------

                if method=='CMA-ES':
                    start_time = time()
                    cma_es = CMA_ES(candidates, 
                                    noise_variance_opt,
                                    kernel_opt, 
                                    num_robots=num_robots,
                                    transform=transform)
                    solution = cma_es.optimize(X_init=Xu_init, 
                                               max_steps=500)
                    solution = solution.reshape(num_robots, num_waypoints, 2)
                    end_time = time()
                    ipp_time = end_time-start_time

                    budget_constraint = cma_es.transform.constraints(solution.reshape(-1, 2))
                    budget_satisfied = budget_constraint > -10.

                    solution_X, solution_y = [], []
                    for r in range(num_robots):
                        X_new, y_new = path2data(solution[r])
                        solution_X.extend(X_new)
                        solution_y.extend(y_new)
                    solution_X = np.array(solution_X)
                    solution_y = np.array(solution_y)

                # ---------------------------------------------------------------------------------

                if method=='BO':
                    start_time = time()
                    bo_model = BayesianOpt(candidates, 
                                           noise_variance_opt,
                                           kernel_opt, 
                                           transform=transform)
                    solution = bo_model.optimize(X_init=Xu_init,
                                                 max_steps=50)
                    solution = solution.reshape(num_robots, num_waypoints, 2)
                    end_time = time()
                    ipp_time = end_time-start_time

                    budget_constraint = bo_model.transform.constraints(solution.reshape(-1, 2))
                    budget_satisfied = budget_constraint > -10.

                    solution_X, solution_y = [], []
                    for r in range(num_robots):
                        X_new, y_new = path2data(solution[r])
                        solution_X.extend(X_new)
                        solution_y.extend(y_new)
                    solution_X = np.array(solution_X)
                    solution_y = np.array(solution_y)

                # ---------------------------------------------------------------------------------

                if method=='Greedy-MI':
                    start_time = time()
                    solution = get_greedy_mi_sol(num_robots*num_waypoints,
                                                 candidates, 
                                                 candidates, 
                                                 noise_variance_opt,
                                                 kernel_opt, 
                                                 transform=transform)
                    solution = solution.reshape(num_robots, num_waypoints, 2)
                    end_time = time()
                    ipp_time = end_time-start_time

                    solution_X, solution_y = [], []
                    for r in range(num_robots):
                        X_new, y_new = path2data(solution[r])
                        solution_X.extend(X_new)
                        solution_y.extend(y_new)
                    solution_X = np.array(solution_X)
                    solution_y = np.array(solution_y)

                # ---------------------------------------------------------------------------------

                if method=='Greedy-SGP':
                    start_time = time()
                    solution = get_greedy_sgp_sol(num_robots*num_waypoints,
                                                  candidates, 
                                                  candidates, 
                                                  noise_variance_opt,
                                                  kernel_opt, 
                                                  transform=transform)
                    solution = solution.reshape(num_robots, num_waypoints, 2)
                    end_time = time()
                    ipp_time = end_time-start_time

                    solution_X, solution_y = [], []
                    for r in range(num_robots):
                        X_new, y_new = path2data(solution[r])
                        solution_X.extend(X_new)
                        solution_y.extend(y_new)
                    solution_X = np.array(solution_X)
                    solution_y = np.array(solution_y)

                # ---------------------------------------------------------------------------------

                # Get RMSE using the oracle hyperparameters
                y_pred, _ = get_reconstruction((solution_X, solution_y), 
                                            X_test, 
                                            noise_variance_opt, 
                                            kernel_opt)
                rmse = get_rmse(y_pred, y_test)

                param_time = gp_time if 'Adaptive' not in method else param_time
                results[num_waypoints][method]['ParamTime'].append(param_time)
                results[num_waypoints][method]['IPPTime'].append(ipp_time)
                results[num_waypoints][method]['RMSE'].append(rmse)
                if distance_budget:
                    results[num_waypoints][method]['Constraint'].append(bool(budget_satisfied))

                print(f'\n{method} Param Time: {param_time:.4f}')
                print(f'{method} IPP Time: {ipp_time:.4f}')
                print(f'{method} RMSE: {rmse:.4f}')
                if distance_budget:
                    print(f'{method} Constraint: {budget_satisfied}')

            # Log the results to a json file
            with open(f'{fname}.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="SP/IPP benchmarking script")
    parser.add_argument("--num_mc", type=int, default=10)
    parser.add_argument("--num_robots", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--distance_budget", action='store_true')
    parser.add_argument("--benchmark_discrete", action='store_true')
    parser.add_argument("--dataset_path", type=str, 
                        default='../datasets/bathymetry/bathymetry.tif')
    args=parser.parse_args()

    # Set the maximum distance (for each path) for the TSP solver
    max_dist = 350 if args.num_robots==1 else 150

    # Limit maximum waypoints/placements for multi robot case
    max_range = 101 if args.num_robots==1 and args.sampling_rate==2 else 51
    xrange = range(5, max_range, 5)

    # Methods to benchmark
    methods = ['Adaptive-SGP',
               'Adaptive-CMA-ES',
               'SGP',
               'CMA-ES']

    # Benchmark BO & discrete methods if benchmark_discrete is True
    # and the remaining parameters are in their base cases
    if args.sampling_rate == 2 and args.num_robots == 1 and \
       not args.distance_budget and args.benchmark_discrete:
        methods = ['BO', 
                   'Greedy-MI',
                   'Greedy-SGP',
                   'Discrete-SGP']

    main(args.dataset_path, 
         args.num_mc, 
         args.num_robots, 
         max_dist, 
         args.sampling_rate, 
         xrange,
         methods,
         args.distance_budget)
