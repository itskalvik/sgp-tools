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

from sgptools.models.continuous_sgp import *
from sgptools.models.core.augmented_gpr import *
from sgptools.models.core.transformations import *

gpflow.config.set_default_likelihood_positive_minimum(1e-4)

np.random.seed(1234)
tf.random.set_seed(1234)


def main(dataset_path, 
         num_mc, 
         num_robots, 
         max_dist, 
         sampling_rate, 
         xrange,
         optimizers,
         distance_budget):
    dataset = dataset_path.split('/')[-1][:-4]
    print(f'Dataset: {dataset}')
    print(f'Num MC: {num_mc}')
    print(f'Num Robots: {num_robots}')
    print(f'Sampling Rate: {sampling_rate}')
    print(f'Dataset Path: {dataset_path}')
    print(f'Range: {xrange}')
    print(f'Distance Budget: {distance_budget}')
    print('optimizers:')
    for optimizer in optimizers:
        print(f'- {optimizer}')

    fname = f'optimizers_{dataset}_{num_robots}R_{sampling_rate}S'
    if distance_budget:
        fname += '_B'

    # Configure discrete/continuous sensing robot model
    if sampling_rate > 2:
        continuous_ipp = True
    else:
        continuous_ipp = False

    # Load the dataset
    dataset = Dataset(dataset_path)
    
    # Get oracle hyperparameters to benchmark rmse
    start_time = time()
    _, noise_variance_opt, kernel_opt = get_model_params(*dataset.get_train(), 
                                                         print_params=True,
                                                         optimizer='scipy')
    end_time = time()
    gp_time = end_time - start_time
    print(f'GP Training Time: {gp_time:.2f}')

    results = dict()
    for num_waypoints in xrange:
        results[num_waypoints] = {o:defaultdict(list) for o in optimizers}

    for _ in range(num_mc):
        for num_waypoints in xrange:
            print(f'\nNum Waypoints: {num_waypoints}', flush=True)

            # Get random hyperparameters
            _, noise_variance, kernel = get_model_params(*dataset.get_train(), 
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
            Xu_init = get_inducing_pts(dataset.X_train, 
                                       num_waypoints*num_robots,
                                       random=True)
            Xu_init, _ = run_tsp(Xu_init, 
                                 num_vehicles=num_robots, 
                                 max_dist=max_dist, 
                                 resample=num_waypoints,
                                 time_limit=120 if num_robots > 1 else 60)

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
                budget = np.inf

            # ---------------------------------------------------------------------------------

            for optimizer in optimizers:
                start_time = time()
                backend = optimizer.split(".")[0]
                method = optimizer.split(".")[1]
                ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                                dataset.X_train, 
                                                noise_variance_opt, 
                                                kernel_opt,
                                                transform,
                                                Xu_init=Xu_init.reshape(-1, 2), 
                                                optimizer=backend,
                                                method=method)
                solution = ipp_sgpr.inducing_variable.Z.numpy()
                solution = solution.reshape(num_robots, num_waypoints, 2)
                end_time = time()
                ipp_time = end_time-start_time

                budget_constraint = ipp_sgpr.transform.constraints(ipp_sgpr.inducing_variable.Z)
                budget_satisfied = budget_constraint > -10.

                # ---------------------------------------------------------------------------------

                # Get the solution sensed data
                solution_X, solution_y = [], []
                for r in range(num_robots):
                    X_new, y_new = dataset.get_sensor_data(solution[r], 
                                                            continuous_sening=continuous_ipp,
                                                            max_samples=1500)
                    solution_X.extend(X_new)
                    solution_y.extend(y_new)
                solution_X = np.array(solution_X)
                solution_y = np.array(solution_y)

                # Get RMSE using the oracle hyperparameters
                if len(solution_X) > 0:
                    y_pred, y_var = get_reconstruction((solution_X, solution_y), 
                                                        dataset.X_test, 
                                                        noise_variance_opt, 
                                                        kernel_opt)
                    rmse = get_rmse(y_pred, dataset.y_test)
                    nlpd = get_nlpd(y_pred, dataset.y_test, y_var)
                    smse = get_smse(y_pred, dataset.y_test, y_var)
                else:
                    rmse = np.nan
                    nlpd = np.nan
                    smse = np.nan

                param_time = gp_time
                results[num_waypoints][optimizer]['ParamTime'].append(param_time)
                results[num_waypoints][optimizer]['IPPTime'].append(ipp_time)
                results[num_waypoints][optimizer]['RMSE'].append(rmse)
                results[num_waypoints][optimizer]['NLPD'].append(nlpd)
                results[num_waypoints][optimizer]['SMSE'].append(smse)
                if distance_budget:
                    results[num_waypoints][optimizer]['Constraint'].append(bool(budget_satisfied))

                print(f'\n{optimizer} Param Time: {param_time:.4f}')
                print(f'{optimizer} IPP Time: {ipp_time:.4f}')
                print(f'{optimizer} RMSE: {rmse:.4f}')
                print(f'{optimizer} SMSE: {smse:.4f}')
                print(f'{optimizer} NLPD: {nlpd:.4f}')
                if distance_budget:
                    print(f'{optimizer} Constraint: {budget_satisfied}')

            # Log the results to a json file
            with open(f'{fname}.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="SP/IPP benchmarking script")
    parser.add_argument("--num_mc", type=int, default=10)
    parser.add_argument("--num_robots", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--distance_budget", action='store_true')
    parser.add_argument("--dataset_path", type=str, 
                        default='../datasets/bathymetry/bathymetry.tif')
    args=parser.parse_args()

    # Set the maximum distance (for each path) for the TSP solver
    max_dist = 350 if args.num_robots==1 else 150

    # Limit maximum waypoints/placements for multi robot case
    max_range = 101 if args.num_robots==1 and args.sampling_rate==2 else 51
    xrange = range(5, max_range, 5)

    # optimizers to benchmark
    optimizers = ['scipy.CG',
                  'scipy.L-BFGS-B',
                  'scipy.BFGS',
                  'scipy.Newton-CG',
                  'tf.SGD',
                  'tf.Adam',
                  'tf.RMSprop',
                  'tf.Nadam']

    main(args.dataset_path, 
         args.num_mc, 
         args.num_robots, 
         max_dist, 
         args.sampling_rate, 
         xrange,
         optimizers,
         args.distance_budget)
