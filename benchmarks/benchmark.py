import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

from sgptools.models.cma_es import *
from sgptools.models.core.osgpr import *
from sgptools.models.continuous_sgp import *
from sgptools.models.core.augmented_gpr import *
from sgptools.models.core.transformations import *

gpflow.config.set_default_likelihood_positive_minimum(1e-4)

np.random.seed(1234)
tf.random.set_seed(1234)


'''
Online IPP (data collection phase is excluded from the total runtime)
The parameters GP is initilized with the parameters of the kernel and variance in ipp_model

Args:
    X_train: numpy array (n, 2), Inputs X data used to initilize the OSGPR inducing points
    ipp_model:  CMA_ES or SGPR model from sgp-tools
    Xu_init: numpy array (num_robots, num_waypoints, 2) Current/Initial Solution
    path2data: Function that takes the path and returns the data from the path
    continuous_ipp: bool, If True, model continuous sensing robots
    ipp_method: str, 'SGP' or 'CMA', method used for IPP
    param_method: str, 'GP' or 'SSGP', method used for parameter updates
    plot: bool, If True, all intermediate IPP solutions are plotted and saved to disk

Returns:
    sol_data_X: Locations X where the robot traversed
    sol_data_y: Ground truth label data from the dataset corresponding to sol_data_X
    total_time_param : Total runtime of the parameter update phase of the online IPP approach 
                       excluding the time taken to get the 
                       data collected along the solution paths.
    total_time_ipp : Total runtime of the IPP update phase of the online IPP approach 
                     excluding the time taken to get the 
                     data collected along the solution paths.
'''
def online_ipp(X_train, ipp_model, Xu_init, path2data, 
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

        if time_step == num_waypoints:
            break # Skip param and path update for the last waypoint

        # Init/update parameter model
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
            plt.scatter(X_train[:, 0], X_train[:, 1])
            for i in range(num_robots):
                plt.scatter(X_new[:, 0], X_new[:, 1])
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
                               method='CG')
            curr_sol = ipp_model.inducing_variable.Z
            curr_sol = ipp_model.transform.expand(curr_sol, 
                                                  expand_sensor_model=False).numpy()
        elif ipp_method == 'CMA':
            curr_sol = ipp_model.optimize(X_init=curr_sol,
                                          max_steps=5000)
        curr_sol = curr_sol.reshape(num_robots, num_waypoints, 2)
        end_time = time()
        total_time_ipp += end_time - start_time
        
    return np.array(sol_data_X), np.array(sol_data_y), total_time_param, total_time_ipp 


def main(dataset_type, dataset_path, num_mc, num_robots, max_dist, sampling_rate, xrange):
    dataset = dataset_path.split('/')[-1][:-4]
    print(f'Dataset: {dataset}')
    print(f'Num MC: {num_mc}')
    print(f'Num Robots: {num_robots}')
    print(f'Sampling Rate: {sampling_rate}')
    print(f'Dataset Path: {dataset_path}')
    print(f'Range: {xrange}')

    # Configure discrete/continuous sensing robot model
    if sampling_rate > 2:
        continuous_ipp = True
        path2data = lambda x : cont2disc(interpolate_path(x, sampling_rate=0.2), X, y)
    else:
        continuous_ipp = False
        path2data = lambda x : cont2disc(x, X, y)

    # Get the data
    X_train, y_train, X_test, y_test, candidates, X, y = get_dataset(dataset_type,
                                                                     dataset_path,
                                                                     num_train=1000)
    
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
        results[num_waypoints] = {'Adaptive-SGP':  defaultdict(list),
                                  'Adaptive-CMA-ES':  defaultdict(list),
                                  'Online-SGP': defaultdict(list),
                                  'Online-CMA-ES': defaultdict(list)}
        if continuous_ipp:
            results[num_waypoints]['Adaptive-Agg-SGP'] = defaultdict(list)
            results[num_waypoints]['Online-Agg-SGP'] = defaultdict(list)

    for _ in range(num_mc):
        for num_waypoints in xrange:
            print(f'\nNum Waypoints: {num_waypoints}', flush=True)

            # Get random hyperparameters
            _, noise_variance, kernel = get_model_params(X_train, y_train, 
                                                         max_steps=0,
                                                         print_params=False)
            # Set lower and upper limits on the hyperparameters
            kernel.variance = gpflow.Parameter(
                np.random.normal(1.0, 0.1),
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(0.1),
                    gpflow.utilities.to_default_float(20.0),
                ),
            )
            kernel.lengthscales = gpflow.Parameter(
                np.random.normal(1.0, 0.1),
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(0.1),
                    gpflow.utilities.to_default_float(20.0),
                ),
            )

            # Generate initial paths
            Xu_init = get_inducing_pts(X_train, num_waypoints*num_robots)
            Xu_init, dist = run_tsp(Xu_init, 
                                    num_vehicles=num_robots, 
                                    max_dist=max_dist, 
                                    resample=num_waypoints)
            print(f"Path length(s): {dist}")

            # Setup the IPP Transform
            transform = IPPTransform(num_robots=num_robots,
                                     sampling_rate=sampling_rate)
        
            # ---------------------------------------------------------------------------------

            # Adaptive SGP
            ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                         X_train, 
                                         noise_variance, 
                                         kernel,
                                         transform,
                                         Xu_init=Xu_init.reshape(-1, 2), 
                                         max_steps=0)
            online_X, online_y, param_time, ipp_time = online_ipp(X_train, 
                                                                  ipp_sgpr, 
                                                                  Xu_init,
                                                                  path2data,
                                                                  continuous_ipp,
                                                                  'SGP',
                                                                  'SSGP' if continuous_ipp else 'GP')
            # Get RMSE from oracle hyperparameters
            y_pred, _ = get_reconstruction((online_X, online_y), 
                                           X_test, 
                                           noise_variance_opt, 
                                           kernel_opt)
            rmse = get_rmse(y_pred, y_test)

            print(f'\nAdaptive-SGP Param Time: {param_time:.4f}')
            print(f'Adaptive-SGP IPP Time: {ipp_time:.4f}')
            print(f'Adaptive-SGP RMSE: {rmse:.4f}')
            results[num_waypoints]['Adaptive-SGP']['ParamTime'].append(gp_time)
            results[num_waypoints]['Adaptive-SGP']['IPPTime'].append(ipp_time)
            results[num_waypoints]['Adaptive-SGP']['RMSE'].append(rmse)

            # ---------------------------------------------------------------------------------

            # Adaptive SGP with covariance aggregation for continuous sensing
            if continuous_ipp:
                ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                             X_train, 
                                             noise_variance, 
                                             kernel,
                                             IPPTransform(num_robots=num_robots,
                                                          sampling_rate=sampling_rate,
                                                          aggregate_fov=True),
                                             Xu_init=Xu_init.reshape(-1, 2), 
                                             max_steps=0)
                online_X, online_y, param_time, ipp_time = online_ipp(X_train, 
                                                                     ipp_sgpr, 
                                                                     Xu_init,
                                                                     path2data,
                                                                     continuous_ipp,
                                                                     'SGP',
                                                                     'SSGP' if continuous_ipp else 'GP')
                # Get RMSE from oracle hyperparameters
                y_pred, _ = get_reconstruction((online_X, online_y), 
                                               X_test, 
                                               noise_variance_opt, 
                                               kernel_opt)
                rmse = get_rmse(y_pred, y_test)

                print(f'\nAdaptive-Agg-SGP Param Time: {param_time:.4f}')
                print(f'Adaptive-Agg-SGP IPP Time: {ipp_time:.4f}')
                print(f'Adaptive-Agg-SGP RMSE: {rmse:.4f}')
                results[num_waypoints]['Adaptive-Agg-SGP']['ParamTime'].append(gp_time)
                results[num_waypoints]['Adaptive-Agg-SGP']['IPPTime'].append(ipp_time)
                results[num_waypoints]['Adaptive-Agg-SGP']['RMSE'].append(rmse)
            
            # ---------------------------------------------------------------------------------

            # Adaptive CMA_ES
            cma_es = CMA_ES(candidates, 
                            noise_variance, 
                            kernel,
                            num_robots=num_robots,
                            transform=transform)
            online_X, online_y, param_time, ipp_time = online_ipp(X_train, 
                                                                  cma_es, 
                                                                  Xu_init,
                                                                  path2data,
                                                                  continuous_ipp,
                                                                  'CMA',
                                                                  'SSGP' if continuous_ipp else 'GP')
            # Get RMSE from oracle hyperparameters
            y_pred, _ = get_reconstruction((online_X, online_y), 
                                           X_test, 
                                           noise_variance_opt, 
                                           kernel_opt)
            rmse = get_rmse(y_pred, y_test)

            print(f'\nAdaptive-CMA-ES Param Time: {param_time:.4f}')
            print(f'Adaptive-CMA-ES IPP Time: {ipp_time:.4f}')
            print(f'Adaptive-CMA-ES RMSE: {rmse:.4f}')
            results[num_waypoints]['Adaptive-CMA-ES']['ParamTime'].append(gp_time)
            results[num_waypoints]['Adaptive-CMA-ES']['IPPTime'].append(ipp_time)
            results[num_waypoints]['Adaptive-CMA-ES']['RMSE'].append(rmse)

            # ---------------------------------------------------------------------------------

            # Online SGP
            start_time = time()
            ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                         X_train, 
                                         noise_variance_opt, 
                                         kernel_opt,
                                         transform,
                                         Xu_init=Xu_init.reshape(-1, 2), 
                                         optimizer='scipy')
            offline_sgp_sol = ipp_sgpr.inducing_variable.Z.numpy()
            offline_sgp_sol = offline_sgp_sol.reshape(num_robots, num_waypoints, 2)
            end_time = time()
            ipp_time = end_time-start_time

            # Get RMSE from oracle hyperparameters
            offline_X, offline_y = [], []
            for r in range(num_robots):
                X_new, y_new = path2data(offline_sgp_sol[r])
                offline_X.extend(X_new)
                offline_y.extend(y_new)
            offline_X = np.array(offline_X)
            offline_y = np.array(offline_y)
            y_pred, _ = get_reconstruction((offline_X, offline_y), 
                                           X_test, 
                                           noise_variance_opt, 
                                           kernel_opt)
            rmse = get_rmse(y_pred, y_test)

            print(f'\nOnline-SGP Time: {ipp_time:.4f}')
            print(f'Online-SGP RMSE: {rmse:.4f}')
            results[num_waypoints]['Online-SGP']['ParamTime'].append(gp_time)
            results[num_waypoints]['Online-SGP']['IPPTime'].append(ipp_time)
            results[num_waypoints]['Online-SGP']['RMSE'].append(rmse)

            # ---------------------------------------------------------------------------------

            # Online SGP with covariance aggregation for continuous sensing
            if continuous_ipp:
                start_time = time()
                ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                             X_train, 
                                             noise_variance_opt, 
                                             kernel_opt,
                                             IPPTransform(num_robots=num_robots,
                                                          sampling_rate=sampling_rate,
                                                          aggregate_fov=True),
                                             Xu_init=Xu_init.reshape(-1, 2), 
                                             optimizer='scipy')
                offline_sgp_sol = ipp_sgpr.inducing_variable.Z.numpy()
                offline_sgp_sol = offline_sgp_sol.reshape(num_robots, num_waypoints, 2)
                end_time = time()
                ipp_time = end_time-start_time

                # Get RMSE from oracle hyperparameters
                offline_X, offline_y = [], []
                for r in range(num_robots):
                    X_new, y_new = path2data(offline_sgp_sol[r])
                    offline_X.extend(X_new)
                    offline_y.extend(y_new)
                offline_X = np.array(offline_X)
                offline_y = np.array(offline_y)
                y_pred, _ = get_reconstruction((offline_X, offline_y), 
                                               X_test, 
                                               noise_variance_opt, 
                                               kernel_opt)
                rmse = get_rmse(y_pred, y_test)

                print(f'\nOnline-Agg-SGP Time: {ipp_time:.4f}')
                print(f'Online-Agg-SGP Cov RMSE: {rmse:.4f}')
                results[num_waypoints]['Online-Agg-SGP']['ParamTime'].append(gp_time)
                results[num_waypoints]['Online-Agg-SGP']['IPPTime'].append(ipp_time)
                results[num_waypoints]['Online-Agg-SGP']['RMSE'].append(rmse)

            # ---------------------------------------------------------------------------------

            # Online CMA-ES
            start_time = time()
            cma_es = CMA_ES(candidates, 
                            noise_variance_opt,
                            kernel_opt, 
                            num_robots=num_robots,
                            transform=transform)
            cma_sol = cma_es.optimize(X_init=Xu_init, 
                                      max_steps=5000)
            cma_sol = cma_sol.reshape(num_robots, num_waypoints, 2)
            end_time = time()
            ipp_time = end_time-start_time

            # Get RMSE from oracle hyperparameters
            cma_X, cma_y = [], []
            for r in range(num_robots):
                X_new, y_new = path2data(cma_sol[r])
                cma_X.extend(X_new)
                cma_y.extend(y_new)
            cma_X = np.array(cma_X)
            cma_y = np.array(cma_y)
            y_pred, _ = get_reconstruction((cma_X, cma_y), 
                                           X_test, 
                                           noise_variance_opt, 
                                           kernel_opt)
            rmse = get_rmse(y_pred, y_test)

            print(f'\nOnline-CMA-ES Time: {ipp_time:.4f}')
            print(f'Online-CMA-ES RMSE: {rmse:.4f}')
            results[num_waypoints]['Online-CMA-ES']['ParamTime'].append(gp_time)
            results[num_waypoints]['Online-CMA-ES']['IPPTime'].append(ipp_time)
            results[num_waypoints]['Online-CMA-ES']['RMSE'].append(rmse)

            # ---------------------------------------------------------------------------------

            # Dump the results to a json file
            with open(f'{dataset}_{num_robots}R_{sampling_rate}S.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="SP/IPP benchmarking script")
    parser.add_argument("--num_mc", type=int, default=10)
    parser.add_argument("--num_robots", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--dataset_path", type=str, 
                        default='datasets/bathymetry/bathymetry.tif')
    args=parser.parse_args()

    max_dist = 350 if args.num_robots==1 else 150
    dataset_type = 'tif'
    max_range = 101 if args.num_robots==1 and args.sampling_rate==2 else 51
    xrange = range(5, max_range, 5)
    main(dataset_type, 
         args.dataset_path, 
         args.num_mc, 
         args.num_robots, 
         max_dist, 
         args.sampling_rate, 
         xrange)
