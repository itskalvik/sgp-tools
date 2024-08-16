import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
from time import time
from collections import defaultdict

from sgptools.utils.data import *
from sgptools.utils.misc import *
from sgptools.utils.metrics import *
from sgptools.utils.tsp import run_tsp
from sgptools.utils.gpflow import get_model_params

from sgptools.models.cma_es import *
from sgptools.models.continuous_sgp import *
from sgptools.models.core.augmented_gpr import *
from sgptools.models.core.transformations import *

np.random.seed(10)
tf.random.set_seed(10)


'''
Online IPP (data collection phase is excluded from the total runtime)
The parameters GP is initilized with the parameters of the kernel and variance in ipp_model

Args:
    X_train: numpy array (n, 2), Inputs X data used to initilize the OSGPR inducing points
    ipp_model:  CMA_ES or SGPR model from sgp-tools
    Xu_init: numpy array (num_robots, num_waypoints, 2) Current/Initial Solution
    path2data: Function that takes the path and returns the data from the path
    continuous_ipp: bool, if True, model continuous sensing robots
    method: str, 'SGP' or 'CMA', IPP method used for online IPP

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
               continuous_ipp=False, method='SGP'):
    total_time_param = 0
    total_time_ipp = 0
    num_robots = Xu_init.shape[0]
    num_waypoints = Xu_init.shape[1]
    curr_sol = Xu_init

    if continuous_ipp:
        offset = 2
    else:
        offset = 1

    sol_data_X = []
    sol_data_y = []
    for time_step in range(offset, num_waypoints+1):
        # Get the new data
        last_visited = curr_sol.copy()[:, time_step-offset:time_step]
        for r in range(num_robots):
            X_new, y_new = path2data(last_visited[r])
            sol_data_X.extend(X_new)
            sol_data_y.extend(y_new)

        if time_step == num_waypoints:
            break # Skip param update for the last waypoint

        # Init/update parameter GP
        # Starting from initial params ensures recovery from bad params
        start_time = time()
        _, noise_variance, kernel = get_model_params(np.array(sol_data_X), 
                                                     np.array(sol_data_y),
                                                     lengthscales=1.0,
                                                     variance=1.0,
                                                     noise_variance=1e-4,
                                                     print_params=False,
                                                     optimizer='scipy')
        end_time = time()
        total_time_param += end_time - start_time

        plt.figure()
        plt.scatter(X_train[:, 0], X_train[:, 1])
        for i in range(num_robots):
            plt.scatter(X_new[:, 0], X_new[:, 1])
        plt.savefig(f'test-{time_step}.png')
        plt.close()
        
        # SGP-IPP update
        Xu_visited = curr_sol.copy()[:, :time_step]
        ipp_model.transform.update_Xu_fixed(Xu_visited)
        ipp_model.update(noise_variance, kernel)
        start_time = time()
        if method == 'SGP':
            _ = optimize_model(ipp_model,
                               kernel_grad=False, 
                               method='CG',
                               optimizer='scipy')
            curr_sol = ipp_model.inducing_variable.Z
            curr_sol = ipp_model.transform.expand(curr_sol, 
                                                  expand_sensor_model=False).numpy()
            curr_sol = project_waypoints(curr_sol, X_train)
        elif method == 'CMA':
            curr_sol = ipp_model.optimize(X_init=curr_sol,
                                          max_steps=5000)
        curr_sol = curr_sol.reshape(num_robots, num_waypoints, 2)
        end_time = time()
        total_time_ipp += end_time - start_time
        
    return np.array(sol_data_X), np.array(sol_data_y), total_time_param, total_time_ipp 


def main(dataset, dataset_path, num_mc, num_robots, max_dist, sampling_rate):
    print(f'Dataset: {dataset}')
    print(f'Num MC: {num_mc}')
    print(f'Num Robots: {num_robots}')
    print(f'Sampling Rate: {sampling_rate}')

    # Configure discrete/continuous sensing robot model
    if sampling_rate > 2:
        continuous_ipp = True
        path2data = lambda x : cont2disc(interpolate_path(x, sampling_rate=0.1), X, y)
    else:
        continuous_ipp = False
        path2data = lambda x : cont2disc(x, X, y)

    # Get the data
    X_train, y_train, X_test, y_test, candidates, X, y = get_dataset(dataset, dataset_path,
                                                                     num_train=1000)
    
    # Get oracle hyperparameters to benchmark rmse
    start_time = time()
    _, noise_variance_opt, kernel_opt = get_model_params(X_train, y_train, 
                                                         print_params=True,
                                                         optimizer='scipy',
                                                         method='CG')
    end_time = time()
    gp_time = end_time - start_time
    print(f'GP Training Time: {gp_time:.2f}')

    results_mc = dict()
    for num_waypoints in range(5, 101, 5):
        print(f'\nNum Waypoints: {num_waypoints}')
        
        # Get random hyperparameters
        _, noise_variance, kernel = get_model_params(X_train, y_train, 
                                                     lengthscales=1.0,
                                                     variance=1.0,
                                                     noise_variance=1e-4,
                                                     max_steps=0,
                                                     print_params=False)

        # Generate initial paths
        Xu_init = get_inducing_pts(X_train, num_waypoints*num_robots)
        Xu_init, _ = run_tsp(Xu_init, 
                             num_vehicles=num_robots, 
                             max_dist=max_dist, 
                             resample=num_waypoints)
        
        # Setup the IPP Transform
        transform = IPPTransform(num_robots=num_robots,
                                 sampling_rate=sampling_rate)
        
        results_param_time = defaultdict(list)
        results_ipp_time = defaultdict(list)
        results_rmse = defaultdict(list)

        # ---------------------------------------------------------------------------------

        # Online Continuous SGP
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
                                                              'SGP')
        # Get RMSE from oracle hyperparameters
        y_pred, _ = get_reconstruction((online_X, online_y), 
                                       X_test, 
                                       noise_variance_opt, 
                                       kernel_opt)
        online_sgp_rmse = get_rmse(y_pred, y_test)

        print(f'\nOnline SGP Param Time: {param_time:.4f}')
        print(f'Online SGP IPP Time: {ipp_time:.4f}')
        print(f'Online SGP RMSE: {online_sgp_rmse:.4f}')
        results_param_time['online_sgp'].append(param_time)
        results_ipp_time['online_sgp'].append(ipp_time)
        results_rmse['online_sgp'].append(online_sgp_rmse)

        # ---------------------------------------------------------------------------------

        # Online Param CMA_ES
        cma_es = CMA_ES(candidates, noise_variance, kernel, 
                        num_robots=num_robots,
                        transform=transform)
        online_X, online_y, param_time, ipp_time = online_ipp(X_train, 
                                                              cma_es, 
                                                              Xu_init,
                                                              path2data,
                                                              continuous_ipp,
                                                              'CMA')
        # Get RMSE from oracle hyperparameters
        y_pred, _ = get_reconstruction((online_X, online_y), 
                                       X_test, 
                                       noise_variance_opt, 
                                       kernel_opt)
        online_cma_rmse = get_rmse(y_pred, y_test)

        print(f'\nOnline CMA Param Time: {param_time:.4f}')
        print(f'Online CMA IPP Time: {ipp_time:.4f}')
        print(f'Online CMA RMSE: {online_cma_rmse:.4f}')
        results_param_time['online_cma'].append(param_time)
        results_ipp_time['online_cma'].append(ipp_time)
        results_rmse['online_cma'].append(online_cma_rmse)
        
        # ---------------------------------------------------------------------------------

        # Oracle Offline Continuous SGP
        start_time = time()
        ipp_sgpr, _ = continuous_sgp(num_waypoints, 
                                     X_train, 
                                     noise_variance_opt, 
                                     kernel_opt,
                                     transform,
                                     Xu_init=Xu_init.reshape(-1, 2), 
                                     optimizer='scipy',
                                     method='CG')
        offline_sgp_sol = ipp_sgpr.inducing_variable.Z.numpy()
        offline_sgp_sol = project_waypoints(offline_sgp_sol, X_train)
        offline_sgp_sol = offline_sgp_sol.reshape(num_robots, num_waypoints, 2)
        end_time = time()
        offline_sgp_time = end_time-start_time

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
        offline_sgp_rmse = get_rmse(y_pred, y_test)

        print(f'\nOracle Offline SGP Time: {offline_sgp_time:.4f}')
        print(f'Oracle Offline SGP RMSE: {offline_sgp_rmse:.4f}')
        results_param_time['oracle_offline_sgp'].append(gp_time)
        results_ipp_time['oracle_offline_sgp'].append(offline_sgp_time)
        results_rmse['oracle_offline_sgp'].append(offline_sgp_rmse)

        # ---------------------------------------------------------------------------------

        # Oracle Offline CMA_ES
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
        cma_time = end_time-start_time

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
        cma_rmse = get_rmse(y_pred, y_test)

        print(f'\nOracle Offline CMA Time: {cma_time:.4f}')
        print(f'Oracle Offline CMA RMSE: {cma_rmse:.4f}')
        results_param_time['oracle_offline_cma'].append(gp_time)
        results_ipp_time['oracle_offline_cma'].append(cma_time)
        results_rmse['oracle_offline_cma'].append(cma_rmse)

        # ---------------------------------------------------------------------------------

        # Dump the results to a json file
        results_mc[num_waypoints] = {'RMSE': results_rmse,
                                     'Param-Time': results_param_time,
                                     'IPP-Time': results_ipp_time}
        with open(f'AIPP_{num_robots}R_{dataset}.json', 'w', encoding='utf-8') as f:
            json.dump(results_mc, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    dataset = 'tif'
    dataset_path = 'datasets/elevation/ak2023_wrangell_dem_J1054332_001_000-1.tif'
    num_mc = 1
    num_robots = 1
    max_dist = 100
    sampling_rate = 5
    
    main(dataset, dataset_path, num_mc, num_robots, max_dist, sampling_rate)