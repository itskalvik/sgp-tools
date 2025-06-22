import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import json
import argparse
import numpy as np
from time import time
from collections import defaultdict

from sgptools.utils.data import *
from sgptools.utils.misc import *
from sgptools.utils.metrics import *
from sgptools.utils.tsp import run_tsp
from sgptools.utils.gpflow import get_model_params

from sgptools.methods import *
from sgptools.core.transformations import *

gpflow.config.set_default_likelihood_positive_minimum(1e-4)
np.random.seed(1234)
tf.random.set_seed(1234)


class IPPBenchmark:
    def __init__(self, 
                 dataset_path, 
                 num_mc, 
                 num_robots, 
                 max_dist, 
                 sampling_rate, 
                 xrange,
                 methods,
                 distance_budget,
                 tsp_time_limit=30,
                 verbose=False):
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
            print(f'- {method}')

        self.num_mc = num_mc
        self.num_robots = num_robots
        self.sampling_rate = sampling_rate
        self.xrange = xrange
        self.distance_budget = distance_budget
        self.tsp_time_limit = tsp_time_limit
        self.max_dist = max_dist
        self.verbose = verbose
    
        self.fname = f'IPP-{dataset}_{num_robots}R_{sampling_rate}S'
        if distance_budget:
            fname += '_B'

        # Configure discrete/continuous sensing robot model
        if self.sampling_rate > 2:
            self.continuous_sening = True
        else:
            self.continuous_sening = False

        # Load the dataset
        self.dataset = Dataset(dataset_path)
        
        # Get oracle hyperparameters to benchmark rmse
        start_time = time()
        _, self.noise_variance_opt, self.kernel_opt = get_model_params(*self.dataset.get_train(), 
                                                            verbose=True)
        end_time = time()
        self.gp_time = end_time - start_time
        print(f'GP Training Time: {self.gp_time:.2f}')

        self.results = dict()
        for num_waypoints in self.xrange:
            self.results[num_waypoints] = {m:defaultdict(list) for m in methods}

    def run(self):
        for _ in range(self.num_mc):
            for num_waypoints in self.xrange:
                print(f'\nNum Waypoints: {num_waypoints}', flush=True)

                # Generate initial paths
                Xu_init = get_inducing_pts(self.dataset.X_train, 
                                        num_waypoints*self.num_robots,
                                        random=True)
                Xu_init, _ = run_tsp(Xu_init, 
                                    num_vehicles=self.num_robots, 
                                    max_dist=self.max_dist, 
                                    resample=num_waypoints,
                                    time_limit=self.tsp_time_limit)

                # Setup the IPP Transform
                transform = IPPTransform(num_robots=self.num_robots,
                                         sampling_rate=self.sampling_rate)
                distances = transform.distance(Xu_init.reshape(-1, 2))
                print(f"Path length(s): {distances}")
        
                if self.distance_budget:
                    # Set the distance budget to the length of the shortest path minus 5.0
                    budget = np.min(distances)-5.0
                    print(f'Distance Budget: {budget:.4f}')
                    transform.distance_budget = budget
                    transform.constraint_weight = 250.
                else:
                    budget_satisfied=True
                    budget = np.inf

                # ---------------------------------------------------------------------------------

                for method in methods:
                    if method == 'ContinuousSGP':
                        candidates = self.dataset.X_train
                    else:
                        candidates = self.dataset.candidates
                    model = get_method(method)(num_waypoints,
                                            candidates,
                                            self.kernel_opt,
                                            self.noise_variance_opt,
                                            transform,
                                            num_robots=self.num_robots)

                    start_time = time()
                    solution = model.optimize()
                    end_time = time()
                    ipp_time = end_time-start_time

                    budget_constraint = model.transform.constraints(solution.reshape(-1, 2))
                    budget_satisfied = budget_constraint > -10.

                    self.evaluate(solution, num_waypoints, method, budget_satisfied, ipp_time, self.gp_time)
                self.log_results()

    def log_results(self):
        # Log the results to a json file
        with open(f'{self.fname}.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

    def evaluate(self, solution,  num_waypoints, method, budget_satisfied, ipp_time, param_time):
        # Get the solution sensed data
        solution_X, solution_y = [], []
        for r in range(self.num_robots):
            X_new, y_new = self.dataset.get_sensor_data(solution[r], 
                                                    continuous_sening=self.continuous_sening,
                                                    max_samples=1500)
            solution_X.extend(X_new)
            solution_y.extend(y_new)
        solution_X = np.array(solution_X)
        solution_y = np.array(solution_y)

        # Get RMSE using the oracle hyperparameters
        if len(solution_X) > 0:
            y_pred, y_var = get_reconstruction((solution_X, solution_y), 
                                                self.dataset.X_test, 
                                                self.noise_variance_opt, 
                                                self.kernel_opt)
            rmse = get_rmse(y_pred, self.dataset.y_test)
            nlpd = get_nlpd(y_pred, self.dataset.y_test, y_var)
            smse = get_smse(y_pred, self.dataset.y_test, y_var)
        else:
            rmse = np.nan
            nlpd = np.nan
            smse = np.nan

        self.results[num_waypoints][method]['ParamTime'].append(param_time)
        self.results[num_waypoints][method]['IPPTime'].append(ipp_time)
        self.results[num_waypoints][method]['RMSE'].append(rmse)
        self.results[num_waypoints][method]['NLPD'].append(nlpd)
        self.results[num_waypoints][method]['SMSE'].append(smse)
        if self.distance_budget:
            self.results[num_waypoints][method]['Constraint'].append(bool(budget_satisfied))

        print(f'\n{method}')
        print("-"*len(method))
        print(f'Param Time: {param_time:.4f}')
        print(f'IPP Time: {ipp_time:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'SMSE: {smse:.4f}')
        print(f'NLPD: {nlpd:.4f}')
        if self.distance_budget:
            print(f'Constraint: {budget_satisfied}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="SP/IPP benchmarking script")
    parser.add_argument("--num_mc", type=int, default=10)
    parser.add_argument("--num_robots", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--distance_budget", action='store_true')
    parser.add_argument("--dataset_path", type=str, 
                        default='../datasets/mississippi.tif')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--tsp_time_limit", type=int, default=-1)
    args=parser.parse_args()

    # Set the maximum distance (for each path) for the TSP solver
    max_dist = 350 if args.num_robots==1 else 150

    # Limit maximum waypoints/placements for multi robot case
    max_range = 101 if args.num_robots==1 and args.sampling_rate==2 else 51
    xrange = range(5, max_range, 5)

    if args.tsp_time_limit > 0:
        tsp_time_limit = args.tsp_time_limit 
    elif args.num_robots==1:
        tsp_time_limit = 30
    else:
        tsp_time_limit = 120
        
    # Methods to benchmark   
    methods = ['BayesianOpt', 
               'CMA', 
               'ContinuousSGP']

    if args.sampling_rate == 2 and \
       args.num_robots == 1 and \
       args.distance_budget:
        methods = ['BayesianOpt', 
                   'CMA', 
                   'ContinuousSGP']

    if args.sampling_rate == 2 and \
       args.num_robots == 1 and \
       not args.distance_budget:
        methods = ['BayesianOpt', 
                   'CMA',
                   'GreedyObjective',
                   'GreedySGP',
                   'ContinuousSGP']

    benchmark = IPPBenchmark(args.dataset_path, 
                args.num_mc, 
                args.num_robots, 
                max_dist, 
                args.sampling_rate, 
                xrange,
                methods,
                args.distance_budget,
                tsp_time_limit=tsp_time_limit,
                verbose=args.verbose)
    benchmark.run()
    