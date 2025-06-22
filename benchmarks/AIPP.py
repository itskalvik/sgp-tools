import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
import numpy as np
from time import time
import tensorflow_probability as tfp

from sgptools.utils.data import *
from sgptools.utils.misc import *
from sgptools.utils.metrics import *
from sgptools.utils.tsp import run_tsp
from sgptools.utils.gpflow import get_model_params

from sgptools.methods import *
from sgptools.core.osgpr import *
from sgptools.core.transformations import *

from IPP import IPPBenchmark

gpflow.config.set_default_likelihood_positive_minimum(1e-4)
np.random.seed(1234)
tf.random.set_seed(1234)


class AIPPBenchmark(IPPBenchmark):
    def __init__(self, 
                 dataset_path, 
                 num_mc, 
                 num_robots, 
                 max_dist, 
                 sampling_rate, 
                 xrange,
                 methods,
                 distance_budget,
                 tsp_time_limit=120,
                 verbose=False,
                 param_model_type='GP'):
        super().__init__(dataset_path, 
                         num_mc, 
                         num_robots, 
                         max_dist, 
                         sampling_rate, 
                         xrange, 
                         methods, 
                         distance_budget, 
                         tsp_time_limit, 
                         verbose)
        self.fname = 'AIPP' + self.fname.split('-')[1]
        self.param_model_type = param_model_type

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
                                    max_dist=max_dist, 
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

                # Get random hyperparameters
                _, noise_variance, kernel = get_model_params(*self.dataset.get_train(), 
                                                             max_steps=0,
                                                             verbose=False)
                
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

                # ---------------------------------------------------------------------------------

                for method in methods:
                    if method == 'ContinuousSGP':
                        candidates = self.dataset.X_train
                    else:
                        candidates = self.dataset.candidates
                    
                    model = get_method(method)(num_waypoints,
                                               candidates,
                                               kernel,
                                               noise_variance,
                                               transform,
                                               num_robots=self.num_robots)

                    solution, param_time, ipp_time = self.simulate_aipp(model, Xu_init)

                    budget_constraint = model.transform.constraints(solution.reshape(-1, 2))
                    budget_satisfied = budget_constraint > -10.

                    self.evaluate(solution, num_waypoints, method, budget_satisfied, ipp_time, param_time)
                self.log_results()

    def simulate_aipp(self, ipp_model, Xu_init):
        continuous = True if self.sampling_rate > 2 else False
        total_time_param = 0
        total_time_ipp = 0
        num_robots = Xu_init.shape[0]
        num_waypoints = Xu_init.shape[1]
        curr_sol = Xu_init
        offset = 2 if continuous else 1
        init_kernel, init_noise_variance = ipp_model.get_hyperparameters()

        if self.param_model_type=='SSGP':
            param_model = init_osgpr(self.dataset.X_train, 
                                    num_inducing=40, 
                                    kernel=init_kernel,
                                    noise_variance=init_noise_variance)

        # Simulate AIPP
        sol_data_X = []
        sol_data_y = []
        for time_step in range(offset, num_waypoints+1):        
            # Get the data from visited location(s)
            last_visited = curr_sol.copy()[:, time_step-offset:time_step]
            data_X_batch = []
            data_y_batch = []
            for r in range(num_robots):
                X_new, y_new = self.dataset.get_sensor_data(last_visited[r], 
                                                    continuous_sening=continuous,
                                                    max_samples=1500)
                data_X_batch.extend(X_new)
                data_y_batch.extend(y_new)            
                sol_data_X.extend(X_new)
                sol_data_y.extend(y_new)

            # Plot the updated path
            if self.verbose:
                plt.scatter(self.dataset.X_train[:, 0], 
                            self.dataset.X_train[:, 1], s=1, c='k')
                for i in range(num_robots):
                    plt.plot(curr_sol[i, :, 0], curr_sol[i, :, 1], marker='o')
                plt.show()
                
            # Skip param and path update if no data was collected
            if len(data_X_batch) == 0:
                continue

            # Skip param and path update for the last waypoint
            if time_step == num_waypoints:
                break

            # Init/update hyperparameters model
            start_time = time()
            if self.param_model_type=='GP':
                # Starting from initial params ensures recovery from bad params
                _, noise_variance, kernel = get_model_params(np.array(sol_data_X), 
                                                            np.array(sol_data_y),
                                                            kernel=init_kernel,
                                                            noise_variance=init_noise_variance,
                                                            verbose=self.verbose,
                                                            optimizer='scipy',
                                                            method='CG')
                # Clip to avoid floats being interpreted as NANs
                noise_variance = np.clip(noise_variance.numpy(), 1e-4, 5.0)
            else:
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

            # IPP update
            Xu_visited = curr_sol.copy()[:, :time_step]
            ipp_model.transform.update_Xu_fixed(Xu_visited)
            ipp_model.update(kernel, noise_variance)
            start_time = time()
            curr_sol = ipp_model.optimize()
            end_time = time()
            total_time_ipp += end_time - start_time
            
        return curr_sol, total_time_param, total_time_ipp


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="SP/IPP benchmarking script")
    parser.add_argument("--num_mc", type=int, default=10)
    parser.add_argument("--num_robots", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--distance_budget", action='store_true')
    parser.add_argument("--dataset_path", type=str, 
                        default='./mississippi.tif')
    parser.add_argument("--param_model_type", type=str, default='GP')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--tsp_time_limit", type=int, default=-1)
    args=parser.parse_args()

    # Set the maximum distance (for each path) for the TSP solver
    max_dist = 350 if args.num_robots==1 else 150

    # Limit maximum waypoints/placements for multi robot case
    max_range = 51 if args.num_robots==1 and args.sampling_rate==2 else 51
    xrange = range(5, max_range, 5)

    if args.tsp_time_limit > 0:
        tsp_time_limit = args.tsp_time_limit 
    elif args.num_robots==1:
        tsp_time_limit = 30
    else:
        tsp_time_limit = 120

    # Methods to benchmark
    methods = ['CMA', 
               'ContinuousSGP']
    
    benchmark = AIPPBenchmark(args.dataset_path, 
                              args.num_mc, 
                              args.num_robots, 
                              max_dist, 
                              args.sampling_rate, 
                              xrange,
                              methods,
                              args.distance_budget,
                              tsp_time_limit=tsp_time_limit,
                              verbose=args.verbose,
                              param_model_type=args.param_model_type)
    benchmark.run()
    