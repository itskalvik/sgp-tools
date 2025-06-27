import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
import numpy as np
from time import time

from sgptools.utils.data import *
from sgptools.utils.misc import *
from sgptools.utils.metrics import *
from sgptools.utils.tsp import run_tsp

from sgptools.methods import *
from sgptools.core.osgpr import *
from sgptools.core.transformations import *

from IPP import IPPBenchmark

gpflow.config.set_default_likelihood_positive_minimum(1e-4)
np.random.seed(1234)
tf.random.set_seed(1234)


class NumCandidatesBenchmark(IPPBenchmark):

    def __init__(self,
                 dataset_path,
                 num_mc,
                 num_robots,
                 max_dist,
                 sampling_rate,
                 xrange,
                 candidates_range,
                 distance_budget,
                 tsp_time_limit=30,
                 verbose=False):
        super().__init__(dataset_path, num_mc, num_robots, max_dist,
                         sampling_rate, xrange, candidates_range,
                         distance_budget, tsp_time_limit, verbose)
        self.fname = 'num_candidates-' + self.fname.split('-')[1]
        self.dataset_path = dataset_path

    def run(self):
        for _ in range(self.num_mc):
            for num_waypoints in self.xrange:
                print(f'\nNum Waypoints: {num_waypoints}', flush=True)

                # Generate initial paths
                Xu_init = get_inducing_pts(self.dataset.X_train,
                                           num_waypoints * self.num_robots,
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
                    budget = np.min(distances) - 5.0
                    print(f'Distance Budget: {budget:.4f}')
                    transform.distance_budget = budget
                    transform.constraint_weight = 250.
                else:
                    budget_satisfied = True
                    budget = np.inf

                # ---------------------------------------------------------------------------------

                for num_candidates in candidates_range:
                    dataset = Dataset(self.dataset_path,
                                      num_candidates=int(num_candidates),
                                      verbose=False)
                    model = get_method("CMA")(num_waypoints,
                                              dataset.candidates,
                                              self.kernel_opt,
                                              self.noise_variance_opt,
                                              transform,
                                              num_robots=self.num_robots)

                    start_time = time()
                    solution = model.optimize()
                    end_time = time()
                    ipp_time = end_time - start_time

                    budget_constraint = model.transform.constraints(
                        solution.reshape(-1, 2))
                    budget_satisfied = budget_constraint > -10.

                    self.evaluate(solution, num_waypoints, num_candidates,
                                  budget_satisfied, ipp_time, self.gp_time)
                self.log_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SP/IPP benchmarking script")
    parser.add_argument("--num_mc", type=int, default=10)
    parser.add_argument("--num_robots", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--distance_budget", action='store_true')
    parser.add_argument("--dataset_path",
                        type=str,
                        default='../datasets/mississippi.tif')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--tsp_time_limit", type=int, default=-1)
    parser.add_argument("--num_waypoints", type=int, default=50)
    args = parser.parse_args()

    # Set the maximum distance (for each path) for the TSP solver
    max_dist = 350 if args.num_robots == 1 else 150

    candidates_range = range(100, 1001, 100)
    candidates_range = [str(num) for num in candidates_range]

    if args.tsp_time_limit > 0:
        tsp_time_limit = args.tsp_time_limit
    elif args.num_robots == 1:
        tsp_time_limit = 30
    else:
        tsp_time_limit = 120

    benchmark = NumCandidatesBenchmark(args.dataset_path,
                                       args.num_mc,
                                       args.num_robots,
                                       max_dist,
                                       args.sampling_rate,
                                       [args.num_waypoints],
                                       candidates_range,
                                       args.distance_budget,
                                       tsp_time_limit=tsp_time_limit,
                                       verbose=args.verbose)
    benchmark.run()
