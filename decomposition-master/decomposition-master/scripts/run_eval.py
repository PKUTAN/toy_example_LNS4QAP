import sys
sys.path.append('.')


from utils import divide_and_conquer
from utils import iterated_rounding
from utils import coordinate_descent
from utils import warmstart
from utils import heuristics
from utils import common
from utils import search

from absl import flags
from absl import app

FLAGS = flags.FLAGS

import gurobipy
import networkx as nx
import numpy as np
import os
import pickle
import random
import time

from multiprocessing import Pool


flags.DEFINE_string('lp_dir', None, 'The lp problem directory.')
flags.DEFINE_string('graph_dir', None, 'The graph gpickle directory.')
flags.DEFINE_string('p_type', 'mvc', 'The optimization problem type.')
flags.DEFINE_integer('iterations', 1, 'The number of iterations to run a certain heuristic.')
flags.DEFINE_integer('time_limit', None, 'Time limit to solve each problem.')
flags.DEFINE_integer('steps', 1, 'The number of steps for partitions.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('mode', None, 'Which evaluation mode to run.')
flags.DEFINE_integer('num_branch', 1, 'Number of clusters to try.')
flags.DEFINE_bool('verbose', False, 'If True, print objective for each step.')
flags.DEFINE_bool('only_one', False, 'If True, only run one cluster.')
flags.DEFINE_integer('min_k', 1, 'Minimum number of clusters.')
flags.DEFINE_integer('max_k', 5, 'Maximum number of clusters.')
flags.DEFINE_integer('min_t', 1, 'Minimum time per cluster.')
flags.DEFINE_integer('max_t', 3, 'Maximum time per cluster.')


def spectral_clustering(var_dict, graph, num_clusters):
    pass


def simple_rounding_strat(x):
    '''A simple rounding strategy.'''

    return int(x >= .5)


def random_rounding_strat(x):
    '''Random rounding strategy.'''
    random_num = random.random()

    return int(random_num < x)


def try_heuristics():
    '''IGNORE. Legacy code.'''
    
    if mode == 'dnd':
        solver_time, obj, _ = divide_and_conquer.divide_and_conquer(model, clusters,
                                                                    var_dict,
                                                                    time_limit=1800)
        print(solver_time, obj)
    elif mode == 'ir':
        rounding_strat = simple_rounding_strat
        obj = iterated_rounding.iterated_rounding(model, clusters, var_dict,
                                                  rounding_strat)
        print(obj)
    elif mode == 'cd':
        for _ in range(iterations):
            start_time = time.time()
            solver_time, obj, total_diff = coordinate_descent.coordinate_descent(
                model, num_clusters, 
                steps=steps, p_type=p_type, var_dict=var_dict, time_limit=time_limit,
                graph=graph)
            end_time = time.time()
            print(end_time - start_time, solver_time, obj, total_diff)
    elif mode == 'ws':
        solver_time, obj = warmstart.warm_start(model, clusters, var_dict, time_limit=10)
        print(solver_time, obj)
    elif mode == 'greedy':
        obj = heuristics.greedy(graph)
        print(obj)
    elif mode == 'hybrid':
        solver_time, obj, sol = divide_and_conquer.divide_and_conquer(model, clusters,
                                                                      var_dict,
                                                                      time_limit=120)
        solver_time, obj = coordinate_descent.coordinate_descent(model, num_clusters,
                                                                 var_dict, 5,
                                                                 time_limit=120, sol=sol)
        print(solver_time, obj)


def compute_random_decomp_stats(lp_dir, output_dir, min_k, max_k,
                                steps, p_type, iterations, min_t, max_t,
                                only_one=False):

    num_prob = len(os.listdir(lp_dir))

    if p_type == 'psulu':
        init_time = 0
    else:
        init_time = 0
    
    for k in range(min_k, max_k+1):
        for t in range(min_t, max_t+1):
            output_k = os.path.join(output_dir, 'random_k_%d_step_%d_time_%d_init_%d'
                                    %(k, steps, t, init_time))
            if not os.path.exists(output_k):
                os.makedirs(output_k)

            stats_output = os.path.join(output_k, 'stats.txt')
            stats_f = open(stats_output, 'w')
            total_diff_stats = []
            final_objs = []
            avg_imp = []
            best_clusters = {}
            best_sol = None
            cluster_file = os.path.join(output_k, 'best_clusters.pkl')

            # problem_paths = os.listdir(lp_dir)
            # problem_paths = [os.path.join(lp_dir, p) for p in problem_paths]
            # num_problems = len(problem_paths)
            # p = Pool(processes=num_problems)
            # data = p.starmap(single_instance_random,
            #                  zip(
            #                  problem_paths,
            #                  [k] * num_problems,
            #                  [steps] * num_problems,
            #                  [iterations] * num_problems,
            #                  [p_type] * num_problems,
            #                  [time_limit] * num_problems,
            #                  [output_k] * num_problems),
            #                  chunksize=1)
            # p.close()
            # for d in data:
            #     total_diff_stats.extend(d[0])
            #     final_objs.extend(d[1])
        
            for f_name in os.listdir(lp_dir):
                if f_name.endswith('pkl'):
                    continue
                
                lp_prefix = f_name.split('.')[0]
                lp_file = os.path.join(lp_dir, lp_prefix + '.lp')
                output_file = os.path.join(output_k, lp_prefix + '.out')
                output = open(output_file, 'w')
                # graph_file = os.path.join(graph_dir, lp_prefix + '.gpickle')

                model = gurobipy.read(lp_file)
                best_obj = 1000000000
                best_cluster = None

                # run Gurobi as is.
                if k == 1:
                    num_iter = 1
                else:
                    num_iter = iterations

                sol = None
                for _ in range(num_iter):
                    if sol is None:
                        var_dict = common.generate_var_dict(model, p_type)
                        sol, sol_vec, start_obj = (
                            common.initialize_solution(
                                var_dict, p_type, model))
                        
                    start_time = time.time()
                    solver_time, obj, total_diff, cluster_list, sol_list, obj_list, time_list = (
                        coordinate_descent.coordinate_descent(
                            model, k, 
                            steps=steps, p_type=p_type, time_limit=t,
                            sol=sol, start_obj=start_obj, var_dict=var_dict,
                            sol_vec=sol_vec, only_one=only_one))
                    end_time = time.time()
                    print(end_time - start_time, solver_time, obj, total_diff)
                    print(end_time - start_time, solver_time, obj, total_diff,
                          file=output)
                    print(' '.join(['%.4f' %x for x in obj_list]),
                          file=output)
                    print(' '.join(['%.4f' %x for x in time_list]),
                          file=output)                    
                        
                    total_diff_stats.append(total_diff)
                    final_objs.append(obj)
                    avg_imp.append(total_diff / (end_time-start_time))
                
                    if obj < best_obj:
                        best_obj = obj
                        best_cluster = cluster_list
                        best_sol = sol_list
                    
                output.close()
                best_clusters[lp_prefix] = (best_cluster, best_sol)
                    
            pickle.dump(best_clusters, open(cluster_file, 'wb'))
            
            print('Mean total rewards: ', np.mean(total_diff_stats), file=stats_f)
            print('Std total rewards: ', np.std(total_diff_stats),
                  file=stats_f)
            print('Min final objectives: ', np.min(final_objs), file=stats_f)
            print('Mean final objectives: ', np.mean(final_objs), file=stats_f)
            print('Std final objectives: ',
                  np.std(final_objs) / np.sqrt(num_iter * num_prob),
                  file=stats_f)
            print('Average improvement: ', np.mean(avg_imp), file=stats_f)
            print('Std improvement: ', np.std(avg_imp) / np.sqrt(num_iter * num_prob),
                  file=stats_f)
            stats_f.close()

            print('Mean total rewards: ', np.mean(total_diff_stats))
            print('Std total rewards: ', np.std(total_diff_stats))
            print('Min final objectives: ', np.min(final_objs))
            print('Mean final objectives: ', np.mean(final_objs))
            print('Std final objectives: ',
                  np.std(final_objs) / np.sqrt(num_iter * num_prob))
            print(stats_output)
        

def single_instance_random(lp_file,
                           num_clusters,
                           steps,
                           iterations,
                           p_type,
                           time_limit,
                           output_k):
    relative_lp_file = lp_file.rsplit('/', maxsplit=1)[1]
    lp_prefix = relative_lp_file.split('.')[0]
    output_file = os.path.join(output_k, lp_prefix + '.out')
    output = open(output_file, 'w')
    total_diff_stats = []
    final_objs = []

    model = gurobipy.read(lp_file)
    for _ in range(iterations):
        start_time = time.time()
        solver_time, obj, total_diff, _, _ = coordinate_descent.coordinate_descent(
            model, num_clusters, 
            steps=steps, p_type=p_type, time_limit=time_limit)
        end_time = time.time()
        print(end_time - start_time, solver_time, obj, total_diff)            
        print(end_time - start_time, solver_time, obj, total_diff, file=output)
        total_diff_stats.append(total_diff)
        final_objs.append(obj)
    output.close()

    return total_diff_stats, final_objs

        
def test_single_instance(lp_file,
                         num_clusters,
                         steps,
                         iterations,
                         p_type,
                         time_limit,
                         verbose):
    model = gurobipy.read(lp_file)
    total_diff_stats = []
    final_objs = []
    for _ in range(iterations):
        start_time = time.time()
        solver_time, obj, total_diff, _, _ = coordinate_descent.coordinate_descent(
            model, num_clusters, 
            steps=steps, p_type=p_type, time_limit=time_limit, verbose=verbose)
        end_time = time.time()
        print(end_time - start_time, solver_time, obj, total_diff)
        total_diff_stats.append(total_diff)
        final_objs.append(obj)
    print('Mean total rewards: ', np.mean(total_diff_stats))
    print('Std total rewards: ', np.std(total_diff_stats))
    print('Min final objectives: ', np.min(final_objs))


def test_bfs(lp_file, num_clusters, steps, num_branch, p_type, time_limit):
    model = gurobipy.read(lp_file)
    total_time, obj, total_diff, _, _ = search.best_first(
        model, num_clusters, steps, num_branch, p_type, time_limit)
    print(total_time, obj, total_diff)


def test_hybrid(lp_file, num_clusters, bfs_steps, 
                num_branch, p_type, time_limit):
    model = gurobipy.read(lp_file)
    total_time, obj, total_diff, sol, cluster_list = search.best_first(
        model, num_clusters, bfs_steps, num_branch, p_type, time_limit)

    var_dict = common.generate_var_dict(model, p_type)
    while True:
        for clusters in cluster_list:
            sol, _, _, obj = coordinate_descent.coordinate_descent_by_clusters(
                model.copy(), clusters, var_dict, sol)
            print('objective: ', obj)


def collect_bfs_data(lp_dir, num_clusters, steps, num_branch, p_type, time_limit):
    output_dir = os.path.join('random_logs/bfs', lp_dir)
    
    for f_name in os.listdir(lp_dir):
        abs_lp = os.path.join(lp_dir, f_name)
        model = gurobipy.read(abs_lp)
        _, _, _, cluster_list = search.best_first(
            model, num_clusters, steps, num_branch, p_type, time_limit)
        
    
def main(argv):
    lp_dir = FLAGS.lp_dir
    graph_dir = FLAGS.graph_dir
    p_type = FLAGS.p_type
    iterations = FLAGS.iterations
    time_limit = FLAGS.time_limit
    min_t = FLAGS.min_t
    max_t = FLAGS.max_t
    
    steps = FLAGS.steps
    output_dir = FLAGS.output_dir
    mode = FLAGS.mode
    num_branch = FLAGS.num_branch
    verbose = FLAGS.verbose
    min_k = FLAGS.min_k
    max_k = FLAGS.max_k
    only_one = FLAGS.only_one

    if mode == 'random_stats':
        output_dir = os.path.join('random_logs', lp_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        compute_random_decomp_stats(lp_dir, output_dir, min_k, max_k,
                                    steps, p_type,
                                    iterations, min_t, max_t, only_one)
    elif mode == 'single_instance':
        test_single_instance(lp_dir, min_k, steps, iterations,
                             p_type, time_limit, verbose)
    elif mode == 'bfs':
        test_bfs(lp_dir, min_k, steps, num_branch, p_type, min_t)
    elif mode == 'hybrid':
        test_hybrid(lp_dir, min_k, steps, num_branch, p_type, time_limit)
    elif mode == 'gcn_no_train':
        pass


if __name__ == "__main__":
    app.run(main)
