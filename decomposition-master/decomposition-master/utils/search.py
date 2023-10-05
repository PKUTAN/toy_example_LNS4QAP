import time

from utils import common
from utils import coordinate_descent


def select_best_branch(model, var_dict, num_clusters, num_branch, sol, p_type):
    '''Select the best branch.
    Arguments:
      model: the integer program.
      var_dict: mapping node index to node variable name.
      num_clusters: number of clusters.
      num_branch: how many clusters to try.
      sol: current solution.

    Returns:
      new solution, new objective value selected as the best.
    '''

    best_sol = None
    best_obj = 1000000
    best_cluster = None
    
    for _ in range(num_branch):
        clusters = common.uniform_random_clusters(var_dict, num_clusters, p_type)
        new_sol, _, _, obj = coordinate_descent.coordinate_descent_by_clusters(
            model.copy(), clusters, var_dict, sol)
        if obj < best_obj:
            best_obj = obj
            best_sol = new_sol
            best_cluster = clusters

    return best_sol, best_obj, best_cluster
        

def best_first(model, num_clusters, steps, num_branch, p_type, time_limit):
    '''Best first search for number of rounds.
    
    Arguments:
      model: the integer program.
      num_clusters: number of clusters.
      steps: how many clusters to use.
      num_branch: how many clusters to try at each step.
      p_type: problem type.
      time_limit: time limit to solve each sub-problem.

    Returns:
      total time, final objective value, improvement over initial objective value.
    '''

    model.setParam('MIPFocus', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    cluster_list = []

    start_time = time.time()
    var_dict = common.generate_var_dict(model, p_type)
    sol, _, start_obj = common.initialize_solution(var_dict, p_type, model)
    # warmstart_problem = common.set_warmstart_vector(model.copy(), sol)
    # warmstart_problem.optimize()
    # start_obj = warmstart_problem.ObjVal
    # sol, _ = common.extract_solution(warmstart_problem, var_dict)

    for _ in range(steps):
        sol, obj, clusters = select_best_branch(
            model, var_dict, num_clusters, num_branch, sol, p_type)
        print('objective: ', obj)
        cluster_list.append(clusters)

    end_time = time.time()
    return end_time - start_time, obj, start_obj - obj, sol, cluster_list


def beam_search(model, num_clusters, steps, beam_size, num_branch, p_type, time_limit):
    pass

