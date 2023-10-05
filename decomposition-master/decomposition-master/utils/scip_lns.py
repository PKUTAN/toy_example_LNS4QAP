from pyscipopt import Model
import networkx as nx
import numpy as np
import random
import time

from utils import common


def lns(model, num_clusters, steps, p_type, var_dict=None,
        time_limit=60, start_obj=0, sol=None, graph=None,
        sol_vec=None, verbose=False, only_one=False):
    """Perform coordinate descent on model. The descent order is random.

    Arguments:
      model: the integer program.
      num_clusters: number of clusters.
      steps: the number of times to cycle through the coordinates.
      p_type: problem type, affecting initializing solutions.
      var_dict: a dict maps node index to node variable name.
      time_limit: time limit for each coordinate descent step.
      sol: (optional) initial solution.
      graph: networkx graph object.
      verbose: if True, print objective after every decomposition.
    """

    model.setRealParam('limits/time', time_limit)
    model.hideOutput()

    total_time = 0

    if var_dict is None:
        var_dict = common.generate_var_dict(model, p_type)

    start_time = time.time()
    if sol is None:
        sol, sol_vec, start_obj = common.initialize_solution(var_dict, p_type, model)

    cluster_list = []
    sol_list = []
    obj_list = [start_obj]

    if p_type == 'psulu':
        time_list = [3.0]
    else:
        time_list = [0.0]

    for _ in range(steps):
        sol_list.append(sol_vec)
        clusters = common.uniform_random_clusters(var_dict, num_clusters, p_type)
        # clusters = common.metis_clusters(graph, num_clusters)
        # clusters = common.random_block_coordinates(var_dict, len(var_dict) // num_clusters)

        # print('-------------------------------------------------------')
        sol, sol_vec, solver_time, obj = lns_by_clusters(
            Model(sourceModel=model), clusters, var_dict, sol, only_one)
        total_time += solver_time
        if verbose:
            print("objective: ", obj)
            cur_time = time.time()
            print("time: ", cur_time - start_time)

        cur_time = time.time()

        cluster_list.append(clusters)
        obj_list.append(obj)
        time_list.append(cur_time-start_time)

    return total_time, obj, start_obj - obj, cluster_list, sol_list, obj_list, time_list


def lns_by_clusters(model, clusters, var_dict, sol, only_one=False):
    """Perform gradient descent by clusters. The order of clusters is randomized.

    Arguments:
      model: the integer program.
      clusters: a dict mapping cluster index to node index.
      var_dict: mapping node index to node variable name.
      sol: current solution.

    Returns:
      new solution. time spent by the solver. new objective.
    """

    # cluster_idx = list(range(len(clusters)))
    # random.shuffle(cluster_idx)

    # order clusters by size from largest to smallest
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    solver_t = 0

    for idx, cluster in sorted_clusters:
        sol, solver_time, obj = neighborhood_search(Model(sourceModel=model), cluster, var_dict, sol)
        solver_t += solver_time

        if only_one:
            break

    sol_vec = None
    for k, var_list in var_dict.items():
        if sol_vec is None:
            sol_vec = np.zeros((len(var_dict), len(var_list)))

        for i, v in enumerate(var_list):
            sol_vec[k, i] = sol[v]

    return sol, sol_vec, solver_t, obj


def neighborhood_search(model, cluster, var_dict, sol):
    """Perform gradient descent on model along coordinates defined by
       variables in cluster,  starting from a current solution sol.

    Arguments:
      model: the integer program.
      cluster: the coordinates to perform gradient descent on.
      var_dict: mapping node index to node variable name.
      sol: a dict representing the current solution.

    Returns:
      new_sol: the new solution.
      time: the time used to perform the descent.
      obj: the new objective value.
    """

    all_var = model.getVars()

    # inverse dict
    inv_dict = {}
    for k, var_list in var_dict.items():
        for v in var_list:
            inv_dict[v] = k

    for v in all_var:
        v_name = v.name
        if v_name not in inv_dict:
            continue

        v_num = inv_dict[v_name]
        if v_num not in cluster:
            model.addCons(v == sol[v_name])



    # for k, var_list in var_dict.items():
    #     for v in var_list:
    #         model_var = model.getVarByName(v)
    #         # warm start variables in the current coordinate set with the existing solution.
    #         if k in cluster:
    #             model_var.start = sol[v]
    #         else:
    #             model.addConstr(model_var == sol[v])

    model.optimize()
    new_sol = {}
    sol = model.getBestSol()

    for v in all_var:
        v_name = v.name
        new_sol[v_name] = round(sol[v])

    # for k, var_list in var_dict.items():
    #     for v in var_list:
    #         var = model.getVarByName(v)
    #         try:
    #             new_sol[v] = round(var.X)
    #         except:
    #             return sol, model.Runtime, -1

    # print('Solver time: %f' %(model.Runtime))
    return new_sol, model.getSolvingTime(), model.getSolObjVal(sol)

    # return new_sol, model.Runtime, common.get_objval(model)
