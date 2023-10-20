import gurobipy
import networkx as nx
import numpy as np
import random
import time
from gurobipy import Model,GRB,quicksum

from utils import common




def run_until_limit(model, k, p_type, per_limit, time_limit):
    model.setParam('MIPFocus', 1)
    model.setParam('OutputFlag', 0)

    var_dict = common.generate_var_dict(model, p_type)
        
    sol, _, start_obj = common.initialize_solution(var_dict, p_type, model)

    cluster_list = []
    start_time = time.time()
    total_solver_time = 0
    step = 0
    
    while True:
        clusters = common.uniform_random_clusters(var_dict, k, p_type)
        model_copy = model.copy()
        model_copy.setParam('TimeLimit', min(per_limit, time_limit))
        sol, _, solver_time, obj = coordinate_descent_by_clusters(model_copy, clusters,
                                                                  var_dict, sol,
                                                                  time_limit)
        total_solver_time += solver_time
        time_limit -= solver_time

        if time_limit <= 0:
            break

        step += 1
        
    end_time = time.time()
    print('Total solver time: ', total_solver_time)
    print('Num step: ', step)
    
    return end_time - start_time, obj, start_obj - obj


def coordinate_descent(model, num_clusters, steps, p_type, var_dict=None,
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

    # focus on finding feasible solutions.
    model.setParam('MIPFocus', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

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
        sol, sol_vec, solver_time, obj = coordinate_descent_by_clusters(
            model.copy(), clusters, var_dict, sol, only_one)
        total_time += solver_time
        if verbose:
            print("objective: ", obj)

        cur_time = time.time()

        cluster_list.append(clusters)
        obj_list.append(obj)
        time_list.append(cur_time-start_time)

    return total_time, obj, start_obj - obj, cluster_list, sol_list, obj_list, time_list


def coordinate_descent_by_clusters(F,D,model, clusters, sol, only_one=False):
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
    prob_index = [i for i in range(D.shape[1])]
    # order clusters by size from largest to smallest
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    obj_best = float('inf')

    for idx, cluster in sorted_clusters:
        if len(cluster) <= 1:
            continue

        out_index = np.setdiff1d(np.array(prob_index),np.array(cluster)).tolist()
        sol, obj = gradient_descent(model.copy(), D, F, sol, cluster, out_index)

        # if obj_best >= obj:
        #     best_sol = sol
        #     obj_best = obj
        if only_one:
            break

    return sol, obj

def gradient_descent(model,D,F,sol,index, out_index):
    N = len(index)

    assert N > 0 ,'N 必须大于0'

    per_sol = common.form_per(sol)
    per_assigned = per_sol[out_index]
    sub_loc = sorted([j for i,j in sol if i in index])

    x = model.addMVar(shape=(N,N),vtype= GRB.BINARY,name='x')
    # import pdb;pdb.set_trace()
    if N == 20:
        objective = (F*(x@D@x.T)).sum()
        model.setObjective(objective,GRB.MINIMIZE)

        model.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
        model.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N)); 

        model.optimize()

        sol = common.solution(model)
        obj = model.objVal

        return sol, obj
    else:
        sub_D = D[sub_loc][:,sub_loc]
        sub_F = F[index][:,index]
        sub_left_D = D[sub_loc] @ per_assigned.T
        sub_right_D = per_assigned @ D[:,sub_loc]
        sub_left_F = F[index][:,out_index]
        sub_right_F = F[out_index][:,index]


        objective = (sub_F*(x@sub_D@x.T)).sum() # 二次项
        objective += (sub_left_F*(x@sub_left_D)).sum() 
        objective += (sub_right_F*(sub_right_D@x.T)).sum()#一次项
        model.setObjective(objective,GRB.MINIMIZE)

        model.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
        model.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N)); 

        model.optimize()
        
        sub_sol = common.solution(model)
        sol_new = common.sub2whole(sub_sol,sol,index)
        obj = common.cal_obj(sol_new,D,F)

        # import pdb; pdb.set_trace()
        
        # gurobi_interm_obj.append(obj)
        # gurobi_interm_time.append(time.time()-start_time)

        return sol_new, obj

# def gradient_descent(model, cluster, sol):
#     """Perform gradient descent on model along coordinates defined by 
#        variables in cluster,  starting from a current solution sol.
    
#     Arguments:
#       model: the integer program.
#       cluster: the coordinates to perform gradient descent on.
#       var_dict: mapping node index to node variable name.
#       sol: a dict representing the current solution.

#     Returns:
#       new_sol: the new solution.
#       time: the time used to perform the descent.
#       obj: the new objective value.
#     """


#     model.optimize()
#     new_sol = {}

#     for k, var_list in var_dict.items():
#         for v in var_list:
#             var = model.getVarByName(v)
#             try:
#                 new_sol[v] = round(var.X)
#             except:
#                 return sol, model.Runtime, -1

#     # print('Solver time: %f' %(model.Runtime))
#     return new_sol, model.Runtime, common.get_objval(model)
