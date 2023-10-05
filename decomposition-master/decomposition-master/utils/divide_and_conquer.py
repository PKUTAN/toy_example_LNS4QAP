import random

from utils import common


def divide_and_conquer(model, clusters, var_dict, time_limit=60):
    """Solver for integer variables one cluster at a time.
    
    Arguments:
      model: the integer program.
      clusters: a dict mapping cluster indices to node indices.
      var_dict: a dict mapping node index to node variable name.
      time_limit: time limit for solving each subproblem.

    Returns:
      time: total time.
      obj: the final objective value.
    """

    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)
    cluster_order = list(range(len(clusters)))
    random.shuffle(cluster_order)
    # relax all integer constraints.
    model = common.relax_integer_constraints(model, var_dict)
    time = 0
    
    for cluster in cluster_order:
        # unrelax the integer constraints in cluster.
        var_to_unrelax = clusters[cluster]
        model = common.unrelax_integer_constraints(
            model,
            dict([(var, var_dict[var]) for var in var_to_unrelax]))
        model.optimize()
        time += model.Runtime
        
        var_to_fix = dict([(var_dict[var], round(model.getVarByName(var_dict[var]).X))
                           for var in var_to_unrelax])
        model = common.fix_integer_values(model, var_to_fix)

    return time, common.get_objval(model), common.extract_solution(model, var_dict)
