import gurobipy
import random

from utils import common


def round_solutions(model, cluster, var_dict, rounding_strat):
    """Perform rounding operations on solutions according to the roudning strategy.

    Arguments:
      model: the integer programming model.
      cluster: a list of node indices.
      var_dict: a dict mapping node index to node variable name.
      rounding_strat: rounding strategy.

    Returns:
      a new model with variables in cluster rounded according to the rounding strategy.
    """

    for var in cluster:
        model_var = model.getVarByName(var_dict[var])
        after_round = rounding_strat(model_var.X)
        model.addConstr(model_var == after_round)

    return model

    
def iterated_rounding(model, clusters, var_dict, rounding_strat):
    """Perform iterated rounding. 

    Arguments:
      model: the integer programming model.
      clusters: a dict maps cluster indices to node indices.
      var_dict: a dict maps node index to node variable name.
      rounding_strat: the rounding strategy to use.
    """

    cluster_order = list(range(len(clusters)))
    random.shuffle(cluster_order)
    model = common.relax_integer_constraints(model, var_dict)
    time = 0
    
    for idx in cluster_order:
        model.optimize()
        print('---------------------------------------------------------------')
        model = round_solutions(model, clusters[idx], var_dict, rounding_strat)
        model.reset()

    model.optimize()

    return common.get_objval(model)
        
