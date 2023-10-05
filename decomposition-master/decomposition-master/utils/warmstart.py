from utils import common


def warm_start(model, clusters, var_dict, time_limit=60):
    '''Warm start the model from sub-model solutions.'''

    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)
    sol = {}
    total_time = 0
    
    for k, v in clusters.items():
        new_model = common.relax_integer_constraints(model.copy(), var_dict)
        new_model = common.unrelax_integer_constraints(
            new_model,
            dict([(var, var_dict[var]) for var in v]))
        new_model.optimize()
        total_time += new_model.Runtime

        for var in v:
            model_var = var_dict[var]
            if new_model.getVarByName(model_var).X > 0:
                sol[model_var] = new_model.getVarByName(model_var).X

    model = common.set_warmstart_vector(model, sol)
    model.optimize()

    return total_time, common.get_objval(model)
