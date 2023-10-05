from absl import flags
from absl import app

import functools
import gurobipy
from gurobipy import GRB
import numpy as np

import os


FLAGS = flags.FLAGS

flags.DEFINE_string('lp_dir', None, 'The directory for lp instances.')
flags.DEFINE_string('output_dir', None, 'The directory to store outputs.')
flags.DEFINE_integer('time_limit', 60, 'Time limit in seconds.')
flags.DEFINE_bool('log_sol', False, 'Whether to log solutios.')
flags.DEFINE_integer('total', -1, '')


def callback(model, where, output):
    if where == GRB.Callback.MIPSOL:
        best_obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        print("%.4f %.4f" %(runtime, best_obj), file=output)


def main(argv):
    time_limit = FLAGS.time_limit
    lp_dir = FLAGS.lp_dir
    output_dir = os.path.join('gurobi_logs_%d' %time_limit, lp_dir)
    log_sol = FLAGS.log_sol
    total = FLAGS.total

    objs = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not log_sol:
        output_abs_filename = os.path.join(output_dir, 'gurobi.log')
        output = open(output_abs_filename, 'w')

    ctr = 0

    for f_name in os.listdir(lp_dir):
        if not f_name.endswith('.lp'):
            continue

        abs_filename = os.path.join(lp_dir, f_name)
        model = gurobipy.read(abs_filename)
        prefix_f_name = f_name.split('.')[0]
        if log_sol:
            output_abs_filename = os.path.join(output_dir, prefix_f_name + '.log')
            output = open(output_abs_filename, 'w')

        def log_callback(model, where):
            callback(model, where, output)
            
        model.setParam('MIPFocus', 1)
        model.setParam('TimeLimit', time_limit)
        # model.setParam('OutputFlag', 0)

        if log_sol:
            model.optimize(log_callback)
            objs.append(model.ObjVal)
        else:
            model.optimize()
            obj = model.ObjVal
            print('%s %.4f' %(prefix_f_name, obj), file=output)

        if log_sol:
            output.close()

        ctr += 1
        if total > 0 and ctr >= total:
            break

    if not log_sol:
        output.close()
    else:
        print(np.mean(objs))
        print(np.std(objs) / np.sqrt(len(objs)))

        
if __name__ == "__main__":
    app.run(main)
