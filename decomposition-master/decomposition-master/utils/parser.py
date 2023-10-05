import gurobipy
import os

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import math

FLAGS = flags.FLAGS

flags.DEFINE_string('in_dir', None, 'Input directory containing CATS format files.')
flags.DEFINE_string('out_dir', None, 'Output directory to store lp files.')
flags.DEFINE_string('mode', None, '')


def parse_cats(f_name):
    '''Convert from CATS file format to lp file format.'''

    model = gurobipy.Model()
    obj = 0
    
    f = open(f_name, 'r')
    lines = f.readlines()
    f.close()

    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith('bids'):
            num_bids = int(line.split()[1])
            break

    item_bid = {}
    
    for i in range(num_bids):
        line = lines[-i-1].strip()
        items = line.split()
        bid_num = int(items[0])
        bid_price = float(items[1])
        bid_var = model.addVar(vtype=gurobipy.GRB.BINARY, name='x%d' %bid_num)
        obj += bid_price * bid_var

        for j in range(2, len(items) - 1):
            item_num = int(items[j])
            if item_num in item_bid:
                item_bid[item_num].append(bid_var)
            else:
                item_bid[item_num] = [bid_var]

    for k, v in item_bid.items():
        model.addConstr(sum(v) <= 1, 'c%d' %k)

    model.setObjective(-obj, gurobipy.GRB.MINIMIZE)
    model.update()

    return model


def parse_random_decomp_log(f_name):
    diff_data = []
    with open(f_name) as f:
        for line in f:
            data = line.split()
            diff = float(data[-1])
            diff_data.append(diff)
    return diff_data


def parse_gurobi_log(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()

    objs = []
    for line in lines:
        obj = float(line.strip().split()[1])
        objs.append(obj)

    print(np.mean(objs))
    print(np.std(objs) / np.sqrt(len(objs)))


def parse_heu_log(f_name):
    stats = pd.read_csv(f_name, sep='\t').to_numpy()
    num = len(stats)
    stats = stats[:, 1:].astype('float')

    print(np.mean(stats, axis=0))
    print(np.std(stats, axis=0) / np.sqrt(len(stats)))
    

def main(argv):
    in_dir = FLAGS.in_dir
    out_dir = FLAGS.out_dir
    mode = FLAGS.mode

    if mode == 'gurobi_log':
        parse_gurobi_log(in_dir)
    elif mode == 'heu_log':
        parse_heu_log(in_dir)
    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
        for cat in os.listdir(in_dir):
            cat_name = cat.split('.')[0]
            abs_filename = os.path.join(in_dir, cat)
            model = parse_cats(abs_filename)
            output_filename = os.path.join(out_dir, cat_name + '.lp')
            model.write(output_filename)

        
if __name__ == "__main__":
    app.run(main)
        

        

    
