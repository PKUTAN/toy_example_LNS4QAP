import sys
sys.path.append('.')

from absl import flags
from absl import app

FLAGS = flags.FLAGS

import gurobipy
import networkx as nx
import numpy as np
import os
import scipy.sparse as sp
from utils import common

flags.DEFINE_string('lp_dir', None, '')
flags.DEFINE_string('p_type', 'cats', '')
flags.DEFINE_integer('num_obst', 0, '')
flags.DEFINE_integer('num_waypoints', 0, '')

def main(argv):
    lp_dir = FLAGS.lp_dir
    p_type = FLAGS.p_type
    num_obst = FLAGS.num_obst
    num_waypoints = FLAGS.num_waypoints
    if p_type == 'cats':
        graph_dir = lp_dir.replace("_lp", "_adj")
    elif p_type == 'psulu' or p_type == 'mvc':
        graph_dir = lp_dir.replace('lpfiles', 'adj')
        
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    for lp_file in os.listdir(lp_dir):
        prefix = lp_file.split('.')[0]
        abs_lp_path = os.path.join(lp_dir, lp_file)
        abs_graph_path = os.path.join(graph_dir, prefix+'.npz')
        model = gurobipy.read(abs_lp_path)
        graph = common.milp_to_adj(model, p_type,
                                   num_obst=num_obst,
                                   num_waypoints=num_waypoints)
        sp_adj = sp.coo_matrix(graph)
        
        sp.save_npz(abs_graph_path, sp_adj)
    

if __name__ == "__main__":
    app.run(main)
