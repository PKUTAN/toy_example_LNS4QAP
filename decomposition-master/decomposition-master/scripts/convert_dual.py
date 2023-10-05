from utils import common

from absl import flags
from absl import app

import os
import networkx as nx
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string('gpickle_dir', None, 'Input graph directory.')
flags.DEFINE_string('lp_dir', None, 'LP directory.')


def generate_var_dict(graph):
    var_dict = {}

    for node in graph.nodes():
        var_name = 'v%d' %node
        var_dict[node] = var_name

    return var_dict
        

def main(argv):
    gpickle_dir = FLAGS.gpickle_dir
    lp_dir = FLAGS.lp_dir

    for gpickle_f in os.listdir(gpickle_dir):
        abs_filename = os.path.join(gpickle_dir, gpickle_f)
        if gpickle_f.startswith('dual'):
            continue
        # dual_graph = common.compute_dual_graph(nx.read_gpickle(abs_filename))
        # out_filename = os.path.join(gpickle_dir, 'dual_%s' %gpickle_f)
        var_filename = os.path.join(lp_dir, gpickle_f.split('.')[0] + '.pkl')
        var_file = open(var_filename, 'wb')
        var_dict = generate_var_dict(nx.read_gpickle(abs_filename))
        pickle.dump(var_dict, var_file)
        var_file.close()
        # nx.write_gpickle(dual_graph, out_filename)


if __name__ == "__main__":
    app.run(main)
        
