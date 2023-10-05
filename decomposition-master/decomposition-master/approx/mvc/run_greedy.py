import sys
sys.path.append('.')

from absl import flags
from absl import app

import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover
import numpy as np
import os
import random

FLAGS = flags.FLAGS

flags.DEFINE_string('gpickle_dir', None, '')

def main(argv):
    gpickle_dir = FLAGS.gpickle_dir
    cover_weights = []

    for gpickle_f in os.listdir(gpickle_dir):
        if not gpickle_f.endswith('gpickle'):
            continue
        
        gpickle_f = os.path.join(gpickle_dir, gpickle_f)
        graph = nx.read_gpickle(gpickle_f)

        for i in range(len(graph.nodes)):
            graph.nodes[i]['weight'] = random.random()
            
        min_cover = min_weighted_vertex_cover(graph)
        node_weight = dict(graph.nodes(data='weight'))
        weight = 0

        # import pdb; pdb.set_trace()
        for node in min_cover:
            weight += node_weight[node]

        cover_weights.append(weight)
    print(np.mean(cover_weights))
    print(np.std(cover_weights) / np.sqrt(len(cover_weights)))

if __name__ == "__main__":
    app.run(main)
