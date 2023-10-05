import argparse, pulp
import networkx as nx
import numpy as np
import random

def gen_graph(max_n, min_n, g_type='barabasi_albert', edge=4, p=0.15, weighted=False):
  cur_n = np.random.randint(max_n - min_n + 1) + min_n
  if g_type == 'erdos_renyi':
      g = nx.erdos_renyi_graph(n = cur_n, p=p)
  elif g_type == 'powerlaw':
      g = nx.powerlaw_cluster_graph(n = cur_n, m = 4, p = 0.05)
  elif g_type == 'barabasi_albert':
      g = nx.barabasi_albert_graph(n = cur_n, m = edge)

  if weighted:
    for i in range(len(g.nodes)):
      g.nodes[i]['weight'] = random.random()
      
  return g

def createOpt(G, weighted=False):
  prob = pulp.LpProblem('MILP Minimum Vertex Cover', pulp.LpMinimize)
  vert = {}
  obj = 0
  for j, (v1, v2) in enumerate(G.edges()):
    if not(v1 in vert):
      vert[v1] = pulp.LpVariable('v'+str(v1), 0, 1, pulp.LpInteger)
      if weighted:
        weight = random.random()
        obj = obj + weight * vert[v1]
      else:
        obj = obj + vert[v1]
    if not(v2 in vert):
      vert[v2] = pulp.LpVariable('v'+str(v2), 0, 1, pulp.LpInteger)
      if weighted:
        weight = random.random()
        obj = obj + weight * vert[v2]
      else:
        obj = obj + vert[v2]

    prob.addConstraint(vert[v1] + vert[v2] >= 1, name='c%d'%j)

  prob.setObjective(obj)
  return prob

def cmdLineParser():
  '''
  Command Line Parser.
  '''
  parser = argparse.ArgumentParser(description='Minimum Vertex Cover')
  parser.add_argument('-o', dest='outPrefix', type=str, action='store', \
                      default=None, help='Output Prefix')
  parser.add_argument('-g', dest='g_type', type=str, action='store', \
                      default='erdos_renyi', help='Graph type')
  parser.add_argument('-max_n', dest='max_n', type=int, action='store', \
                      default=700, help='max number of nodes')
  parser.add_argument('-min_n', dest='min_n', type=int, action='store', \
                      default=500, help='min number of nodes')
  parser.add_argument('-weighted', dest='weighted', action='store_true')
  return parser.parse_args()

def generateInstance(max_n, min_n, g_type, outPrefix=None, weighted=False, edge=4, p=0.15):
  G = gen_graph(max_n, min_n, g_type, edge, p)
  P = createOpt(G, weighted)

  if outPrefix != None:
    # Write out
    nx.write_gpickle(G, outPrefix.replace('lpfiles', 'gpickle') + '.gpickle')
    P.writeLP(outPrefix + '.lp')
 
def main():
  args = cmdLineParser()
  G = gen_graph(args.max_n, args.min_n, args.g_type, args.weighted)
  P = createOpt(G, args.weighted)

  if args.outPrefix != None:
    # Write out
    nx.write_gpickle(G, args.outPrefix + '.gpickle')
    P.writeLP(args.outPrefix + '.lp')
    
if __name__ == '__main__':
  main()

