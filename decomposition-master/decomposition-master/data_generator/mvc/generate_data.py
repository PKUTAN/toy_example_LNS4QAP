import argparse
from probgen import generateInstance
import os

nodeLimit = 2000000
run_cmd = 'gurobi_cl ResultFile={0} TimeLimit=3600 NodeLimit=%d LogFile={2} {1}'%nodeLimit

def main(outFolder, offset, max_n, min_n, g_type, N=200, weighted=False, edge=4, p=0.15):
  print('Creating %s'%outFolder)
  os.system('mkdir -p %s'%outFolder)
  os.system('mkdir -p %s'%outFolder.replace('lpfiles', 'gpickle'))

  solve = False
  if 'unlabeled' in outFolder:
      solve = False

  for i in range(N):
    outPrefix = outFolder + '/input%d'%(offset+i)
    generateInstance(max_n, min_n, g_type, outPrefix=outPrefix,
                     weighted=weighted, edge=edge, p=p)
    if solve:
        cF = outPrefix + '.lp'
        solPrefix = outPrefix.replace('lpfiles', 'sol') + '.sol'
        logPrefix = outPrefix.replace('lpfiles', 'log') + '.log'
        os.system('mkdir -p %s'%os.path.dirname(solPrefix))
        os.system('mkdir -p %s'%os.path.dirname(logPrefix))
        os.system(run_cmd.format(solPrefix, cF, logPrefix))

  return

def cmdLineParser():
  '''
  Command Line Parser.
  '''
  parser = argparse.ArgumentParser(description='Minimum Vertex Cover')
  parser.add_argument('-o', dest='outFolder', type=str, action='store', \
                      default='../../../data/mvc_test/', help='Output Folder')
  parser.add_argument('-g', dest='g_type', type=str, action='store', \
                      default='erdos_renyi', help='Graph type')
  parser.add_argument('-s', dest='step', type=int, action='store', \
                      default=250, help='Step size')
  parser.add_argument('-N', dest='num', type=int, action='store', \
                      default=10, help='Number of instances of each type')
  parser.add_argument('-M', dest='numSteps', type=int, action='store', \
                      default=1, help='Number of steps')
  parser.add_argument('-max_n', dest='max_n', type=int, action='store', \
                      default=100, help='max number of nodes')
  parser.add_argument('-min_n', dest='min_n', type=int, action='store', \
                      default=100, help='min number of nodes')
  parser.add_argument('-weighted', dest='weighted', action='store_true')
  parser.add_argument('-edge', dest='edge', type=int, action='store', \
                      default=100, help='number of edges to attach.')
  parser.add_argument('-p', dest='p', type=float, action='store', \
                      default=0.15, help='edge probability')
  return parser.parse_args()

if __name__ == '__main__':
  args = cmdLineParser()
  folders = ['test', 'labeled', 'valid']
  unlabeled = ['unlabeled%d'%i for i in range(1)]
  folders.extend(unlabeled)

  max_n = args.max_n
  for kk, cstep in enumerate(range(args.numSteps)):
    if kk != 0:
      min_n = max_n
      max_n  = max_n + args.step
    else:
      min_n = args.min_n
      max_n = args.max_n

    offset = 0
    for prefix in folders:
      if args.g_type == 'erdos_renyi':
        folderPrefix = args.outFolder + '%d_%.2f/lpfiles/%s/'%(max_n, args.p, prefix)
      else:
        folderPrefix = args.outFolder + '%d_%d/lpfiles/%s/'%(max_n, args.edge, prefix)        
      main(folderPrefix, offset, max_n, min_n, args.g_type, args.num,
           args.weighted, args.edge, args.p)
      offset = offset + args.num
      
