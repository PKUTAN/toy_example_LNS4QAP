from __future__ import print_function
import networkx as Nx
import luigi, argparse, glob, fnmatch
from pymhlib.demos.vertex_cover import VertexCoverInstance, VertexCoverSolution
import random, os, re, pulp
import numpy  as np
import pandas as pd
import tempfile

############
# HeuristicVC:
#     ./fastvc ../test-data/frb59-26-1.mis 10 60
# 
# NuMVC:
#     ./numvc ../test-data/frb59-26-1.mis 0 10 60
#
############

def createOpt(G):
    prob = pulp.LpProblem('MILP Minimum Vertex Cover', pulp.LpMinimize)
    vert = {}
    obj = 0
    for j, (v1, v2) in enumerate(G.edges()):
        if not(v1 in vert):
            vert[v1] = pulp.LpVariable('v'+str(v1), 0, 1, pulp.LpInteger)
            obj = obj + vert[v1]
        if not(v2 in vert):
            vert[v2] = pulp.LpVariable('v'+str(v2), 0, 1, pulp.LpInteger)
            obj = obj + vert[v2]

        prob.addConstraint(vert[v1] + vert[v2] >= 1, name='c%d'%j)

    prob.setObjective(obj)
    return prob

class GpickleToMIS:
    def __init__(self, inpF, scratchF, refresh=False):
        self.inpF = inpF
        self.fname, ext = os.path.splitext(self.inpF)

        assert ext == '.gpickle', "Accepts only gpickles"
        assert os.path.isfile(self.inpF), "File does not exist"

        self.misF = self.fname + '.mis'
        self.lpF  = self.fname + '.lp'

        # Now check existence of scratch
        self.scratchF = scratchF + '/' + os.path.dirname(self.fname) + '/'
        if not os.path.isdir(self.scratchF):
            os.makedirs(self.scratchF)

        self.refresh  = refresh

        # Create the MIS file and store in the same place as Gpickle
        self.toMIS()

        return

    def toMIS(self):
        if not os.path.isfile(self.misF) or self.refresh:
            G = Nx.read_gpickle(self.inpF)
            lines = 'p edge %d %d\n'%(len(G.nodes), len(G.edges))
            for n1, n2 in G.edges:
                lines += 'e %d %d\n'%(n1 + 1, n2 + 1)

            with open(self.misF, 'w') as F:
                F.write(lines)

        return

    def toMVC(self):
        if not os.path.isfile(self.lpF) or self.refresh:
            G = Nx.read_gpickle(self.inpF)
            optP = createOpt(G)
            optP.writeLP(self.lpF)

        return

    def getMIS(self):
        return self.misF

    def getLP(self):
        self.toMVC()
        return self.lpF

    def getLog(self, str):
        return self.scratchF + os.path.basename(self.fname) + '.%s.log'%str

class SolverVC(luigi.Task):
    task_namespace = 'solvers'

    # Must pass parameters
    probF      = luigi.Parameter()
    scratchF   = luigi.Parameter()
    timelimit  = luigi.IntParameter(default=60)
    gurobi     = luigi.BoolParameter(default=True)

    def input(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        misF = gpickle2mis.getLP()
        return luigi.LocalTarget(misF)

    def output(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.gurobi:
            cstr = 'gurobi'
        else:
            cstr = 'scip'

        logF = gpickle2mis.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inp = self.input().path
        out = self.output().path

        # Create input instances
        if self.gurobi:
            with tempfile.NamedTemporaryFile(suffix='.sol') as temp:
                os.system('gurobi_cl TimeLimit=%d ResultFile=%s %s'%(self.timelimit, temp.name, inp))
                temp.seek(0)
                clines = temp.readlines()

            # Read Gurobi log
            cobj = -1
            for cl in clines:
                if 'Objective value' in str(cl):
                   cobj = float(re.findall(r'\d+', str(cl))[0])
                   break
                
        else:
            #TBD
            pass

        with open(out, 'w') as F:
            F.write('Best Obj %d'%cobj)

        return

    def complete(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.gurobi:
            cstr = 'gurobi'
        else:
            cstr = 'scip'

        logF = gpickle2mis.getLog(cstr)
        return os.path.isfile(logF)

class GreedyVC(luigi.Task):
    task_namespace = 'fastvc'

    # Must pass parameters
    probF    = luigi.Parameter()
    scratchF = luigi.Parameter()
    greedy   = luigi.BoolParameter(default=True)

    def input(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        misF = gpickle2mis.getMIS()
        return luigi.LocalTarget(misF)

    def output(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.greedy:
            cstr = 'greedy'
        else:
            cstr = '2approx'

        logF = gpickle2mis.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inp = self.input().path
        out = self.output().path

        # Create input instances
        instance = VertexCoverInstance(inp)
        solution = VertexCoverSolution(instance)
        if self.greedy:
            solution.greedy_construction()
            obj = solution.obj()
        else:
            solution.two_approximation_construction()
            solution.remove_redundant()
            obj = solution.obj()

        with open(out, 'w') as F:
            F.write('Best Obj %d'%obj)

        return

    def complete(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.greedy:
            cstr = 'greedy'
        else:
            cstr = '2approx'

        logF = gpickle2mis.getLog(cstr)
        return os.path.isfile(logF)

class HeuristicVC(luigi.Task):
    task_namespace = 'fastvc'

    # Must pass parameters
    probF      = luigi.Parameter()
    scratchF   = luigi.Parameter()
    timelimit  = luigi.IntParameter(default=60)
    seed       = luigi.IntParameter(default=60)
    fast       = luigi.BoolParameter(default=True)

    def input(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        misF = gpickle2mis.getMIS()
        return luigi.LocalTarget(misF)

    def output(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.fast:
            cstr = 'fastvc.%d'%self.seed
        else:
            cstr = 'numvc.%d'%self.seed

        logF = gpickle2mis.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inp = self.input().path
        out = self.output().path
        if self.fast:
            os.system('./approx/mvc/FastVC/fastvc %s %d %d > %s'%(inp, self.seed, self.timelimit, out))
        else:
            os.system('./approx/mvc/NuMVC_heap/numvc %s 0 %d %d > %s'%(inp, self.seed, self.timelimit, out))
        return

    def complete(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.fast:
            cstr = 'fastvc.%d'%self.seed
        else:
            cstr = 'numvc.%d'%self.seed

        logF = gpickle2mis.getLog(cstr)
        return os.path.isfile(logF)

class ParallelRun(luigi.WrapperTask):
    probF     = luigi.Parameter()
    scratchF  = luigi.Parameter()
    timelimit = luigi.IntParameter(default=60)
    refresh   = luigi.BoolParameter(default=False)

    # Number of runs for random seed
    nruns     = luigi.IntParameter(default=1)

    def requires(self):
        gurobiproc = []
        itr   = 1
        other = []
        for root, dirnames, filenames in os.walk(self.probF):
           for filename in fnmatch.filter(filenames, '*.gpickle'):
               cfile = os.path.join(root, filename)

               ## Fast vertex cover
               fastvcproc = []
               numvcproc  = []
               for j in range(self.nruns):
                   seed = random.randint(1, 1000)
                   fastvcproc += [HeuristicVC(probF=cfile, scratchF=self.scratchF, 
                                                timelimit=self.timelimit, seed=seed)]
                   numvcproc  += [HeuristicVC(probF=cfile, scratchF=self.scratchF, 
                                          timelimit=self.timelimit, seed=seed, fast=False)]

               other += [GreedyVC(probF=cfile, scratchF=self.scratchF, greedy=True), 
                         GreedyVC(probF=cfile, scratchF=self.scratchF, greedy=False)]

               ## first run HeuristicVC
               if not (itr%4):
                  luigi.build(other, workers=len(other), local_scheduler=True)
                  other = []
                  itr = 1

               itr = itr + 1

        return

def readlog(logF):
    # Read log
    # import pdb; pdb.set_trace()
    with open(logF, 'r') as F:
       clines = F.readlines()

    obj = -1
    for cl in clines:
       if 'Best' in cl:
          try:
             cobj = re.findall(r'\d+', cl)
             if len(cobj) == 1:
                 obj = float(cobj[0])
             else:
                 obj = float(cobj[0]+'.'+cobj[1])
             break
          except:
             pass

    return obj

def compileLogs(probF, scratchF, outlogF):
    '''
    Compiles all the logs and creates a single file
    '''

    firstpass = True
    allobj = {}
    allobj['method'] = []

    # import pdb; pdb.set_trace()
    
    for root, dirnames, filenames in os.walk(probF):
       for filename in fnmatch.filter(filenames, '*.gpickle'):
           cfile = os.path.join(root, filename)
           fname, _ = os.path.splitext(os.path.basename(cfile))

           # Get log folder
           if firstpass:
               gpickle2mis = GpickleToMIS(cfile, scratchF)
               logF = gpickle2mis.getLog('greedy')
               logF = os.path.dirname(logF)
               firstpass = False

           # Read each input files heursitics and compile them 
           cobj = {}
           for cF in glob.glob(logF + '/%s*'%fname):
               _, ext = cF.split(fname)
               method = re.sub(r'\d+|\.', '', ext).replace('log', '')
               obj = readlog(cF)

               if obj < 0:
                   # Objective always positive
                   continue

               if method in cobj:
                   cobj[method].append(obj)
               else:
                   cobj[method] = [obj, ]

           if len(cobj) != 2:
               print('skipping')
               continue

           # Take mean of variables seed
           allobj['method'].append(fname)
           for method, v in cobj.items():
               if method in allobj:
                   allobj[method].append(np.mean([v]))
               else:
                   allobj[method] = [np.nanmean([v]), ]

    # Save as csv
    df = pd.DataFrame(allobj)
    df.to_csv(outlogF, index=False, float_format='%.2f', sep='\t')
    return

def cmdLineParser():
    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser(description='run baselines')
    parser.add_argument('-i','--input', dest='inputF', type=str,
                        action='store', default='./data/',
                        help='Input folder for processing')
    parser.add_argument('-l','--logF', dest='logF', type=str,
                        action='store', default='./data/baseline.log',
                        help='Output log')
    parser.add_argument('-s','--scratch', dest='scratchF', type=str,
                        action='store', default='/tmp/jssong/data_heuristics',
                        help='Scratch folder for processing')
    parser.add_argument('-t','--time', dest='timelimit', type=int,
                        action='store', default=1800,
                        help='Time limit for the data points')
    parser.add_argument('-r','--refresh', dest='refresh', type=bool,
                        action='store', default=True,
                        help='Refresh MIS for the graphs')

    return parser.parse_args()

if __name__ == '__main__':
    args = cmdLineParser()
    inpF      = args.inputF
    scratchF  = args.scratchF
    timelimit = args.timelimit
    refresh   = args.refresh
    logF      = args.logF

    # Prepare the input for each iteration
    luigi.build([ParallelRun(probF=inpF, scratchF=scratchF, timelimit=timelimit, refresh=refresh)], 
                workers=1, local_scheduler=True)

    compileLogs(inpF, scratchF, logF)


