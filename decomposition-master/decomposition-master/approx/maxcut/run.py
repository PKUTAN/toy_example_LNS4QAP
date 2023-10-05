import argparse, os, fnmatch
import pulp, luigi, tempfile
import random, re, glob
from MaxCut import maxcut_random, maxcut_greedy
import networkx as Nx
import numpy as np
import pandas as pd

def _eval_cut(G, chi):
    # calculates total weight across a cut
    #
    # input:
    #   G: a numpy array representing an adjacency matrix
    #   chi: an array where all elements are +1 or -1, representing which side of the cut
    #   that vertex is in.
    #
    #
    total, V = 0, G.shape[0]
    for i in range(V):
        for j in range(i + 1, V):
            if chi[i] != chi[j]:
                total += G[i, j]
    return total

def createOpt(G):
    prob = pulp.LpProblem('MILP Maximum Cut', pulp.LpMinimize)
    edgeVar = {}
    for j, (v1, v2) in enumerate(G.edges()):
        e12 = getEdgeVar(v1, v2, edgeVar)
        for u in G._adj[v1]:
            neigh = G._adj[u]
            if v2 in neigh:
                e23 = getEdgeVar(v2, u, edgeVar)
                e13 = getEdgeVar(v1, u, edgeVar)

                prob.addConstraint(e12 <= e13 + e23)
                prob.addConstraint(e12 + e13 + e23 <= 2)

    obj = 0
    for (v1, v2) in G.edges():
        e12 = getEdgeVar(v1, v2, edgeVar)
        try:
            obj = obj + (G[v1][v2]['weight'])*e12
        except:
            obj = obj + e12

    prob.setObjective(-1*obj) # Note that this is LpMinimum
    return prob

def getEdgeVar(v1, v2, vert):
    u1 = min(v1, v2)
    u2 = max(v1, v2)
    if not ((u1, u2) in vert):
        vert[(u1, u2)] = pulp.LpVariable('u%d_%d'%(u1, u2), 0, 1, pulp.LpInteger)

    return vert[(u1, u2)]

class GpickleToMIS:
    def __init__(self, inpF, scratchF, refresh=False):
        self.inpF = inpF
        self.fname, ext = os.path.splitext(self.inpF)
        self.basename = os.path.basename(self.fname)
        self.refresh  = refresh

        assert ext == '.gpickle', "Accepts only gpickles"
        assert os.path.isfile(self.inpF), "File does not exist"

        self.lpF  = self.fname + '.lp'
        self.lpF  = self.lpF.replace('gpickle', 'lpfiles')

        # Now check existence of scratch
        self.scratchF = scratchF + '/' + os.path.dirname(self.fname) + '/'

        if not os.path.isdir(self.scratchF):
            os.makedirs(self.scratchF)

        self.toMIS()

        self.refresh  = refresh

        return

    def toMIS(self):
        if (not os.path.isfile(self.getMIS())) or self.refresh:
            G = Nx.read_gpickle(self.inpF)
            lines = '%d %d\n'%(len(G.nodes), len(G.edges))

            tag = {}
            for n1, n2 in G.edges:
                cmin = min(n1, n2)
                cmax = max(n1, n2)
                if (cmin, cmax) not in tag: 
                    try:
                        lines += '%d %d %f\n'%(cmin + 1, cmax + 1, G[n1][n2]['weight'])
                    except:
                        lines += '%d %d 1\n'%(cmin + 1, cmax + 1)

                    tag[(cmin, cmax)] = (cmin, cmax)

            with open(self.getMIS(), 'w') as F:
                F.write(lines)

        return

    def toMaxCut(self):
        if not os.path.isfile(self.lpF) or self.refresh:
            G = Nx.read_gpickle(self.inpF)
            optP = createOpt(G)
            optP.writeLP(self.lpF)

        return

    def getLP(self):
        #self.toMaxCut()
        return self.lpF

    def getMIS(self):
        scratchMISF  = self.scratchF.replace('/gpickle/', '/mis/')

        if not os.path.isdir(scratchMISF):
            os.makedirs(scratchMISF)

        misF = scratchMISF + '/%s.mis'%(self.basename)
        return misF

    def getInpF(self):
        return self.inpF        

    def getLog(self, str):
        scratchLogF  = self.scratchF.replace('/gpickle/', '/log/')

        if not os.path.isdir(scratchLogF):
            os.makedirs(scratchLogF)

        logF = scratchLogF + '/%s.%s.log'%(self.basename, str)
        return logF

class GreedyMC(luigi.Task):
    task_namespace = 'fastvc'

    # Must pass parameters
    probF    = luigi.Parameter()
    scratchF = luigi.Parameter()
    greedy   = luigi.BoolParameter(default=True)

    def input(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        inpF = gpickle2mis.getInpF()
        return luigi.LocalTarget(inpF)

    def output(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.greedy:
            cstr = 'greedy'
        else:
            cstr = '2approx'

        logF = gpickle2mis.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inpF = self.input().path
        if os.path.isfile(inpF):
            inpG = Nx.read_gpickle(inpF)
            inpG = Nx.to_numpy_matrix(inpG)
        else:
            raise Exception('File does not exist: %s'%inpF)

        out  = self.output().path

        # Create input instances
        if self.greedy:
            obj = _eval_cut(inpG, maxcut_greedy(inpG, False))
        else:
            obj = _eval_cut(inpG, maxcut_random(inpG, False))

        with open(out, 'w') as F:
            F.write('Best Obj %f'%obj)

        return

    def complete(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.greedy:
            cstr = 'greedy'
        else:
            cstr = '2approx'

        logF = gpickle2mis.getLog(cstr)
        return os.path.isfile(logF)

class HeuristicMC(luigi.Task):
    task_namespace = 'fastvc'

    # Must pass parameters
    probF      = luigi.Parameter()
    scratchF   = luigi.Parameter()
    heuristic  = luigi.Parameter()
    timelimit  = luigi.IntParameter(default=60)
    seed       = luigi.IntParameter(default=60)

    def input(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        inpFmis = gpickle2mis.getMIS()
        return luigi.LocalTarget(inpFmis)

    def output(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        cstr = '%s.%d'%(self.heuristic, self.seed)

        logF = gpickle2mis.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inp = self.input().path
        out = self.output().path

        # fM - maxcut file name: h  - heuristic name
        # s  - seed : r  - run time : ps - print screen
        with tempfile.NamedTemporaryFile(suffix='.sol') as temp:
            os.system('./approx/maxcut/MQLib/bin/MQLib -fM %s -h %s -s %d -r %d -ps > %s'
                        %(inp, self.heuristic, self.seed, self.timelimit, temp.name))
            temp.seek(0)
            clines = temp.readlines()

        # process MQLib log
        # import pdb; pdb.set_trace()
        clines = clines[-1]
        clines = clines.decode('utf-8')
        chi = [int(v) for v in clines.rstrip().split(' ')]
        
        # Get graph
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        inpF = gpickle2mis.getInpF()
        if os.path.isfile(inpF):
            inpG = Nx.read_gpickle(inpF)
            inpG = Nx.to_numpy_matrix(inpG)
        else:
            raise Exception('File does not exist: %s'%inpF)
            
        obj = _eval_cut(inpG, chi)
        with open(out, 'w') as F:
            F.write('Best Obj %f'%obj)

        return

    def complete(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        cstr = '%s.%d'%(self.heuristic, self.seed)
        logF = gpickle2mis.getLog(cstr)
        return os.path.isfile(logF)

class SolverMC(luigi.Task):
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
                   cl = cl.decode('utf-8')
                   cl = cl.replace('# Objective value = ', '')
                   cobj = np.abs(float(cl))
                   break
                
        else:
            #TBD
            pass

        with open(out, 'w') as F:
            F.write('Best Obj %f'%cobj)

        return

    def complete(self):
        gpickle2mis = GpickleToMIS(self.probF, self.scratchF)
        if self.gurobi:
            cstr = 'gurobi'
        else:
            cstr = 'scip'

        logF = gpickle2mis.getLog(cstr)
        return os.path.isfile(logF)

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

           if len(cobj) != 6:
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


def readlog(logF):
    # import pdb; pdb.set_trace()
    # Read log
    with open(logF, 'r') as F:
       clines = F.readlines()

    obj = -1
    for cl in clines:
       if 'Best' in cl:
          try:
             cobj = re.findall(r'\d+', cl)
             # if len(cobj) == 1:
             obj = float(cobj[0]+'.'+cobj[1])
             break
          except:
             pass

    return obj

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
               seed = random.randint(1, 1000)

               other += [HeuristicMC(probF=cfile, scratchF=self.scratchF, 
                                      heuristic="BURER2002", timelimit=self.timelimit, seed=seed), 
                         HeuristicMC(probF=cfile, scratchF=self.scratchF, 
                                      heuristic="DESOUSA2013", timelimit=self.timelimit, seed=seed), 
                         HeuristicMC(probF=cfile, scratchF=self.scratchF, 
                                      heuristic="DUARTE2005", timelimit=self.timelimit, seed=seed), 
                         HeuristicMC(probF=cfile, scratchF=self.scratchF, 
                                      heuristic="LAGUNA2009CE", timelimit=self.timelimit, seed=seed), 
                         GreedyMC(probF=cfile, scratchF=self.scratchF, greedy=True),
                         GreedyMC(probF=cfile, scratchF=self.scratchF, greedy=False)]

               ## first run HeuristicVC
               if not (itr%2):
                 luigi.build(other, workers=len(other), local_scheduler=True)
                 other = []
                 itr = 1

               itr = itr + 1

        ## reminder
        if len(other) != 0:
          luigi.build(other, workers=len(other), local_scheduler=True)

        return

def cmdLineParser():
    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser(description='run baselines')
    parser.add_argument('-i','--input', dest='inputF', type=str,
                        action='store', default='./maxcut_test/gpickle/',
                        help='Input folder for processing')
    parser.add_argument('-l','--logF', dest='logF', type=str,
                        action='store', default='./heuristics_log/maxcut_baseline.log',
                        help='Output log')
    parser.add_argument('-s','--scratch', dest='scratchF', type=str,
                        action='store', default='/tmp/jssong/data_heuristics',
                        help='Scratch folder for processing')
    parser.add_argument('-t','--time', dest='timelimit', type=int,
                        action='store', default=5,
                        help='Time limit for the data points')
    parser.add_argument('-r','--refresh', dest='refresh', type=bool,
                        action='store', default=False,
                        help='Refresh the data point generation')

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


