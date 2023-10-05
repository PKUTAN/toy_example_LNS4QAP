import argparse, os, fnmatch
import pulp, luigi, tempfile
# import factorgraph as fg
import random, re, glob
import numpy as np
import pandas as pd
import gurobipy as G

GRB = G.GRB

class CATStoLP:
    def __init__(self, inpF, scratchF, refresh=False):
        self.inpF = inpF
        self.fname, ext = os.path.splitext(self.inpF)
        self.basename = os.path.basename(self.fname)
        self.refresh  = refresh

        assert ext == '.txt', "Accepts only txt from CATS generator"
        assert os.path.isfile(self.inpF), "File does not exist"

        self.lpF  = self.fname + '.lp'
        self.lpF  = self.lpF.replace('cats', 'lpfiles')

        # Now check existence of scratch
        self.scratchF = scratchF + '/' + os.path.dirname(self.fname) + '/'

        if not os.path.isdir(self.scratchF):
            os.makedirs(self.scratchF)

        self.refresh  = refresh
        return

    def getLP(self):
        return self.lpF

    def getInpF(self):
        return self.inpF        

    def getLog(self, str):
        scratchLogF  = self.scratchF.replace('/cats/', '/log/')

        if not os.path.isdir(scratchLogF):
            os.makedirs(scratchLogF)

        logF = scratchLogF + '/%s.%s.log'%(self.basename, str)
        return logF

def parseLP(prob):
    # Get objective
    obj = prob.getObjective()
    cnstr = {}
    cobj  = {}
    for idx in range(obj.size()):
        var = obj.getVar(idx)
        coeff = obj.getCoeff(idx)
        cobj[var.VarName] = coeff
        cnstr[var] = []

    # Get variables belong to a constraint
    varKeys = [*cnstr]
    for var in varKeys:
        cns = prob.getCol(var)
        for j in range(cns.size()):
            cnstr[var].append(cns.getConstr(j).__str__())

        # Rename
        cnstr[var.VarName] = cnstr[var]
        del cnstr[var]

    return (cobj, cnstr)

def runGreedy(obj, cnstr):
    absObj = []
    key    = []
    for k in obj.keys():
        absObj.append(np.abs(obj[k]))
        key.append(k)

    # iteratively add the best bid
    flag = np.zeros((len(key), ))
    coveredConstraint = set()
    sol = []
    greedyObj = 0
    while np.where(flag)[0].shape[0] != flag.shape[0]:
        bestidx = np.argmax(absObj)
        cKey    = key[bestidx]

        # Constraints of the current bid
        cc      = set(cnstr[cKey])    
        
        # Check conflicting constraints
        violatedCnstr = coveredConstraint.intersection(cc)
        if len(violatedCnstr) == 0:
            coveredConstraint = coveredConstraint.union(cc)
            greedyObj = greedyObj + absObj[bestidx]
            sol.append(cKey)

        # Mark the bid as visited
        flag[bestidx] = 1
        absObj[bestidx] = -1

    return greedyObj

def createFactorGraph(obj, cntrs, max_iter):
    # Make an empty graph
    g = fg.Graph()
    # Get max objective value
    cobj_max = 0
    for k in obj.keys():
        cobj = np.abs(obj[k])
        cobj_max = max(cobj_max, cobj)
        
    # Add relevant binary variables
    for k in obj.keys():
        cobj = np.abs(obj[k])

        # Create binary variable
        g.rv(k, 2)
        
        # Add single factors
        g.factor([k], potential=np.array([0.01, cobj/cobj_max]))

    # Add factors for each constraints
    cntr2var = {}
    for k in cntrs.keys():
        cc = cntrs[k]
        for ccv in cc:
            if ccv in cntr2var:
                cntr2var[ccv].append(k) 
            else:
                cntr2var[ccv] = [k]

    # Create joint factors
    for k in cntr2var.keys():
        jointVars = cntr2var[k]
        
        # create potential
        # Strong penalty for breaking constraints
        jointFactorShape = (2,) * len(jointVars)
        jointFactor = -100*np.ones(jointFactorShape, dtype=float)

        # Just enough to satisfy the constraints
        allzeros = (0,)*len(jointVars)
        jointFactor[allzeros] = 100 
        for i in range(len(jointVars)):
            allzeros = [0,]*len(jointVars)
            allzeros[i] = 1
            jointFactor[tuple(allzeros)] = 100

        g.factor(jointVars, potential=jointFactor) 
        break

    # Solve and get partial solution
    iter_, converged = g.lbp(max_iters=1000)
    marginals = g.rv_marginals()
    sol = {}
    for k, v in marginals:
        sol[k] = np.argmax(v)

    return sol

class beliefProp(luigi.Task):
    task_namespace = 'beliefProp'

    # Must pass parameters
    probF    = luigi.Parameter()
    scratchF = luigi.Parameter()
    max_iter = luigi.IntParameter(default=50)

    def input(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        inpF = cats2lp.getLP()
        return luigi.LocalTarget(inpF)

    def output(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        cstr = 'bp'
        logF = cats2lp.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inpF = self.input().path
        if os.path.isfile(inpF):
            inpProb = G.read(inpF)
        else:
            raise Exception('File does not exist: %s'%inpF)

        out  = self.output().path

        # Create input instances
        (cobj, cnstr) = parseLP(inpProb)
        partialSol = createFactorGraph(cobj, cnstr, self.max_iter)
        raise Exception('Not working currently')
        with open(out, 'w') as F:
            F.write('Best Obj %f'%obj)

        return

    def complete(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        cstr = 'bp'
        logF = cats2lp.getLog(cstr)
        return os.path.isfile(logF)

class GreedyCATS(luigi.Task):
    task_namespace = 'greedyCATS'

    # Must pass parameters
    probF    = luigi.Parameter()
    scratchF = luigi.Parameter()

    def input(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        inpF = cats2lp.getLP()
        return luigi.LocalTarget(inpF)

    def output(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        cstr = 'greedy'
        logF = cats2lp.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inpF = self.input().path
        if os.path.isfile(inpF):
            inpProb = G.read(inpF)
        else:
            raise Exception('File does not exist: %s'%inpF)

        out  = self.output().path

        # Create input instances
        (cobj, cnstr) = parseLP(inpProb)
        obj = runGreedy(cobj, cnstr)
        with open(out, 'w') as F:
            F.write('Best Obj %f'%obj)

        return

    def complete(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        cstr = 'greedy'
        logF = cats2lp.getLog(cstr)
        return os.path.isfile(logF)

def greedyRounding(obj, cnstr, sol):
    absObj = []
    key    = []
    csol   = []
    for k in obj.keys():
        absObj.append(np.abs(obj[k]))
        key.append(k)
        csol.append(sol[k])

    # iteratively add the best bid
    csol = np.array(csol)
    flag = np.zeros((len(key), ))
    coveredConstraint = set()
    sol = []
    greedyObj = 0
    while np.where(flag)[0].shape[0] != flag.shape[0]:
        if np.where(csol > 0)[0].shape[0] > 0:
            # Prefer to choose solution from LP over the objective
            bestidx = np.argmax(csol)
        else:
            # Once LP solutions are rounded, then iteratively check objective
            bestidx = np.argmax(absObj)

        # Constraints of the current bid
        cKey = key[bestidx]
        cc = set(cnstr[cKey])    
        
        # Check conflicting constraints
        violatedCnstr = coveredConstraint.intersection(cc)
        if len(violatedCnstr) == 0:
            coveredConstraint = coveredConstraint.union(cc)
            greedyObj = greedyObj + absObj[bestidx]
            sol.append(cKey)

        # Mark the bid as visited
        flag[bestidx] = 1
        absObj[bestidx] = -1
        csol[bestidx] = -1

    return greedyObj

class SolverLP(luigi.Task):
    task_namespace = 'solversLP'

    # Must pass parameters
    probF      = luigi.Parameter()
    scratchF   = luigi.Parameter()
    timelimit  = luigi.IntParameter(default=60)

    def input(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        inpF = cats2lp.getLP()
        return luigi.LocalTarget(inpF)

    def output(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        cstr = 'lp_rounding'
        logF = cats2lp.getLog(cstr)
        return luigi.LocalTarget(logF)

    def run(self):
        inp = self.input().path
        out = self.output().path

        if os.path.isfile(inp):
            prob = G.read(inp)
            (cobj, cnstr) = parseLP(prob)
        else:
            raise Exception('File does not exist: %s'%inpF)

        # Create input instances
        with tempfile.NamedTemporaryFile(suffix='.lp') as tempProb:
            # Read the file and do LP approximation
            obj = prob.getObjective()
            for idx in range(obj.size()):
                var = obj.getVar(idx)
                var.setAttr('VType', GRB.CONTINUOUS)

            # Write as temp file
            prob.write(tempProb.name)

            with tempfile.NamedTemporaryFile(suffix='.sol') as temp:
                os.system('gurobi_cl TimeLimit=%d ResultFile=%s %s'%(self.timelimit, temp.name, tempProb.name))
                temp.seek(0)
                clines = temp.readlines()

        # LP rounding 
        sol = {}
        for cl in clines:
            cc = cl.decode('UTF-8')
            if cc[0] == '#':
                continue
            
            var, val = cc.split(' ')
            val = float(val.replace('/n', ''))
            sol[var] = val

        obj = greedyRounding(cobj, cnstr, sol)
        obj = np.abs(obj)
        with open(out, 'w') as F:
            F.write('Best Obj %f'%obj)

        return

    def complete(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        cstr = 'lp_rounding'
        logF = cats2lp.getLog(cstr)
        return os.path.isfile(logF)

class Solver(luigi.Task):
    task_namespace = 'solvers'

    # Must pass parameters
    probF      = luigi.Parameter()
    scratchF   = luigi.Parameter()
    timelimit  = luigi.IntParameter(default=60)
    gurobi     = luigi.BoolParameter(default=True)

    def input(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        inpF = cats2lp.getLP()
        return luigi.LocalTarget(inpF)

    def output(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        if self.gurobi:
            cstr = 'gurobi'
        else:
            cstr = 'scip'

        logF = cats2lp.getLog(cstr)
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
                   cobj = float(str(cl.decode('UTF-8')).replace('# Objective value =', ''))
                   #cobj = float(re.findall(r'\d+', str(cl))[0])
                   break

            cobj = np.abs(cobj)
        else:
            #TBD
            pass

        with open(out, 'w') as F:
            F.write('Best Obj %f'%cobj)

        return

    def complete(self):
        cats2lp = CATStoLP(self.probF, self.scratchF)
        if self.gurobi:
            cstr = 'gurobi'
        else:
            cstr = 'scip'

        logF = cats2lp.getLog(cstr)
        return os.path.isfile(logF)

class ParallelRun(luigi.WrapperTask):
    probF     = luigi.Parameter()
    scratchF  = luigi.Parameter()
    timelimit = luigi.IntParameter(default=60)
    refresh   = luigi.BoolParameter(default=False)
    bp_iter   = luigi.IntParameter(default=50)

    # Number of runs for random seed
    nruns     = luigi.IntParameter(default=1)

    def requires(self):
        gurobiproc = []
        itr   = 1
        other = []
        for root, dirnames, filenames in os.walk(self.probF):
           for filename in fnmatch.filter(filenames, '*.txt'):
               cfile = os.path.join(root, filename)
               seed = random.randint(1, 1000)
               other += [GreedyCATS(probF=cfile, scratchF=self.scratchF), 
                         SolverLP(probF=cfile, scratchF=self.scratchF, timelimit=self.timelimit),]

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

def compileLogs(probF, scratchF, outlogF):
    '''
    Compiles all the logs and creates a single file
    '''

    firstpass = True
    allobj = {}
    allobj['method'] = []
    for root, dirnames, filenames in os.walk(probF):
       for filename in fnmatch.filter(filenames, '*.txt'):
           cfile = os.path.join(root, filename)
           fname, _ = os.path.splitext(os.path.basename(cfile))

           # Get log folder
           if firstpass:
               cats2lp = CATStoLP(cfile, scratchF)
               cstr = 'lp_rounding'
               logF = cats2lp.getLog(cstr)
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

           # Take mean of variables seed
           allobj['method'].append(fname)
           for method, v in cobj.items():
               if method in allobj:
                   allobj[method].append(np.mean([v]))
               else:
                   allobj[method] = [np.nanmean([v]), ]

    # Save as csv
    df = pd.DataFrame(allobj)
    if not os.path.isdir(os.path.dirname(outlogF)):
        os.mkdir(os.path.dirname(outlogF))

    # import pdb; pdb.set_trace()
    df.to_csv(outlogF, index=False, float_format='%.2f', sep='\t')
    return

def readlog(logF):
    # Read log
    with open(logF, 'r') as F:
       clines = F.readlines()

    obj = -1
    for cl in clines:
       if 'Best' in cl:
          try:
             #cobj = re.findall(r'\d+', cl)
             cobj = float(cl.strip('Best Obj'))
             #if len(cobj) == 1:
             #    obj = float(cobj[0])
             break
          except:
             pass

    return cobj

def cmdLineParser():
    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser(description='run baselines')
    parser.add_argument('-i','--input', dest='inputF', type=str,
                        action='store', default='./approx/CATS/test-data/cats/',
                        help='Input folder for processing')
    parser.add_argument('-l','--logF', dest='logF', type=str,
                        action='store', default='./heuristics_log/',
                        help='Output log')
    parser.add_argument('-s','--scratch', dest='scratchF', type=str,
                        action='store', default='/tmp/jssong/data_heuristics',
                        help='Scratch folder for processing')
    parser.add_argument('-t','--time', dest='timelimit', type=int,
                        action='store', default=5,
                        help='Time limit for the data points')
    parser.add_argument('-b','--bp_iter', dest='bp_iter', type=int,
                        action='store', default=50,
                        help='belief propagation max iterations')
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
    bp_iter   = args.bp_iter

    # Prepare the input for each iteration
    luigi.build([ParallelRun(probF=inpF, scratchF=scratchF, bp_iter=bp_iter,
                             timelimit=timelimit, refresh=refresh)], 
                workers=1, local_scheduler=True)

    #import pdb; pdb.set_trace()
    outF = logF + 'baseline.csv'
    compileLogs(inpF, scratchF, outF)


