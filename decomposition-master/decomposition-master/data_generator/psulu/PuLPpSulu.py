#!/usr/bin/python
from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import ndimage, misc, stats
from pSulu_api import pSulu

import math
import numpy as np
import pickle
import pulp
import argparse
import sys, pdb, math
import yaml as Y


class PathSolver:
    def __init__(self, N, u_max, xInit, xT, A, B):
        '''
        Mixed integer programming based path planning
        '''
        # Initialization parameters
        self.xInit = xInit
        self.xT    = xT

        # create solver instance
        self.prob   = pulp.LpProblem("MILP Path Solver", pulp.LpMinimize)

        # Remove noOverlap to update the constraints on recursive calls
        self.prob.noOverlap = 0

        # Result x and y positions
        self.wayPoint           = [] 
        self.activePoints       = []
        self.activePtsIdx       = []
        self.activeObstIdx      = []

        # Local hidden variables
        self.__N        = N
        self.__u_max    = u_max
        self.__A        = A
        self.__B        = B

        self.__nu       = len(self.__B[0])
        self.__nx       = len(self.__A[0])
        self.__M        = 10000 
        self.__epsilon  = 0.0001
        self.__nObst    = None
        self.__dObst    = None
        self.__obstIdx  = {}
        self.__active   = None
        self.__zVal     = []
        self.__H        = []
        self.__G        = []
        self.__sol      = []

        # create problem variables
        self.__createVar()
   
    def __createVar(self):
        # Real variables 
        self.u = pulp.LpVariable.dicts("u", (range(self.__N), range(self.__nu)), \
                                                -self.__u_max, self.__u_max)
        self.absu = pulp.LpVariable.dicts("absu", (range(self.__N), range(self.__nu)), \
                                                self.__epsilon, self.__u_max)
        self.x = pulp.LpVariable.dicts("x", (range(self.__N+1), range(self.__nx)))

    def __addObjective(self):
        # Problem Objective
        self.prob += pulp.lpSum([[self.absu[i][j] for i in range(self.__N)] \
                                                   for j in range(self.__nu)])

    def __addVarConstraint(self, zInit):
        '''
        Add constraints on the variables (Includes state transition constraint)
        '''
        # Constraints on state parameters
        # x[0] == xInit 
        for x_var, xi in zip(self.x[0].values(), self.xInit):
          self.prob.addConstraint(x_var == xi)

        for x_var, xi in zip(self.x[self.__N].values(), self.xT):
          self.prob.addConstraint(x_var == xi)

        # Constraints on intermediate variables
        for k in range(self.__N): 
            # absu >= u
            # absu + u >= 0
            for i in range(self.__nu):
              self.prob.addConstraint(self.absu[k][i] - self.u[k][i] >= 0)
              self.prob.addConstraint(self.absu[k][i] + self.u[k][i] >= 0)

            # State Transition modelled as a constraint
            # x[k+1] == A*x[k] + B*u[:,k]
            for x_var, a, b in zip(self.x[k+1].values(), self.__A, self.__B):
                self.prob.addConstraint((x_var - pulp.lpSum([(ai * xi) for ai, xi in zip(a, self.x[k].values())]) \
                            - pulp.lpSum([(bi * ui) for bi, ui in zip(b, self.u[k].values())])) == 0)


        # \sum_{i} z_{i} == dim(z_{i}) - 1 constraint
        for k in range(self.__N+1): 
            for i in range(self.__nObst):
                self.prob.addConstraint(pulp.lpSum([self.z[k][i][j] for j in range(self.__dObst)]) == \
                                                  self.__dObst-1)

        # Initialize z variables
        for k in range(len(zInit)):
            stepnum  = zInit[k]['stepnum']
            obstname = zInit[k]['obstname']
            obstnum  = zInit[k]['obstnum']
            cside    = zInit[k]['side']
 
            if (stepnum < 1) or (stepnum > self.__N + 1):
                print(stepnum)
                raise Exception('Step number in initialization outside range')

            if (cside < 0) or (cside > 3):
                raise Exception('Number of sides should be in the range [0, 3]')
           
            # Initialize them to satify basic constraint \sum(z[timestep][obst]) = 1
            sides = range(self.__dObst)
            sides.remove(cside)
            for i in sides:
                self.prob.addConstraint(self.z[stepnum][obstnum][i] == 1)

            # Fix the side of the obstacle using z
            self.prob.addConstraint(self.z[stepnum][obstnum][cside] == 0)

    def __addObstPt(self, idx, delta):
        '''
        Adds obstacle constraint for each way point
        '''
        # import pdb; pdb.set_trace()
        z = self.z[idx]
        x = list(self.x[idx].values())[:2]

        # H*x[k+1] - M*z[k] <= G 
        nConstraint = 0
        for mDelta, Hi, Gi, Zi in zip(delta, self.__H, self.__G, z.values()):
            # For each obstacle
            for m, h, g, zi in zip(mDelta, Hi, Gi, Zi.values()):

                # For each hyperplane of the obstacle
                if (str('constraint%d_%d'%(nConstraint, idx)) in self.prob.constraints): 
                    # if key already exists (http://www.coin-or.org/PuLP/pulp.html)
                    # it still needs to be updated 
                    # (that's what were doing -- overwriting the old key with the new data, so
                    del self.prob.constraints[str('constraint%d_%d'%(nConstraint, idx))] 
                    # delete the old key first, then add the new one below
                    # this gets rid of the pulp 
                    # "('Warning: overlapping constraint names:', 'constraint43_19')" 
                    # types of message-printouts 
                    # (http://pulp.readthedocs.org/en/latest/user-guide/troubleshooting.html)

                self.prob.addConstraint((pulp.lpSum([((-hi)*xi) for hi, xi in zip(h,x)]) \
                                         - self.__M*zi + m + g) <= 0,
                                         name='constraint%d_%d'%(nConstraint, idx))


                # Used for naming the constraints to replace on recursive calls with delta
                nConstraint = nConstraint + 1

    def writeOut(self, name):
        '''
        Write out the solution
        '''
        self.prob.writeLP(name + '.lp')
        return

    def __addObstConstraint(self, delta=None):
        '''
        Adds obstacle constraints based on the H and G matrix to all points
        '''
        if delta is None:
            delta = np.zeros((self.__N+1, self.__H.shape[0], self.__H.shape[1]))

        for k in range(self.__N+1): 
            # Adding constraints on the obstacle for each point
            self.__addObstPt(k, delta[k])

    def activeObstWayPt(self, x, mDelta, Hi, Gi, Zi):
        '''
        Test is the way points is active for a particular waypoint
        '''
        for m, h, g, z in zip(mDelta, Hi, Gi, Zi):
             hDotx = np.sum([hi*xi for hi, xi in zip(h, x)])

             # To acount for floating point inaccuracy
             if ((hDotx - self.__M*z - m - g) >= 0):
                return True
     
        return False
    
    def getActiveBoundaries(self):
        '''
        Check the active boundaries at each time point for each obstacle
        '''
        active_boundaries = [[False for i in range(self.__nObst)]\
                                for j in range(self.__N+1)]

        for t_i in range(self.__N+1):
            for obs_j in range(self.__nObst):
                for dobs_k in range(self.__dObst):
                    if pulp.value(self.z[t_i][obs_j][dobs_k]) < 1:
                        
                        ## debug
                        # print 'Z: ' +  str(self.z[t_i][obs_j][dobs_k])
                        # print 'Z value' + str(pulp.value(self.z[t_i][obs_j][dobs_k]))
                        # wp = self.wayPoint[t_i][:2]                        
                        # input("Press Enter to continue...")
                        ##debug

                        active_boundaries[t_i][obs_j] = dobs_k
                        break
        return active_boundaries

    def activewayPoint(self, idx, delta, mask=None):
        '''
        Checks if waypoint with index is active
        '''
        z = self.__zVal[idx]
        x = self.wayPoint[idx+1][:2]

        if mask is None:
            mask = [False for i in  range(self.__nObst)]

        # H*x[k+1] - M*z[k] <= G 
        for j, (mDelta, Hi, Gi, Zi, cMask) in enumerate(zip(delta, self.__H, self.__G, z, mask)):

            # Mask is used to look at only the obstacles not already tested for delta update
            if cMask is False:
                if self.activeObstWayPt(x, mDelta, Hi, Gi, Zi) is False:
                    continue
                else:
                    return j
            else:
                # Obstacle already parsed
                continue     

        return None

    def findActivePoints(self, mdelta, mask=None):
        
        '''
        Finds the active way points after each MILP run
        '''
        if mdelta is None:
            mdelta = np.zeros((self.__N-1, self.__H.shape[0], self.__H.shape[1]))

        if mask is None:
            mask = [[False for i in range(self.__nObst)] for j in range((self.__N)-1)]

        activePoints = []
        activePtsIdx = []
        activeObstIdx = []
        for k, cdelta, cmask in zip(range(self.__N-1), mdelta, mask):
            h = self.activewayPoint(k, cdelta, cmask) #
            if h  is not None:
                cWayPoint = self.wayPoint[k+1]
                activePoints.append(cWayPoint)
                activePtsIdx.append(k)
                activeObstIdx.append(h)
            
        return (activePoints, activePtsIdx, activeObstIdx)

    def addObstacles(self, H, G, zInit):
        '''
        Adds obstacles with H and G. This function initiates all the constraints for optimization
        '''
        nObst         = len(G)
        self.__H      = H
        self.__G      = G
        self.__nObst  = self.__G.shape[0]
        self.__dObst  = self.__G[0].shape[0]
        
        # z variable dim (nPoints x nObst)
        self.z = pulp.LpVariable.dicts("z", (range(self.__N+1), range(self.__nObst), \
                                        range(self.__dObst)), 0, 1, pulp.LpInteger)
        self.__addObjective()
        self.__addVarConstraint(zInit)
        self.__addObstConstraint()

    def solve(self, mdelta=None, outF=None):
        '''
        Solves and extracts the output variables into xVal and yVal
        '''
        # Modify the constraints with new delta

        #import pdb; pdb.set_trace()
        self.__addObstConstraint(mdelta)

        self.writeOut('MILP')
        # Solve the optimization problem
        self.prob.solve(pulp.GUROBI_CMD(options=[("TimeLimit", 10)]))

        # Populate the solution variable
        print("Status: ", pulp.LpStatus[self.prob.status])
        if 'Optimal' in pulp.LpStatus[self.prob.status]:
          self.feasible = True
        else:
          self.feasible = False

        # Get solution from waypoints
        if self.feasible:
           self.wayPoint = []
           for i in range(self.__N + 1):
              self.wayPoint.append([pulp.value(self.x[i][0]), pulp.value(self.x[i][1])])  

           self.__zVal = []
           for i in range(self.__N-1):
              self.__zVal.append([[pulp.value(self.z[i][j][k]) for k in range(self.__dObst)] \
                                       for j in range(self.__nObst)])

           # Find the active points in the optimized way points
           (self.activePoints, self.activePtsIdx, self.activeObstIdx) = \
                                                                self.findActivePoints(mdelta)

           self._sol = {}
           for var in self.prob._variables:
             self._sol[str(var)] = pulp.value(var) 

           if outF is not None:
             with open(outF + '.yaml', 'w') as outfile:
                Y.dump(self._sol, outfile, default_flow_style=False, explicit_start=True)

           return True
        else:
           return False


    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
      
        # Plot obstacles
        for g in self.__G:
            ax.add_patch(
                patches.Rectangle(
                    (g[0], g[1]),
                    np.abs(g[0] + g[2]),
                    np.abs(g[1] + g[3]),
                    alpha=0.5)
                )

        # Plot Waypoints
        for pt1, pt2 in zip(self.wayPoint[:-1], self.wayPoint[1:]):
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c='b')
            ax.scatter(pt1[0], pt1[1], c='g')
            ax.scatter(pt2[0], pt2[1], c='g')

        plt.show()

    # Class Interface functions
    def getWayPoints(self):
        return self.wayPoint

    def getActivePoints(self):
        return (self.activePtsIdx, self.activeObstIdx)

    def getObjective(self):
        return pulp.value(self.prob.objective)



            
class IRA(pSulu):
    '''
    Iterative Risk Allocation 
    '''
    def __init__(self, configFile):
        '''
        Initialize from the config file - For compatibility with Claudio's pSulu Input
        '''
        # Initializing base class
        super(self.__class__, self).__init__(configFile)

        # pSulu Parameters
        self.alpha  = 0.2
 
        # Obstacles related variables
        self.__H      = self.obstMap.obstNormal
        self.__G      = self.obstMap.obstOffset
        self.__V      = self.obstMap.obstVert
        self.__nObst  = self.obstMap.nObstacles
        self.__zInit  = self.obstMap.zInit

        # Transition parameters
        self.__N      = self.waypoints + 1
        self.__A, self.__B  = self.__initTransformationParam()
        self.__Q, self.__P0 = self.__initCovarianceParam()

        self.__delta      = self.__initDelta()
        self.__coVarMat   = self.computeCov()
        self.__wayPoints  = []
        self.__J          = []
        self.__deltaStep  = 10**(-3)
        self.mask         = None

        # Chance constraint condition
        if ((self.chance_constraint > 0.5) or (self.chance_constraint < 0)):
            raise Exception('Chance Constraint should be between 0 and 0.5 for convergence theoretical guarantees')

        # Formalizing the MILP solver
        u_max     = self.max_velocity
        self.MILP = PathSolver(self.__N-1, u_max, \
                               self.start_location, self.end_location, \
                               self.__A, self.__B)
        self.MILP.addObstacles(self.__H, self.__G, self.__zInit)

    @staticmethod
    def __initTransformationParam(dt=1):
        # Optimization variables
        # Transition Parameters
        A = [[1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        B = [[dt/2, 0], 
            [0, dt/2], 
            [1, 0],
            [0, 1]]
        return (A,B)
    
    def __initCovarianceParam(self):
        P0 = np.eye(len(self.__A[0])) 
        P0[0,0] = self.coVarX
        P0[1,1] = self.coVarY
        P0[2,2] = 0.0001
        P0[3,3] = 0.0001
        Qstate = np.eye(len(self.__A[0])//2)*0.001
        Qzeros = np.zeros([len(self.__A[0])//2, len(self.__A[0])//2])
        Q = np.bmat([[Qstate, Qzeros], [Qzeros, Qzeros]])
        Q = np.array(Q)
        return (Q,P0)

    def readObstMap(self, obstMap):
        '''
        Reads the H and G matrix that define the obstacle 
        '''
        import scipy.io as SIO
        matContent = SIO.loadmat(obstMap)

        # Obstacles
        H = np.rollaxis(matContent['obstMap']['H'][0][0], 2)
        G = matContent['obstMap']['g'][0][0]
        V = np.rollaxis(matContent['obstMap']['V'][0][0], 2)

        return (H, G, V)

    def get_fixed_feature_ira_delta(self):
        G = np.reshape(self._IRA__G, (1, self._IRA__G.size))
        H = np.reshape(self._IRA__H, (1, self._IRA__H.size))
        delta = self._IRA__delta
        if isinstance(delta, list):
            delta = np.asarray(delta)
            delta = np.reshape(delta, (1, delta.size))
        else:
            delta = np.reshape(delta, (1, delta.size))
        feature = np.concatenate([G, H, delta], axis=1)
        #assert(feature.size == 60)
        return feature
        
    def get_fixed_feature(self):
        G = np.reshape(self._IRA__G, (1, self._IRA__G.size))
        H = np.reshape(self._IRA__H, (1, self._IRA__H.size))
        delta = self.transformDelta()
        if isinstance(delta, list):
            delta = np.asarray(delta)
            delta = np.reshape(delta, (1, delta.size))
        else:
            delta = np.reshape(self.delta, (1, delta.size))
        feature = np.concatenate([G, H, delta], axis=1)
        assert(feature.size == 304)
        return feature

    def writeInfo(self, itr):
        '''Write out the compact representation of the problem via
        H, G and mdelta.'''

        f = open('INFO' + str(itr), "wb")
        np.savetxt(f, self.get_fixed_feature_ira_delta())
        f.close()

    def predict_solution(self, model, delta):
        dim = 56
        fixed_feature = self.get_fixed_feature_ira_delta()
        X_test = np.zeros((self.__N * self.__nObst, dim))
        z = self.MILP.z
        history = np.reshape([-1 for i in range(self.__N * self.__nObst)],
                             (1, self.__N * self.__nObst))
        predictions = []
        
        for i in range(self.__N):
            for j in range(self.__nObst):
                #import pdb; pdb.set_trace()
                feature = np.concatenate([fixed_feature, history], axis=1)
                pred = model.predict(feature)
                zero_index = np.argmax(pred)
                # predict backtrack (should not happen here)
                assert(zero_index < 4)
                if zero_index == 4:
                    zero_index = 0
                predictions.append(zero_index)
                history[0, i*self.__nObst+j] = zero_index
            
        N = int(self.__N)
        nObst = int(self.__nObst)
        for i in range(N):
            for j in range(nObst):
                zero_index = predictions[i*nObst+j]
                for k in range(4):
                    if k == zero_index:
                        self.MILP.prob.addConstraint(z[i][j][k]==0)
                    else:
                        self.MILP.prob.addConstraint(z[i][j][k]==1)
                        
    def solve(self, model=None, new_delta=None):
        '''
        Iteratively calls MILP with different delta
        '''
        # Iterate until convergence 
        J = float('inf')
        itr = 1
        while True:
          self.writeInfo(itr)
          # Solve the optimization with new delta
          oldJ = J
          #if new_delta:
          #    self._IRA__delta = new_delta
              
          #if model:
          #    self._IRA__delta = new_delta
          #    self.predict_solution(model, self._IRA__delta)
          #if not model:
          #    self.MILP.writeOut('MILP'+str(itr))

          # import pdb; pdb.set_trace()
          M = self.transformDelta()
          self.feasible = self.MILP.solve(M, 'sol'+str(itr))

          # import pdb; pdb.set_trace()
          # Write the optimization problem out
          
          if not self.feasible: 
             print('MILP Infeasible')
             return
 
          J = self.MILP.getObjective()

          # Saving way points
          self.__J.append(J)
          self.__wayPoints.append(self.MILP.getWayPoints())

          #self.plot()
          
          print("Objective Function Value: %f"%J)

          if (oldJ - J < 0.001):
            break 
          else:
            # Compute Residue for the delta
            (nActive, deltaResidual, newDelta) = self.__calcDeltaResidue(M)
            if nActive > 0:
              riskInc = deltaResidual/nActive
              self.__delta = self.__reallocate(newDelta, riskInc)

          # Iteration count
          itr = itr + 1

          break

    def __reallocate(self, newDelta, riskInc):
      '''
      Reallocated the risk to active points
      '''
      idx = np.where(np.equal(newDelta, None))
      for iTime, cObst in zip(idx[0], idx[1]):
           _newDelta_ = self.__delta[iTime][cObst] + riskInc

           # Clips the delta between 0 to 0.5 for theoretical guarantees
           newDelta[iTime][cObst] = max(0, min(_newDelta_, 0.5))

      return newDelta
      
    def __calcDeltaResidue(self, M):
      '''
      Computes the Delta Residue to redistribute
      '''
      newDelta      = [[None for x in range(self.__H.shape[0])] for y in range(self.__N)]
      deltaResidual = 0
      nActive       = self.__H.shape[0] * self.__N
 
      # find active constraints
      activeBoundaries = self.MILP.getActiveBoundaries() ## need to do
      waypoints        = self.MILP.getWayPoints()
      
      # For each way point
      for iTime in range(self.__N):
        # For each obstacle
        wp = waypoints[iTime]
        for cObst in range(self.__nObst):
          # find the equation of the active constraint
          activeIdx = activeBoundaries[iTime][cObst]
          activeH = self.__H[cObst][activeIdx]
          activeG = self.__G[cObst][activeIdx]

          # redistribute when risk used is less than allocated, recalculate risk
          if(M[iTime][cObst][activeIdx] + activeG <= np.dot(activeH,wp)):
            usedRisk = self.evaluateRisk(iTime,activeH,activeG)
            newDelta[iTime][cObst] = (self.__delta[iTime][cObst] * self.alpha + \
                                       (1-self.alpha)*usedRisk)
            deltaResidual = deltaResidual + (self.__delta[iTime][cObst] - \
                                            newDelta[iTime][cObst])
            nActive = nActive - 1

      return (nActive, deltaResidual, newDelta)

    def evaluateRisk(self, iTime, activeH, activeG):
        '''
        Evaluates Risk given active H and G for a waypoint
        '''
        # find waypoint
        wp = self.MILP.wayPoint[iTime][:2]
        P  = self.__coVarMat[iTime][:2,:2]
        
        # find affine transform param
        h_P_h = np.dot(activeH,np.dot(P,activeH))        
        safety_margin = np.dot(activeH,wp) - activeG
              
        # print 'safety ' + str(safety_margin)
        # print 'hPh ' + str(h_P_h)
        return stats.norm.sf(safety_margin,0,np.sqrt(h_P_h))

    def get_covariance(self):
        return self.__coVarMat


    def get_corners(self):
        return self.__V

    
    def plot(self):
        '''
        Plots all the way points over all iterations
        '''
        from pylab import rand
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        colormap = plt.cm.gist_ncar

        ## Different color points 
        clrMap = [colormap(i) for i in np.linspace(0, 0.9, len(self.__wayPoints))]
      
        ## Plot obstacles
        for corners in self.__V:
          x = corners[:,0]
          y = corners[:,1]
          plt.fill(x, y)

        ## Plot Waypoints
        if self.feasible:
          wayPt = self.__wayPoints[-1]
          covar = self.__coVarMat

          for i, (pt1, pt2) in enumerate(zip(wayPt[:-1], wayPt[1:])):
            #ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=clrMap[-1])
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c='r')
            ax.scatter(pt1[0], pt1[1], c='g')
            ax.scatter(pt2[0], pt2[1], c='g')
            if (i > 0) and (i<(len(wayPt) -1)):
              ells = Ellipse(np.array(wayPt[i]), width=3*(covar[i-1][0][0]), \
                                        height=3*(covar[i-1][1][1]), angle=0)
              ax.add_artist(ells)
              ells.set_alpha(0.4)
              ells.set_facecolor('g')

        plt.show()

    def plot_objective(self):
        plt.plot(self.__J, 'bo', self.__J, 'k')
        plt.ylabel("Objective Function value")
        plt.xlabel("IRA iteration")
        plt.show()

    def __deltaToDistance(self, idelta, H, idx):
        '''
        Computes the mdelta for each points
        '''        
        coVarMat = self.__coVarMat[idx][:2,:2]
        mdelta = []
        for delta, iH in zip(idelta, H): 
            cM = []
            for h in iH:
                cM.append(stats.norm.isf(delta, 0, \
                                np.sqrt(np.dot(h, np.dot(coVarMat, np.transpose(h))))))
            mdelta.append(cM)

        return mdelta

    def __deltaGToDistance(self, delta, H):
        '''
        Computes the mdelta for each points
        '''        
        idx = self.__N-1
        coVarMat = self.__coVarMat[idx][:2,:2]

        mdelta = []
        for idelta in delta:
          M = []
          for cdelta, iH in zip(idelta, H): 
              M.append(0)
          mdelta.append(M)

        return mdelta

    def transformDelta(self):
        '''
        Computes the m(\delta) 
        '''
        M = []
        H = self.__H
        for idx in range(self.__N):        
            idelta = self.__delta[idx]
            m = self.__deltaToDistance(idelta, H, idx)
            M.append(m)

        return M

    def __initDelta(self):
        '''
        Delta is the risk factor corresponding to each individual point
        '''
        delta = np.ones((self.__N, self.__H.shape[0])) 
        delta = delta * self.chance_constraint
        return delta

    def computeCov(self):
        '''
        Computes the covariance at each points
        '''
        A = self.__A
        Q = self.__Q
        coVarMat = [self.__P0]
        for idx in range(self.__N-1):
            coVarMat.append(np.dot(A, np.dot(coVarMat[idx], np.transpose(A))) + Q)
        return coVarMat

    def getWayPoints(self):
        return self.__wayPoints[-1]

    def isFeasible(self):
        return self.feasible

    def obstacle_features(self):
        # TODO: return G, H
        features = []
        #for corners in self.__V:
        #  x = corners[:,0]
        #  y = corners[:,1]
        #  features.extend([np.mean(x), np.mean(y)])
        
        return self.__H, self.__G

    
def firstPassCommandLine():
    
    # Creating the parser for the input arguments
    parser = argparse.ArgumentParser(description='Path Planning based on MILP')

    # Positional argument - Input XML file
    #parser.add_argument('-u_max', '--u_max', type=int, default=1.25,
    #                    help='Maximum value of velocity', dest='u_max')
    #parser.add_argument('-nSteps', '--N', type=int, default=20,
    #                    help='Number of steps to the destination', dest='nSteps')
    #parser.add_argument('-i', type=str, default=None,
    #                    help='Input Obstacle Map (.mat file)', dest='obstMap')

    # Parse input
    args = parser.parse_args()
    return args

def main(args):
    # Create IRA instance and solve it
    inp = args.inputFile
    with open(inp) as f:
        config = Y.load(f)

    config['waypoints'] = args.num_wp
    with open(inp, 'w') as f:
        Y.dump(config, f)
    
    itrRA = IRA(inp)
    #import pdb; pdb.set_trace()
    #itrRA.solve()
    #itrRA.plot()
    
    itrRA.MILP.writeOut("./scratch/MILP")
    # return itrRA.MILP.getObjective(), itrRA.obstacle_features()
        
if __name__ == '__main__':
    args = firstPassCommandLine()
    obj, features = main(args)
    print(features)
