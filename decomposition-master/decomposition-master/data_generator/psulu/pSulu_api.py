import yaml as Y
import argparse
import pdb
import os, sys
import numpy as np

from numpy import linalg as LA

#Tiago: should this class be called ObstacleMap. Obstacle sounds like one object only.
#       Use explicit name of variables as oppose to H, V, g etc. it make it hard to read ;)
class ObstacleMap():
  '''
  Maintains the obstacle Map
  Currently doesn't support
  '''
  def __init__(self, obstMapFile=None):
    self.nObstacles     = None
    self.nSides         = None

    # Standard names used in the paper
    # H -> Polytope boundaries, V -> Corner Vertices, g->Plane Constant
    self.obstNormal   = None #H
    self.obstVert     = None #V
    self.obstOffset   = None #G
    self.obstName     = None

    if obstMapFile is not None:
      # Initializing the nSides to 4 for now - To be generalized
      self.nSides = 4

      self.__readObstMap__(obstMapFile)
      (self.obstNormal, self.obstOffset) = self.__computeHG__()

  def __readObstMap__(self, obstMapFile):
    '''
    Read Obstacle map from the YAML
    '''

    # Read Obstacle YAML map
    print('Reading Obstacle Map: %s'%obstMapFile)
    stream = open(obstMapFile, 'r')
    envParams = Y.load(stream)

    # Initialize the class objects
    obstacles = envParams['environment']['obstacles']
    self.nObstacles = len(obstacles.keys())

    # Read obstancles from the environment file
    self.obstVert = np.zeros((self.nSides,2,self.nObstacles))
    self.obstName = []
    self.zInit    = []
    for i, obstName in enumerate(obstacles.keys()):
      self.obstVert[:,:,i] = obstacles[obstName]['corners']
      self.obstName.append(obstName)
      try:
         # Search for initialization variables
         cKeys = obstacles[obstName].keys()
         for k in cKeys:
           if 'init' in k:
              stepnum = int(k.replace('init_',''))
              self.zInit.append({'stepnum': stepnum, 'obstname': obstName, 'obstnum': i,\
                                        'side': obstacles[obstName][k]})
      except:
         continue

    return

  def __computeHG__(self):
    '''
    Computes H and G from the corners
    '''
    # Initialize
    obstVert  = np.array(self.obstVert)
    nObst     = self.nObstacles
    nSides    = self.nSides

    # Lets compute sides
    obstNormal = np.zeros((self.nSides,2,self.nObstacles))
    obstOffset = np.zeros((nObst,nSides))
    for obst in range(nObst):
        for side in range(nSides-1):
          obstNormal[side,:,obst] = [obstVert[side+1,1,obst]-obstVert[side,1,obst], \
                                        obstVert[side,0,obst]-obstVert[side+1,0,obst]]
          obstNormal[side,:,obst] = obstNormal[side,:,obst]/\
                                        LA.norm(obstVert[side+1,:,obst]-obstVert[side,:,obst],2)
          obstOffset[obst,side]   = np.dot(obstNormal[side,:,obst], obstVert[side,:,obst])

        obstNormal[nSides-1,:,obst] = [obstVert[0,1,obst]-obstVert[nSides-1,1,obst], \
                                        obstVert[nSides-1,0,obst]-obstVert[0,0,obst]];
        obstNormal[nSides-1,:,obst] = obstNormal[nSides-1,:,obst]/\
                                        LA.norm(obstVert[0,:,obst]-obstVert[nSides-1,:,obst],2)
        obstOffset[obst,nSides-1]   = np.dot(obstNormal[nSides-1,:,obst], \
                                                obstVert[nSides-1,:,obst])

    obstNormal = np.rollaxis(obstNormal, 2)
    self.obstVert = np.rollaxis(obstVert, 2)
    return (obstNormal, obstOffset)

class pSulu(object):
  '''
  Base class for different pSulu Implementations
  '''

  def __init__(self, configFile=None):
    self.environment        = None
    self.start_location     = None
    self.end_location       = None
    self.chance_constraint  = None
    self.waypoints          = None
    self.max_velocity       = None
    self.max_control        = None
    self.coVarX             = None
    self.coVarY             = None

    if configFile is not None:
      self.__parseConfig__(configFile)
      self.obstMap = self.readEnvironment()

      # Fixing Covariance
      self.coVarY = self.coVarY or 0.001
      self.coVarX = self.coVarX or 0.001

  def setParameters(parameters):
    '''
    Parameters of the class are initialized using the dictionary
    '''
    # Initialize the class objects
    for params in vars(self).keys():
      try:
        setattr(self, params, configParams[params])
      except:
        print('Attribute %s missing in the parameters'%params)

    self.obstMap = self.readEnvironment()
    return

  def __parseConfig__(self, configFile):
    '''
    Reads the config file and initialises the objects
    '''

    # Read the config file
    #pdb.set_trace()
    stream = open(configFile, 'r')
    configParams = Y.load(open(configFile, 'r'))

    # Initialize the class objects
    for params in vars(self).keys():
      try:
        setattr(self, params, configParams[params])
      except:
        print('Attribute %s missing in the parameters'%params)

    # Convert string to native types
    self.baseFolder = os.path.dirname(os.path.abspath(configFile)) + '/'
    self.__convertType__()
    return

  def __convertType__(self):
    '''
    Converts string to native datatype - Hardcoded for now
    To be improvised later
    '''
    self.environment = self.environment.strip('[()]')
    self.start_location = [float(xx) for xx in self.start_location]
    self.end_location = [float(xx) for xx in self.end_location]
    
  def readEnvironment(self):
    '''
    Reads the obstacle map
    '''
    obstMap = ObstacleMap(self.environment)
    return obstMap

def main(args):
  inputConfig = args.inputConfig

  # Creating instance
  psulu = pSulu(inputConfig)
  return

def firstPassCommandLine():

  # Creating the parser for the input arguments
  parser = argparse.ArgumentParser(description='Path Planning based on MILP')

  # Positional argument - Input XML file
  parser.add_argument('-input', '--i', type=str,
                      default='data_generator/psulu/config/param.yaml',
                      help='Input Configuration File', dest='inputConfig')

  # Parse input
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = firstPassCommandLine()
  main(args)
