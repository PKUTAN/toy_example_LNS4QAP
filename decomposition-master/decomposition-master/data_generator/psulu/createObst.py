from __future__ import division
import argparse
import yaml
import random
import numpy as np
import math

def firstPassCommandLine():
    
    # Creating the parser for the input arguments
    parser = argparse.ArgumentParser(description='Path Planning based on MILP')

    # Positional argument - Input XML file
    parser.add_argument('-n', type=int, default=20,
                        help='Number of obstacles', dest='numObst')
    parser.add_argument('-o', type=str, default='data_generator/psulu/config/newEnvi.yaml',
                        help='Output File name', dest='outFile')
    parser.add_argument('-maxL', type=int, default=0.05,
                        help='Max Length', dest='maxLen')
    parser.add_argument('-minL', type=int, default=0.05,
                        help='Min Length', dest='minLen')
    parser.add_argument('-maxW', type=int, default=0.05,
                        help='Max Width', dest='maxWid')
    parser.add_argument('-minW', type=int, default=0.05,
                        help='Min Width', dest='minWid')
    parser.add_argument('-data_dir', type=str, default='data/psulu',
                        help='Dir to save LP files', dest='data_dir')
    parser.add_argument('-num_wp', type=int, default=19,
                        help='Number of waypoints', dest='num_wp')
    parser.add_argument('-i', type=str, default='data_generator/psulu/config/param.yaml',
                        help='Input File (Yaml Formal)', dest='inputFile')

    # Parse input
    args = parser.parse_args()
    return args

def getRand(minval=0, maxval=1):
    '''
    Generates a random number with uniform distribution in a given range
    '''	
    return random.uniform(minval, maxval)

def main(args):
    # Parameters
    numObst = args.numObst
    minL    = args.minLen 
    maxL    = args.maxLen
    minW    = args.minWid
    maxW    = args.maxWid
    outF    = args.outFile
  
    # Creating the obstacles
    obst = {}
    obst['environment'] = {}
    obst['environment']['obstacles'] = {}
    for n in range(numObst):
       cObst = {}
       cObst['shape'] = 'polygon'

       # Get random Length and width
       width  = getRand(minW, maxW)
       length = getRand(minL, maxL)
       rectTemplate = np.array([[0,0], [width, 0], [width, length], [0, length]])

       # Get a random rotation
       angle = 2*np.pi*getRand()
       rotationMat = np.array([[math.cos(angle), -math.sin(angle)], \
			       [math.sin(angle),  math.cos(angle)]]).T
       offset = np.array([width/2, length/2])
       rotatedRect = np.dot(rectTemplate - offset, rotationMat) + offset

       # Get random displacement
       dispW = getRand(0.2, 0.8)
       dispL = getRand(0.2, 0.8)
       rotatedRect = rotatedRect + np.array([dispW, dispL])

       # Compile the corners into the dictionary
       cObst['corners'] = rotatedRect.tolist()
       obstName = 'obs_%d'%(n)
       obst['environment']['obstacles'][obstName] = cObst

    with open(outF, 'w') as outfile:
       yaml.dump(obst, outfile, explicit_start=True)

if __name__ == '__main__':
    args = firstPassCommandLine()
    main(args)
