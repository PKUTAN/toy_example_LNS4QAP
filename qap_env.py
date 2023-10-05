import torch
import torch.nn as nn
import numpy as np
from gym.utils import seeding
import os
import sys
import logging
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch.optim as optim



class qap_env():
    def __init__(self, instance,iter_steps):
        F,D,per,sol,name, opt_obj = instance
        self.n = F.shape[0]
        self.steps = iter_steps
        self.action_space = (self.n)
        self.obs_space = (self.n)
        self.obs = torch.zeros((1, 1, self.n, self.n))
        self.results = []
        self.best = -500
        self.f = open("./result/result.txt", 'w')

        f = open("./data/n_edges_710.dat", "r")
        for line in f:
            self.net = eval(line)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.obs = torch.zeros((1, 1, self.n, self.n))
        return self.obs

    def transform(self, x):
        up = nn.Upsample(size=84, mode='bilinear', align_corners=False)
        return up(x)*255
    
    def step(self, action):
        pass
        


def QAP(instance):
    return qap_env(instance)

if __name__ == '__main__':
    pass
