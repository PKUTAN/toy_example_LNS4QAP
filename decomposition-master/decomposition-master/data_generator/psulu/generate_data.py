#!/usr/bin/python
import createObst
import PuLPpSulu
import os
import sys
import pickle
import random


if __name__ == "__main__":
    # folders = {'train': 100,
    #            'valid': 10,
    #            'test': 50}

    folders = {'valid': 10}
    
    total = 0
    instance_obstacle_map = {}
    objs = {}
    obs_features = {}
    run_idx = 0
    obst_args = createObst.firstPassCommandLine()
    data_dir = obst_args.data_dir
    num_obst = obst_args.numObst
    # psulu_args = PuLPpSulu.firstPassCommandLine()
    num_wp = obst_args.num_wp
    old_milp_path = "./scratch/MILP.lp"
    old_obst_path = "data_generator/psulu/config/newEnvi.yaml"
    
    for k, v in folders.items():
        lp_dir = os.path.join(data_dir, '%d_%d' %(num_obst, num_wp+1), 'lpfiles', k)
        obst_dir = os.path.join(data_dir, '%d_%d' %(num_obst, num_wp+1), 'obstacles', k)

        if not os.path.exists(lp_dir):
            os.makedirs(lp_dir)

        if not os.path.exists(obst_dir):
            os.makedirs(obst_dir)
            
        for _ in range(v):
            createObst.main(obst_args)
            PuLPpSulu.main(obst_args)
        
            new_milp_path = os.path.join(lp_dir, "input%i.lp" %(run_idx))
            new_obst_path = os.path.join(obst_dir, "envi%i.yaml" %(run_idx))
            os.rename(old_milp_path, new_milp_path)
            os.rename(old_obst_path, new_obst_path)

            run_idx += 1
