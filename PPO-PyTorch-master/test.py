import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import roboschool
from gurobipy import GRB,Model,quicksum

from PPO import PPO
from dataset import QAPLIB

def LNS4instances(model,D,F,action,sol,verbose = False):    
    prob_index = [i for i in range(D.shape[1])]

    next_sols = []
    next_objs = []

    if action.shape[1] <= 1:
        # f,d,sol = f.cpu().numpy(), d.cpu().numpy(),sol.cpu().numpy()
        obj = cal_obj_np(sol,D,F)
        
        return sol,obj
    
    # f,d,sol = f.cpu().numpy(), d.cpu().numpy(),sol.cpu().numpy()

    index = sorted(action[0].tolist())
    out_index = np.setdiff1d(np.array(prob_index),np.array(index)).tolist()
    sol,obj = LNS4RL(model.copy(),D,F,sol,index,out_index,0)
    # next_sols.append(torch.tensor(sol_))
    # next_objs.append(torch.tensor([obj_]))
        
    return  sol, obj

def LNS4RL(model,D,F,sol,index, out_index, start_time):
    N = len(index)

    assert N > 0 ,'N 必须大于0'

    per_sol = form_per_np(sol)
    per_assigned = per_sol[out_index]
    sub_loc = sorted([j for i,j in sol if i in index])

    x = model.addMVar(shape=(N,N),vtype= GRB.BINARY,name='x')
    
    if N == 20:
        objective = (F*(x@D@x.T)).sum()
        model.setObjective(objective,GRB.MINIMIZE)

        model.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
        model.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N)); 

        model.optimize()

        sol = solution(model)
        obj = model.objVal

        return sol, obj
    else:
        sub_D = D[sub_loc][:,sub_loc]
        sub_F = F[index][:,index]
        sub_left_D = D[sub_loc] @ per_assigned.T
        sub_right_D = per_assigned @ D[:,sub_loc]
        sub_left_F = F[index][:,out_index]
        sub_right_F = F[out_index][:,index]


        objective = (sub_F*(x@sub_D@x.T)).sum() # 二次项
        objective += (sub_left_F*(x@sub_left_D)).sum() 
        objective += (sub_right_F*(sub_right_D@x.T)).sum()#一次项
        model.setObjective(objective,GRB.MINIMIZE)

        model.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
        model.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N)); 

        model.optimize()
        
        sub_sol = solution(model)
        sol_new = sub2whole(sub_sol,sol,index)
        obj = cal_obj_np(sol_new,D,F)

        # import pdb; pdb.set_trace()
        
        # gurobi_interm_obj.append(obj)
        # gurobi_interm_time.append(time.time()-start_time)

        return sol_new, obj

def form_per(sol,device):
    N = len(sol)
    per = torch.zeros((N,N)).to(device)
    
    for i,j in sol:
        per[i,j] = 1

    return per

def form_per_np(sol):
    N = len(sol)
    per = np.zeros((N,N))
    
    for i,j in sol:
        per[i,j] = 1

    return per

def cal_obj(sol,D,F,device):
    per = form_per(sol,device)
    obj = torch.sum(F*(per@D@per.T))  
    return obj

def cal_obj_np(sol,D,F):
    per = form_per_np(sol)
    obj = np.sum(F*(per@D@per.T))  
    return obj

def sub2whole(sub_sol,sol,index):
    sol_out = sol[:]
    sub_prob_size = len(sub_sol)
    unassigned_loc = sorted([loc for idx, loc in sol_out if idx in index])
    unassigned_flow = index

    for i in range(sub_prob_size):
        sol_out[unassigned_flow[i]][1] = unassigned_loc[sub_sol[i][1]]
    
    return sol_out
             
def solution(model):
    sol = []
    for v in model.getVars():
        if v.x == 1:
            sol.append(eval(v.VarName[1:]))
    return sol

def mycallback(model, where):
    # 如果是在找到一个新的解决方案时
    if where == GRB.Callback.MIPSOL:
        # 获取当前解的目标值
        objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        gurobi_interm_obj.append(objval)
        gurobi_interm_time.append(runtime)

def initial_sol(F,D):
    '''
        We offer mutiple initial methods including:
        1) random initialize 
        2) learnable network initialize
    '''
    
    N = F.shape[0]
    
    ####random initialize
    prob_index = [i for i in range(N)]
    # init_loc = random.sample(prob_index,len(prob_index))
    # sol = [[i,j] for i,j in enumerate(init_loc)]
    sol = [[i,i] for i in range(N)]

    obj = cal_obj_np(sol,D,F)

    ####learnable network initialize(TODO)

    return sol,obj

def featurize(F,D,cur_sol):
    n,n = F.shape
    # import pdb;pdb.set_trace()
    
    features = np.zeros((n,3*n))

    sol = form_per_np(cur_sol)
    loc_fea = np.matmul(sol,D)
    features = np.concatenate([F,sol,loc_fea] , axis= -1)

    return features 

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "erdos20"
    has_continuous_action_space = False
    max_ep_len = 2      # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 100    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    train_set = QAPLIB('train','erdos')
    F,D,per,sol,name, opt_obj = train_set.get_pair(0)

    # state space dimension
    # state_dim = env.observation_space.shape[0]
    state_dim = 60

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 20

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, local_size= 10)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    model = Model('QAP')
    model.Params.TimeLimit = 5    
    model.Params.OutputFlag = 0

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}_29_1.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    obj_result = []
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        init_sol, init_obj = initial_sol(F,D)
        state = featurize(F,D,init_sol)

        current_ep_reward = 0
        cur_sol = init_sol
        cur_obj = init_obj

        for t in range(1, max_ep_len+1):

            action, _  = ppo_agent.select_action(state,'train')
            # import pdb; pdb.set_trace()
            indices = np.array([np.where(action == 1)[1]])
            next_sol,next_obj = LNS4instances(model,F,D,indices,cur_sol)
            
            state = featurize(F,D,next_sol)
            cur_sol = next_sol
            if render:
                env.render()
                time.sleep(frame_delay)

            while t == max_ep_len+1:
                done = True
            else:
                done = False

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        obj_result.append(next_obj)
        print('Episode: {} \t\t obj: {}'.format(ep, next_obj))
        ep_reward = 0
    
    # env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(np.array(obj_result).mean()))
    print('min:',obj_result.min())

    print("============================================================================================")


if __name__ == '__main__':

    test()
