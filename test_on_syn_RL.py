import torch
import torch.nn.functional as F
import numpy as np
import random
import time
from sklearn.decomposition import PCA
from gurobipy import Model,GRB,quicksum
import networkx as nx
import argparse
from copy import deepcopy

from ddpg import DDPG
from dataset import QAPDataset
from data_loader import get_dataloader
from util import *

def LNS4instances(model,D,F,action,sol,verbose = False):
    prob_index = [i for i in range(D.shape[1])]

    next_sols = []
    next_objs = []

    for f,d,action,sol in zip(D,F,action,sol):
        f,d,sol = f.cpu().numpy(), d.cpu().numpy(),sol.cpu().numpy()
        index = sorted(action.tolist())
        out_index = np.setdiff1d(np.array(prob_index),np.array(index)).tolist()
        sol_,obj_ = LNS4RL(model.copy(),d,f,sol,index,out_index,0)
        next_sols.append(torch.tensor(sol_))
        next_objs.append(torch.tensor([obj_]))
        
    return  next_sols, next_objs

def LNS4RL(model,D,F,sol,index, out_index, start_time):
    N = len(index)
    per_sol = form_per_np(sol)
    per_assigned = per_sol[out_index]
    sub_loc = sorted([j for i,j in sol if i in index])

    x = model.addMVar(shape=(N,N),vtype= GRB.BINARY,name='x')
    
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

def initial_sol(F,D,device):
    '''
        We offer mutiple initial methods including:
        1) random initialize 
        2) learnable network initialize
    '''
    
    N = F.shape[0]
    
    ####random initialize
    prob_index = [i for i in range(N)]
    init_loc = random.sample(prob_index,len(prob_index))
    sol = [[i,j] for i,j in enumerate(init_loc)]
    obj = cal_obj(sol,D,F,device)

    ####learnable network initialize(TODO)

    return torch.tensor(sol),obj

def featurize(F,D,cur_sol,device):
    b,n,n = F.shape
    # import pdb;pdb.set_trace()
    
    features = torch.zeros((b,n,3*n)).to(device)

    for i in range(b):
        sol = form_per(cur_sol[i],device)
        loc_fea = torch.matmul(sol,D[i])
        features[i] = torch.cat([F[i],sol,loc_fea] , dim= -1)

    return features 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='RL for solving QAP using LNS')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=10, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=80, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=490, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--gpu', default= 'cuda:0',type=str,help='"cuda:0" for GPU utilization; "cpu" for CPU utilization')
    args = parser.parse_args()

    device = args.gpu
    if args.seed > 0:
        np.random.seed(args.seed)
    
    batchsize = 16 
    train_dataset = QAPDataset('QAPLIB',900,sets = 'train')
    train_dataloader = get_dataloader(train_dataset,batchsize)
    N = len(train_dataset)

    agent = DDPG(3*30, 1, batchsize, args)

    num_epochs = 10
    rollout_steps = 100
    local_size = 8

    model = Model('QAP')
    model.Params.TimeLimit = 5    
    model.Params.OutputFlag = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        step = 0
        for instances in train_dataloader:
            Fs,Ds,names = instances['Fi'],instances['Fj'],instances['name']
            batch_size, prob_size, _ = Fs.shape

            Fs = Fs.to(device)
            Ds = Ds.to(device)

            init_sols = []
            init_objs = []
            for F,D in zip(Fs,Ds):
                init_sol,obj = initial_sol(F,D,device)
                init_sols.append(init_sol)
                init_objs.append(torch.tensor([obj]))
            init_sols = torch.stack(init_sols,dim= 0).to(device)
            init_objs = torch.stack(init_objs,dim= 0).to(device)

            init_features = deepcopy(featurize(Fs,Ds,init_sols,device))

            # import pdb ; pdb.set_trace()
            cur_objs = init_objs
            features = init_features
            cur_sols = init_sols

            agent.is_training = True

            for t_rollout in range(rollout_steps):
                #reset when the episode starts
                if t_rollout == 0:
                    agent.reset(to_numpy(features))
                
                #agent's action selections
                with torch.no_grad():
                    if t_rollout <= args.warmup:
                        actions = agent.random_action()
                        actions_index = torch.topk(torch.from_numpy(actions),local_size)[1].numpy()
                    else:
                        actions = agent.select_action(features)
                        actions_index = torch.topk(torch.from_numpy(actions),local_size)[1].numpy()
                
                # import pdb; pdb.set_trace()
                next_sols,next_objs = LNS4instances(model,Fs,Ds,actions_index,cur_sols)
                
                next_sols = torch.stack(next_sols,dim=0).to(device)
                next_objs = torch.stack(next_objs,dim=0).to(device)

                features = deepcopy(featurize(Fs,Ds,next_sols,device))

                reward = cur_objs - next_objs
                reward = reward/1000

                if rollout_steps and t_rollout == rollout_steps-1:
                    done = True
                else:
                    done = False
                
                # import pdb; pdb.set_trace()
                agent.observe(to_numpy(reward),to_numpy(features),done)

                #update
                cur_objs = next_objs
                cur_sols = next_sols
                
                             
                if t_rollout > args.warmup:
                    agent.update_policy()
                    # print('policy_loss:{},critic_loss:{}'.format(agent.actor_loss,agent.critic_loss))
            
            print('epoch:{},episode:{},t_rollout:{},name:{},reward:{}'.format(epoch,step,t_rollout,names,next_objs))
            step += 1

                


                




            

