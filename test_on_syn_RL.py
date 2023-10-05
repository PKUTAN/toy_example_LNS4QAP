import torch.nn.functional as F
import numpy as np
import random
import time
from sklearn.decomposition import PCA
from gurobipy import Model,GRB,quicksum
import networkx as nx

from dataset import QAPDataset
from data_loader import get_dataloader


def LNS_QAP(model,D,F ,prob_size ,local_size ,steps  ,limited_times = 3 ,verbose = False):
    prob_index = [i for i in range(prob_size)]
    model.Params.TimeLimit = limited_times    
    model.Params.OutputFlag = 0
    start_time = time.time()
    init_loc = random.sample(prob_index,len(prob_index))

    sol = [[i,j] for i,j in enumerate(init_loc)]
    best_obj = obj = float('inf')
    best_sol = None
    count = 0
    for step in range(steps):
        ##仅仅是需要一个中间模块（RL-learning based），来生成index
        index = sorted(random.sample(prob_index,local_size))
        out_index = np.setdiff1d(np.array(prob_index),np.array(index)).tolist()
        sol ,new_obj= LNS(model.copy(),D,F,sol,index, out_index, start_time)
        if verbose == True:
            print('step:{},obj:{},time:{},local size:{}'.format(step,new_obj,time.time()-start_time,local_size))
        if (obj - new_obj) == 0.:
            local_size = min(20,local_size+1)
        else:
            local_size = 10
            count = 0

        if local_size == 20:
            count += 1
        if count == 10:
            break

        obj = new_obj
        if best_obj > obj:
            best_obj = obj
            best_sol = sol
    end_time = time.time()

    return  best_sol, (end_time-start_time), best_obj


def LNS(model,D,F,sol,index, out_index, start_time):
    # import pdb; pdb.set_trace()
    N = len(index)
    per_sol = form_per(sol)
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
    obj = cal_obj(sol_new,D,F)
    # import pdb; pdb.set_trace()
    
    gurobi_interm_obj.append(obj)
    gurobi_interm_time.append(time.time()-start_time)

    return sol_new, obj

def form_per(sol):
    N = len(sol)
    per = np.zeros((N,N))
    
    for i,j in sol:
        per[i,j] = 1

    return per

def cal_obj(sol,D,F):
    per = form_per(sol)
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
    init_loc = random.sample(prob_index,len(prob_index))
    sol = [[i,j] for i,j in enumerate(init_loc)]
    obj = cal_obj(sol,D,F)

    ####learnable network initialize(TODO)

    return sol,obj

def featurize(F,D,cur_sol):
    b,n,n = F.shape
    
    feature = np.zeros((b,n,))

    for i in range(b):

if __name__ == '__main__':
    train_dataset = QAPDataset('QAPLIB',900,sets = 'train')
    train_dataloader = get_dataloader(train_dataset,batch_size =1)
    N = len(train_dataset)

    batchsize = 1
    num_epochs = 100
    rollout_steps = 200

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for instances in train_dataloader:
            Fs,Ds,names = instances['Fi'],instances['Fj'],instances['name']

            

            init_sols = []
            init_objs = []
            for F,D in zip(Fs,Ds):
                init_sol,obj = initial_sol(F,D)
                init_sols.append(init_sol)
                init_objs.append(obj)

            init_feature = featurize(Fs,Ds,init_sols)

            total_reward = 0
            obj = init_objs
            feature = init_feature
            sol = init_sols
            for t_rollout in range(rollout_steps):





            import pdb; pdb.set_trace()

