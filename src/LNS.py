import numpy as np
import random
import time
from gurobipy import Model,GRB,quicksum

from .utils import form_per, solution,sub2whole,cal_obj

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


def LNS(model,D,F,sol,index, out_index,start_time):
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
