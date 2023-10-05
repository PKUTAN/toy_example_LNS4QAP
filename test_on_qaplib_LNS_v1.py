import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
from pathlib import Path
import random
import re
import pygmtools as pygm
import time
import copy

# cls_list = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
cls_list = ['erdos']
class BaseDataset:
    def __init__(self):
        pass

    def get_pair(self, cls, shuffle):
        raise NotImplementedError

class QAPLIB(BaseDataset):
    def __init__(self, sets, cls, fetch_online=False):
        super(QAPLIB, self).__init__()
        self.classes = ['qaplib']
        self.sets = sets

        if cls is not None and cls != 'none':
            idx = cls_list.index(cls)
            self.cls_list = [cls_list[idx]]
        else:
            self.cls_list = cls_list

        self.data_list = []
        # self.qap_path = Path('./data/taillard45e')
        # self.qap_path = Path('./data/qapdata')
        self.qap_path = Path('./data/synthetic_data/erdos30_0.6/train')
        for inst in self.cls_list:
            for dat_path in self.qap_path.glob(inst + '*.dat'):
                name = dat_path.name[:-4]
                prob_size = int(re.findall(r"\d+", name)[0])
                if (self.sets == 'test' and prob_size > 90) \
                    or (self.sets == 'train' and prob_size > 1000):
                    continue
                self.data_list.append(name)

        # remove trivial instance esc16f
        if 'esc16f' in self.data_list:
            self.data_list.remove('esc16f')

        # define compare function
        def name_cmp(a, b):
            a = re.findall(r'[0-9]+|[a-z]+', a)
            b = re.findall(r'[0-9]+|[a-z]+', b)
            for _a, _b in zip(a, b):
                if _a.isdigit() and _b.isdigit():
                    _a = int(_a)
                    _b = int(_b)
                cmp = (_a > _b) - (_a < _b)
                if cmp != 0:
                    return cmp
            if len(a) > len(b):
                return -1
            elif len(a) < len(b):
                return 1
            else:
                return 0

        def cmp_to_key(mycmp):
            'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # sort data list according to the names
        self.data_list.sort(key=cmp_to_key(name_cmp))

    def __len__(self):
        return len(self.data_list)
    
    def get_pair(self, idx, shuffle=None):
        """
        Get QAP data by index
        :param idx: dataset index
        :param shuffle: no use here
        :return: (pair of data, groundtruth permutation matrix)
        """
        name = self.data_list[idx]

        dat_path = self.qap_path / (name + '.dat')
        if Path.exists(self.qap_path / (name + '.sln')):
            sln_path = self.qap_path / (name + '.sln')
            sln_file = sln_path.open()
        else:
            sln_file = None
        dat_file = dat_path.open()
        

        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield int(_)

        dat_list = [[_ for _ in split_line(line)] for line in dat_file]
        if sln_file != None:
            sln_list = [[_ for _ in split_line(line)] for line in sln_file]
        else:
            sln_list = None

        prob_size = dat_list[0][0]

        # read data
        r = 0
        c = 0
        Fi = [[]]
        Fj = [[]]
        F = Fi
        for l in dat_list[1:]:
            F[r] += l
            c += len(l)
            assert c <= prob_size
            if c == prob_size:
                r += 1
                if r < prob_size:
                    F.append([])
                    c = 0
                else:
                    F = Fj
                    r = 0
                    c = 0
        Fi = np.array(Fi, dtype=np.float32)
        Fj = np.array(Fj, dtype=np.float32)
        assert Fi.shape == Fj.shape == (prob_size, prob_size)
        #K = np.kron(Fj, Fi)

        # read solution
        if sln_list != None:
            sol = sln_list[0][1]
            obj = sln_list[0][-1]
            perm_list = []
            for _ in sln_list[1:]:
                perm_list += _
            assert len(perm_list) == prob_size
            perm_mat = np.zeros((prob_size, prob_size), dtype=np.float32)
            for r, c in enumerate(perm_list):
                perm_mat[r, c - 1] = 1

            return Fi, Fj, perm_mat, sol, name,obj
        else:
            return Fi,Fj,None, None ,name, None
    
def LNS_QAP(model,D,F ,prob_size ,local_size ,steps  ,limited_times = 3 ,verbose = False):
    prob_index = [i for i in range(prob_size)]
    # m = Model('Init')
    # m.Params.TimeLimit = 20    
    # m.Params.OutputFlag = 0

    # ##presolve to get a maybe better initial start
    # N = D.shape[0]
    # x = m.addMVar(shape = (N,N), vtype= GRB.BINARY,name='x')
    # m.setObjective(quicksum(quicksum(F*(x@D@x.T))),GRB.MINIMIZE)
    # m.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N))
    # m.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N))
    # m.Params.TimeLimit = 0
    # m.optimize()
    
    # sol = solution(m)
    
    start_time = time.time()
    init_loc = random.sample(prob_index,len(prob_index))

    sol = [[i,j] for i,j in enumerate(init_loc)]
    best_obj = obj = float('inf')
    best_sol = None
    count = 0
    model.Params.OutputFlag = 0
    for step in range(steps):
        ##仅仅是需要一个中间模块（RL-learning based），来生成index
        index = sorted(random.sample(prob_index,local_size))
        out_index = np.setdiff1d(np.array(prob_index),np.array(index)).tolist()
        sol ,new_obj= LNS(model.copy(),D,F,sol,index, out_index, limited_times,start_time)
        if verbose == True:
            print('step:{},obj:{},time:{},local size:{}'.format(step,new_obj,time.time()-start_time,local_size))
        if (obj - new_obj) == 0.:
            local_size = min(15,local_size+1)
        else:
            local_size = 10
            count = 0

        if local_size == 15:
            count += 1
        if count == 10:
            break

        obj = new_obj
        if best_obj > obj:
            best_obj = obj
            best_sol = sol
    end_time = time.time()

    return  best_sol, (end_time-start_time), best_obj


def LNS(model,D,F,sol,index, out_index, limit_time,start_time):
    # import pdb; pdb.set_trace()
    N = len(index)
    per_sol = form_per(sol)
    per_assigned = per_sol[out_index]
    sub_loc = sorted([j for i,j in sol if i in index])
    model.Params.TimeLimit = limit_time 
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
    
if __name__ == '__main__':
    train_set = QAPLIB('train','erdos')
    
    gurobi_interm_obj = []
    gurobi_interm_time = []
    from gurobipy import Model,GRB,quicksum
    import os
    
    for i in range(900):
        F,D,per,sol,name, opt_obj = train_set.get_pair(i)
        # import pdb;pdb.set_trace()

        N = F.shape[0]

        print('The QAP problem is:{}, and the best solution is:{}'.format(name,sol))
        print("####################################################")
        m = Model('QAP')

        sol, time_duration , obj= LNS_QAP(m,D,F,N,10,1000,limited_times=10,verbose=True)
        print('The solution is:{} , the objective value is: {}, the time duration is:{}'.format(sol,obj,time_duration))
        # print('gap is :{}'.format((obj-opt_obj)/opt_obj))

        # save_LNS_BKS = './data/synthetic_data/erdos30_0.6/train/' + name +'_random_init_random_LNS.sln'

        # with open(save_LNS_BKS, "w") as file:
        #     file.write('obj:{}, sol:{}, time:{}'.format(obj,sol,time_duration))

        # save_obj_path = './result_gurobi/' + name +'LNS_obj.npy'
        # save_time_path = './result_gurobi/' + name +'LNS_time.npy'

        # np.save(save_obj_path,gurobi_interm_obj)
        # np.save(save_time_path,gurobi_interm_time)