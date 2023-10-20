import gurobipy as gp
import numpy as np
import re
from pathlib import Path
import time
import random

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
        self.qap_path = Path('./data/synthetic_data/erdos10_0.6/')
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

def LNS_QAP(model ,prob_size ,local_size ,steps ,limited_times = 3 ,verbose = False):
    prob_index = [i for i in range(prob_size)]
    model.Params.TimeLimit = limited_times
    
    
    model.optimize()
    sol = solution(model)

    count = 0
    obj = 0
    model.Params.OutputFlag = 0
    start_time = time.time()
    for _ in range(steps):
        index = random.sample(prob_index,local_size)
        sol ,new_obj= LNS(model.copy() ,sol,index)
        if verbose == True:
            print(new_obj)
        if new_obj == obj:
            count += 1
        else:
            count = 0
        if count == 5:
            break
        obj = new_obj
    end_time = time.time()
    return sol , (end_time-start_time), obj


def LNS(model,sol,index):
    
    for var in sol:
        node, position = var
        if node not in index:
            model.addConstr(x[node,position] == 1)
    # import pdb; pdb.set_trace()        
    model.optimize()
    sol_new = solution(model)
    return sol_new , model.ObjVal

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
    F,D,per,sol,name, opt_obj = train_set.get_pair(0)

    N = F.shape[0]
    log_path = './log/LNS_QAP/' + name + '.log'

    print('The QAP problem is:{}, and the best solution is:{}'.format(name,sol))
    print("####################################################")
    m = Model('QAP')
    m.Params.TimeLimit =1000
    x = m.addMVar(shape = (N,N), vtype= GRB.BINARY,name='x')
    m.setObjective(quicksum(quicksum(F*(x@D@x.T))),GRB.MINIMIZE)

    

    # count =0
    # I = []
    # J = []
    # for i in range(N):
    #     if count == 70:
    #         break
    #     for j in range(N):
    #         if per[i][j] == 1:
    #             m.addConstr(x[i,j] == 1)
    #             I.append(i)
    #             J.append(j)
    #             count += 1
    # for i in range(60):
    #     m.addConstr(x[i,i] == 1)
    m.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N))
    m.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N))
    # m.Params.Method = 4
    # m.Params.Presolve = 0
    m.optimize(mycallback)

    save_obj_path = './result_gurobi/' + name +'_obj.npy'
    save_time_path = './result_gurobi/' + name +'_time.npy'

    np.save(save_obj_path,gurobi_interm_obj)
    np.save(save_time_path,gurobi_interm_time)

    # sol, time_duration , obj= LNS_QAP(m,N,15,100,limited_times=5,verbose=True)
    # print('The solution is:{} , the objective value is: {}, the time duration is:{}'.format(sol,obj,time_duration))
    # print('gap is :{}'.format((obj-opt_obj)/opt_obj))












    # x = m.addMVar(shape=(N, N), vtype=GRB.BINARY, name='x')

    # # Linearization variable
    # y = m.addMVar(shape=(N, N, N, N), vtype=GRB.BINARY, name='y')

    # # Objective function
    # obj = quicksum(F[i, j] * D[k, l] * y[i, j, k, l] for i in range(N) for j in range(N) for k in range(N) for l in range(N))
    # m.setObjective(obj, GRB.MINIMIZE)

    # # Row and column constraints for x
    # m.addConstrs(quicksum(x[i, j] for j in range(N)) == 1 for i in range(N))
    # m.addConstrs(quicksum(x[i, j] for i in range(N)) == 1 for j in range(N))

    # # Linearization constraints
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             for l in range(N):
    #                 m.addConstr(y[i, j, k, l] <= x[i, k])
    #                 m.addConstr(y[i, j, k, l] <= x[j, l])
    #                 m.addConstr(y[i, j, k, l] >= x[i, k] + x[j, l] - 1)

    # import pdb; pdb.set_trace()
    # # Optimize the model
    # m.optimize()

    # Binary decision variable
    # x = m.addMVar(shape=(N, N), vtype=GRB.BINARY, name='x')

    # # Set the initial solution as a unit matrix
    # initial_solution = np.eye(N)
    # for i in range(N):
    #     for j in range(N):
    #         x[i, j].start = initial_solution[i, j]

    # # Initial objective and constraints (based on the unit matrix)
    # objective_expr = 0
    # for i in range(N):
    #     for j in range(N):
    #         objective_expr += F[i, j] * D[i,j] * x[i, i] * x[j, j]

    # m.setObjective(objective_expr, GRB.MINIMIZE)

    # # Row and column constraints for x
    # m.addConstrs(quicksum(x[i, j] for j in range(N)) == 1 for i in range(N))
    # m.addConstrs(quicksum(x[i, j] for i in range(N)) == 1 for j in range(N))

    # # Optimize the model
    # m._x = x
    # m._N = N
    # m._F = F
    # m._D = D
    # m.optimize(callback_function)













    
    # A = [eval(v.VarName[1:])+eval(u.VarName[1:]) for v in m.getVars() for u in m.getVars() if v.x >0.99 and u.x > 0.99]
    # import pdb; pdb.set_trace()
    # m.setObjective(quicksum(F[i,k]*D[j,l] for i,k,j,l in A),GRB.MINIMIZE)

    # for i in range(N):
    #     for j in range(N):
    #         if i==j:
    #             x[i,j].Start = 1.0
    #         else:
    #             x[i,j].Start = 0.0
    
    # m.update()
    # import pdb;pdb.set_trace()
    # m.setObjective(quicksum(F[i,j]*D[k,l] for i in range(N) for j in range(N) for k in range(N) for l in range(N) if x[i,k].x >0.99 and x[j,l] > 0.99),GRB.MINIMIZE)
    # m.setObjective(quicksum(F[i,j]*((x[i]@D)@x.T[:j]) for i in range(N) for j in range(N)))

    # m.addConstr(x[0,0] == 1);
    # m.addConstr(x[1,1] == 1);
    # m.addConstr(x[2,2] == 1);
    # m.addConstr(x[3,3] == 1);
    # m.addConstr(x[4,4] == 1);
    # m.addConstr(x[5,5] == 1);
    # m.addConstr(x[6,6] == 1);
    # m.addConstr(x[7,7] == 1);
    # m.addConstr(x[8,8] == 1);
    # m.addConstr(x[9,9] == 1);
    # m.addConstr(x[10,10] == 1);
    # m.addConstr(x[11,11] == 1);
    # m.addConstr(x[12,12] == 1);
    # m.addConstr(x[13,13] == 1);
    # m.addConstr(x[14,14] == 1);
    # m.addConstr(x[15,15] == 1);
    # m.addConstr(x[16,16] == 1);
    # m.addConstr(x[17,17] == 1);
    # m.addConstr(x[18,18] == 1);
    # m.addConstr(x[19,19] == 1);
    # m.addConstr(x[20,20] == 1);
    # m.addConstr(x[21,23] == 1);
    # m.addConstr(x[22,25] == 1);
    # m.addConstr(x[23,17] == 1);
    # m.addConstr(x[24,2] == 1);
    # m.addConstr(x[25,13] == 1);
    # m.addConstr(x[26,6] == 1);
    # m.addConstr(x[27,4] == 1);
    # m.addConstr(x[28,8] == 1);
    # m.addConstr(x[29,3] == 1);
    
    # m.Params.TimeLimit = 320
    # m.optimize()
    # sol, time_duration , obj= LNS_QAP(m,N,8,100,limited_times=10,verbose=True)
    # print('The solution is:{} , the objective value is: {}, the time duration is:{}'.format(sol,obj,time_duration))
    # print('gap:{}'.format((obj-opt_obj)/opt_obj))