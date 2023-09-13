import gurobipy as gp
import numpy as np
import re
from pathlib import Path
import time
import random

cls_list = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']

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
        self.qap_path = Path('./data/qapdata/')
        for inst in self.cls_list:
            for dat_path in self.qap_path.glob(inst + '*.dat'):
                name = dat_path.name[:-4]
                prob_size = int(re.findall(r"\d+", name)[0])
                if (self.sets == 'test' and prob_size > 90) \
                    or (self.sets == 'train' and prob_size > 90):
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
        sln_path = self.qap_path / (name + '.sln')
        dat_file = dat_path.open()
        sln_file = sln_path.open()

        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield int(_)

        dat_list = [[_ for _ in split_line(line)] for line in dat_file]
        sln_list = [[_ for _ in split_line(line)] for line in sln_file]

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
        sol = sln_list[0][1]
        perm_list = []
        for _ in sln_list[1:]:
            perm_list += _
        assert len(perm_list) == prob_size
        perm_mat = np.zeros((prob_size, prob_size), dtype=np.float32)
        for r, c in enumerate(perm_list):
            perm_mat[r, c - 1] = 1

        return Fi, Fj, perm_mat, sol, name


def LNS_QAP(model ,prob_size ,local_size ,steps ,limited_times = 3 ,verbose = False):
    prob_index = [i for i in range(prob_size)]
    model.Params.TimeLimit = limited_times
    total_time = 0 
    
    start_time = time.time()
    model.optimize()
    sol = solution(model)
    
    for _ in range(steps):
        index = random.sample(prob_index,local_size)
        sol ,obj= LNS(model.copy() ,sol,index)
        if verbose == True:
            print(obj)
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

if __name__ == '__main__':
    from gurobipy import Model,GRB,quicksum

    train_set = QAPLIB('train','tho')
    F,D,per,sol,name = train_set.get_pair(0)
                # if F.sum()==None:
                #     continue
    N = F.shape[0]
    
    m = Model('QAP')
    x = m.addMVar(shape = (N,N), vtype= GRB.BINARY,name='x')
    m.setObjective(quicksum(quicksum(F*(x.T@D@x))),GRB.MINIMIZE)
    
    m.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
    m.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N));
    # m.addConstr(x[0,7] == 1);
    # m.addConstr(x[1,5] == 1);
    # m.addConstr(x[2,19] == 1);
    # m.addConstr(x[3,16] == 1);
    # m.addConstr(x[4,18] == 1);
    # m.addConstr(x[5,11] == 1);
    # m.addConstr(x[6,28] == 1);
    # m.addConstr(x[7,14] == 1);
    # m.addConstr(x[8,0] == 1);
    # m.addConstr(x[9,1] == 1);
    # m.addConstr(x[10,29] == 1);
    # m.addConstr(x[11,10] == 1);
    # m.addConstr(x[12,12] == 1);
    # m.addConstr(x[13,27] == 1);
    # m.addConstr(x[14,22] == 1);
    # m.addConstr(x[15,26] == 1);
    # m.addConstr(x[16,15] == 1);
    # m.addConstr(x[17,21] == 1);
    # m.addConstr(x[18,9] == 1);
    # m.addConstr(x[19,20] == 1);
    # m.addConstr(x[20,24] == 1);
    # m.addConstr(x[21,23] == 1);
    # m.addConstr(x[22,25] == 1);
    # m.addConstr(x[23,17] == 1);
    # m.addConstr(x[24,2] == 1);
    # m.addConstr(x[25,13] == 1);
    # m.addConstr(x[26,6] == 1);
    # m.addConstr(x[27,4] == 1);
    # m.addConstr(x[28,8] == 1);
    # m.addConstr(x[29,3] == 1);


    m.Params.OutputFlag = 0
    sol, time_duration , obj= LNS_QAP(m,N,20,10,limited_times=5,verbose=True)
    print('The solution is:{} , the objective value is: {}, the time duration is:{}'.format(sol,obj,time_duration))