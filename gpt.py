from gurobipy import Model,GRB,quicksum

train_set = QAPLIB('train','tai')
F,D,per,sol,name ,opt_obj = train_set.get_pair(2)
            # if F.sum()==None:
            #     continue
N = F.shape[0]

m = Model('QAP')
x = m.addMVar(shape = (N,N), vtype= GRB.BINARY,name='x')

m.setObjective(quicksum(quicksum(F*((x@D)@x.T))), GRB.MINIMIZE)
m.addConstrs(quicksum(x[i,j] for j in range(N))==1 for i in range(N));
m.addConstrs(quicksum(x[i,j] for i in range(N))==1 for j in range(N));

m.optimize()