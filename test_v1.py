import gurobipy as gp
from gurobipy import GRB

def mycallback(model, where):
    # 如果是在找到一个新的解决方案时
    if where == GRB.Callback.MIPSOL:
        # 获取当前解的目标值
        objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        print(f'Found a new solution with objective: {objval}')

# 创建一个新的模型
m = gp.Model()

# 定义变量和约束等（这里是一个简化的示例）
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
m.setObjective(x + 2 * y, GRB.MAXIMIZE)
m.addConstr(x + 2 * y <= 2.5)

# 开始优化，并在求解过程中使用回调函数
m.optimize(mycallback)
