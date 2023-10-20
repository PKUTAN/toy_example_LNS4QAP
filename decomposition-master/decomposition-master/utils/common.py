import random
import networkx as nx
import numpy as np
# import nxmetis
import torch
import scipy.sparse as sp

from scipy import stats
# from pygcn import utils

from torch.distributions import Categorical


def relax_integer_constraints(model, var_relax):
    """Relax the integer constraints for variables in var_relax.

    Arguments:
      model: the integer program.
      var_relax: a dict mapping node index to node variable name.

    Returns:
      a new model with var_relax relaxed to be continuous.
    """

    for node, var in var_relax.items():
        var_to_relax = model.getVarByName(var)
        var_to_relax.setAttr('VType', 'C')

    return model


def fix_integer_values(model, var_fix):
    """Fix integer constraints according to var_fix.

    Arguments:
      model: the integer program.
      var_fix: a dict mapping variable name to values.

    Returns:
      a new model with variables in var_fix fixed to values.
    """

    for k, v in var_fix.items():
        var = model.getVarByName(k)
        model.addConstr(var == v)

    return model


def unrelax_integer_constraints(model, var_unrelax):
    """Unrelax integer constraints.

    Arguments:
      model: the integer program.
      var_unrelax: a dict mapping node index to node variable name.

    Returns:
      a new model with var_unrelax un-relaxed to be integers.
    """

    for node, var in var_unrelax.items():
        var_to_unrelax = model.getVarByName(var)
        var_to_unrelax.setAttr('VType', 'B')

    return model


def get_objval(model):
    try:
        obj = model.ObjVal
    except:
        obj = 1000000
    return obj


def set_warmstart_vector(model, warmstart_v):
    """Warm start a model with values in warmstart_v.

    Arguments:
      model: the integer program.
      warmstart_v: a dict mapping variable name to a value.

    Returns:
      a model with a warmstart solution.
    """

    for k, v in warmstart_v.items():
        var = model.getVarByName(k)
        var.start = v

    return model


def extract_solution(model, var_dict):
    """Extract solution from variables in var_dict."""

    sol = {}
    sol_vec = None

    for k, var_list in var_dict.items():
        if sol_vec is None:
            sol_vec = np.zeros((len(var_dict), len(var_list)))
            
        for i, v in enumerate(var_list):
            var = model.getVarByName(v)
            sol[v] = var.X
            sol_vec[k, i] = var.X

    return sol, sol_vec


def vec_to_dict(sol_vec, var_dict):
    sol = {}
    for k, var_list in var_dict.items():
        for i, v in enumerate(var_list):
            sol[v] = sol_vec[k, i]

    return sol


def random_clusters(var_dict, num_clusters):
    
    node_indices = sorted(list(var_dict.keys()))
    random.shuffle(node_indices)

    clusters = {}
    node_per_cluster = len(node_indices) // num_clusters

    for i in range(num_clusters):
        if i < num_clusters - 1:
            clusters[i] = node_indices[i*node_per_cluster:(i+1)*node_per_cluster]
        else:
            clusters[i] = node_indices[i*node_per_cluster:]

    return clusters


def uniform_random_clusters(var_dict, num_clusters, p_type):
    '''Return a random clustering. Each node is assigned to a cluster
    a equal probability.'''

    choices = list(range(num_clusters))
    clusters = dict([(i, []) for i in range(num_clusters)])

    for k in var_dict.keys():
        cluster_choice = random.choice(choices)
        clusters[cluster_choice].append(k)
        
        # if p_type == 'maxcut' or p_type == 'mvc' or p_type == 'cats':
        #     clusters[cluster_choice].append(k)
        # elif p_type == 'psulu':
        #     if k in assigned_keys:
        #         continue
        #     var_name = var_dict[k]
        #     name_prefix = var_name.rsplit('_', maxsplit=1)[0]
        #     for i in range(4):
        #         new_var_name = name_prefix + '_%d' %i
        #         new_var_key = inv_dict[new_var_name]
        #         assigned_keys.add(new_var_key)
        #         clusters[cluster_choice].append(new_var_key)

    return clusters
        

def metis_clusters(graph, num_clusters):
    import pdb; pdb.set_trace()
    _, clusters = nxmetis.partition(graph, num_clusters)
    clusters = dict([(i, clusters[i]) for i in range(len(clusters))])

    return clusters
    
 
def compute_dual_graph(graph):

    return nx.line_graph(graph)


def random_block_coordinates(var_dict, max_block=50):
    node_indices = list(var_dict.keys())
    random.shuffle(node_indices)

    return {0: node_indices[:max_block]}


def generate_var_dict(model, p_type):
    model_vars = model.getVars()
    num_vars = 0
    var_dict = {}

    if p_type == 'cats' or p_type == 'mvc':
        num_vars = len(model_vars)
    elif p_type == 'maxcut':
        for model_var in model_vars:
            if model_var.VarName.startswith('v'):
                num_vars += 1
    elif p_type == 'psulu':
        num_x = 0
        num_z = 0

        for model_var in model_vars:
            if model_var.VarName.startswith('z'):
                num_z += 1
            elif model_var.VarName.startswith('x'):
                num_x += 1

        assert(num_z % num_x == 0)
        assert(num_x % 4 == 0)
        num_obst = num_z // num_x
        num_waypoint = num_x // 4

        var_idx = 0
        for w in range(num_waypoint):
            for o in range(num_obst):
                prefix = 'z_%d_%d' %(w, o)
                var_dict[var_idx] = [prefix + '_%d' %i
                                     for i in range(4)]
                var_idx += 1

        # var_idx = 0
        # for o in range(num_obst):
        #     var_dict[var_idx] = ['z_%d_%d_%d' %(w, o, i)
        #                          for w in range(num_waypoint)
        #                          for i in range(4)]
        #     var_idx += 1

        # var_idx = 0
        # for w in range(num_waypoint):
        #     var_dict[var_idx] = ['z_%d_%d_%d' %(w, o, i)
        #                          for o in range(num_obst)
        #                          for i in range(4)]
        #     var_idx += 1
            
    else:
        raise Exception('Problem type %s is not supported.' %p_type)

    if p_type == 'cats':
        var_dict = dict([(i, ["x%d" %i]) for i in range(num_vars)])
    elif p_type == 'mvc' or p_type == 'maxcut':
        var_dict = dict([(i, ["v%d" %i]) for i in range(num_vars)])

    return var_dict


def sample_partitions(cluster_probs, mode='train'):
    '''cluster_probs[i][j] is the probability that node i belongs to cluster j.
    
    Returns: 
      A sampled partition represented by a dict.
      The log probability of sampling this partition.
    '''
    # import pdb; pdb.set_trace()
    num_clusters = len(cluster_probs[0])
    cluster_range = list(range(num_clusters))

    partitions = dict([(i, []) for i in range(num_clusters)])
    log_prob = 0
    entropy = 0

    if mode == 'train'  or mode == 'evaluate':
        rvs = Categorical(cluster_probs)
        samples = rvs.sample()
        if sum(samples) > 10:
            eraze = sum(samples) - 10
            sam_np = samples.cpu().numpy()
            sam_idx = np.array(np.where(sam_np == 1)[0])
            
            sam_0 = cluster_probs[sam_idx][:,1]
            sam_0_idx = torch.argsort(sam_0)
            for i in range(eraze):
                idx = sam_idx[sam_0_idx[i]]
                samples[idx] = 0
        elif sum(samples) < 10:
            eraze = 10 - sum(samples) 
            sam_np = samples.cpu().numpy()
            sam_idx = np.array(np.where(sam_np == 0)[0])
            
            sam_0 = cluster_probs[sam_idx][:,0]
            sam_0_idx = torch.argsort(-sam_0)
            for i in range(eraze):
                idx = sam_idx[sam_0_idx[i]]
                samples[idx] = 1

                
        log_probs = rvs.log_prob(samples)
        entropies = rvs.entropy()
        # import pdb; pdb.set_trace()

        for i in range(len(cluster_probs)):
            partitions[samples[i].item()].append(i)
            log_prob += log_probs[i]
            entropy += entropies[i].item()
    elif mode == 'train':
        samples = torch.argmax(cluster_probs, dim=1)
        for i in range(len(cluster_probs)):
            partitions[samples[i].item()].append(i)
            log_prob += torch.log(cluster_probs[i][samples[i].item()])
    
    #for i in range(len(cluster_probs)):
    #    rv = Categorical(cluster_probs[i])
    #    cluster = rv.sample()
    #    log_prob += rv.log_prob(cluster)
    #    partitions[cluster.item()].append(i)
    #    entropy += rv.entropy()

    # print([len(cluster) for cluster in partitions.values()])
    # print(entropy / len(cluster_probs))

    return partitions, log_prob, entropy / len(cluster_probs)


# def get_adj_matrix(graph):
#     '''Compute the adjacency matrix of a graph. For weighted graphs, the adjacency matrix is
#     weighted as well. Include a self-loop for each node.'''

#     adj = nx.to_scipy_sparse_matrix(graph)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     adj = utils.normalize(adj + sp.eye(adj.shape[0]))
        
#     return utils.sparse_mx_to_torch_sparse_tensor(adj)    


def initialize_mvc_objective(model):
    objective = model.getObjective()
    obj = 0

    for i in range(objective.size()):
        obj += objective.getCoeff(i)

    return obj


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


def solution(model):
    sol = []
    for v in model.getVars():
        if v.x == 1:
            sol.append(eval(v.VarName[1:]))
    return sol

def sub2whole(sub_sol,sol,index):
    sol_out = sol[:]
    sub_prob_size = len(sub_sol)
    unassigned_loc = sorted([loc for idx, loc in sol_out if idx in index])
    unassigned_flow = index

    for i in range(sub_prob_size):
        sol_out[unassigned_flow[i]][1] = unassigned_loc[sub_sol[i][1]]
    
    return sol_out

def initial_sol(F,D):
    '''
        We offer mutiple initial methods including:
        1) random initialize 
        2) learnable network initialize
    '''
    
    N = F.shape[1]
    
    ####random initialize
    prob_index = [i for i in range(N)]
    # init_loc = random.sample(prob_index,len(prob_index))
    # sol = [[i,j] for i,j in enumerate(init_loc)]
    sol = [[i,i] for i in range(N)]

    obj = cal_obj(sol,D,F)

    ####learnable network initialize(TODO)

    return sol,obj   

def featurize(F,D,cur_sol):
    n,n = F.shape
    # import pdb;pdb.set_trace()
    
    features = np.zeros((n,3*n))

    sol = form_per(cur_sol)
    loc_fea = np.matmul(sol,D)
    features = np.concatenate([F,sol,loc_fea] , axis = -1)

    return features 


def initialize_solution(p_type, model, init_time=0, use_init=False):
    '''Initialize a feasible solution.

    Arguments:
      var_dict: a dict maps node index to node variable name.

    Returns:
      a dict maps node variable name to a value.
      a torch tensor view of the dict.
    '''

    sol = {}
    # the sol_vec needs to be of type float for later use with Pytorch.
    sol_vec = None
    init_obj = 0


    # import pdb; pdb.set_trace()
    # model_copy = model.copy()
    # model_copy.setParam('MIPFocus', 1)
    # model_copy.setParam('TimeLimit', init_time)
    # model_copy.optimize()

    # try:
    #     sol, sol_vec = extract_solution(model_copy, var_dict)
    #     init_obj = model_copy.ObjVal
    # except:
    #     model_copy.reset()
    #     model_copy.setParam('TimeLimit', 100000)
    #     model_copy.setParam('SolutionLimit', 1)
    #     model_copy.optimize()
    #     sol, sol_vec = extract_solution(model_copy, var_dict)
    #     init_obj = model_copy.ObjVal            

    return sol, sol_vec, init_obj


def primal_gap(obj, best_obj):
    '''Compute the primal gap as defined in Measuring the impact of primal heuristics by 
    Timo Berthold.'''

    if obj == 0 and best_obj == 0:
        return 0

    if obj * best_obj < 0:
        return 1

    return abs(obj - best_obj) / max(abs(obj), abs(best_obj))


def primal_integral(objs, best_obj):
    '''Compute the primal integral.

    Arguments:
      objs: list of tuples (time, obj)
      best_obj: the best objective to compute primal gap from.
    '''

    p_integral = 0
    t = 0
    
    for obj in objs:
        p_gap = primal_gap(obj[1], best_obj)
        p_integral += (obj[0] - t) * p_gap
        t = obj[0]

    return p_integral
    

def lst_to_str(lst):
    return ' '.join(str(x) for x in lst)


def compute_std(total_rewards, n_traj):
    assert(len(total_rewards) % n_traj == 0)
    sum_total_rewards = [sum(total_rewards[i::n_traj]) for i in range(n_traj)]
    return np.std(sum_total_rewards) / np.sqrt(n_traj)


def milp_to_graph(model):
    '''Compute the bipartite graph representation of a MILP.'''

    bi_graph = nx.Graph()
    # B.add_nodes_from([1, 2, 3, 4], bipartite=0)
    # B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
    # Add edges only between nodes of opposite node sets
    # B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])

    # model.getRow(constr)
    # obj_coeffs = model.getAttr('Obj', dvars)
    
    constrs = model.getConstrs()
    dvars = model.getVars()

    dvar_nodes = list(range(len(dvars)))
    constr_nodes = list(range(len(dvars), len(dvars)+len(constrs)))
    edge_list = []


    for constr_num, constr in enumerate(constrs):
        row = model.getRow(constr)
        
        for i in range(row.size()):
            dvar_num = int(row.getVar(i).VarName[1:])
            coeff = row.getCoeff(i)
            edge_list.append((dvar_nodes[dvar_num],
                              constr_nodes[constr_num],
                              coeff))

    bi_graph.add_weighted_edges_from(edge_list)
    return bi_graph


def cats_milp_to_adj(model):
    constrs = model.getConstrs()
    dvars = model.getVars()

    adj = np.zeros((len(dvars), len(constrs)))
    
    for constr_num, constr in enumerate(constrs):
        row = model.getRow(constr)
        
        for i in range(row.size()):
            dvar_num = int(row.getVar(i).VarName[1:])
            coeff = row.getCoeff(i)
            adj[dvar_num, constr_num] = coeff

    return adj


def psulu_milp_to_adj(model, num_obst, num_waypoints):
    # order of vars: absu, u, x, z
    num_absu = 2 * (num_waypoints - 1)
    num_u = num_absu
    num_x = 2 * num_waypoints
    constrs = model.getConstrs()
    dvars = model.getVars()
    num_vars = len(dvars) - 3 * num_obst * num_waypoints 
    # assert(num_absu + num_u + num_x + num_obst * num_waypoints == num_vars)

    #adj = np.zeros((num_vars, len(constrs)))
    adj = np.zeros((num_x+num_obst*num_waypoints, len(constrs)))
    valid_constrs = 0

    for constr_num, constr in enumerate(constrs):
        row = model.getRow(constr)
        valid = False
        has_x = False
        has_z = False

        for i in range(row.size()):
            varname = row.getVar(i).VarName
            if varname.startswith('x') and (
                    varname.endswith('0') or varname.endswith('1')) :
                has_x = True
            elif varname.startswith('z'):
                has_z = True

        if not has_x or not has_z:
            continue
        
        for i in range(row.size()):
            varname = row.getVar(i).VarName
            items = varname.split('_')
            
            # if varname.startswith('absu'):
            #     var_idx = int(items[1]) * 2 + int(items[2])
            # elif varname.startswith('u'):
            #     var_idx = num_absu + int(items[1]) * 2 + int(items[2])
            # elif varname.startswith('x'):
            #     var_idx = num_absu + num_u + int(items[1]) * 2 + int(items[2])
            # else:
            #     var_idx = (num_absu + num_u + num_x +
            #                int(items[1]) * num_waypoints +
            #                int(items[2]))
            
            if varname.startswith('x') and (
                    varname.endswith('0') or varname.endswith('1')) :
                var_idx = int(items[1]) * 2 + int(items[2])
                valid = True
            elif varname.startswith('z'):
                var_idx = (num_x +
                           int(items[1]) * num_waypoints +
                           int(items[2]))
                valid = True
            else:
                continue
            
            coeff = row.getCoeff(i)
            if varname.startswith('z'):
                coeff = 1

            adj[var_idx, valid_constrs] = coeff

        if valid:
            valid_constrs += 1

    adj = adj[:, :valid_constrs]
    # print(adj.shape)
            
    return adj


def milp_to_adj(model, p_type, **kwargs):
    '''Compute the bipartite graph adjacency matrix representation of a MILP.'''

    if p_type == 'cats' or p_type == 'mvc':
        return cats_milp_to_adj(model)
    elif p_type == 'psulu':
        return psulu_milp_to_adj(model, kwargs['num_obst'], kwargs['num_waypoints'])
    else:
        raise NotImplementedError('Problem type %s not implemented!' %p_type)


def compute_accuracy(output, labels):
    predictions = torch.argmax(output, dim=1)
    correct = (predictions == labels).sum().item()
    total = torch.numel(labels)

    return correct / total


def convert_to_network(network_str):
    '''Convert a string representation to a list of integers.'''
    
    network = network_str.split(',')
    network = [int(x) for x in network]
    return network
