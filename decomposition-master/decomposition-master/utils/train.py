import sys
sys.path.append('.')

import gurobipy
import os
import pickle
import shutil
import time
import torch
torch.manual_seed(0)

import networkx as nx
import numpy as np
np.random.seed(0)

from absl import app
from absl import flags

from algo import compare
from algo import reinforce

from multiprocessing import Pool

from sklearn.decomposition import PCA
import scipy.sparse as sp

from algo import supervise
from utils import common
from utils import coordinate_descent
from utils import models


FLAGS = flags.FLAGS

flags.DEFINE_integer('nfeat', None, 'Number of features for each node.')
flags.DEFINE_integer('nhid', None, 'Number of hidden units.')
flags.DEFINE_float('dropout', 0.0, 'Dropout proportion.')
flags.DEFINE_float('lr', None, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay.')
flags.DEFINE_integer('K', None, 'Number of clusters.')
flags.DEFINE_integer('temp', None, 'Temperature for computing soft cluster assignments.')
flags.DEFINE_string('dir_problems', None, 'Directory of the problems.')
flags.DEFINE_string('dir_graphs', None, 'Directory of the graphs.')
#flags.DEFINE_string('p_type', None, 'Problem type.')
#flags.DEFINE_integer('steps', None, 'Number of steps for each episode.')
flags.DEFINE_integer('iterations', None, 'Number of iterations for training.')
#flags.DEFINE_integer('time_limit', None, 'Max time for solving each problem.')
flags.DEFINE_integer('n_traj', None,
                     'Number of trajectories to sample for each iteration.')
flags.DEFINE_integer('kmeans_iter', None, 'Number of iterations to run kmeans for.')
#flags.DEFINE_string('network', None, 'Network architecture for the GCN.')
#flags.DEFINE_string('mode', 'train', 'train or test.')
#flags.DEFINE_string('model_dir', None, 'Path to a trained model.')
flags.DEFINE_string('algo', None, 'compare or reinforce.')


def train(agent,
          problems,
          graphs,
          var_dicts,
          steps,
          iterations,
          n_traj,
          p_type,
          log_stats,
          model_dir):
    '''Train an agent on a collection of problems for iterations number of iterations. 
    For each iteration, for each problem, the agent takes step number of steps.

    Arguments:
      agent: a learning agent.
      problems: a collection of integer programs.
      graphs: a collection of graphs that map to integer programs.
      var_dicts: a collection of dicts mapping node indices to variable names.
      steps: the number of steps for each problem.
      iterations: the number of iterations to run the training.
      n_traj: the number of trajectories to sample for each iteration.
      p_type: the problem type. This decides how the initial solution is generated.
      log_stats: the log file to save training stats.
      model_dir: directory to save trained models.

    Returns:
      a trained agent.
    '''
    log_output = open(log_stats, 'w')
    mean_rewards = []
    std_rewards = []
    mean_objs = []
    min_objs = []
    
    for ite in range(iterations):
        total_rewards = []
        obj = None
        # train_rewards and train_log_probs are
        # lists of problem_rewards and problem_log_probs (see below)
        train_rewards = []
        train_log_probs = []
        start_time = time.time()
        final_objs = []
        train_features = []
        train_clusters = []
        
        for problem, graph, var_dict in zip(problems, graphs, var_dicts):
            problem_clusters = []
            problem_features = []
            # problem_rewards and problem_log_probs are lists of lists.
            problem_log_probs = []
            problem_rewards = []
            init_sol, init_sol_feature, init_obj = common.initialize_solution(
                var_dict, p_type, problem)
            # warmstart_problem = common.set_warmstart_vector(problem.copy(), init_sol)
            # warmstart_problem.optimize()
            # init_obj = warmstart_problem.ObjVal
            # init_sol, init_sol_feature = common.extract_solution(warmstart_problem,
            #                                                      var_dict)
            # adj = common.get_adj_matrix(graph)
            # normalized_laplacian = nx.normalized_laplacian_matrix(graph).todense()
            if p_type == 'cats':
                laplacian = graph
            else:
                laplacian = nx.laplacian_matrix(graph).todense()
            # adjacency = nx.adjacency_matrix(graph).todense()
            
            pca = PCA(n_components=agent.model.network[0]-1)
        
            pca_laplacian = pca.fit_transform(laplacian)

            for _ in range(n_traj):

                # record log probs and rewards for this individual trajectory
                log_probs = []
                rewards = []
                clusters = []
                features = []
                # init_node_features = torch.eye(len(graph))

                combined = np.concatenate((pca_laplacian, init_sol_feature), axis=1)
                init_feature = torch.as_tensor(combined).float()
                # init_features = init_node_features
                
                total_reward = 0
                obj = init_obj
                feature = init_feature
                sol = init_sol

                for i in range(steps):
                    # init_node_features = torch.randn((len(graph), agent.nout))
                    partitions, log_prob = (
                        agent.select_action(feature, mode='train'))
                    # import pdb; pdb.set_trace()
                    new_sol, new_sol_vec, _, new_obj = \
                        coordinate_descent.coordinate_descent_by_clusters(
                            problem.copy(), partitions, var_dict, sol)
                    new_sol_feature = np.c_[new_sol_vec]
                    combined = np.concatenate((pca_laplacian, new_sol_feature), axis=1)
                    feature = torch.as_tensor(combined).float()
                    print('objective: ', new_obj)

                    # new_sol_feature = torch.tensor(new_sol_vec).unsqueeze(0).t()

                    # new_features = new_features.cpu()
                    # features = torch.cat((new_features, new_sol_features), dim=1)
                    # feature = torch.cat((init_node_feature, new_sol_feature), dim=1)
                    reward = obj - new_obj
                    # print('reward: ', reward)
                    total_reward += reward
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    clusters.append(partitions)
                    features.append(feature)

                    obj = new_obj
                    sol = new_sol

                print('----------')
                total_rewards.append(total_reward)
                final_objs.append(new_obj)
                problem_log_probs.append(log_probs)
                problem_rewards.append(rewards)
                problem_clusters.append(clusters)
                problem_features.append(features)
                
            train_log_probs.append(problem_log_probs)
            train_rewards.append(problem_rewards)
            train_features.append(problem_features)
            train_clusters.append(problem_clusters)

        print('Mean total rewards: %.2f' %(np.mean(total_rewards)))
        print('Std total rewards: %.2f' %(np.std(total_rewards)))
        print('Mean final objectives: %.2f' %(np.mean(final_objs)))
        print('Min final objectives: %.2f' %(np.min(final_objs)))
        mean_rewards.append(np.mean(total_rewards))
        std_rewards.append(np.std(total_rewards))
        mean_objs.append(np.mean(final_objs))
        min_objs.append(np.min(final_objs))
        end_time = time.time()

        agent.update(train_log_probs, train_rewards)
        # agent.update_with_features(train_features,
        #                            train_clusters,
        #                            train_rewards)
        
        torch.save(agent, os.path.join(model_dir, 'iter_%d.pt' %ite))
        print('Iteration time: %.2f\n' %(end_time-start_time))

    print(common.lst_to_str(mean_rewards), file=log_output)
    print(common.lst_to_str(std_rewards), file=log_output)
    print(common.lst_to_str(mean_objs), file=log_output)
    print(common.lst_to_str(min_objs), file=log_output)
    log_output.close()

    print(log_stats)

    return agent


def evaluate(agent, problems, graphs, var_dicts, steps, n_traj, p_type):
    eval_rewards = []
    total_rewards = []
    final_objs = []
    
    for problem, graph, var_dict in zip(problems, graphs, var_dicts):
        # problem_rewards is a list of lists.
        problem_rewards = []
        init_sol, init_sol_features, init_obj = common.initialize_solution(
            var_dict, p_type, problem)
        #warmstart_problem = common.set_warmstart_vector(problem.copy(), init_sol)
        #warmstart_problem.optimize()
        #init_obj = warmstart_problem.ObjVal
        #init_sol, init_sol_features = common.extract_solution(warmstart_problem,
        #                                                      var_dict)
        #adj = common.get_adj_matrix(graph)
        normalized_laplacian = nx.normalized_laplacian_matrix(graph).todense()
        laplacian = nx.laplacian_matrix(graph).todense()
        pca = PCA(n_components=agent.model.network[-1])
        init_node_features = torch.tensor(
            pca.fit_transform(laplacian)).float()

        for _ in range(n_traj):
            # record rewards for this individual trajectory
            rewards = []
            init_features = torch.cat((init_node_features, init_sol_features), dim=1)
            
            total_reward = 0
            obj = init_obj
            features = init_features
            sol = init_sol
            solver_time = 0
            start_time = time.time()

            for i in range(steps):
                partitions, log_prob, new_features = agent.select_action(
                    adj, features, mode='evaluate')
                new_sol, new_sol_vec, t, new_obj = \
                    coordinate_descent.coordinate_descent_by_clusters(
                        problem.copy(), partitions, var_dict, sol)
                new_sol_features = torch.tensor(new_sol_vec).unsqueeze(0).t()
                features = torch.cat((init_node_features, new_sol_features), dim=1)
                reward = obj - new_obj
                total_reward += reward
                rewards.append(reward)
                obj = new_obj
                sol = new_sol
                solver_time += t

            end_time = time.time()
            print('Total time: %f' %(end_time-start_time))
            print('Solver time: %f' %solver_time)
                
            total_rewards.append(total_reward)
            final_objs.append(new_obj)
            problem_rewards.append(rewards)

        eval_rewards.append(problem_rewards)
    print('Mean total rewards: %.2f' %(np.mean(total_rewards)))
    print('Std total rewards: %.2f' %(np.std(total_rewards)))


def run_one(agent):
    pass


def convert_to_network(network_str):
    '''Convert a string representation to a list of integers.'''
    
    network = network_str.split(',')
    network = [int(x) for x in network]
    return network
    
        
def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    nfeat = FLAGS.nfeat
    nhid = FLAGS.nhid
    dropout = FLAGS.dropout
    lr = FLAGS.lr
    weight_decay = FLAGS.weight_decay
    K = FLAGS.K
    temp = FLAGS.temp
    dir_problems = FLAGS.dir_problems
    dir_graphs = FLAGS.dir_graphs
    p_type = FLAGS.p_type
    steps = FLAGS.steps
    iterations = FLAGS.iterations
    time_limit = FLAGS.time_limit
    n_traj = FLAGS.n_traj
    kmeans_iter = FLAGS.kmeans_iter
    mode = FLAGS.mode
    model_dir = FLAGS.model_dir
    algo = FLAGS.algo

    if p_type == 'maxcut' or p_type == 'mvc':
        dir_graphs = dir_problems.replace('lpfiles', 'gpickle')
    elif p_type == 'cats':
        dir_graphs = dir_problems.replace('_lp', '_adj')
    elif p_type == 'psulu':
        dir_graphs = dir_problems.replace('lpfiles', 'adj')
        
    problems = []
    var_dicts = []
    graphs = []

    # if mode == 'train':
    #     network = convert_to_network(FLAGS.network)
    #     problem_dirs = dir_problems.rsplit(sep='/', maxsplit=1)
    #     problem_prefix = problem_dirs[0]
    #     problem_suffix = problem_dirs[1]
    #     timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    #     # information to save to a log dir:
    #     # model details (save a copy of train.sh)
    #     # training statistics
    #     log_dir = os.path.join(problem_prefix,
    #                            '%s_train_logs' %problem_suffix,
    #                            timestamp)
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)

    #     # save a copy of train.sh
    #     shutil.copyfile('scripts/train.sh', os.path.join(log_dir, 'train.sh'))

    #     # file to save training stats
    #     log_stats = os.path.join(log_dir, 'stats.txt')
    #     # file to save the trained model
    #     model_dir = os.path.join(log_dir, 'models')

    if FLAGS.network:
        network = convert_to_network(FLAGS.network)

    if mode == 'train':
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        log_stats = os.path.join(model_dir, 'log_stats')
        if not os.path.exists(log_stats):
            os.makedirs(log_stats)
            
        log_stats = os.path.join(log_stats, 'stats.txt')
        
        #cluster_model = models.GCNClusterNet(network, dropout, K, temp, device)
        # cluster_model = models.FeedforwardClusterNet(network, K, temp, device)
        cluster_model = models.FeedforwardSoftmaxNet(network, device)
        if algo == 'reinforce':
            agent = reinforce.ReinforceAgent(cluster_model, lr, weight_decay, device,
                                             kmeans_iter)
        elif algo == 'compare':
            agent = compare.CompareAgent(cluster_model, lr, weight_decay, device,
                                         kmeans_iter)
        else:
            raise NotImplementedError
    else:
        agent = torch.load(model_dir)
    
    for problem_file in os.listdir(dir_problems):
        if problem_file.endswith('lp'):
            problem_prefix = problem_file.split('.')[0]
            abs_problem_filename = os.path.join(dir_problems, problem_file)
            abs_var_dict_filename = os.path.join(dir_problems, problem_prefix+'.pkl')
            if p_type == 'cats':
                abs_graph_filename = os.path.join(dir_graphs, problem_prefix+'.npz')        
            else:
                abs_graph_filename = os.path.join(dir_graphs, problem_prefix+'.gpickle')
            problem = gurobipy.read(abs_problem_filename)
            problem.setParam('MIPFocus', 1)
            problem.setParam('TimeLimit', time_limit)
            problem.setParam('OutputFlag', 0)
            problems.append(problem)

            if os.path.exists(abs_var_dict_filename):
                var_dicts.append(pickle.load(open(abs_var_dict_filename, "rb")))
            else:
                var_dicts.append(common.generate_var_dict(problem, p_type))

            if p_type == 'cats':
                graphs.append(sp.load_npz(abs_graph_filename).todense())
            else:
                graphs.append(nx.read_gpickle(abs_graph_filename))

        
    if mode == 'train':
        agent = train(agent,
                      problems,
                      graphs,
                      var_dicts,
                      steps,
                      iterations,
                      n_traj,
                      p_type,
                      log_stats,
                      model_dir)
    else:
        evaluate(agent,
                 problems,
                 graphs,
                 var_dicts,
                 steps,
                 n_traj,
                 p_type)
        # supervise.evaluate(model_dir, network, None, None, dir_graphs, dir_problems,
        #                    device, steps, p_type, time_limit, alg_name='rl')
    

if __name__ == "__main__":
    app.run(main)
