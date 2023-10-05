import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import time

from itertools import chain

import pickle
import networkx as nx
import os
import gurobipy

from absl import flags
from absl import app

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.sparse as sp

from utils import coordinate_descent
from utils import common
from utils import models

FLAGS = flags.FLAGS

flags.DEFINE_string('network', None, 'Neural network architecture.')
flags.DEFINE_string('graph_dir', None, 'Directory with gpickle files.')
flags.DEFINE_string('lp_dir', None, 'Directory with lp files.')
flags.DEFINE_string('cluster_file', None, 'Path to the file with best clusters.')
flags.DEFINE_integer('epoch', 50, 'Number of epoches.')
flags.DEFINE_string('mode', 'train', 'train or evaluate.')
flags.DEFINE_string('model_dir', None, 'Path to a model.')
flags.DEFINE_string('p_type', 'maxcut', '')
flags.DEFINE_integer('seed', 0, 'Random seed to use.')
flags.DEFINE_string('cache_dir', None, '')
flags.DEFINE_integer('time_limit', 1, '')
flags.DEFINE_integer('steps', 1, '')


class Net(nn.Module):
    def __init__(self, network):
        super(Net, self).__init__()
        
        self.network = network
        self.layers = []

        for idx in range(len(network)-1):
            self.layers.append(nn.Linear(network[idx], network[idx+1]))

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))

        x = F.softmax(self.layers[-1](x))
        return x


def clusters_to_labels(clusters):
    '''Convert a cluster to per-node labels.'''
    # import pdb; pdb.set_trace()

    labels = [0 for _ in range(sum([len(v) for v in clusters.values()]))]
    sorted_clusters = sorted(clusters.values(), key=lambda x: len(x), reverse=True)

    # for k, v in clusters.items():
    #     for node in v:
    #         labels[node] = k

    for i, cluster in enumerate(sorted_clusters):
        for node in cluster:
            labels[node] = i

    return labels


def labels_to_clusters(labels, k=5):
    clusters = dict([(i, []) for i in range(k)])

    for i, l in enumerate(labels):
        if l >= 0:
            clusters[l].append(i)

    print([len(cluster) for cluster in clusters.values()])        

    return clusters


def partial_labels_to_clusters(labels, k=5):
    clusters = dict([(i, []) for i in range(k)])

    for i, l in labels:
        clusters[l].append(i)

    return clusters


def initialize_problem(lp_file, time_limit, p_type, optimize=True, use_init=False):

    problem = gurobipy.read(lp_file)
    problem.setParam('MIPFocus', 1)
    problem.setParam('TimeLimit', time_limit)
    problem.setParam('OutputFlag', 0)

    var_dict = common.generate_var_dict(problem, p_type)

    init_sol = None
    init_sol_features = None
    init_obj = None
    
    if optimize:
        init_sol, init_sol_features, init_obj = common.initialize_solution(
            var_dict, p_type, problem, use_init=use_init)

    return problem, var_dict, init_obj, init_sol, init_sol_features


def supervise_training(network, lp_dir, graph_dir, cluster_file,
                       epoch, device, model_dir, p_type='maxcut', seed=0):

    model_file = os.path.join(model_dir,
                              '_'.join([str(x) for x in network]) + '_seed_%d' %(seed))
    # contains cluster and current solution
    best_clusters = pickle.load(open(cluster_file, 'rb'))
    features = []
    labels = []
    feat_dir = os.path.join('/tmp/jssong', cluster_file.rsplit('/', maxsplit=1)[0])

    feat_file = os.path.join(
        feat_dir,
        'train_feat_lap_%d.pkl' %network[0])

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    has_feat = False
    if os.path.exists(feat_file):
        has_feat = True
        with open(feat_file, 'rb') as f:
            features = pickle.load(f)

    # has_pca = False
    # has_init = False
    # init_sol_feats = []
    # lap_pca_feats = []
    num_step = len(list(best_clusters.values())[0][0])
    pca_feats = {}
    
    for step in range(num_step):
        step_feats = []
        step_labels = []
        for idx, f_name in enumerate(os.listdir(graph_dir)):
            graph_file = os.path.join(graph_dir, f_name)
            prefix = f_name.split('.')[0]

            if not prefix in best_clusters:
                continue
        
            best_cluster, best_sol = best_clusters[prefix]
            cluster = best_cluster[step]
            sol = best_sol[step]

            if not has_feat:
                if prefix not in pca_feats:
                    if p_type == 'maxcut' or p_type == 'mvc':
                        graph = nx.read_gpickle(graph_file)
                        laplacian = nx.laplacian_matrix(graph).todense()
                        # laplacian = nx.normalized_laplacian_matrix(graph).todense()
                        # laplacian = nx.adjacency_matrix(graph).todense()
                    elif p_type == 'cats' or p_type == 'psulu':
                        graph = sp.load_npz(graph_file).todense()
                        laplacian = graph
                    lp_file = os.path.join(lp_dir, prefix + '.lp')
                    # normalized_laplacian = nx.normalized_laplacian_matrix(graph).todense()
                    # adjacency = nx.adjacency_matrix(graph).todense()

                    # _, _, _, _, init_sol_feat = initialize_problem(
                    #     lp_file, time_limit, p_type)

                    if p_type == 'psulu':
                        pca = PCA(n_components=network[0]-4)
                    else:
                        pca = PCA(n_components=network[0]-1)                        

                    if not prefix in best_clusters:
                        continue

                    #combined = np.concatenate((laplacian, init_sol_feat), axis=1)
                    #pca_feat = torch.tensor(pca.fit_transform(combined)).float()

                    pca_laplacian = pca.fit_transform(laplacian)
                    if p_type == 'psulu':
                        obs, wp = lp_dir.split('/', maxsplit=3)[2].split('_')
                        obs = int(obs)
                        wp = int(wp)          
                        pca_laplacian = pca_laplacian[-obs*wp:]
                        pca_laplacian = normalize(pca_laplacian)
                        
                    pca_feats[prefix] = pca_laplacian
                    combined = np.concatenate((pca_laplacian, sol), axis=1)
                    
                    #pca_feat = torch.as_tensor(pca.fit_transform(combined)).float()
                    pca_feat = torch.as_tensor(combined).float()

                    #all_feat = np.concatenate((laplacian, sol), axis=1)
                    #pca_feat = torch.tensor(all_feat).float()
                    step_feats.append(pca_feat)
                else:
                    combined = np.concatenate((pca_feats[prefix], sol), axis=1)

                    #step_feats.append(
                    #    torch.as_tensor(pca.fit_transform(combined)).float())
                    step_feats.append(torch.as_tensor(combined).float())
            else:
                step_feats = features[step]
            
            step_labels.extend(clusters_to_labels(cluster))

        train_models(step_feats, step_labels, network, device, epoch, model_file, step)
        
        if not has_feat:
            features.append(step_feats)

    if not has_feat:
        with open(feat_file, 'wb') as f:
            pickle.dump(features, f)


def train_models(features, labels, network, device, epoch, model_file, step, 
                 save=True):
    
    model_file = model_file + '_step_%d' %(step)
    features = torch.cat(features)
    labels = torch.as_tensor(labels)

    loss_f = nn.CrossEntropyLoss()
    net = models.FeedforwardSoftmaxNet(network, device)
    # k = network[-1]
    # net = models.FeedforwardClusterNet(network[:-1], k, 10, device)
    
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)
    features = features.to(device)
    labels = labels.to(device)
    net = net.to(device)

    for i in range(1, epoch+1):
        optimizer.zero_grad()
        output = net(features)
        train_acc = common.compute_accuracy(output, labels)
        #valid_acc = common.compute_accuracy(output[num_train:], labels[num_train:])        
        loss = loss_f(output, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('loss:', loss)
            print('train accuracy: ', train_acc)
            #print('valid accuracy: ', valid_acc)
            if save:
                epoch_model_file = model_file + '_epoch_%d.pt' %i
                torch.save(net, epoch_model_file)
    return net


def evaluate(model_dir, network, seed, epoch, graph_dir, lp_dir, device,
             steps=1, p_type='maxcut', time_limit=1, alg_name='bc'):
    use_init = False

    if 'mvc' in lp_dir and '1500' in lp_dir:
        use_init = True
        
    output_dir = os.path.join('model_logs', lp_dir, alg_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if alg_name != 'rl':
        model_file = os.path.join(model_dir,
                                  '_'.join([str(x) for x in network]) + '_seed_%d' %(seed))
    else:
        model_file = model_dir

    models = []
    for i in range(steps):
        if alg_name != 'rl':
            step_model_file = model_file + '_step_%d_epoch_%d.pt' %(i, epoch)
            models.append(torch.load(step_model_file))            
        else:
            step_model_file = model_file
            model = torch.load(step_model_file)
            models.append(model.model)

    total_diff = []
    final_objs = []

    for idx, f_name in enumerate(os.listdir(lp_dir)):
        lp_file = os.path.join(lp_dir, f_name)
        prefix = f_name.split('.')[0]

        objs = []
        times = []
        output_file = os.path.join(output_dir, prefix + '.out')
        stat_output = open(output_file, 'w')

        if p_type == 'maxcut' or p_type == 'mvc':
            graph_file = os.path.join(graph_dir, prefix + '.gpickle')            
            graph = nx.read_gpickle(graph_file)
            laplacian = nx.laplacian_matrix(graph).todense()
            # laplacian = nx.normalized_laplacian_matrix(graph).todense()            
            # laplacian = nx.adjacency_matrix(graph).todense()            
        elif p_type == 'cats' or p_type == 'psulu':
            graph_file = os.path.join(graph_dir, prefix + '.npz')
            graph = sp.load_npz(graph_file).todense()
            laplacian = graph

        # adjacency = nx.adjacency_matrix(graph).todense()
        # normal_lap = nx.normalized_laplacian_matrix(graph).todense()
        if p_type == 'psulu':
            pca = PCA(n_components=network[0]-4)
        else:
            pca = PCA(n_components=network[0]-1)
            
        num_clusters = network[-1]

        # labels = (output > 0.5).nonzero()
        # clusters = partial_labels_to_clusters(labels.tolist(), k=3)

        problem, var_dict, init_obj, init_sol, init_sol_feat = (
            initialize_problem(lp_file, time_limit, p_type, use_init=use_init))

        pca_start = time.time()
        pca_laplacian = pca.fit_transform(laplacian)
        pca_end = time.time()
        print('PCA time: ', pca_end - pca_start)

        if p_type == 'psulu':
            obs, wp = lp_dir.split('/', maxsplit=3)[2].split('_')
            obs = int(obs)
            wp = int(wp)
            pca_laplacian = pca_laplacian[-obs*wp:]
            pca_laplacian = normalize(pca_laplacian)
        
        #pca_laplacian = laplacian

        objs.append(init_obj)
        if p_type == 'psulu':
            times.append(3.0)
        else:
            times.append(0.0)

        start_time = time.time()
        current_obj = init_obj
        use_random = False
        solver_t = 0

        for i in range(steps):
            net = models[i]
            if use_random:
                clusters = common.uniform_random_clusters(var_dict, num_clusters, p_type)
            else:
                #inf_start = time.time()
                combined = np.concatenate((pca_laplacian, init_sol_feat), axis=1)
                # pca_feat = torch.as_tensor(pca.fit_transform(combined)).float()
                pca_feat = torch.as_tensor(combined).float()
                features = pca_feat

                features = features.to(device)
                output = net(features)

                #labels = post_processing_two(output)
                #clusters = labels_to_clusters(labels, net.network[-1])

                if p_type == 'maxcut':
                    labels = post_processing(output)
                    clusters = labels_to_clusters(labels, net.network[-1])
                    # import pdb; pdb.set_trace()
                else:
                    labels = torch.argmax(output, dim=1)
                    clusters = labels_to_clusters(
                        labels.squeeze().tolist(), net.network[-1])
                    
                #inf_end = time.time()
                #print('Inf time: ', inf_end-inf_start)

            new_sol, new_sol_vec, t, new_obj = (
                coordinate_descent.coordinate_descent_by_clusters(
                    problem.copy(), clusters, var_dict, init_sol))
            init_sol_feat = np.c_[new_sol_vec]
            solver_t += t
            init_sol = new_sol

            cur_time = time.time()

            print('objective: ', new_obj)
            objs.append(new_obj)
            times.append(cur_time-start_time)
            # if math.isclose(current_obj, new_obj, rel_tol=0.0001):
            #     use_random = True
            #     print('Use random!')
            # else:
            #     use_random = False
            use_random = False

            current_obj = new_obj

        end_time = time.time()
        print(' '.join(['%.4f' %x for x in objs]),
              file=stat_output)
        print(' '.join(['%.4f' %x for x in times]),
              file=stat_output)
        stat_output.close()

        print('total diff: ', init_obj - new_obj)
        print('total time: ', end_time - start_time)
        print('obj: ', new_obj)
        print('solver time: ', solver_t)
        total_diff.append(init_obj - new_obj)
        final_objs.append(new_obj)

    # if not has_init:     
    #     with open(init_sol_feat_file, 'wb') as f:
    #         pickle.dump(init_sol_feats, f)

    # if not has_pca:
    #     with open(lap_pca_file, 'wb') as f:
    #         pickle.dump(lap_pca_feats, f)
            

    print('mean diff: ', np.mean(total_diff))
    print('std diff: ', np.std(total_diff))
    print('mean obj: ', np.mean(final_objs))
    print('std obj: ', np.std(final_objs) / np.sqrt(len(final_objs)))


def post_processing(probs):
    k = probs.size()[1]
    n = probs.size()[0]
    mean_cluster_size = n // k
    labels = [0 for i in range(n)]
    label_count = dict([(i, 0) for i in range(k)])
    probs = probs.cpu().detach().numpy()
    
    for i in range(n):
        prob = probs[i]
        order = np.argsort(-prob)
        for j in range(k):
            if label_count[order[j]] < mean_cluster_size:
                labels[i] = order[j]
                label_count[order[j]] += 1
                break

    return labels


def post_processing_two(probs):
    k = probs.size()[1]
    n = probs.size()[0]
    mean_cluster_size = n // k
    probs = probs.cpu().detach().numpy()
    labels = [-1 for i in range(n)]
    label_count = dict([(i, 0) for i in range(k)])

    sort_idx = np.argsort(-probs, axis=None)
    assigned = set()
    
    for idx in sort_idx:
        row = idx // k
        col = idx % k

        if row in assigned:
            continue

        if label_count[col] < mean_cluster_size:
            labels[row] = col
            label_count[col] += 1
            assigned.add(row)

    return labels


def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if FLAGS.network:
        network = common.convert_to_network(FLAGS.network)
    cluster_file = FLAGS.cluster_file
    epoch = FLAGS.epoch
    mode = FLAGS.mode
    model_dir = FLAGS.model_dir
    lp_dir = FLAGS.lp_dir
    p_type = FLAGS.p_type
    seed = FLAGS.seed
    cache_dir = FLAGS.cache_dir
    time_limit = FLAGS.time_limit
    steps = FLAGS.steps

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    torch.manual_seed(seed)
    np.random.seed(seed)

    if p_type == 'maxcut' or p_type == 'mvc':
        graph_dir = lp_dir.replace('lpfiles', 'gpickle')
    elif p_type == 'cats':
        graph_dir = lp_dir.replace('_lp', '_adj')
    elif p_type == 'psulu':
        graph_dir = lp_dir.replace('lpfiles', 'adj')

    if mode == 'train':
        supervise_training(network, lp_dir, graph_dir, cluster_file,
                           epoch, device, model_dir, p_type=p_type, seed=seed)
    elif mode == 'eval':
        evaluate(model_dir, network, seed, epoch, graph_dir, lp_dir, device,
                 steps=steps, p_type=p_type, time_limit=time_limit)
    
        
if __name__ == "__main__":
    app.run(main)
