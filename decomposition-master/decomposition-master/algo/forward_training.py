import sys
sys.path.append('.')

import torch

from absl import flags
from absl import app

from algo import supervise

import networkx as nx
import numpy as np
import pickle
import os
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import time

from utils import common
from utils import coordinate_descent


FLAGS = flags.FLAGS

#flags.DEFINE_string('network', None, 'Neural network architecture.')
#flags.DEFINE_string('graph_dir', None, 'Directory with gpickle files.')
#flags.DEFINE_string('lp_dir', None, 'Directory with lp files.')
#flags.DEFINE_string('cluster_file', None, 'Path to the file with best clusters.')
#flags.DEFINE_integer('epoch', 50, 'Number of epoches.')
#flags.DEFINE_string('mode', 'train', 'train or evaluate.')
#flags.DEFINE_string('model_file', None, 'Path to a model.')
#flags.DEFINE_string('p_type', 'maxcut', '')
#flags.DEFINE_integer('seed', 0, 'Random seed to use.')
#flags.DEFINE_string('cache_dir', None, '')
#flags.DEFINE_integer('time_limit', 1, '')
#flags.DEFINE_integer('steps', 1, '')


def forward_training(lp_dir,
                     graph_dir,
                     epoch,
                     device,
                     step,
                     p_type,
                     time_limit,
                     network,
                     model_dir,
                     best_first_clusters,
                     feat_file,
                     num_iter=5, 
                     seed=0):
    
    cur_sol = {}
    # cache pca_feats
    has_feat = False

    if os.path.exists(feat_file):
        has_feat = True
        with open(feat_file, 'rb') as f:
            pca_feats = pickle.load(f)
    else:
        pca_feats = {}
        
    model_file = os.path.join(model_dir,
                              '_'.join([str(x) for x in network]) + '_seed_%d' %(seed))
    problem_dict = {}
    var_dict = {}
    sol_dict = {}
    sol_feat_dict = {}
    num_cluster = network[-1]

    for i in range(step):
        step_feats = []
        step_labels = []
        if i == 0:
            best_clusters = best_first_clusters
            
            for lp_file in os.listdir(lp_dir):
                lp_f = os.path.join(lp_dir, lp_file)
                prefix = lp_file.split('.')[0]
                if p_type == 'mvc' or p_type == 'maxcut':
                    graph_file = os.path.join(graph_dir, prefix+'.gpickle')
                elif p_type == 'cats' or p_type == 'psulu':
                    graph_file = os.path.join(graph_dir, prefix+'.npz')
                cluster = best_clusters[prefix][0][0]
                
                if p_type == 'maxcut' or p_type == 'mvc':
                    graph = nx.read_gpickle(graph_file)
                    laplacian = nx.laplacian_matrix(graph).todense()
                elif p_type == 'cats' or p_type == 'psulu':
                    graph = sp.load_npz(graph_file).todense()
                    laplacian = graph

                problem, var, _, _, _ = (
                    supervise.initialize_problem(lp_f, time_limit, p_type,
                                                 optimize=False))
                
                init_sol_feat = best_clusters[prefix][1][0]
                init_sol = common.vec_to_dict(init_sol_feat, var)

                problem_dict[prefix] = problem
                var_dict[prefix] = var
                sol_feat_dict[prefix] = init_sol_feat
                sol_dict[prefix] = init_sol

                if not has_feat:
                    if p_type == 'psulu':
                        pca = PCA(n_components=network[0]-4)
                    else:
                        pca = PCA(n_components=network[0]-1)
                        
                    pca_laplacian = pca.fit_transform(laplacian)

                    if p_type == 'psulu':
                        obs, wp = lp_dir.split('/', maxsplit=3)[2].split('_')
                        obs = int(obs)
                        wp = int(wp)                        
                        pca_laplacian = pca_laplacian[-obs*wp:]
                        pca_laplacian = normalize(pca_laplacian)
                        
                    pca_feats[prefix] = pca_laplacian
                    combined = np.concatenate((pca_laplacian, init_sol_feat), axis=1)
                    pca_feat = torch.as_tensor(combined).float()
                else:
                    pca_feat = torch.as_tensor(
                        np.concatenate(
                            (pca_feats[prefix], init_sol_feat), axis=1)).float()
                step_feats.append(pca_feat)
                step_labels.extend(supervise.clusters_to_labels(cluster))                
        else:
            best_clusters = collect_best_clusters(
                problem_dict, var_dict, sol_dict, sol_feat_dict,
                num_iter, num_cluster, p_type, time_limit)

            for lp_f in os.listdir(lp_dir):
                prefix = lp_f.split('.')[0]
                cluster = best_clusters[prefix][0]
                combined = np.concatenate((pca_feats[prefix], sol_feat_dict[prefix]), axis=1)
                step_feats.append(torch.as_tensor(combined).float())

                step_labels.extend(supervise.clusters_to_labels(cluster))

        step_model = supervise.train_models(step_feats, step_labels, network, device,
                                            epoch, model_file, i)

        if i < step - 1:
            for k in problem_dict:
                # for each problem, predict one cluster
                problem = problem_dict[k]
                var = var_dict[k]
                sol = sol_dict[k]
                sol_feat = sol_feat_dict[k]
                feat = torch.as_tensor(
                    np.concatenate((pca_feats[k], sol_feat), axis=1)).float()
                feat = feat.to(device)
                output = step_model(feat)
                labels = torch.argmax(output, dim=1)
                clusters = supervise.labels_to_clusters(labels.squeeze().tolist(),
                                                        num_cluster)
                new_sol, new_sol_vec, _, _ = (
                    coordinate_descent.coordinate_descent_by_clusters(
                        problem.copy(), clusters, var, sol))
                sol_dict[k] = new_sol
                sol_feat_dict[k] = np.c_[new_sol_vec]

    if not has_feat:
        with open(feat_file, 'wb') as f:
            pickle.dump(pca_feats, f)


def collect_best_clusters(problem_dict,
                          var_dict,
                          sol_dict,
                          sol_feat_dict,
                          num_iter,
                          num_cluster,
                          p_type,
                          time_limit):
    best_clusters = {}

    for k in problem_dict:
        best_obj = 100000000
        best_cluster = None

        for _ in range(num_iter):
            start_time = time.time()
            solver_time, obj, total_diff, cluster_list, sol_list, _, _ = (
                coordinate_descent.coordinate_descent(
                    problem_dict[k], num_cluster,
                    steps=1, p_type=p_type,
                    var_dict=var_dict[k],
                    time_limit=time_limit,
                    sol=sol_dict[k],
                    sol_vec=sol_feat_dict[k]))
            end_time = time.time()
            print(end_time - start_time, solver_time, obj, total_diff)

            if obj < best_obj:
                best_obj = obj
                best_cluster = cluster_list

        best_clusters[k] = best_cluster

    return best_clusters


def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if FLAGS.network:
        network = common.convert_to_network(FLAGS.network)
    cluster_file = FLAGS.cluster_file
    epoch = FLAGS.epoch
    mode = FLAGS.mode
    model_dir = FLAGS.model_dir
    lp_dir = FLAGS.lp_dir
    if lp_dir.endswith('/'):
        lp_dir = lp_dir[:-1]
        
    p_type = FLAGS.p_type
    seed = FLAGS.seed
    time_limit = FLAGS.time_limit
    steps = FLAGS.steps

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    torch.manual_seed(seed)
    np.random.seed(seed)

    best_clusters = pickle.load(open(cluster_file, 'rb'))

    feat_dir = os.path.join(
        '/tmp/jssong',
        cluster_file.rsplit('/',maxsplit=1)[0])
    dir_name = lp_dir.rsplit('/', maxsplit=1)[1]
    feat_file = os.path.join(
        feat_dir,
        '%s_pca_feat_%d.pkl' %(dir_name, network[0]-1))

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    
    if mode == 'train':
        model_file = os.path.join(
            model_dir,
            '_'.join([str(x) for x in network]) + '_seed_%d' %(seed))

    if p_type == 'maxcut' or p_type == 'mvc':
        graph_dir = lp_dir.replace('lpfiles', 'gpickle')
    elif p_type == 'cats':
        graph_dir = lp_dir.replace('_lp', '_adj')
    elif p_type == 'psulu':
        graph_dir = lp_dir.replace('lpfiles', 'adj')

    if mode == 'train':
        forward_training(lp_dir, graph_dir, epoch, device, steps,
                         p_type, time_limit, network,
                         model_dir, best_clusters, feat_file, seed=seed)
    elif mode == 'eval':
        supervise.evaluate(model_dir, network, seed, epoch, graph_dir, lp_dir, device,
                           steps=steps, p_type=p_type, time_limit=time_limit, alg_name='ft')
        
        
if __name__ == "__main__":
    app.run(main)
