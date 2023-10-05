'''Comparison-based optimization. Maximize the log-likelihood of
   the best decompositions.'''

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from utils import common


# def clusters_to_labels(clusters):
#     '''Convert a cluster to per-node labels.'''

#     labels = [0 for _ in range(sum([len(v) for v in clusters.values()]))]

#     for k, cluster in clusters.items():
#         for node in cluster:
#             labels[node] = k

#     return labels


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


class CompareAgent(object):

    def __init__(self, model, lr, weight_decay, device, kmeans_iter):
        self.model = model
        self.lr = lr
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        self.model.to(self.device)
        self.kmeans_iter = kmeans_iter
        self.best_clusters = {}
        self.best_rewards = {}


    def update(self, log_probs, rewards):

        # log_probs and rewards are lists of lists.
        # Each list contains log_probs and rewards for a single problem.

        policy_loss = 0

        for log_prob, reward in zip(log_probs, rewards):
            # select the top half of rewards
            traj_rewards = [sum(rs) for rs in reward]
            prob_reward = list(zip(traj_rewards, log_prob))
            prob_reward = sorted(prob_reward, key=lambda x: x[0], reverse=True)
            mean_rewards = np.mean(traj_rewards)

            # maximize log probs of good clusters
            # minimize log probs of bad clusters
            for i in range(len(prob_reward)):
                for l_p in prob_reward[i][1]:
                    if i == 0:
                        policy_loss += -l_p
                    # else:
                    #     policy_loss += l_p
                    
                #if prob_reward[i][0] > mean_rewards:
                #    for l_p in prob_reward[i][1]:
                #        policy_loss += -l_p
                #else:
                #    for l_p in prob_reward[i][1]:
                #        policy_loss += l_p
                        
            # for i in range(len(prob_reward) // 2):
            #     for l_p in prob_reward[i][1]:
            #         policy_loss += -l_p

            # # minimize log probs of bad clusters
            # for i in range(len(prob_reward) // 2, len(prob_reward)):
            #     for l_p in prob_reward[i][1]:
            #         policy_loss += l_p

    def update_with_features(self, features, clusters, rewards):

        policy_loss = 0
        train_feat = []
        train_cluster = []

        for feature, cluster, reward in zip(features, clusters, rewards):
            # select the top half of rewards
            traj_rewards = [sum(rs) for rs in reward]
            prob_reward = list(zip(traj_rewards, feature, cluster))
            prob_reward = sorted(prob_reward, key=lambda x: x[0], reverse=True)
            mean_rewards = np.mean(traj_rewards)

            # maximize log probs of good clusters
            # minimize log probs of bad clusters

            feat, target_cluster = prob_reward[0][1], prob_reward[0][2]
            train_feat.append(feat)
            train_cluster.append(target_cluster)

        for i in range(10):
            policy_loss = 0
            for problem_feat, problem_cluster in zip(train_feat, train_cluster):
                for feat, cluster in zip(problem_feat, problem_cluster):
                    feat = feat.to(self.device)
                    # _, cluster_probs, _, _ = self.model(
                    #     feat,
                    #     num_iter=self.kmeans_iter)
                    cluster_probs = self.model(feat)                    
                    loss = self.cluster_log_prob(
                        cluster_probs, cluster)
                    policy_loss += loss
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            print('policy_loss: ', policy_loss.item())


    def cluster_log_prob(self, cluster_probs, cluster):
        loss_f = nn.CrossEntropyLoss()
        loss = 0
        labels = clusters_to_labels(cluster)
        labels = torch.tensor(labels).to(self.device)
        loss = loss_f(cluster_probs, labels)

        return loss
    

    def select_action(self, adj, features, mode):
        features = features.to(self.device)
        adj = adj.to(self.device)
        new_features = None

        # _, cluster_probs, new_features, _ = self.model(features, adj,
        #                                                num_iter=self.kmeans_iter)
        # _, cluster_probs, new_features, _ = self.model(features,
        #                                                num_iter=self.kmeans_iter)
        cluster_probs = self.model(features)

        partitions, log_prob, entropy = common.sample_partitions(cluster_probs,
                                                                 mode)
        # print('entropy: ', entropy)

        return partitions, log_prob, new_features
        
