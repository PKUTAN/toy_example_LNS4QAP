import torch.optim as optim
import numpy as np

from utils import common

from itertools import chain


EPS = np.finfo(np.float32).eps.item()


class ReinforceAgent(object):

    def __init__(self, model, lr, weight_decay, device, kmeans_iter):
        self.model = model
        self.lr = lr
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        self.model.to(self.device)
        self.kmeans_iter = kmeans_iter


    def normalize_returns(self, train_rewards):
        '''Normalize returns by subtracting mean and divide by standared deviation.'''
        
        train_returns = []

        for problem_rewards in train_rewards:
            problem_returns = []
            for rewards in problem_rewards:
                returns = [0.0 for _ in range(len(rewards))]
                for idx, r in reversed(list(enumerate(rewards))):
                    if idx == len(rewards) - 1:
                        returns[idx] = r
                    else:
                        returns[idx] = r + returns[idx+1]
                problem_returns.append(returns)
            mean_returns = np.mean(problem_returns, axis=0)
            std_returns = np.std(problem_returns, axis=0)

            for i, returns in enumerate(problem_returns):
                problem_returns[i] = [returns[j] - mean_returns[j]
                                      for j in range(len(returns))]
                
            train_returns.append(problem_returns)

        return train_returns
    
        
    def update(self, log_probs, rewards):

        # import pdb; pdb.set_trace()
        n = len(rewards)
        returns = self.normalize_returns(rewards)
        policy_loss = 0
        log_probs = list(chain.from_iterable(chain.from_iterable(log_probs)))
        returns = list(chain.from_iterable(chain.from_iterable(returns)))

        for log_prob, R in zip(log_probs, returns):
            policy_loss += -log_prob * R

        # policy_loss /= n

        self.optimizer.zero_grad()
        policy_loss.backward()

        # import pdb; pdb.set_trace()
        # for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
        #     print('grad norm: ', p.grad.data.norm(2).item())
        #     print('param norm: ', p.data.norm(2).item())
            
        self.optimizer.step()
        print('policy_loss: ', policy_loss.item())


    def select_action(self, features, mode='train'):
        features = features.to(self.device)
        # adj = adj.to(self.device)

        #_, cluster_probs, new_features, _ = self.model(features, adj,
        #                                               num_iter=self.kmeans_iter)
        cluster_probs = self.model(features)        

        partitions, log_prob, entropy = common.sample_partitions(cluster_probs, mode)
        # print('entropy: ', entropy)

        return partitions, log_prob
