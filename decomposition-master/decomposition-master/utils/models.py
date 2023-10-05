from sklearn.cluster import KMeans
import torch

import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

from pygcn import utils


class GCN(nn.Module):
    '''
    2-layer GCN with dropout
    '''
    def __init__(self, network, dropout):
        super(GCN, self).__init__()

        self.layers = []
        for idx in range(len(network)-1):
            self.layers.append(GraphConvolution(network[idx], network[idx+1]))

        # self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        # self.gc2 = GraphConvolution(nhid, nout, bias=False)    
        # self.gc3 = GraphConvolution(nhid, nout, bias=False)
        self.layers = nn.ModuleList(self.layers)
        self.dropout = dropout

    def forward(self, x, adj):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, adj)
            # x = F.dropout(x, self.dropout, training=self.training)

        x = self.layers[-1](x, adj)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)        
        # x = self.gc2(x, adj)
        return x


def cluster(data, k, temp, num_iter, device, init=None, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        data_np = data.detach().cpu().numpy()
        # norm = (data_np**2).sum(axis=1)
        # init = cluster.k_means_._k_init(
        #     data_np, k, norm, sklearn.utils.check_random_state(None))
        init = KMeans(n_clusters=k).fit(data_np).cluster_centers_
        # import pdb; pdb.set_trace()
        init = torch.tensor(init, requires_grad=True)

        if num_iter == 0: return init
    mu = init.to(device)
    #mu = torch.diag(1./torch.norm(init, dim=1, p=2)) @ init
    n = data.shape[0]
    d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for t in range(num_iter):
        #get distances between all data points and cluster centers
        #dist = torch.cosine_similarity(
        #    data[:, None].expand(n, k, d).reshape((-1, d)),
        #    mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        # cluster_mean = r.t() @ data
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu

    #dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)),
    #                               mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
    
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist


class GCNClusterNet(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    '''
    def __init__(self, network, dropout, K, cluster_temp, device):
        super(GCNClusterNet, self).__init__()

        self.GCN = GCN(network, dropout)
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        # self.init =  torch.rand(self.K, nout)
        # self.init = self.init.to(device)
        self.device = device
        self.network = network
        
    def forward(self, x, adj, num_iter=1):
        embeds = self.GCN(x, adj)
        # mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, self.device,
        #                         cluster_temp=self.cluster_temp, init=self.init)
        # mu, r, dist = cluster(embeds, self.K, 1, 1, self.device,
        #                       cluster_temp=self.cluster_temp,
        #                       init=mu_init.detach().clone())
        mu, r, dist = cluster(embeds, self.K, 1, num_iter, self.device,
                              cluster_temp=self.cluster_temp, init=None)
        # import pdb; pdb.set_trace()
        return mu, r, embeds, dist


class FeedforwardNet(nn.Module):
    def __init__(self, network):
        super(FeedforwardNet, self).__init__()
        
        self.network = network
        self.layers = []

        for idx in range(len(network)-1):
            self.layers.append(nn.Linear(network[idx], network[idx+1]))

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))

        # x = F.softmax(self.layers[-1](x))
        x = self.layers[-1](x)
        return x


class FeedforwardSoftmaxNet(nn.Module):
    def __init__(self, network, device):
        super(FeedforwardSoftmaxNet, self).__init__()
        
        self.network = network
        self.layers = []
        self.device = device

        for idx in range(len(network)-1):
            self.layers.append(nn.Linear(network[idx], network[idx+1]))

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))

        # x = F.softmax(self.layers[-1](x))
        x = F.softmax(self.layers[-1](x))
        return x    


class FeedforwardClusterNet(nn.Module):
    def __init__(self, network, K, cluster_temp, device):
        super(FeedforwardClusterNet, self).__init__()

        self.net = FeedforwardNet(network)
        self.K = K
        self.cluster_temp = cluster_temp
        self.device = device
        self.network = network

        
    def forward(self, x, num_iter=1):
        embeds = self.net(x)
        # mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, self.device,
        #                         cluster_temp=self.cluster_temp, init=self.init)
        # mu, r, dist = cluster(embeds, self.K, 1, 1, self.device,
        #                       cluster_temp=self.cluster_temp,
        #                       init=mu_init.detach().clone())
        mu, r, dist = cluster(embeds, self.K, 1, num_iter, self.device,
                              cluster_temp=self.cluster_temp, init=None)
        # import pdb; pdb.set_trace()
        return r
        # return mu, r, embeds, dist
            

class DeepGCN(nn.Module):
    '''GraphConvolution layers followed by feedforward layers.'''

    def __init__(self, gcn_network, ff_network, K, device):
        super(DeepGCN, self).__init__()

        self.GCN = GCN(network, dropout)
        
