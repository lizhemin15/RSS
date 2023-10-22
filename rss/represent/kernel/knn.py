from sklearn.decomposition import PCA
from sklearn import neighbors
import torch.nn as nn
import torch
from rss.represent.tensor import TF
import rss.toolbox as tb
import numpy as np


class KNN_net(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        # decomposition with tucker
        self.G_net = TF(parameter)
        self.sizes = parameter['sizes']
        self.weights = parameter['weights']
        _, self.G_cor = tb.get_cor(xshape=parameter['sizes'],xrange=1) # Numpy array, (\prod xshape,len(xshape))
        self.update_neighbor()

    def forward(self,x):
        G = self.G_net(None) # Torch, shape: parameter['sizes']
        G = tb.reshape2(G) # Torch, shape: (\prod xshape,1)
        return torch.mean(G[self.neighbor_index]*self.neighbor_dist.to(G.device).to(torch.float32),dim=1).reshape(self.sizes)

    def update_neighbor(self,mode='cor',n_neighbors=1,mask=None):
        # get neighbor_index
        # mode = 'cor','patch','PCA'
        # X shape (n_samples,n_features)
        if mask == None:
            trainx = self.G_cor
            testx = self.G_cor
        else:
            trainx = self.G_cor[(mask==1).reshape(-1,)]
            testx = self.G_cor
        if mode == 'cor':
            pass
        elif mode == 'patch':
            pass
        elif mode == 'PCA':
            pass
        else:
            raise('Wrong mode = ',mode)
        neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(trainx)
        dist, self.neighbor_index = neigh.kneighbors(testx)
        # cal weight
        if self.weights == 'distance':
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
                inf_mask = np.isinf(dist)
                inf_row = np.any(inf_mask, axis=1)
                dist[inf_row] = inf_mask[inf_row]
                dist = dist-np.min(dist,axis=1,keepdims=True)+1e-7
                dist = dist/np.sum(dist,axis=1)
        elif self.weights == 'uniform':
            pass
        else:
            raise('Wrong weighted method=',self.weights)
        dist = np.expand_dims(dist, axis=2)
        self.neighbor_dist = torch.tensor(dist)



def KNN(parameter):
    de_para_dict = {'sizes':[100,100],'dim_cor':[100,100],'mode':'tucker','weights':'distance'}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return KNN_net(parameter)

