from sklearn.decomposition import PCA
from sklearn import neighbors
import torch.nn as nn
import torch
from rss.represent.tensor import DMF,TF
from rss.represent.unn import UNN
import rss.toolbox as tb
import numpy as np
from einops import rearrange

class TDKNN_net(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        # decomposition with tucker
        self.G_net = TF(parameter)
        self.sizes = parameter['sizes']
        self.weights = parameter['weights']
        self.weights_alpha = parameter.get('weights_alpha', 1)
        self.mode = parameter['mode']
        self.G_cor_list = self.gen_cor(parameter['sizes'],xrange=1.01231546) # List of (xshape[i],1)
        self.G_cor_test_list = self.gen_cor(parameter['sizes'],xrange=1) # List of (xshape[i],1)
        self.update_neighbor()

    def forward(self,x):
        Mx_i_list = []
        for i in range(len(self.G_net.net_list)):
            M_i = self.G_net.net_list[i].weight # Torch, shape:  parameter['sizes'][i] \times parameter['dim_cor'][i] 
            Mn_i = M_i[self.neighbor_index_list[i],:] # Torch, shape: parameter['sizes'][i] \times n_neighbors \times parameter['dim_cor'][i]
            #print('Mn_i shape:',Mn_i.shape)
            Mx_i = torch.sum(Mn_i*self.neighbor_dist_list[i].to(x.device).to(torch.float32),dim=1) # Weighted : Torch, shape: parameter['sizes'][i] \times parameter['dim_cor'][i] 
            #print('Mx_i shape:',Mx_i.shape)
            Mx_i_list.append(Mx_i)
        self.Mx_i_list = Mx_i_list
        return self.G_net.tucker_product(self.G_net.G,Mx_i_list) # Torch, shape: parameter['sizes']

        # self.G = self.G_net(x) # Torch, shape: parameter['sizes']
        # G = tb.reshape2(self.G) # Torch, shape: (\prod xshape,1)
        # return torch.sum(G[self.neighbor_index]*self.neighbor_dist.to(G.device).to(torch.float32),dim=1).reshape(self.sizes)

    def gen_cor(self,xshape,xrange=1):
        G_cor_list = []
        for shape_now in xshape:
            _, G_cor_i = tb.get_cor(xshape=[shape_now],xrange=xrange) # Numpy array, (\prod xshape,len(xshape))
            G_cor_list.append(G_cor_i)
        return G_cor_list

    def update_neighbor(self, mode='cor', n_neighbors=1, mask=None, n_components=1, labda=1):
        # 更新邻居索引
        # mode: 'cor', 'patch', 'PCA'
        # X shape (n_samples, n_features)
        self.mask = mask
        if mode == 'cor':
            feature_list = self.G_cor_list.copy()
            feature_test_list = self.G_cor_test_list.copy()
        elif mode == 'y':
            Mx_i_list = self.Mx_i_list # self.forward(None).detach().cpu().numpy()
            feature_list = []
            feature_test_list = []
            for i,Mx_i in enumerate(Mx_i_list):
                #TODO add codes here
                # Mx_i: Torch, shape: parameter['sizes'][i] \times  parameter['dim_cor'][i]
                Mx_i = Mx_i.numpy()
                feature = mix_feature(self.G_cor_list[i], Mx_i, labda=labda)
                feature_test = mix_feature(self.G_cor_test_list[i], Mx_i, labda=labda)
                feature_list.append(feature)
                feature_test_list.append(feature_test)
        else:
            raise('Wrong mode = ', mode)

        trainx_list = feature_list
        testx_list = feature_test_list
        self.neighbor_index_list = []
        self.neighbor_dist_list = []
        for i in range(len(trainx_list)):
            trainx = trainx_list[i]
            testx = testx_list[i]
            neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
            neigh.fit(trainx)
            dist, neighbor_index = neigh.kneighbors(testx)
            self.neighbor_index_list.append(neighbor_index)

            # 计算权重
            if self.weights == 'distance':
                with np.errstate(divide="ignore"):
                    dist = 1.0 / dist
                    inf_mask = np.isinf(dist)
                    inf_row = np.any(inf_mask, axis=1)
                    dist[inf_row] = inf_mask[inf_row]
                    dist = dist**self.weights_alpha
                    #dist = dist - np.min(dist, axis=1, keepdims=True) + 1e-7
                    #dist = dist / np.sum(dist, axis=1, keepdims=True)
            elif self.weights == 'softmax':
                dist = np.exp(-dist / self.weights_alpha**2)
            elif self.weights == 'uniform':
                dist = np.ones(dist.shape)
            else:
                raise('Wrong weighted method=', self.weights)

            dist = dist / np.sum(dist, axis=1, keepdims=True)
            neighbor_dist = torch.tensor(dist)
            #self.neighbor_dist = torch.nn.functional.softmax(self.neighbor_dist)
            neighbor_dist = neighbor_dist.unsqueeze(2)
            self.neighbor_dist_list.append(neighbor_dist)


def normalization(x_in):
    min_value = np.min(x_in.reshape(-1, 1, x_in.shape[-1]), axis=0)
    max_value = np.max(x_in.reshape(-1, 1, x_in.shape[-1]), axis=0)
    return (x_in-min_value)/(max_value-min_value)

def mix_feature(x_in,feature,labda=1):
    x_in_norm = normalization(x_in)/x_in.shape[-1]
    feature_norm = normalization(feature)/feature.shape[-1]
    return np.concatenate((x_in_norm,feature_norm*labda),axis=1)
    
def TDKNN(parameter):
    de_para_dict = {'sizes':[100,100],'dim_cor':[100,100],'weights':'distance','mode':'tucker'}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return TDKNN_net(parameter)



