from sklearn.decomposition import PCA
from sklearn import neighbors
import torch.nn as nn
import torch
from rss.represent.tensor import DMF,TF
from rss.represent.unn import UNN
import rss.toolbox as tb
import numpy as np
from einops import rearrange

class KNN_net(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        # decomposition with tucker
        if parameter['mode'] in ['tucker','tensor']:
            self.G_net = TF(parameter)
        elif parameter['mode'] in ['DMF']:
            sizes = []
            sizes.append(parameter['sizes'][0])
            sizes.extend(parameter['dim_cor'])
            sizes.append(parameter['sizes'][1])
            self.G_net = DMF({'sizes':sizes})
        elif parameter['mode'] in ['UNet','ResNet','skip']:
            self.G_net = UNN({'net_name':parameter['mode'],'input_depth': 1,'num_output_channels':1})
        else:
            raise('Wrong mode = ',parameter['mode'])
        self.sizes = parameter['sizes']
        self.weights = parameter['weights']
        self.weights_alpha = parameter.get('weights_alpha', 1)
        self.mode = parameter['mode']
        _, self.G_cor = tb.get_cor(xshape=parameter['sizes'],xrange=1.01231546) # Numpy array, (\prod xshape,len(xshape))
        _, self.G_cor_test = tb.get_cor(xshape=parameter['sizes'],xrange=1) # Numpy array, (\prod xshape,len(xshape))
        self.update_neighbor()

    def forward(self,x):
        self.G = self.G_net(x) # Torch, shape: parameter['sizes']
        G = tb.reshape2(self.G) # Torch, shape: (\prod xshape,1)
        return torch.sum(G[self.neighbor_index]*self.neighbor_dist.to(G.device).to(torch.float32),dim=1).reshape(self.sizes)

    def update_neighbor(self, mode='cor', n_neighbors=1, mask=None, n_components=1, labda=1):
        # 更新邻居索引
        # mode: 'cor', 'patch', 'PCA'
        # X shape (n_samples, n_features)
        self.mask = mask
        if mode == 'cor':
            feature = self.G_cor.copy()
            feature_test = self.G_cor_test.copy()
        elif mode == 'patch':
            G = self.G.detach().cpu().numpy()#self.forward(None).detach().cpu().numpy()
            if len(G.shape) == 2:
                feature_patch = patch_feature(G, n_components)
                feature_patch = rearrange(feature_patch, 'a b c -> (a b) (c)')
                feature = mix_feature(self.G_cor, feature_patch, labda=labda)
                feature_patch_test = patch_feature(G, n_components)
                feature_patch_test = rearrange(feature_patch_test, 'a b c -> (a b) (c)')
                feature_test = mix_feature(self.G_cor_test, feature_patch_test, labda=labda)
            else:
                raise('neighbor mode patch only suppose the 2-dimension shape, not suppose your data shape:', G.shape)
        elif mode == 'PCA':
            G = self.G.detach().cpu().numpy()#self.forward(None).detach().cpu().numpy()
            if len(G.shape) == 2:
                feature_PCA = PCA_feature(G, n_components)
                feature_PCA = rearrange(feature_PCA, 'a b c -> (a b) (c)')
                feature = mix_feature(self.G_cor, feature_PCA, labda=labda)
                feature_PCA_test = PCA_feature(G, n_components)
                feature_PCA_test = rearrange(feature_PCA_test, 'a b c -> (a b) (c)')
                feature_test = mix_feature(self.G_cor, feature_PCA_test, labda=labda)
            else:
                raise('neighbor mode PCA only suppose the 2-dimension shape, not suppose your data shape:', G.shape)
        else:
            raise('Wrong mode = ', mode)

        if mask == None:
            trainx = feature
            testx = feature_test
        else:
            trainx = feature.copy()
            trainx[(mask==0).reshape(-1,)] = 1e6
            testx = feature

        neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(trainx)
        dist, self.neighbor_index = neigh.kneighbors(testx)

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
        self.neighbor_dist = torch.tensor(dist)
        #self.neighbor_dist = torch.nn.functional.softmax(self.neighbor_dist)
        self.neighbor_dist = self.neighbor_dist.unsqueeze(2)
    
def my_meshgrid(x,y):
    if len(x.shape) == 1:
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
    n_components = x.shape[-1]
    result = np.zeros((x.shape[0], y.shape[0], n_components*2))
    result[:, :, :n_components] += y[np.newaxis,:,:]
    result[:, :, n_components:] = x[:,np.newaxis,:]
    return result

def normalization(x_in):
    min_value = np.min(x_in.reshape(-1, 1, x_in.shape[-1]), axis=0)
    max_value = np.max(x_in.reshape(-1, 1, x_in.shape[-1]), axis=0)
    return (x_in-min_value)/(max_value-min_value)

def PCA_feature(X,n_components=5):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    feature_col = pca.components_.T
    pca.fit(X.T)
    feature_row = pca.components_.T
    return my_meshgrid(feature_col,feature_row)

def patch_feature(X,extend_size=1):
    # patch size = (2*extend_size+1)**2
    # 先写矩阵形式的，后续再写张量版本
    X_extend = np.zeros((X.shape[0]+extend_size*2,X.shape[1]+extend_size*2,(2*extend_size+1)**2))
    for i in range(2*extend_size+1):
        for j in range(2*extend_size+1):
            X_extend[i:X.shape[0]+i,j:X.shape[1]+j,i*(2*extend_size+1)+j] = X
    return X_extend[extend_size:-extend_size,extend_size:-extend_size,:]

def mix_feature(x_in,feature,labda=1):
    x_in_norm = normalization(x_in)/x_in.shape[-1]
    feature_norm = normalization(feature)/feature.shape[-1]
    return np.concatenate((x_in_norm,feature_norm*labda),axis=1)
    
def KNN(parameter):
    de_para_dict = {'sizes':[100,100],'dim_cor':[100,100],'mode':'tucker','weights':'distance'}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return KNN_net(parameter)



