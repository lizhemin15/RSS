from torch import nn
import torch
import math

class TensorFactorization(nn.Module):
    def __init__(self, dim_ori, dim_cor,mode='tucker'):
        super().__init__()
        self.mode = mode
        if self.mode == 'tucker':
            stdv = 1 / math.sqrt(dim_cor[0])*1e-3
            self.G = torch.nn.Parameter((torch.randn(dim_cor)-0.5)*2*stdv)
            net_list = []
            for i in range(len(dim_cor)):
                net_list.append(nn.Linear(in_features=dim_cor[i], out_features=dim_ori[i], bias=False))
            self.net_list = nn.ModuleList(net_list)
        elif self.mode == 'cp':
            pass
        elif self.mode == 'tensor':
            stdv = 1 / math.sqrt(dim_ori[0])*1e-3
            self.G = torch.nn.Parameter((torch.randn(dim_ori)-0.5)*2*stdv)
        else:
            raise('Not suppose mode = ',self.mode)
    
    def forward(self,x):
        # x is a list, every element is a tensor
        if self.mode == 'tucker':
            pre = []
            for i in range(len(self.net_list)):
                pre.append(self.net_list[i].weight)
            self.pre = pre
            return self.tucker_product(self.G,pre)
        elif self.mode == 'tensor':
            return self.G
        
    def tucker_product(self,G,pre):
        abc_str = 'abcdefghijklmnopqrstuvwxyz'
        Gdim = G.dim()
        for i in range(Gdim):
            einstr = abc_str[:Gdim]+','+abc_str[Gdim]+abc_str[i]+'->'+abc_str[:Gdim].replace(abc_str[i],abc_str[Gdim])
            if i == 0:
                Gnew = torch.einsum(einstr,[G,pre[i]])
            else:
                Gnew = torch.einsum(einstr,[Gnew,pre[i]])
        return Gnew
    
def TF(parameter):
    de_para_dict = {'sizes':[100,100],'dim_cor':[100,100],'mode':'tucker'}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return TensorFactorization(parameter['sizes'],parameter['dim_cor'],parameter['mode'])