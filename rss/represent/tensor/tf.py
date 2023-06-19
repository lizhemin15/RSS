from torch import nn
import torch
import math
class TensorFactorization(nn.Module):
    def __init__(self, dim_ori, dim_cor,mode='tucker'):
        super().__init__()
        net_list = []
        for i in range(len(dim_cor)):
            net_list.append(nn.Linear(in_features=dim_cor[i], out_features=dim_ori[i], bias=False))
        self.net_list = nn.ModuleList(net_list)
        self.mode = mode
        if mode == 'tucker':
            stdv = 1 / math.sqrt(dim_cor[0])*1e-3
            self.G = torch.nn.Parameter((torch.randn(dim_cor)-0.5)*2*stdv)
    
    def forward(self,x):
        # x is a list, every element is a tensor
        pre = []
        for i in range(len(self.net_list)):
            pre.append(self.net_list[i].weight)
        self.pre = pre
        return self.tucker_product(self.G,pre)
        
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