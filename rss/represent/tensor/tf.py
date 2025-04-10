from torch import nn
import torch
import math
from rss.represent.inr import SIREN
from rss.represent.kan import get_kan

class TensorFactorization(nn.Module):
    def __init__(self, dim_ori, dim_cor,mode='tucker',w0=1,w0_initial=30.):
        super().__init__()
        self.mode = mode
        stdv = 1 / math.sqrt(dim_cor[0])*1e-3
        self.G = torch.nn.Parameter((torch.randn(dim_cor)-0.5)*2*stdv)
        if self.mode == 'tucker':
            net_list = []
            for i in range(len(dim_cor)):
                net_list.append(nn.Linear(in_features=dim_cor[i], out_features=dim_ori[i], bias=False))
            self.net_list = nn.ModuleList(net_list)
        elif self.mode == 'cp':
            # wait for define
            pass
        elif self.mode == 'tensor':
            # G is defined
            pass
        elif self.mode in ['tucker_inr','tucker_kan']:
            net_list = []
            self.input_list = []
            for i in range(len(dim_cor)):
                if self.mode == 'tucker_inr':
                    net_list.append(SIREN({'dim_in':1,'dim_hidden':256,'dim_out':dim_cor[i],'num_layers':2,'w0':w0,'w0_initial':w0_initial,'use_bias':True}))
                elif self.mode == 'tucker_kan':
                    net_list.append(get_kan({'net_name':"EFF_KAN",'dim_in':1,'dim_hidden':20,'dim_out':dim_cor[i],'num_layers':3,'spline_type':'spline','grid_size':100}))
                self.input_list.append(torch.linspace(-1,1,dim_ori[i]).reshape(-1,1))
            self.net_list = nn.ModuleList(net_list)

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
        elif self.mode in ['tucker_inr','tucker_kan']:
            pre = []
            for i in range(len(self.net_list)):
                net_now = self.net_list[i]
                input_now = self.input_list[i].to(self.G.device)
                pre.append(net_now(input_now))
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
    
def TF(parameter):
    de_para_dict = {'sizes':[100,100],'dim_cor':[100,100],'mode':'tucker','w0':1,'w0_initial':30.}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return TensorFactorization(dim_ori=parameter['sizes'],dim_cor=parameter['dim_cor'],mode=parameter['mode'],
                                w0=parameter['w0'],w0_initial=parameter['w0_initial'])