from rss.represent.inr import MLP,SIREN
import torch.nn as nn
from rss.represent.tensor import DMF,TF
from rss.represent.utils import reshape2
from rss.represent.interpolation import Interpolation
from rss.represent.unn import UNN
from rss.represent.kernel import KNN,TDKNN

def get_nn(parameter={}):
    net_name = parameter.get('net_name','SIREN')
    if net_name == None:
        net_name = 'None'
    if net_name == 'composition':
        net = Composition(parameter)
    elif net_name == 'MLP':
        net = MLP(parameter)
    elif net_name == 'SIREN':
        net = SIREN(parameter)
    elif net_name == 'DMF':
        net = DMF(parameter)
    elif net_name == 'TF':
        net = TF(parameter)
    elif net_name == 'Interpolation':
        net = Interpolation(parameter)
    elif net_name in ['UNet','ResNet','skip']:
        net = UNN(parameter)
    elif net_name == 'KNN':
        net = KNN(parameter)
    elif net_name == 'TDKNN':
        net = TDKNN(parameter)
    else:
        raise('Wrong net_name = ',net_name)
    return net



class Composition(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.net_list_para = parameter.get('net_list',[{'net_name':'SIREN'}])
        net_list = []
        for _,net_para in enumerate(self.net_list_para):
            net_list.append(get_nn(net_para))
        self.net_list = nn.ModuleList(net_list)

    def forward(self, x_in):
        for i,net in enumerate(self.net_list):
            if i == 0:
                x = net(x_in)
                continue
            if self.net_list_para[i]['net_name'] == 'Interpolation':
                x = net(x=x_in,tau=x)
            else:
                x = net(x)
        return x
        
class Contenate(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.net_list_para = parameter.get('net_list',[{'net_name':'SIREN'}])
        net_list = []
        for _,net_para in enumerate(self.net_list_para):
            net_list.append(get_nn(net_para))
        self.net_list = nn.ModuleList(net_list)


    def forward(self,x_list):
        # Contenate multiple input together to a single net
        pass



























