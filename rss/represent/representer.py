from rss.represent.inr import MLP,SIREN
import torch.nn as nn
from rss.represent.tensor import DMF,TF
from rss.represent.utils import reshape2

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

    def forward(self, x):
        # Too ugly here
        for i,net in enumerate(self.net_list):
            if self.net_list_para[i]['net_name'] in ['DMF','TF'] and len(self.net_list_para)>1:
                x = net(x).reshape(-1,self.net_list_para[i+1]['dim_in'])
            else:
                x = net(x)
        if self.net_list_para[0]['net_name'] in ['DMF','TF']:
            return x.reshape((self.net_list_para[0]['sizes'][0],self.net_list_para[0]['sizes'][-1]))
        else:
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



























