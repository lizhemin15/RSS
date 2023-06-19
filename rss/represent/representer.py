from rss.represent.inr import MLP,SIREN
import torch.nn as nn

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
    else:
        raise('Wrong net_name = ',net_name)
    return net



class Composition(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        net_list_para = parameter.get('net_list',[{'net_name':'SIREN'}])
        net_list = []
        for net_para in net_list_para:
            net_list.append(get_nn(net_para))
        self.net_list = nn.ModuleList(net_list)

    def forward(self, x):
        for net in self.net_list:
            x = net(x)
        return x