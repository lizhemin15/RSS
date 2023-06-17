import torch.nn.functional as F
import torch
from torch import nn

def get_act(act):
    if act == 'relu':
        act_return = F.relu
    elif act == 'sigmoid':
        act_return = F.sigmoid
    elif act == 'tanh':
        act_return = torch.tanh
    elif act == 'softmax':
        act_return = F.softmax
    elif act == 'threshold':
        act_return = F.threshold
    elif act == 'hardtanh':
        act_return = F.hardtanh
    elif act == 'elu':
        act_return = F.elu
    elif act == 'relu6':
        act_return = F.relu6
    elif act == 'leaky_relu':
        act_return = F.leaky_relu
    elif act == 'prelu':
        act_return = F.prelu
    elif act == 'rrelu':
        act_return = F.rrelu
    elif act == 'logsigmoid':
        act_return = F.logsigmoid
    elif act == 'hardshrink':
        act_return = F.hardshrink
    elif act == 'tanhshrink':
        act_return = F.tanhshrink
    elif act == 'softsign':
        act_return = F.softsign
    elif act == 'softplus':
        act_return = F.softplus
    elif act == 'softmin':
        act_return = F.softmin
    elif act == 'softmax':
        act_return = F.softmax
    elif act == 'log_softmax':
        act_return = F.log_softmax
    elif act == 'softshrink':
        act_return = F.softshrink
    elif act == 'sin':
        act_return = torch.sin
    elif act == 'identity':
        act_return = nn.Identity()
    else:
        print('Wrong act name:',act)