import torch.nn.functional as F
import torch as t
from torch import nn
from einops import rearrange

abc_str = 'abcdefghijklmnopqrstuvwxyz'


def get_act(act):
    if act == 'relu':
        act_return = F.relu
    elif act == 'sigmoid':
        act_return = F.sigmoid
    elif act == 'tanh':
        act_return = t.tanh
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
        act_return = t.sin
    elif act == 'identity':
        act_return = nn.Identity()
    else:
        print('Wrong act name:',act)
    return act_return



def get_opt(opt_type='Adam',parameters=None,lr=1e-3,weight_decay=0):
    # Initial the optimizer of parameters in network
    if opt_type == 'Adadelta':
        optimizer = t.optim.Adadelta(parameters,lr=lr)
    elif opt_type == 'Adagrad':
        optimizer = t.optim.Adagrad(parameters,lr=lr)
    elif opt_type == 'Adam':
        optimizer = t.optim.Adam(parameters,lr=lr,weight_decay=weight_decay)
    elif opt_type == 'RegAdam':
        optimizer = t.optim.Adam(parameters,lr=lr, weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        optimizer = t.optim.AdamW(parameters,lr=lr)
    elif opt_type == 'SparseAdam':
        optimizer = t.optim.SparseAdam(parameters,lr=lr)
    elif opt_type == 'Adamax':
        optimizer = t.optim.Adamax(parameters,lr=lr)
    elif opt_type == 'ASGD':
        optimizer = t.optim.ASGD(parameters,lr=lr)
    elif opt_type == 'LBFGS':
        optimizer = t.optim.LBFGS(parameters,lr=lr)
    elif opt_type == 'SGD':
        optimizer = t.optim.SGD(parameters,lr=lr)
    elif opt_type == 'NAdam':
        optimizer = t.optim.NAdam(parameters,lr=lr)
    elif opt_type == 'RAdam':
        optimizer = t.optim.RAdam(parameters,lr=lr)
    elif opt_type == 'RMSprop':
        optimizer = t.optim.RMSprop(parameters,lr=lr)
    elif opt_type == 'Rprop':
        optimizer = t.optim.Rprop(parameters,lr=lr)
    elif opt_type == 'Lion':
        from lion_pytorch import Lion
        optimizer = Lion(parameters, lr = lr)
    else:
        raise('Wrong optimization type')
    return optimizer


def to_device(obj,device):
    if t.cuda.is_available() and device != 'cpu':
        obj = obj.cuda(device)
    return obj

def reshape2(data):
    # 
    xshape = data.shape
    einstr = add_space(abc_str[:len(xshape)])+' -> ('+add_space(abc_str[:len(xshape)])+') ()'
    return rearrange(data,einstr)

def add_space(oristr):
    addstr = ''
    for i in range(len(oristr)):
        addstr += oristr[i]
        addstr += ' '
    return addstr