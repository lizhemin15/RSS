import torch.nn.functional as F
import torch as t
from torch import nn
from einops import rearrange

abc_str = 'abcdefghijklmnopqrstuvwxyz'


def get_act(act,coef_cos=1,coef_exp=1):
    act_dict = {
        'relu': F.relu,
        'sigmoid': F.sigmoid,
        'tanh': t.tanh,
        'softmax': F.softmax,
        'threshold': F.threshold,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'relu6': F.relu6,
        'leaky_relu': F.leaky_relu,
        'prelu': F.prelu,
        'rrelu': F.rrelu,
        'logsigmoid': F.logsigmoid,
        'hardshrink': F.hardshrink,
        'tanhshrink': F.tanhshrink,
        'softsign': F.softsign,
        'softplus': F.softplus,
        'softmin': F.softmin,
        'log_softmax': F.log_softmax,
        'softshrink': F.softshrink,
        'sin': t.sin,
        'identity': nn.Identity(),
        'exp': t.exp,
        'gabor': lambda x: t.cos(x*coef_cos) * t.exp(x*coef_exp)
    }
    
    if act in act_dict:
        return act_dict[act]
    else:
        print('Wrong act name:', act)
        return None



def get_opt(opt_type='Adam', parameters=None, lr=1e-3, weight_decay=0):
    # 初始化网络参数的优化器

    optimizer_dict = {
        'Adadelta': t.optim.Adadelta,
        'Adagrad': t.optim.Adagrad,
        'Adam': t.optim.Adam,
        'RegAdam': t.optim.Adam,
        'AdamW': t.optim.AdamW,
        'SparseAdam': t.optim.SparseAdam,
        'Adamax': t.optim.Adamax,
        'ASGD': t.optim.ASGD,
        'LBFGS': t.optim.LBFGS,
        'SGD': t.optim.SGD,
        'NAdam': t.optim.NAdam,
        'RAdam': t.optim.RAdam,
        'RMSprop': t.optim.RMSprop,
        'Rprop': t.optim.Rprop,
    }

    if opt_type == 'Lion':
        from lion_pytorch import Lion
        optimizer_dict['Lion'] = Lion

    if opt_type in optimizer_dict:
        optimizer_class = optimizer_dict[opt_type]
        optimizer = optimizer_class(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Wrong optimization type')
    
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


def patchify(data, patch_size=32, stride=1):
    patches = data.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    # 将patches展平，并移动维度顺序为 (num_patches, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches