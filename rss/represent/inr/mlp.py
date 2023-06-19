import torch
from torch import nn
import torch.nn.functional as F
from rss.represent.utils import get_act

valid_act_list = ['sigmoid','tanh','relu','leaky_relu','selu']

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

def cal_gain(act):
    if act in valid_act_list:
        return nn.init.calculate_gain(act)
    else:
        return 1.0

class Layer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bias = True, activation = 'tanh', drop_out=False,init_mode=None,monoto_mode=0):
        super().__init__()
        self.dim_in = dim_in
        self.activation_name = activation
        self.init_mode = init_mode
        self.monoto_mode = monoto_mode
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.get_act()
        if drop_out:
            self.dropout = nn.Dropout(p=0.1)
        self.drop_if = drop_out


    def init_(self, weight, bias):
        dim = self.dim_in
        w_std = 1 / dim
        if self.init_mode == 'xavier_uniform':
            weight = nn.init.xavier_uniform_(weight,gain=cal_gain(self.activation_name))
        elif self.init_mode == None or self.init_mode == 'xavier_normal':
            weight = nn.init.xavier_normal_(weight,gain=cal_gain(self.activation_name))
        elif self.init_mode == 'kaiming_uniform':
            act = self.activation_name if self.activation_name in valid_act_list else 'relu'
            weight = nn.init.kaiming_uniform_(weight,nonlinearity=act)
        elif self.init_mode == 'kaiming_normal':
            act = self.activation_name if self.activation_name in valid_act_list else 'relu'
            weight = nn.init.kaiming_normal_(weight,nonlinearity=act)
        else:
            raise('Do not support init mode = ',self.init_mode)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        if self.monoto_mode == 0:
            out =  F.linear(x, self.weight, self.bias)
        elif self.monoto_mode == 1:
            out =  F.linear(x, torch.abs(self.weight), self.bias)
        elif self.monoto_mode == -1:
            out =  F.linear(x, -torch.abs(self.weight), self.bias)
        else:
            raise('Wrong monoto_mode = ',self.monoto_mode)
        if self.drop_if:
            out = self.dropout(out)
        out = self.act(out)
        return out

    def get_act(self):
        act = self.activation_name
        self.act = get_act(act)



class INR(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, use_bias = True,
                 final_activation = None, drop_out = [0],activation = 'tanh', init_mode = None,monoto_mode=0):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Layer(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                use_bias = use_bias,
                activation = activation,
                drop_out = 0,
                init_mode=init_mode,
                monoto_mode=monoto_mode
            ))

        final_activation = 'identity' if not exists(final_activation) else final_activation
        self.last_layer = Layer(dim_in = dim_hidden, dim_out = dim_out, use_bias = use_bias,
                                 activation = final_activation, drop_out=drop_out[-1],monoto_mode=monoto_mode)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)



def MLP(parameter):
    de_para_dict = {'dim_in':2,'dim_hidden':100,'dim_out':1,'num_layers':4,'activation':'tanh'}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    # print('MLP : ',parameter)
    return INR(dim_in=parameter['dim_in'], dim_hidden=parameter['dim_hidden'], dim_out=parameter['dim_out'], 
               num_layers=parameter['num_layers'], activation = parameter['activation'])


