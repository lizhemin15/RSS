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
    def __init__(self, dim_in, dim_out, use_bias = True, activation = 'relu', drop_out=False,init_mode=None,monoto_mode=0):
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
                 final_activation = None, drop_out = [0],activation = 'relu', init_mode = None,monoto_mode=0,asi_if=False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if
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
        if self.asi_if:
            self.last_layer_asi = Layer(dim_in = dim_hidden, dim_out = dim_out, use_bias = use_bias,
                                 activation = final_activation, drop_out=drop_out[-1],monoto_mode=monoto_mode)
            self.last_layer_asi.weight.data.copy_(self.last_layer.weight.data)
            if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                self.last_layer_asi.bias.data.copy_(self.last_layer.bias.data)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.asi_if:
            return (self.last_layer(x)- self.last_layer_asi(x))*1.4142135623730951/2
        else:
            return self.last_layer(x)





class GaussianSplatting(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, activation='gabor', coef_cos=1, coef_exp=1):
        super().__init__()
        # Hidden layer parameters initialization
        self.mean = nn.Parameter(torch.zeros(dim_hidden, dim_in, 1))  # mean is now a parameter

        self.W1 = nn.Parameter(torch.randn(dim_hidden, dim_in))  # W1 is now a parameter
        self.b1 = nn.Parameter(torch.randn(1, dim_in, dim_hidden))    # b1 is now a parameter
        
        # Output layer parameters initialization
        self.W2 = nn.Parameter(torch.randn(dim_out, dim_hidden))  # W2 is now a parameter
        self.act = get_act(activation, coef_cos=coef_cos, coef_exp=coef_exp)

    def forward(self, x):
        # Hidden layer forward pass
        # Input shape: (batch_size, input_dim)
        # W1 shape: (input_dim, hidden_dim), b1 shape: (1, dim_in, hidden_dim)
        hidden = F.linear(x+self.b1, self.W1, None)
        hidden = self.act(hidden)

        # Output layer forward pass
        # hidden shape: (batch_size, hidden_dim), W2 shape: (hidden_dim, output_dim)
        output = torch.matmul(hidden, self.W2)

        return output



def MLP(parameter):
    de_para_dict = {'dim_in':2,'dim_hidden':100,'dim_out':1,'num_layers':4,'activation':'relu','asi_if':False}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    # print('MLP : ',parameter)
    return INR(dim_in=parameter['dim_in'], dim_hidden=parameter['dim_hidden'], dim_out=parameter['dim_out'], 
               num_layers=parameter['num_layers'], activation = parameter['activation'], asi_if = parameter['asi_if'])

def Splatting(parameter):
    pass

