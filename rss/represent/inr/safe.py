import torch
from torch import nn
import torch.nn.functional as F
from rss.represent.utils import get_act

class SafeLayer(nn.Module):
    def __init__(self, dim_in, dim_out, degree=3, use_bias=True, drop_out=False, init_mode=None, monoto_mode=0):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.degree = degree
        self.init_mode = init_mode
        self.monoto_mode = monoto_mode
        
        # 线性层参数
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        
        # 切比雪夫多项式系数
        self.cheby_coeffs = nn.Parameter(torch.empty(dim_out, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (dim_out * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
        
        # Dropout
        if drop_out:
            self.dropout = nn.Dropout(p=0.1)
        self.drop_if = drop_out

    def init_(self, weight, bias):
        dim = self.dim_in
        w_std = 1 / dim
        if self.init_mode == 'xavier_uniform':
            weight = nn.init.xavier_uniform_(weight)
        elif self.init_mode == None or self.init_mode == 'xavier_normal':
            weight = nn.init.xavier_normal_(weight)
        elif self.init_mode == 'kaiming_uniform':
            weight = nn.init.kaiming_uniform_(weight, nonlinearity='relu')
        elif self.init_mode == 'kaiming_normal':
            weight = nn.init.kaiming_normal_(weight, nonlinearity='relu')
        else:
            raise('Do not support init mode = ', self.init_mode)
        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        # 线性变换
        if self.monoto_mode == 0:
            out = F.linear(x, self.weight, self.bias)
        elif self.monoto_mode == 1:
            out = F.linear(x, torch.abs(self.weight), self.bias)
        elif self.monoto_mode == -1:
            out = F.linear(x, -torch.abs(self.weight), self.bias)
        else:
            raise('Wrong monoto_mode = ', self.monoto_mode)
            
        # Dropout
        if self.drop_if:
            out = self.dropout(out)
            
        # 切比雪夫多项式激活
        # 归一化到[-1,1]
        out = torch.tanh(out)
        # 扩展维度以计算多项式
        out = out.view((-1, self.dim_out, 1)).expand(-1, -1, self.degree + 1)
        # 计算切比雪夫多项式
        out = out.acos()
        out *= self.arange
        out = out.cos()
        # 组合多项式
        out = torch.einsum('bod,od->bo', out, self.cheby_coeffs)
        
        return out

class SafeINR(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, degree=3, use_bias=True,
                 drop_out=[0], init_mode=None, monoto_mode=0, asi_if=False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if
        self.layers = nn.ModuleList([])
        
        # 隐藏层
        for ind in range(num_layers):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else dim_hidden
            self.layers.append(SafeLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                degree=degree,
                use_bias=use_bias,
                drop_out=0,
                init_mode=init_mode,
                monoto_mode=monoto_mode
            ))
            
        # 输出层
        self.last_layer = SafeLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            degree=degree,
            use_bias=use_bias,
            drop_out=drop_out[-1],
            init_mode=init_mode,
            monoto_mode=monoto_mode
        )
        
        # ASI模式
        if self.asi_if:
            self.last_layer_asi = SafeLayer(
                dim_in=dim_hidden,
                dim_out=dim_out,
                degree=degree,
                use_bias=use_bias,
                drop_out=drop_out[-1],
                init_mode=init_mode,
                monoto_mode=monoto_mode
            )
            self.last_layer_asi.weight.data.copy_(self.last_layer.weight.data)
            if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                self.last_layer_asi.bias.data.copy_(self.last_layer.bias.data)
            self.last_layer_asi.cheby_coeffs.data.copy_(self.last_layer.cheby_coeffs.data)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        if self.asi_if:
            return (self.last_layer(x) - self.last_layer_asi(x)) * 1.4142135623730951/2
        else:
            return self.last_layer(x)

def SAFE(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 100,
        'dim_out': 1,
        'num_layers': 4,
        'degree': 3,
        'asi_if': False
    }
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
    return SafeINR(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        degree=parameter['degree'],
        asi_if=parameter['asi_if']
    )
