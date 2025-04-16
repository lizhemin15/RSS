import torch
import torch.nn as nn
import numpy as np
import math

class ChebyFinerLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=5, bias=True, omega_0=30, scale_req_grad=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.omega_0 = omega_0
        self.scale_req_grad = scale_req_grad
        
        # 线性变换
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # 切比雪夫系数
        self.cheby_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (in_features * (degree + 1)))
        
        # 注册缓冲区
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                       np.sqrt(6 / self.in_features) / self.omega_0)
    
    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
    
    def chebyshev_transform(self, x):
        # 归一化到[-1, 1]区间
        x = torch.tanh(x)
        # 扩展输入维度
        x = x.view((-1, self.in_features, 1)).expand(-1, -1, self.degree + 1)
        # 应用acos
        x = x.acos()
        # 乘以arange [0 .. degree]
        x *= self.arange
        # 应用cos
        x = x.cos()
        return x
    
    def forward(self, input):
        # 线性变换
        x = self.linear(input)
        
        # 生成scale
        scale = self.generate_scale(x)
        
        # 切比雪夫变换
        cheby_x = self.chebyshev_transform(input)
        
        # 计算切比雪夫插值
        cheby_out = torch.einsum("bid,oid->bo", cheby_x, self.cheby_coeffs)
        
        # 应用Finer激活函数
        out = torch.sin(self.omega_0 * scale * x) + cheby_out
        
        return out

class ChebyFiner(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 degree=5, omega_0=30, bias=True, scale_req_grad=False, asi_if=False):
        super().__init__()
        self.net = []
        
        # 第一层
        self.net.append(ChebyFinerLayer(in_features, hidden_features, degree=degree, 
                                       omega_0=omega_0, scale_req_grad=scale_req_grad))
        
        # 隐藏层
        for i in range(hidden_layers):
            self.net.append(ChebyFinerLayer(hidden_features, hidden_features, degree=degree, 
                                           omega_0=omega_0, scale_req_grad=scale_req_grad))
        
        # 最后一层
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0,
                                         np.sqrt(6 / hidden_features) / omega_0)
        
        self.net = nn.Sequential(*self.net)
        self.last_layer = final_linear
        self.asi_if = asi_if
        
        if self.asi_if:
            self.last_layer_asi = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                self.last_layer_asi.weight.copy_(self.last_layer.weight)
                if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                    self.last_layer_asi.bias.copy_(self.last_layer.bias)
    
    def forward(self, coords):
        output = self.net(coords)
        if self.asi_if:
            return (self.last_layer(output) - self.last_layer_asi(output)) * 1.4142135623730951 / 2
        else:
            return self.last_layer(output)

def CHEBYFINER(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256, 
        'dim_out': 1,
        'num_layers': 4,
        'degree': 5,
        'omega_0': 30,
        'scale_req_grad': False,
        'bias': True,
        'asi_if': False
    }  
    
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
        
    return ChebyFiner(
        in_features=parameter['dim_in'],    
        hidden_features=parameter['dim_hidden'],
        hidden_layers=parameter['num_layers'],
        out_features=parameter['dim_out'],
        degree=parameter['degree'],
        omega_0=parameter['omega_0'],
        bias=parameter['bias'], 
        scale_req_grad=parameter['scale_req_grad'],
        asi_if=parameter['asi_if']
    ) 