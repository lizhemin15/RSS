import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class HybridFreqLayer(nn.Module):
    """基于FINER的混合频率层，结合低频和高频分支"""
    def __init__(self, in_features, out_features, bias=True, 
                 low_freq_w0=1.0, high_freq_w0=30.0, 
                 fusion_type='attention', is_first=False,
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.fusion_type = fusion_type
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad = scale_req_grad
        
        # 低频分支
        self.low_freq_linear = nn.Linear(in_features, out_features, bias=bias)
        self.low_freq_w0 = low_freq_w0
        
        # 高频分支
        self.high_freq_linear = nn.Linear(in_features, out_features, bias=bias)
        self.high_freq_w0 = high_freq_w0
        
        # 融合机制
        if fusion_type == 'attention':
            self.attention = nn.Linear(out_features * 2, 2)
        elif fusion_type == 'gate':
            self.gate = nn.Linear(out_features * 2, out_features)
            self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self.init_weights()
        if self.first_bias_scale is not None and self.is_first:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 第一层使用较小的初始化范围
                self.low_freq_linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
                self.high_freq_linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # 后续层使用SIREN的初始化方法
                self.low_freq_linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.low_freq_w0,
                                                    np.sqrt(6 / self.in_features) / self.low_freq_w0)
                self.high_freq_linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.high_freq_w0,
                                                     np.sqrt(6 / self.in_features) / self.high_freq_w0)
    
    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.low_freq_linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                self.high_freq_linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
    
    def generate_scale(self, x):
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
    
    def forward(self, x):
        # 低频分支
        low_freq = self.low_freq_linear(x)
        low_freq_scale = self.generate_scale(low_freq)
        low_freq_act = torch.sin(self.low_freq_w0 * low_freq_scale * low_freq)
        
        # 高频分支
        high_freq = self.high_freq_linear(x)
        high_freq_scale = self.generate_scale(high_freq)
        high_freq_act = torch.sin(self.high_freq_w0 * high_freq_scale * high_freq)
        
        # 融合两个分支
        if self.fusion_type == 'attention':
            # 注意力融合
            concat_features = torch.cat([low_freq_act, high_freq_act], dim=-1)
            attention_weights = F.softmax(self.attention(concat_features), dim=-1)
            out = attention_weights[:, 0:1] * low_freq_act + attention_weights[:, 1:2] * high_freq_act
        elif self.fusion_type == 'gate':
            # 门控融合
            concat_features = torch.cat([low_freq_act, high_freq_act], dim=-1)
            gate = self.sigmoid(self.gate(concat_features))
            out = gate * low_freq_act + (1 - gate) * high_freq_act
        else:
            # 简单平均融合
            out = 0.5 * low_freq_act + 0.5 * high_freq_act
        
        return out

class HybridFreqNet(nn.Module):
    """基于FINER的混合频率网络，结合低频和高频优势"""
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, 
                 low_freq_w0=1.0, high_freq_w0=30.0, 
                 fusion_type='attention', use_bias=True, 
                 first_bias_scale=None, scale_req_grad=False,
                 final_activation=None, asi_if=False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if
        
        # 创建混合频率层
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            is_first = i == 0
            self.layers.append(HybridFreqLayer(
                in_features=dim_in if is_first else dim_hidden,
                out_features=dim_hidden,
                low_freq_w0=low_freq_w0,
                high_freq_w0=high_freq_w0,
                fusion_type=fusion_type,
                is_first=is_first,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad
            ))
        
        # 最终输出层
        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = nn.Linear(dim_hidden, dim_out)
        
        # 抗对称初始化
        if self.asi_if:
            self.last_layer_asi = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                self.last_layer_asi.weight.copy_(self.last_layer.weight)
                if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                    self.last_layer_asi.bias.copy_(self.last_layer.bias)
    
    def forward(self, x):
        # 前向传播
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        if self.asi_if:
            return (self.last_layer(x) - self.last_layer_asi(x)) * 1.4142135623730951 / 2
        else:
            return self.last_layer(x)

def HYBRID_FREQ(parameter):
    """创建基于FINER的混合频率网络的工厂函数"""
    # 默认参数
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'low_freq_w0': 1.0,
        'high_freq_w0': 30.0,
        'fusion_type': 'attention',
        'use_bias': True,
        'first_bias_scale': None,
        'scale_req_grad': False,
        'final_activation': None,
        'asi_if': False
    }
    
    # 更新参数
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
    
    return HybridFreqNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        low_freq_w0=parameter['low_freq_w0'],
        high_freq_w0=parameter['high_freq_w0'],
        fusion_type=parameter['fusion_type'],
        use_bias=parameter['use_bias'],
        first_bias_scale=parameter['first_bias_scale'],
        scale_req_grad=parameter['scale_req_grad'],
        final_activation=parameter['final_activation'],
        asi_if=parameter['asi_if']
    ) 