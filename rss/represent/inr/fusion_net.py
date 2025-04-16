import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange

class FusionLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        grid_size=5,
        spline_order=3,
        w0=1.0,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        use_bias=True,
        drop_out=False
    ):
        super().__init__()
        self.in_features = dim_in
        self.out_features = dim_out
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.w0 = w0
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.drop_out = drop_out

        # 初始化网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(dim_in, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 初始化权重
        self.base_weight = nn.Parameter(torch.Tensor(dim_out, dim_in))
        self.spline_weight = nn.Parameter(
            torch.Tensor(dim_out, dim_in, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(dim_out, dim_in)
            )
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(dim_out))
        else:
            self.register_parameter('bias', None)

        if drop_out:
            self.dropout = nn.Dropout(p=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化基础权重
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        # 初始化样条权重
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
            
            if self.bias is not None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        return bases.contiguous()

    def curve2coeff(self, x, y):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 基础变换
        base_output = F.linear(self.base_activation(x), self.base_weight, self.bias)
        
        # 样条变换
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # 正弦激活
        out = torch.sin(self.w0 * (base_output + spline_output))
        
        if self.drop_out:
            out = self.dropout(out)
            
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.input_dim = input_dim
        self.gaussian_pe = gaussian_pe
        self.gaussian_variance = gaussian_variance

        if self.gaussian_pe:
            # 初始化高斯位置编码
            self.register_buffer(
                'gaussian_matrix',
                torch.randn(input_dim, num_encoding_functions) * gaussian_variance
            )
        else:
            # 初始化标准位置编码
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions
                )
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions
                )

    def forward(self, x):
        if self.gaussian_pe:
            # 应用高斯位置编码
            x_proj = x @ self.gaussian_matrix
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        else:
            # 应用标准位置编码
            x_proj = x.unsqueeze(-1) * self.frequency_bands
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FusionNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        grid_size=5,
        spline_order=3,
        w0=1.0,
        w0_initial=30.0,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        use_bias=True,
        drop_out=False,
        num_encoding_functions=6,
        gaussian_pe=False,
        gaussian_variance=38,
        asi_if=False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if

        # 位置编码
        self.positional_encoding = PositionalEncoding(
            num_encoding_functions=num_encoding_functions,
            input_dim=dim_in,
            gaussian_pe=gaussian_pe,
            gaussian_variance=gaussian_variance
        )

        # 计算编码后的维度
        encoded_dim = dim_in * (2 * num_encoding_functions + 1) if not gaussian_pe else dim_in * (2 * num_encoding_functions)

        # 创建网络层
        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = encoded_dim if is_first else dim_hidden

            self.layers.append(FusionLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                grid_size=grid_size,
                spline_order=spline_order,
                w0=layer_w0,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                use_bias=use_bias,
                drop_out=drop_out
            ))

        # 输出层
        self.last_layer = FusionLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            grid_size=grid_size,
            spline_order=spline_order,
            w0=w0,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            use_bias=use_bias,
            drop_out=drop_out
        )

        if self.asi_if:
            self.last_layer_asi = FusionLayer(
                dim_in=dim_hidden,
                dim_out=dim_out,
                grid_size=grid_size,
                spline_order=spline_order,
                w0=w0,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                use_bias=use_bias,
                drop_out=drop_out
            )
            self.last_layer_asi.load_state_dict(self.last_layer.state_dict())

    def forward(self, x):
        # 应用位置编码
        x = self.positional_encoding(x)

        # 前向传播
        for layer in self.layers:
            x = layer(x)

        if self.asi_if:
            return (self.last_layer(x) - self.last_layer_asi(x)) * 1.4142135623730951/2
        else:
            return self.last_layer(x)

def FUSION_NET(parameter):
    # 默认参数字典
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'grid_size': 5,
        'spline_order': 3,
        'w0': 1.0,
        'w0_initial': 30.0,
        'scale_noise': 0.1,
        'scale_base': 1.0,
        'scale_spline': 1.0,
        'enable_standalone_scale_spline': True,
        'base_activation': nn.SiLU,
        'grid_eps': 0.02,
        'grid_range': [-1, 1],
        'use_bias': True,
        'drop_out': False,
        'num_encoding_functions': 6,
        'gaussian_pe': False,
        'gaussian_variance': 38,
        'asi_if': False
    }
    
    # 更新参数
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
    
    # 创建并返回 FusionNet 对象
    return FusionNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        grid_size=parameter['grid_size'],
        spline_order=parameter['spline_order'],
        w0=parameter['w0'],
        w0_initial=parameter['w0_initial'],
        scale_noise=parameter['scale_noise'],
        scale_base=parameter['scale_base'],
        scale_spline=parameter['scale_spline'],
        enable_standalone_scale_spline=parameter['enable_standalone_scale_spline'],
        base_activation=parameter['base_activation'],
        grid_eps=parameter['grid_eps'],
        grid_range=parameter['grid_range'],
        use_bias=parameter['use_bias'],
        drop_out=parameter['drop_out'],
        num_encoding_functions=parameter['num_encoding_functions'],
        gaussian_pe=parameter['gaussian_pe'],
        gaussian_variance=parameter['gaussian_variance'],
        asi_if=parameter['asi_if']
    ) 