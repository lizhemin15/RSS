import numpy as np
import torch
from torch import nn

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
        
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True, asi_if=False):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        # self.net.append(final_linear)
        self.last_layer = final_linear
        self.asi_if = asi_if
        if self.asi_if:
            self.last_layer_asi = nn.Linear(hidden_features, out_features,
                                 dtype=dtype)
            with torch.no_grad():
                self.last_layer_asi.weight.copy_(self.last_layer.weight)
                if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                    self.last_layer_asi.bias.copy_(self.last_layer.bias)

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.asi_if:
            output = self.net(coords)
            output = (self.last_layer(output)- self.last_layer_asi(output))*1.4142135623730951/2
        else:
            output = self.net(coords)
            output = self.last_layer(output)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output
    

def WIRE(parameter):
    # 默认参数字典
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 100,
        'num_layers': 4,
        'dim_out': 1,
        'w0_initial': 10,
        'w0': 10.0,
        'scale': 10.0,
        'pos_encode': False,
        'sidelength': 512,
        'fn_samples': None,
        'use_nyquist': True,
        'asi_if': False
    }
    
    # 更新参数
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now

    # 创建 INR 实例
    return INR(parameter['dim_in'], 
               parameter['dim_hidden'], 
               parameter['num_layers'], 
               parameter['dim_out'], 
               first_omega_0=parameter['w0_initial'], 
               hidden_omega_0=parameter['w0'], 
               scale=parameter['scale'], 
               pos_encode=parameter['pos_encode'], 
               sidelength=parameter['sidelength'], 
               fn_samples=parameter['fn_samples'], 
               use_nyquist=parameter['use_nyquist'],
               asi_if=parameter['asi_if'])



