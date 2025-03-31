from torch import nn
import torch

## Gauss
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.exp(-(self.scale * self.linear(input))**2)
    
class Gauss(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, scale=30.0):
        super().__init__()
        self.nonlin = GaussLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, scale=scale))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output
    
def GAUSS(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 100, 
        'dim_out': 1,
        'num_layers': 4,
        'w0_initial': 30.0
    }
    
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
        
    return Gauss(
        in_features=parameter['dim_in'],
        hidden_features=parameter['dim_hidden'],
        hidden_layers=parameter['num_layers'],
        out_features=parameter['dim_out'],
        scale=parameter['w0_initial']
    )

    