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
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, scale=30.0, asi_if=False):
        super().__init__()
        self.nonlin = GaussLayer
        self.asi_if = asi_if
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, scale=scale))

        final_linear = nn.Linear(hidden_features, out_features)                
        # self.net.append(final_linear)
        self.last_layer = final_linear
        self.asi_if = asi_if
        if self.asi_if:
            self.last_layer_asi = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                self.last_layer_asi.weight.copy_(self.last_layer.weight)
                if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                    self.last_layer_asi.bias.copy_(self.last_layer.bias)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        if self.asi_if:
            output = (self.last_layer(output)- self.last_layer_asi(output))*1.4142135623730951/2
        else:
            output = self.last_layer(output)
        return output
    
def GAUSS(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256, 
        'dim_out': 1,
        'num_layers': 4,
        'w0_initial': 10.0,
        'asi_if': False
    }
    
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
        
    return Gauss(
        in_features=parameter['dim_in'],
        hidden_features=parameter['dim_hidden'],
        hidden_layers=parameter['num_layers'],
        out_features=parameter['dim_out'],
        scale=parameter['w0_initial'],
        asi_if=parameter['asi_if']
    )

    