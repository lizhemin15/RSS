import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, drop_out=False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        if drop_out:
            self.dropout = nn.Dropout(p=0.1)
        self.drop_if = drop_out


    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        if self.drop_if:
            out = self.dropout(out)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, drop_out=[0,0,0,0,0], asi_if = False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                drop_out = 0
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, drop_out=drop_out[-1])
        self.asi_if = asi_if
        if self.asi_if:
            self.last_layer_asi = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, drop_out=drop_out[-1])
            self.last_layer_asi.weight.data.copy_(self.last_layer.weight.data)
            if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                self.last_layer_asi.bias.data.copy_(self.last_layer.bias.data)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')
        if self.asi_if:
            return (self.last_layer(x)- self.last_layer_asi(x))*1.4142135623730951/2
        else:
            return self.last_layer(x)



def SIREN(parameter):
    de_para_dict = {'dim_in':2,'dim_hidden':100,'dim_out':1,'num_layers':4,'w0':1,'w0_initial':30.,'use_bias':True, 'final_activation':None, 'asi_if':False}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    # print('SIREN : ',parameter)
    return SirenNet(parameter['dim_in'], parameter['dim_hidden'], parameter['dim_out'], parameter['num_layers'],
                    parameter['w0'], parameter['w0_initial'], parameter['use_bias'], parameter['final_activation'], asi_if=parameter['asi_if'])





# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)






# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out
