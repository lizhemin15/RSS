import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helpers
def exists(val):
    return val is not None

# Adaptive Polynomial Wave Activation (APW)
class APWActivation(nn.Module):
    """
    Implements the activation: x + (1 + c * x^2) * sin(omega * x)
    omega and c are learnable parameters.
    """
    def __init__(self, dim_out, initial_omega=30.0, initial_c=0.0):
        super().__init__()
        # Learnable parameters, initialized to reasonable values.
        # Shape (1, dim_out) allows per-neuron parameters if desired via broadcasting.
        self.omega = nn.Parameter(torch.full((1, dim_out), float(initial_omega)))
        # Initialize c close to zero, so initially it behaves like x + sin(omega*x)
        self.c = nn.Parameter(torch.full((1, dim_out), float(initial_c)))

    def forward(self, x):
        # x shape: (batch_size, dim_out)
        # omega, c shape: (1, dim_out)

        # Calculate amplitude scale factor: 1 + c * x^2
        # Ensure c >= 0? Or allow negative c? Let's allow negative for now.
        # Using x.square() is generally preferred over x**2
        amp_scale = 1.0 + self.c * x.square()

        # Calculate the sinusoidal term
        sin_term = torch.sin(self.omega * x)

        # Combine terms: x + scaled_sin
        output = x + amp_scale * sin_term

        # Optional: Check for NaNs/Infs, though less likely than with exp()
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaNs or Infs detected in APWActivation")
            output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)

        return output

# APW Layer: Linear + APWActivation
class APWLayer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bias=True, is_first=False,
                 initial_omega=30.0, initial_c=0.0,
                 linear_w_std_factor=1.0): # Using a simpler factor for std dev
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = APWActivation(dim_out, initial_omega, initial_c)

        # Initialization for the linear layer (Simplified SIREN-like)
        with torch.no_grad():
            if self.is_first:
                # SIREN first layer: uniform(-1/dim_in, 1/dim_in)
                w_std = (1.0 / dim_in) * linear_w_std_factor
            else:
                # SIREN hidden layers: uniform(-sqrt(6/dim_in)/omega0, sqrt(6/dim_in)/omega0)
                # Since omega is learnable, let's use the initial_omega for init.
                # Using c=6 from SIREN paper.
                w_std = (math.sqrt(6.0 / dim_in) / initial_omega) * linear_w_std_factor

            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias and self.linear.bias is not None:
                # Initialize bias similarly or maybe just zero?
                # SIREN uses the same uniform init for bias.
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        linear_out = self.linear(x)
        return self.activation(linear_out)

# APW Network
class APWNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 use_bias=True, final_activation=None, asi_if=False,
                 # APW Layer specific params
                 w0_initial=30.0, w0_hidden=30.0,
                 c_initial=0.0, c_hidden=0.0, # Separate init for c?
                 # Linear layer init params
                 linear_w_std_factor=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if

        self.layers = nn.ModuleList()

        for ind in range(num_layers):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else dim_hidden
            current_omega = w0_initial if is_first else w0_hidden
            current_c = c_initial if is_first else c_hidden # Allow different c init for first layer?

            self.layers.append(APWLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                is_first=is_first,
                initial_omega=current_omega,
                initial_c=current_c,
                linear_w_std_factor=linear_w_std_factor
            ))

        # Final linear layer
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

        # Initialize final linear layer (Simplified SIREN-like)
        with torch.no_grad():
             # Use the hidden layer's omega for initialization scale
             final_w_std = (math.sqrt(6.0 / dim_hidden) / w0_hidden) * linear_w_std_factor
             nn.init.uniform_(self.last_layer.weight, -final_w_std, final_w_std)
             if use_bias and self.last_layer.bias is not None:
                  nn.init.uniform_(self.last_layer.bias, -final_w_std, final_w_std)

        # Optional ASI logic
        if self.asi_if:
            self.last_layer_asi = nn.Linear(dim_hidden, dim_out, bias=use_bias)
            with torch.no_grad():
                self.last_layer_asi.weight.data.copy_(self.last_layer.weight.data)
                if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                     self.last_layer_asi.bias.data.copy_(self.last_layer.bias.data)

        # Store final activation
        self.final_activation = nn.Identity() if not exists(final_activation) else final_activation

    def forward(self, x, mods=None):
        for layer in self.layers:
            x = layer(x)

        output = self.last_layer(x)
        output = self.final_activation(output)

        if self.asi_if:
             output_asi = self.last_layer_asi(x)
             output_asi = self.final_activation(output_asi)
             output = (output - output_asi) * (math.sqrt(2.) / 2.)

        return output

# Factory Function
def APW(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'use_bias': True,
        'final_activation': None,
        'asi_if': False,
        # APW specific
        'w0_initial': 30.0, # Omega for first layer
        'w0_hidden': 30.0,  # Omega for hidden layers
        'c_initial': 0.01,  # Modulation factor c for first layer (start small)
        'c_hidden': 0.01,   # Modulation factor c for hidden layers (start small)
        # Linear init specific
        'linear_w_std_factor': 1.0 # Multiplier for SIREN-like weight init std dev
    }

    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now

    return APWNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        use_bias=parameter['use_bias'],
        final_activation=parameter['final_activation'],
        asi_if=parameter['asi_if'],
        w0_initial=parameter['w0_initial'],
        w0_hidden=parameter['w0_hidden'],
        c_initial=parameter['c_initial'],
        c_hidden=parameter['c_hidden'],
        linear_w_std_factor=parameter['linear_w_std_factor']
    ) 