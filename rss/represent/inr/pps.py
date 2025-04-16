import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helpers
def exists(val):
    return val is not None

# Polynomial Phase Sine Activation (PPS)
class PPSActivation(nn.Module):
    """
    Implements the activation: sin(omega1 * x + omega2 * x^2)
    omega1 and omega2 are learnable parameters.
    """
    def __init__(self, dim_out, initial_omega1=30.0, initial_omega2=0.0):
        super().__init__()
        # Learnable phase parameters.
        # omega1 controls the linear phase component (frequency at origin).
        # omega2 controls the quadratic phase component (frequency change rate).
        self.omega1 = nn.Parameter(torch.full((1, dim_out), float(initial_omega1)))
        # Initialize omega2 to zero or small value to start close to SIREN.
        self.omega2 = nn.Parameter(torch.full((1, dim_out), float(initial_omega2)))

    def forward(self, x):
        # x shape: (batch_size, dim_out)
        # omega1, omega2 shape: (1, dim_out)

        # Calculate the phase: omega1 * x + omega2 * x^2
        phase = self.omega1 * x + self.omega2 * x.square()

        # Calculate final activation
        output = torch.sin(phase)

        # Optional: Check for NaNs/Infs
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaNs or Infs detected in PPSActivation")
            output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)

        return output

# PPS Layer: Linear + PPSActivation
class PPSLayer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bias=True, is_first=False,
                 initial_omega1=30.0, initial_omega2=0.0,
                 linear_w_std_factor=1.0): # Factor for linear weight initialization
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = PPSActivation(dim_out, initial_omega1, initial_omega2)

        # Initialization for the linear layer (SIREN-like, using omega1 as reference)
        with torch.no_grad():
            if self.is_first:
                # SIREN first layer: uniform(-1/dim_in, 1/dim_in)
                w_std = (1.0 / dim_in) * linear_w_std_factor
            else:
                # SIREN hidden layers: uniform(-sqrt(6/dim_in)/omega0, sqrt(6/dim_in)/omega0)
                # Use initial_omega1 as the reference frequency for scaling.
                w_std = (math.sqrt(6.0 / dim_in) / initial_omega1) * linear_w_std_factor

            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias and self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        linear_out = self.linear(x)
        return self.activation(linear_out)

# PPS Network
class PPSNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 use_bias=True, final_activation=None, asi_if=False,
                 # PPS Layer specific params
                 w1_initial=30.0, w1_hidden=30.0,
                 w2_initial=0.01, w2_hidden=0.01, # Separate init for omega2 (start small)
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
            current_omega1 = w1_initial if is_first else w1_hidden
            current_omega2 = w2_initial if is_first else w2_hidden # Allow different omega2 init

            self.layers.append(PPSLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                is_first=is_first,
                initial_omega1=current_omega1,
                initial_omega2=current_omega2,
                linear_w_std_factor=linear_w_std_factor
            ))

        # Final linear layer
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

        # Initialize final linear layer (Simplified SIREN-like, using hidden omega1)
        with torch.no_grad():
             final_w_std = (math.sqrt(6.0 / dim_hidden) / w1_hidden) * linear_w_std_factor
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
def PPS(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'use_bias': True,
        'final_activation': None,
        'asi_if': False,
        # PPS specific
        'w1_initial': 30.0, # Omega1 for first layer
        'w1_hidden': 30.0,  # Omega1 for hidden layers
        'w2_initial': 0.01, # Omega2 for first layer (start small)
        'w2_hidden': 0.01,  # Omega2 for hidden layers (start small)
        # Linear init specific
        'linear_w_std_factor': 1.0 # Multiplier for SIREN-like weight init std dev
    }

    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now

    return PPSNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        use_bias=parameter['use_bias'],
        final_activation=parameter['final_activation'],
        asi_if=parameter['asi_if'],
        w1_initial=parameter['w1_initial'],
        w1_hidden=parameter['w1_hidden'],
        w2_initial=parameter['w2_initial'],
        w2_hidden=parameter['w2_hidden'],
        linear_w_std_factor=parameter['linear_w_std_factor']
    ) 