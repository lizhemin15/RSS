import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helpers
def exists(val):
    return val is not None

# Fixed Quadratic Phase Sine Activation (FQPS)
class FQPSActivation(nn.Module):
    """
    Implements the activation: sin(omega1 * x + C * x^2)
    omega1 is learnable, C is a fixed hyperparameter.
    """
    def __init__(self, dim_out, initial_omega1=30.0, fixed_c=0.01):
        super().__init__()
        # Learnable linear phase component frequency.
        self.omega1 = nn.Parameter(torch.full((1, dim_out), float(initial_omega1)))
        # Fixed quadratic phase component coefficient.
        # Store it as a buffer if you don't want it optimized but want it part of the state_dict,
        # or just as an attribute if it's truly fixed constant.
        # Let's use a simple attribute.
        self.fixed_c = float(fixed_c)

    def forward(self, x):
        # x shape: (batch_size, dim_out)
        # omega1 shape: (1, dim_out)

        # Calculate the phase: omega1 * x + fixed_c * x^2
        phase = self.omega1 * x + self.fixed_c * x.square()

        # Calculate final activation
        output = torch.sin(phase)

        # Optional: Check for NaNs/Infs
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaNs or Infs detected in FQPSActivation")
            output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)

        return output

# FQPS Layer: Linear + FQPSActivation
class FQPSLayer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bias=True, is_first=False,
                 initial_omega1=30.0, fixed_c=0.01,
                 linear_w_std_factor=1.0):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        # Pass the fixed hyperparameter C to the activation
        self.activation = FQPSActivation(dim_out, initial_omega1, fixed_c)

        # Initialization for the linear layer (SIREN-like, using omega1 as reference)
        with torch.no_grad():
            # Use initial_omega1 as the reference frequency for scaling.
            if self.is_first:
                w_std = (1.0 / dim_in) * linear_w_std_factor
            else:
                # Avoid division by zero if initial_omega1 is somehow zero
                w_std = (math.sqrt(6.0 / dim_in) / max(initial_omega1, 1e-6)) * linear_w_std_factor

            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias and self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        linear_out = self.linear(x)
        return self.activation(linear_out)

# FQPS Network
class FQPSNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 use_bias=True, final_activation=None, asi_if=False,
                 # FQPS Layer specific params
                 w1_initial=30.0, w1_hidden=30.0,
                 c_initial=0.01, c_hidden=0.01, # Fixed C values for first/hidden layers
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
            current_fixed_c = c_initial if is_first else c_hidden # Use the appropriate fixed C

            self.layers.append(FQPSLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                is_first=is_first,
                initial_omega1=current_omega1,
                fixed_c=current_fixed_c,
                linear_w_std_factor=linear_w_std_factor
            ))

        # Final linear layer
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

        # Initialize final linear layer (Simplified SIREN-like, using hidden omega1)
        with torch.no_grad():
             # Avoid division by zero
             final_w_std = (math.sqrt(6.0 / dim_hidden) / max(w1_hidden, 1e-6)) * linear_w_std_factor
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
def FQPS(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'use_bias': True,
        'final_activation': None,
        'asi_if': False,
        # FQPS specific
        'w1_initial': 30.0, # Learnable Omega1 for first layer
        'w1_hidden': 30.0,  # Learnable Omega1 for hidden layers
        'c_initial': 0.01, # Fixed C for first layer (hyperparameter)
        'c_hidden': 0.01,  # Fixed C for hidden layers (hyperparameter)
        # Linear init specific
        'linear_w_std_factor': 1.0 # Multiplier for SIREN-like weight init std dev
    }

    # Ensure required hyperparameters 'c_initial' and 'c_hidden' are present
    if 'c_initial' not in parameter:
        parameter['c_initial'] = de_para_dict['c_initial']
        print(f"FQPS: Using default c_initial: {parameter['c_initial']}")
    if 'c_hidden' not in parameter:
        parameter['c_hidden'] = de_para_dict['c_hidden']
        print(f"FQPS: Using default c_hidden: {parameter['c_hidden']}")

    # Update other parameters
    for key in de_para_dict.keys():
        if key not in ['c_initial', 'c_hidden']: # Avoid overwriting C values if provided
            param_now = parameter.get(key, de_para_dict.get(key))
            parameter[key] = param_now

    return FQPSNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        use_bias=parameter['use_bias'],
        final_activation=parameter['final_activation'],
        asi_if=parameter['asi_if'],
        w1_initial=parameter['w1_initial'],
        w1_hidden=parameter['w1_hidden'],
        c_initial=parameter['c_initial'], # Pass fixed C values
        c_hidden=parameter['c_hidden'],   # Pass fixed C values
        linear_w_std_factor=parameter['linear_w_std_factor']
    ) 