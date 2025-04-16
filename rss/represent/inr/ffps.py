import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helpers
def exists(val):
    return val is not None

# Fully Fixed Phase Sine Activation (FFPS)
class FFPSActivation(nn.Module):
    """
    Implements the activation: sin(C1 * x + C2 * x^2)
    C1 and C2 are fixed hyperparameters.
    """
    def __init__(self, fixed_c1=30.0, fixed_c2=0.01):
        super().__init__()
        # Fixed phase coefficients passed during initialization.
        self.fixed_c1 = float(fixed_c1)
        self.fixed_c2 = float(fixed_c2)

        # Ensure C1 is not zero for initialization purposes later
        if abs(self.fixed_c1) < 1e-9:
             print("Warning: FFPSActivation initialized with fixed_c1 close to zero.")
             # self.fixed_c1 = 1e-6 # Or handle as error?

    def forward(self, x):
        # x shape: (batch_size, feature_dim)

        # Calculate the phase: C1 * x + C2 * x^2
        phase = self.fixed_c1 * x + self.fixed_c2 * x.square()

        # Calculate final activation
        output = torch.sin(phase)

        # Optional: Check for NaNs/Infs
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaNs or Infs detected in FFPSActivation")
            output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)

        return output

# FFPS Layer: Linear + FFPSActivation
class FFPSLayer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bias=True, is_first=False,
                 fixed_c1=30.0, fixed_c2=0.01,
                 linear_w_std_factor=1.0):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        # Pass the fixed hyperparameters C1 and C2 to the activation
        self.activation = FFPSActivation(fixed_c1, fixed_c2)

        # Initialization for the linear layer (SIREN-like, using fixed C1 as reference)
        with torch.no_grad():
            # Use fixed_c1 as the reference frequency for scaling.
            # Ensure fixed_c1 is not effectively zero for division.
            reference_freq = max(abs(fixed_c1), 1e-6)

            if self.is_first:
                # Standard first layer init
                w_std = (1.0 / dim_in) * linear_w_std_factor
            else:
                # Standard hidden layer init scaled by reference frequency
                w_std = (math.sqrt(6.0 / dim_in) / reference_freq) * linear_w_std_factor

            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias and self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        linear_out = self.linear(x)
        return self.activation(linear_out)

# FFPS Network
class FFPSNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 use_bias=True, final_activation=None, asi_if=False,
                 # FFPS Layer specific fixed params
                 c1_initial=30.0, c1_hidden=30.0,
                 c2_initial=0.01, c2_hidden=0.01,
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
            current_c1 = c1_initial if is_first else c1_hidden
            current_c2 = c2_initial if is_first else c2_hidden

            self.layers.append(FFPSLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                is_first=is_first,
                fixed_c1=current_c1,
                fixed_c2=current_c2,
                linear_w_std_factor=linear_w_std_factor
            ))

        # Final linear layer
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

        # Initialize final linear layer (SIREN-like, using hidden C1)
        with torch.no_grad():
             reference_freq_hidden = max(abs(c1_hidden), 1e-6)
             final_w_std = (math.sqrt(6.0 / dim_hidden) / reference_freq_hidden) * linear_w_std_factor
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
def FFPS(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'use_bias': True,
        'final_activation': None,
        'asi_if': False,
        # FFPS specific fixed hyperparameters
        'c1_initial': 30.0, # Fixed C1 for first layer
        'c1_hidden': 30.0,  # Fixed C1 for hidden layers
        'c2_initial': 0.01, # Fixed C2 for first layer
        'c2_hidden': 0.01,  # Fixed C2 for hidden layers
        # Linear init specific
        'linear_w_std_factor': 1.0 # Multiplier for SIREN-like weight init std dev
    }

    # Ensure required fixed hyperparameters C1 and C2 are present, using defaults if not
    required_fixed = ['c1_initial', 'c1_hidden', 'c2_initial', 'c2_hidden']
    for key in required_fixed:
        if key not in parameter:
            parameter[key] = de_para_dict[key]
            print(f"FFPS: Using default {key}: {parameter[key]}")

    # Update other parameters (non-fixed ones)
    for key in de_para_dict.keys():
        if key not in required_fixed:
            param_now = parameter.get(key, de_para_dict.get(key))
            parameter[key] = param_now

    return FFPSNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        use_bias=parameter['use_bias'],
        final_activation=parameter['final_activation'],
        asi_if=parameter['asi_if'],
        # Pass fixed C values
        c1_initial=parameter['c1_initial'],
        c1_hidden=parameter['c1_hidden'],
        c2_initial=parameter['c2_initial'],
        c2_hidden=parameter['c2_hidden'],
        linear_w_std_factor=parameter['linear_w_std_factor']
    ) 