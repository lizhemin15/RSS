import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helpers
def exists(val):
    return val is not None

# Exponentiated Oscillatory Activation (EOA)
class EOAActivation(nn.Module):
    """
    Implements the activation: exp(alpha * x) * sin(omega * exp(beta * x))
    alpha, beta, omega are learnable parameters.
    """
    def __init__(self, dim_out, initial_alpha=0.0, initial_beta=0.0, initial_omega=10.0):
        super().__init__()
        # Initialize alpha and beta close to zero, omega with a reasonable default
        # Use nn.Parameter for learnable scalars. Ensure they are float tensors.
        # Parameter shape should match the output dimension if we want per-neuron control,
        # or be scalar if shared across the layer's output dimension.
        # Let's start with scalar parameters shared across the layer for simplicity.
        # Use torch.full to initialize Parameter with a specific value.
        self.alpha = nn.Parameter(torch.full((1, dim_out), float(initial_alpha))) # Shape (1, dim_out) for broadcasting
        self.beta = nn.Parameter(torch.full((1, dim_out), float(initial_beta)))
        self.omega = nn.Parameter(torch.full((1, dim_out), float(initial_omega)))

        # Small constant for numerical stability, especially for exp(beta*x) if beta*x is large negative
        self.eps = 1e-8

    def forward(self, x):
        # Ensure parameters are used correctly. Parameters will broadcast to match x's shape if needed.
        # x shape: (batch_size, dim_out)
        # alpha/beta/omega shape: (1, dim_out)

        # Calculate frequency term: omega * exp(beta * x)
        # Clamp beta * x to avoid excessively large values in exp?
        # Maybe clip the output of exp(beta*x) to avoid extreme frequencies?
        # Let's proceed without clipping first, rely on initialization and optimizer.
        freq_term = self.omega * torch.exp(self.beta * x)

        # Calculate amplitude term: exp(alpha * x)
        amp_term = torch.exp(self.alpha * x)

        # Calculate final activation
        # Add eps inside sin for potential gradient stability near zero frequency?
        # Let's try without first.
        output = amp_term * torch.sin(freq_term)

        # Check for NaNs/Infs
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaNs or Infs detected in EOAActivation")
            # Consider adding debugging info here, like input stats, parameter values
            # print("x stats:", x.min().item(), x.max().item(), x.mean().item())
            # print("alpha:", self.alpha.data.mean().item(), "beta:", self.beta.data.mean().item(), "omega:", self.omega.data.mean().item())
            # Maybe clamp output or return 0?
            output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)

        return output

# EOA Layer: Linear + EOAActivation
class EOALayer(nn.Module):
    def __init__(self, dim_in, dim_out, use_bias=True, is_first=False,
                 initial_alpha=0.0, initial_beta=0.0, initial_omega=10.0,
                 linear_w_std_factor=1.0, linear_w_c=6.0):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = EOAActivation(dim_out, initial_alpha, initial_beta, initial_omega)

        # Initialization for the linear layer (inspired by SIREN, but might need adjustment for EOA)
        # SIREN init depends on omega_0. Here omega is learnable and part of a complex activation.
        # Let's use a simpler scaled uniform/normal init first, maybe closer to standard Kaiming/Xavier
        # Or adapt SIREN's logic loosely.
        with torch.no_grad():
            # Loosely based on SIREN: std depends on input dim and a factor 'c'
            # The division by 'initial_omega' in SIREN was to account for the Sine activation scaling.
            # EOA's scaling is complex. Let's try a simple std dev.
            if self.is_first:
                # First layer might benefit from larger weights if input coords are [-1, 1]
                w_std = (1.0 / dim_in) * linear_w_std_factor
            else:
                # Hidden layers
                w_std = (math.sqrt(linear_w_c / dim_in)) * linear_w_std_factor

            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias and self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        linear_out = self.linear(x)
        return self.activation(linear_out)

# EOA Network
class EOANet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 use_bias=True, final_activation=None, asi_if=False,
                 # EOA Layer specific params
                 initial_alpha=0.0, initial_beta=0.0,
                 w0_initial=10.0, w0_hidden=10.0,
                 # Linear layer init params
                 linear_w_std_factor=1.0, linear_w_c=6.0):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if

        self.layers = nn.ModuleList()

        for ind in range(num_layers):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else dim_hidden
            current_omega = w0_initial if is_first else w0_hidden

            self.layers.append(EOALayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                is_first=is_first,
                initial_alpha=initial_alpha,
                initial_beta=initial_beta,
                initial_omega=current_omega,
                linear_w_std_factor=linear_w_std_factor,
                linear_w_c=linear_w_c
            ))

        # Final linear layer
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

        # Initialize final linear layer (e.g., standard init or adapted SIREN-like)
        with torch.no_grad():
             # Using a standard-like init for the final layer
             final_w_std = (math.sqrt(linear_w_c / dim_hidden)) * linear_w_std_factor
             nn.init.uniform_(self.last_layer.weight, -final_w_std, final_w_std)
             if use_bias and self.last_layer.bias is not None:
                  nn.init.uniform_(self.last_layer.bias, -final_w_std, final_w_std)

        # Optional ASI logic (copied, might need adjustment based on EOA behavior)
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
def EOA(parameter):
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'use_bias': True,
        'final_activation': None,
        'asi_if': False,
        # EOA specific
        'initial_alpha': 0.0,
        'initial_beta': 0.0,
        'w0_initial': 10.0, # Base omega for first layer
        'w0_hidden': 10.0,  # Base omega for hidden layers
        # Linear init specific
        'linear_w_std_factor': 0.5, # Reduce std dev slightly compared to pure SIREN maybe?
        'linear_w_c': 6.0
    }

    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now

    return EOANet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        use_bias=parameter['use_bias'],
        final_activation=parameter['final_activation'],
        asi_if=parameter['asi_if'],
        initial_alpha=parameter['initial_alpha'],
        initial_beta=parameter['initial_beta'],
        w0_initial=parameter['w0_initial'],
        w0_hidden=parameter['w0_hidden'],
        linear_w_std_factor=parameter['linear_w_std_factor'],
        linear_w_c=parameter['linear_w_c']
    ) 