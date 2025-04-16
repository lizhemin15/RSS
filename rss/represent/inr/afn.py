import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# helpers

def exists(val):
    return val is not None

# Learnable frequency Sine layer
class AdaptiveSineLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.0, c=6., is_first=False, use_bias=True, drop_out_p=0.0):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias
        self.drop_out_p = drop_out_p
        self.c = c

        # Learnable frequency parameter per layer
        self.w0 = nn.Parameter(torch.tensor(float(w0))) # Ensure it's a float tensor

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        # Initialize weights using the initial w0 value
        self.init_weights(weight, bias, initial_w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

        # Dropout layer (applied after linear)
        self.dropout = nn.Dropout(p=self.drop_out_p) if self.drop_out_p > 0 else nn.Identity()

    def init_weights(self, weight, bias, initial_w0):
        """
        Initialize weights according to SIREN paper.
        Use the initial w0 value for calculation, the parameter w0 will be learned.
        """
        dim = self.dim_in
        w_std = (1. / dim) if self.is_first else (math.sqrt(self.c / dim) / initial_w0)
        nn.init.uniform_(weight, -w_std, w_std)

        if exists(bias):
            nn.init.uniform_(bias, -w_std, w_std)

    def forward(self, x):
        linear_out = F.linear(x, self.weight, self.bias)
        # Apply dropout after the linear transformation, before activation
        linear_out = self.dropout(linear_out)
        # Apply activation with the learnable frequency
        return torch.sin(self.w0 * linear_out)

# Adaptive Frequency Network (AFN)
class AdaptiveFrequencyNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.0, w0_initial=30.0, use_bias=True, final_activation=None, drop_out_p=0.0, asi_if=False, c=6.0):
        """
        Initialize the Adaptive Frequency Network.

        Args:
            dim_in (int): Input dimension.
            dim_hidden (int): Hidden layer dimension.
            dim_out (int): Output dimension.
            num_layers (int): Number of hidden layers.
            w0 (float): Initial frequency for hidden layers (becomes learnable).
            w0_initial (float): Initial frequency for the first layer (becomes learnable).
            use_bias (bool): Whether to use bias in layers.
            final_activation (nn.Module, optional): Activation for the final layer. Defaults to None (Identity).
            drop_out_p (float): Dropout probability for hidden layers. Defaults to 0.0.
            asi_if (bool): Whether to use Anti-Symmetric Initialization for the final layer. Defaults to False.
            c (float): Parameter for weight initialization variance calculation (from SIREN). Defaults to 6.0.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.asi_if = asi_if

        self.layers = nn.ModuleList()
        current_dropout_p = drop_out_p

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(AdaptiveSineLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0, # Pass initial value, wrapped in nn.Parameter inside the layer
                c=c,
                use_bias=use_bias,
                is_first=is_first,
                drop_out_p=current_dropout_p # Apply dropout to all hidden layers if drop_out_p > 0
            ))

        # Final linear layer
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

        # Initialize final linear layer based on SIREN principles
        with torch.no_grad():
            # SIREN's final layer init uses the hidden layer omega (w0) and c=6
            final_w_std = math.sqrt(c / dim_hidden) / w0
            nn.init.uniform_(self.last_layer.weight, -final_w_std, final_w_std)
            if use_bias and self.last_layer.bias is not None:
                 nn.init.uniform_(self.last_layer.bias, -final_w_std, final_w_std)

        # Optional Anti-Symmetric Initialization (ASI) for the last layer
        if self.asi_if:
            self.last_layer_asi = nn.Linear(dim_hidden, dim_out, bias=use_bias)
            with torch.no_grad():
                # Initialize identically to the original last layer initially
                self.last_layer_asi.weight.data.copy_(self.last_layer.weight.data)
                if self.last_layer.bias is not None and self.last_layer_asi.bias is not None:
                     self.last_layer_asi.bias.data.copy_(self.last_layer.bias.data)

        # Store final activation (if any)
        self.final_activation = nn.Identity() if not exists(final_activation) else final_activation

    def forward(self, x, mods=None):
        # `mods` argument is kept for potential compatibility with wrappers like SirenWrapper, but not used here.
        for layer in self.layers:
            x = layer(x)

        output = self.last_layer(x)

        # Apply final activation if one was provided
        output = self.final_activation(output)

        # Apply ASI if enabled
        if self.asi_if:
             output_asi = self.last_layer_asi(x)
             output_asi = self.final_activation(output_asi) # Apply activation here too
             # The factor sqrt(2)/2 = 1/sqrt(2) is kept from other implementations like WIRE/GAUSS/FINER
             output = (output - output_asi) * (math.sqrt(2.) / 2.)

        return output

# Factory function for easy instantiation
def AFN(parameter):
    """
    Factory function to create an AdaptiveFrequencyNet instance from a parameter dictionary.
    Provides default values based on common INR practices.
    """
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,        # Number of *hidden* layers
        'w0': 30.0,             # Initial frequency for hidden layers (learnable)
        'w0_initial': 30.0,     # Initial frequency for the first layer (learnable)
        'use_bias': True,
        'final_activation': None,# No activation after final linear layer by default
        'drop_out_p': 0.0,      # No dropout by default
        'asi_if': False,        # ASI disabled by default
        'c': 6.0                # SIREN initialization constant
    }

    # Update parameters from the input dictionary, using defaults if keys are missing
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now

    # Create the network instance
    return AdaptiveFrequencyNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        w0=parameter['w0'],
        w0_initial=parameter['w0_initial'],
        use_bias=parameter['use_bias'],
        final_activation=parameter['final_activation'],
        drop_out_p=parameter['drop_out_p'],
        asi_if=parameter['asi_if'],
        c=parameter['c']
    ) 