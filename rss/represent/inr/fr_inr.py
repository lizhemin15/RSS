"""
Fourier Reparameterized INR (FR-INR)

Reference: FR-INR: Fourier Reparameterized Implicit Neural Representations
https://github.com/labshuhanggu/fr-inr

FR-INR replaces standard linear layers with Fourier-reparameterized layers,
where the weight matrix is decomposed as W = lambda @ B, with B being fixed
Fourier bases and lambda being learnable coefficients. This provides implicit
spectral bias control and better convergence for high-frequency signals.

Supported modes:
- 'relu': Standard ReLU network (baseline)
- 'relu+fr': ReLU + Fourier reparameterized hidden layers
- 'sin': SIREN network (baseline)
- 'sin+fr': SIREN + Fourier reparameterized hidden layers (sin_fr_layer)
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FourierReparamLinear(nn.Module):
    """Fourier reparameterized linear layer for ReLU-based INR.

    Weight is parameterized as W = lambda @ B, where:
    - B: fixed Fourier bases (low_freq + high_freq cosine functions)
    - lambda: learnable coefficients controlling spectral composition
    - alpha: scaling factor for bases (controls spectral bias strength)

    This ensures implicit spectral regularization: the weight matrix
    is constrained to live in the span of the Fourier bases.
    """

    def __init__(self, in_features, out_features, high_freq_num=128,
                 low_freq_num=128, phi_num=32, alpha=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num = high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha = alpha

        self.bases = self._init_bases()
        self.lamb = self._init_lamb()
        self.bias = nn.Parameter(torch.Tensor(self.out_features, 1), requires_grad=True)
        self._init_bias()

    def _init_bases(self):
        """Construct fixed Fourier bases B.

        Each row of B is a cosine function cos(freq * x + phi) evaluated
        on a grid of in_features points. Low-frequency bases have fractional
        frequencies (1/low_freq_num, 2/low_freq_num, ...), high-frequency
        bases have integer frequencies (1, 2, ...).
        """
        phi_set = np.array([2 * math.pi * i / self.phi_num for i in range(self.phi_num)])
        high_freq = np.array([i + 1 for i in range(self.high_freq_num)])
        low_freq = np.array([(i + 1) / self.low_freq_num for i in range(self.low_freq_num)])

        if len(low_freq) != 0:
            T_max = 2 * math.pi / low_freq[0]
        else:
            T_max = 2 * math.pi / min(high_freq)

        points = np.linspace(-T_max / 2, T_max / 2, self.in_features)
        total_basis_num = (self.high_freq_num + self.low_freq_num) * self.phi_num
        bases = torch.Tensor(total_basis_num, self.in_features)

        i = 0
        for freq in low_freq:
            for phi in phi_set:
                bases[i, :] = torch.tensor([math.cos(freq * x + phi) for x in points])
                i += 1
        for freq in high_freq:
            for phi in phi_set:
                bases[i, :] = torch.tensor([math.cos(freq * x + phi) for x in points])
                i += 1

        bases = self.alpha * bases
        bases = nn.Parameter(bases, requires_grad=False)
        return bases

    def _init_lamb(self):
        """Initialize learnable coefficients lambda.

        Uses uniform initialization scaled by the norm of each basis row,
        following the Xavier-style initialization adapted for Fourier
        reparameterization.
        """
        total_basis_num = (self.high_freq_num + self.low_freq_num) * self.phi_num
        lamb = torch.Tensor(self.out_features, total_basis_num)

        with torch.no_grad():
            m = total_basis_num
            for i in range(m):
                dominator = torch.norm(self.bases[i, :], p=2)
                lamb[:, i] = nn.init.uniform_(
                    lamb[:, i],
                    -np.sqrt(6 / m) / dominator,
                    np.sqrt(6 / m) / dominator
                )

        lamb = nn.Parameter(lamb, requires_grad=True)
        return lamb

    def _init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)

    def forward(self, x):
        weight = torch.matmul(self.lamb, self.bases)
        output = torch.matmul(x, weight.transpose(0, 1))
        output = output + self.bias.T
        return output


class SinFRLayer(nn.Module):
    """Fourier reparameterized SIREN layer (sin activation with FR weight).

    Same as FourierReparamLinear but with sin(omega_0 * output) activation.
    The initialization of lambda is scaled by omega_0, following SIREN's
    initialization convention.
    """

    def __init__(self, in_features, out_features, high_freq_num=128,
                 low_freq_num=128, phi_num=32, alpha=0.01, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num = high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha = alpha
        self.omega_0 = omega_0

        self.bases = self._init_bases()
        self.lamb = self._init_lamb()
        self.bias = nn.Parameter(torch.Tensor(self.out_features, 1), requires_grad=True)
        self._init_bias()

    def _init_bases(self):
        """Same Fourier basis construction as FourierReparamLinear."""
        phi_set = np.array([2 * math.pi * i / self.phi_num for i in range(self.phi_num)])
        high_freq = np.array([i + 1 for i in range(self.high_freq_num)])
        low_freq = np.array([(i + 1) / self.low_freq_num for i in range(self.low_freq_num)])

        if len(low_freq) != 0:
            T_max = 2 * math.pi / low_freq[0]
        else:
            T_max = 2 * math.pi / min(high_freq)

        points = np.linspace(-T_max / 2, T_max / 2, self.in_features)
        total_basis_num = (self.high_freq_num + self.low_freq_num) * self.phi_num
        bases = torch.Tensor(total_basis_num, self.in_features)

        i = 0
        for freq in low_freq:
            for phi in phi_set:
                bases[i, :] = torch.tensor([math.cos(freq * x + phi) for x in points])
                i += 1
        for freq in high_freq:
            for phi in phi_set:
                bases[i, :] = torch.tensor([math.cos(freq * x + phi) for x in points])
                i += 1

        bases = self.alpha * bases
        bases = nn.Parameter(bases, requires_grad=False)
        return bases

    def _init_lamb(self):
        """Initialize lambda with SIREN-scaled initialization.

        The range is divided by omega_0, consistent with SIREN's
        weight initialization convention.
        """
        total_basis_num = (self.high_freq_num + self.low_freq_num) * self.phi_num
        lamb = torch.Tensor(self.out_features, total_basis_num)

        with torch.no_grad():
            m = total_basis_num
            for i in range(m):
                dominator = torch.norm(self.bases[i, :], p=2)
                lamb[:, i] = nn.init.uniform_(
                    lamb[:, i],
                    -np.sqrt(6 / m) / dominator / self.omega_0,
                    np.sqrt(6 / m) / dominator / self.omega_0
                )

        lamb = nn.Parameter(lamb, requires_grad=True)
        return lamb

    def _init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)

    def forward(self, x):
        weight = torch.matmul(self.lamb, self.bases)
        output = torch.matmul(x, weight.transpose(0, 1))
        output = output + self.bias.T
        return torch.sin(self.omega_0 * output)


class SineLayer(nn.Module):
    """Standard SIREN layer for first/last layer of sin+fr networks."""

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class FRINRNet(nn.Module):
    """Fourier Reparameterized INR network.

    Supports four modes:
    - 'relu': Standard MLP with ReLU activation
    - 'relu+fr': MLP with Fourier-reparameterized hidden layers + ReLU
    - 'sin': Standard SIREN
    - 'sin+fr': SIREN with Fourier-reparameterized hidden layers

    The FR variants replace standard linear layers in hidden layers with
    FourierReparamLinear or SinFRLayer, constraining the weight matrix
    to the span of predefined Fourier bases. This provides implicit
    spectral regularization without explicit frequency control.
    """

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                 mode='relu+fr', outermost_linear=True,
                 high_freq_num=128, low_freq_num=128, phi_num=32,
                 alpha=0.05, first_omega_0=30.0, hidden_omega_0=30.0,
                 pe=False, sidelength=None):
        super().__init__()
        self.pe = pe

        if pe:
            # Positional encoding (NeRF-style)
            if dim_in == 2:
                num_freq = 4
                if sidelength is not None:
                    s = min(sidelength) if isinstance(sidelength, (list, tuple)) else sidelength
                    nyquist_rate = 1 / (2 * (2 * 1 / s))
                    num_freq = int(math.floor(math.log(nyquist_rate, 2)))
            elif dim_in == 3:
                num_freq = 10
            else:
                num_freq = 4
            pe_dim_out = dim_in + 2 * dim_in * num_freq
            self.num_freq = num_freq
            self.pe_dim_in = dim_in
            self.pe_dim_out = pe_dim_out
            dim_in = pe_dim_out

        layers = []

        # First layer
        if mode in ('sin', 'sin+fr'):
            layers.append(SineLayer(dim_in, dim_hidden, is_first=True, omega_0=first_omega_0))
        elif mode in ('relu', 'relu+fr'):
            layers.append(nn.Linear(dim_in, dim_hidden))
            layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            if mode == 'relu':
                layers.append(nn.Linear(dim_hidden, dim_hidden))
                layers.append(nn.ReLU())
            elif mode == 'relu+fr':
                layers.append(FourierReparamLinear(
                    dim_hidden, dim_hidden,
                    high_freq_num=high_freq_num,
                    low_freq_num=low_freq_num,
                    phi_num=phi_num,
                    alpha=alpha
                ))
                layers.append(nn.ReLU())
            elif mode == 'sin':
                layers.append(SineLayer(dim_hidden, dim_hidden, is_first=False, omega_0=hidden_omega_0))
            elif mode == 'sin+fr':
                layers.append(SinFRLayer(
                    dim_hidden, dim_hidden,
                    high_freq_num=high_freq_num,
                    low_freq_num=low_freq_num,
                    phi_num=phi_num,
                    alpha=alpha,
                    omega_0=hidden_omega_0
                ))

        # Last layer
        if outermost_linear:
            final_linear = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                if mode in ('sin', 'sin+fr'):
                    final_linear.weight.uniform_(
                        -np.sqrt(6 / dim_hidden) / hidden_omega_0,
                        np.sqrt(6 / dim_hidden) / hidden_omega_0
                    )
                else:
                    final_linear.weight.uniform_(
                        -np.sqrt(6 / dim_hidden),
                        np.sqrt(6 / dim_hidden)
                    )
            layers.append(final_linear)
        else:
            if mode in ('relu', 'relu+fr'):
                layers.append(nn.Linear(dim_hidden, dim_out))
                layers.append(nn.ReLU())
            elif mode in ('sin', 'sin+fr'):
                layers.append(SineLayer(dim_hidden, dim_out, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def _apply_pe(self, x):
        """Apply positional encoding."""
        x = x.view(x.shape[0], -1, self.pe_dim_in)
        x_pe = x.clone()
        for i in range(self.num_freq):
            for j in range(self.pe_dim_in):
                c = x[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                x_pe = torch.cat((x_pe, sin, cos), axis=-1)
        return x_pe.reshape(x.shape[0], -1, self.pe_dim_out)

    def forward(self, x):
        if self.pe:
            x = self._apply_pe(x)
        return self.net(x)


def FRINR(parameter):
    """Factory function for FRINR, following RSS convention.

    Default parameters:
        dim_in: 2 (input dimension, 2 for images)
        dim_hidden: 256 (hidden layer width)
        dim_out: 1 (output dimension)
        num_layers: 4 (number of hidden layers, including first)
        mode: 'relu+fr' (network mode)
        outermost_linear: True (linear output layer)
        high_freq_num: 128 (number of high-frequency bases)
        low_freq_num: 128 (number of low-frequency bases)
        phi_num: 32 (number of phase shifts per frequency)
        alpha: 0.05 (basis scaling; 0.05 for relu, 0.01 for sin)
        first_omega_0: 30.0 (omega for first SIREN layer)
        hidden_omega_0: 30.0 (omega for hidden SIREN layers)
        pe: False (use positional encoding)
        sidelength: None (image sidelength for PE Nyquist calculation)
    """
    de_para_dict = {
        'dim_in': 2,
        'dim_hidden': 256,
        'dim_out': 1,
        'num_layers': 4,
        'mode': 'relu+fr',
        'outermost_linear': True,
        'high_freq_num': 128,
        'low_freq_num': 128,
        'phi_num': 32,
        'alpha': 0.05,
        'first_omega_0': 30.0,
        'hidden_omega_0': 30.0,
        'pe': False,
        'sidelength': None,
    }
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now

    # Auto-adjust alpha based on mode if not explicitly set
    if 'alpha' not in parameter:
        if parameter['mode'] in ('sin', 'sin+fr'):
            parameter['alpha'] = 0.01
        else:
            parameter['alpha'] = 0.05

    return FRINRNet(
        dim_in=parameter['dim_in'],
        dim_hidden=parameter['dim_hidden'],
        dim_out=parameter['dim_out'],
        num_layers=parameter['num_layers'],
        mode=parameter['mode'],
        outermost_linear=parameter['outermost_linear'],
        high_freq_num=parameter['high_freq_num'],
        low_freq_num=parameter['low_freq_num'],
        phi_num=parameter['phi_num'],
        alpha=parameter['alpha'],
        first_omega_0=parameter['first_omega_0'],
        hidden_omega_0=parameter['hidden_omega_0'],
        pe=parameter['pe'],
        sidelength=parameter['sidelength'],
    )