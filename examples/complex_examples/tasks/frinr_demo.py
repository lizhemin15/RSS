"""
FRINR (Fourier Reparameterized INR) examples for image tasks.

FRINR replaces standard linear layers with Fourier-reparameterized layers,
where the weight matrix W = lambda @ B (B = fixed Fourier bases, lambda = learnable).
This provides implicit spectral bias control and better convergence for
high-frequency signals.

Reference: FR-INR: Fourier Reparameterized Implicit Neural Representations
https://github.com/labshuhanggu/fr-inr
"""

import rss
import os


def frinr_image_completion():
    """Image completion using FRINR with relu+fr mode."""
    result = rss.task.run(
        'completion',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'FRINR',
            'mode': 'relu+fr',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'num_layers': 4,
            'high_freq_num': 128,
            'low_freq_num': 128,
            'phi_num': 32,
            'alpha': 0.05
        },
        data_p={'random_rate': 0.7},  # 70% pixels missing
        train_p={'train_epoch': 100}
    )
    result['model'].show()


def frinr_sin_completion():
    """Image completion using FRINR with sin+fr mode (SIREN + Fourier reparam)."""
    result = rss.task.run(
        'completion',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'FRINR',
            'mode': 'sin+fr',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'num_layers': 4,
            'high_freq_num': 128,
            'low_freq_num': 128,
            'phi_num': 32,
            'alpha': 0.01,           # smaller alpha for sin mode
            'first_omega_0': 30.0,
            'hidden_omega_0': 30.0
        },
        data_p={'random_rate': 0.7},
        train_p={'train_epoch': 100}
    )
    result['model'].show()


def frinr_denoising():
    """Image denoising using FRINR."""
    result = rss.task.run(
        'denoising',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'FRINR',
            'mode': 'relu+fr',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'num_layers': 4,
            'high_freq_num': 128,
            'low_freq_num': 128,
            'phi_num': 32,
            'alpha': 0.05
        },
        data_p={'noise_mode': 'gaussian', 'noise_parameter': 25},  # sigma=25
        train_p={'train_epoch': 200}
    )
    result['model'].show()


def frinr_with_inrr():
    """Image completion using FRINR + INRR regularization.

    This combines FRINR's spectral bias with INRR's adaptive similarity-based
    regularization for potentially better results on textured images.
    """
    result = rss.task.run(
        'completion',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'FRINR',
            'mode': 'relu+fr',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'num_layers': 4,
            'high_freq_num': 128,
            'low_freq_num': 128,
            'phi_num': 32,
            'alpha': 0.05
        },
        reg_p={
            'reg_name': 'INRR',
            'coef': 0.1,
            'mode': 0,
            'inr_parameter': {
                'net_name': 'SIREN',
                'dim_in': 1,
                'dim_out': 256,
                'w0_initial': 30
            }
        },
        data_p={'random_rate': 0.7},
        train_p={'train_epoch': 100}
    )
    result['model'].show()


def frinr_low_freq_only():
    """FRINR with only low-frequency bases.

    Setting high_freq_num=0 restricts the weight matrix to low-frequency
    Fourier components, providing stronger implicit smoothing.
    """
    result = rss.task.run(
        'completion',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'FRINR',
            'mode': 'relu+fr',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'num_layers': 4,
            'high_freq_num': 0,      # no high-frequency bases
            'low_freq_num': 128,
            'phi_num': 32,
            'alpha': 0.05
        },
        data_p={'random_rate': 0.7},
        train_p={'train_epoch': 100}
    )
    result['model'].show()


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("1. FRINR relu+fr image completion...")
    frinr_image_completion()

    print("\n2. FRINR sin+fr image completion...")
    frinr_sin_completion()

    print("\n3. FRINR denoising...")
    frinr_denoising()

    print("\n4. FRINR + INRR regularization...")
    frinr_with_inrr()

    print("\n5. FRINR low-frequency only...")
    frinr_low_freq_only()
