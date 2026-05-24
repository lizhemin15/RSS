"""
Minimal reproducible example for TIP paper: Image Denoising

This script reproduces the image denoising results from the INRR paper
using different regularization methods for comparison.

------------------------------------------------------------------------
How to run FRINR examples
------------------------------------------------------------------------
FRINR (Fourier Reparameterized INR) can be used as a drop-in replacement
for SIREN in the network parameter. Below are ready-to-use commands:

# 1) FRINR (relu+fr) + INRR
python -c "
import rss
result = rss.task.run('denoising',
    data_path='data/img/Barbara.jpg',
    net_p={'net_name': 'FRINR', 'mode': 'relu+fr', 'dim_in': 2,
           'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
           'high_freq_num': 128, 'low_freq_num': 128, 'phi_num': 32,
           'alpha': 0.05},
    reg_p={'reg_name': 'INRR', 'coef': 0.1, 'mode': 0,
           'inr_parameter': {'net_name': 'SIREN', 'dim_in': 1,
                             'dim_out': 256, 'w0_initial': 30, 'num_layers': 3}},
    data_p={'noise_mode': 'gaussian', 'noise_parameter': 25},
    train_p={'train_epoch': 500})
result['model'].show()
"

# 2) FRINR (sin+fr) + INRR
python -c "
import rss
result = rss.task.run('denoising',
    data_path='data/img/Barbara.jpg',
    net_p={'net_name': 'FRINR', 'mode': 'sin+fr', 'dim_in': 2,
           'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
           'high_freq_num': 128, 'low_freq_num': 128, 'phi_num': 32,
           'alpha': 0.01, 'first_omega_0': 30.0, 'hidden_omega_0': 30.0},
    reg_p={'reg_name': 'INRR', 'coef': 0.1, 'mode': 0,
           'inr_parameter': {'net_name': 'SIREN', 'dim_in': 1,
                             'dim_out': 256, 'w0_initial': 30, 'num_layers': 3}},
    data_p={'noise_mode': 'gaussian', 'noise_parameter': 25},
    train_p={'train_epoch': 500})
result['model'].show()
"

# 3) FRINR (relu+fr) without regularization (implicit spectral bias)
python -c "
import rss
result = rss.task.run('denoising',
    data_path='data/img/Barbara.jpg',
    net_p={'net_name': 'FRINR', 'mode': 'relu+fr', 'dim_in': 2,
           'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
           'high_freq_num': 128, 'low_freq_num': 128, 'phi_num': 32,
           'alpha': 0.05},
    data_p={'noise_mode': 'gaussian', 'noise_parameter': 25},
    train_p={'train_epoch': 500})
result['model'].show()
"

# 4) FRINR (relu+fr) + INRR+ (with TV blend)
python -c "
import rss
result = rss.task.run('denoising',
    data_path='data/img/Barbara.jpg',
    net_p={'net_name': 'FRINR', 'mode': 'relu+fr', 'dim_in': 2,
           'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
           'high_freq_num': 128, 'low_freq_num': 128, 'phi_num': 32,
           'alpha': 0.05},
    reg_p={'reg_name': 'INRR', 'coef': 0.1, 'mode': 0,
           'inr_parameter': {'net_name': 'SIREN', 'dim_in': 1,
                             'dim_out': 256, 'w0_initial': 30, 'num_layers': 3},
           'inrr_alpha': 0.5, 'nabla_matrix_order_k': 1},
    data_p={'noise_mode': 'gaussian', 'noise_parameter': 25},
    train_p={'train_epoch': 500})
result['model'].show()
"

FRINR modes:
  - 'relu'    : Standard ReLU network (baseline, same as MLP)
  - 'relu+fr' : ReLU + Fourier reparameterized hidden layers (recommended)
  - 'sin'     : Standard SIREN network (baseline)
  - 'sin+fr'  : SIREN + Fourier reparameterized hidden layers

Key parameters:
  - high_freq_num: number of high-frequency bases (integers 1,2,...)
  - low_freq_num : number of low-frequency bases (fractional 1/N,2/N,...)
  - phi_num      : number of phase shifts per frequency
  - alpha        : basis scaling (0.05 for relu, 0.01 for sin)
------------------------------------------------------------------------
"""

import rss
import os


def inrr_denoising():
    """Image denoising with INRR regularization (the proposed method)."""
    result = rss.task.run(
        'denoising',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'SIREN',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'w0_initial': 50,
            'num_layers': 4
        },
        reg_p={
            'reg_name': 'INRR',
            'coef': 0.1,
            'mode': 0,
            'inr_parameter': {
                'net_name': 'SIREN',
                'dim_in': 1,
                'dim_out': 256,
                'w0_initial': 30,
                'num_layers': 3
            }
        },
        data_p={
            'noise_mode': 'gaussian',
            'noise_parameter': 25  # sigma=25
        },
        train_p={'train_epoch': 500}
    )
    result['model'].show()
    return result


def inrr_plus_denoising():
    """Image denoising with INRR+ (INRR combined with TV nabla matrix)."""
    result = rss.task.run(
        'denoising',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'SIREN',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'w0_initial': 50,
            'num_layers': 4
        },
        reg_p={
            'reg_name': 'INRR',
            'coef': 0.1,
            'mode': 0,
            'inr_parameter': {
                'net_name': 'SIREN',
                'dim_in': 1,
                'dim_out': 256,
                'w0_initial': 30,
                'num_layers': 3
            },
            'inrr_alpha': 0.5,
            'nabla_matrix_order_k': 1
        },
        data_p={
            'noise_mode': 'gaussian',
            'noise_parameter': 25
        },
        train_p={'train_epoch': 500}
    )
    result['model'].show()
    return result


def tv_denoising():
    """Image denoising with TV regularization (baseline comparison)."""
    result = rss.task.run(
        'denoising',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'SIREN',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'w0_initial': 50,
            'num_layers': 4
        },
        reg_p={
            'reg_name': 'TV',
            'coef': 0.1,
            'mode': 0
        },
        data_p={
            'noise_mode': 'gaussian',
            'noise_parameter': 25
        },
        train_p={'train_epoch': 500}
    )
    result['model'].show()
    return result


def air_denoising():
    """Image denoising with AIR regularization (baseline comparison)."""
    result = rss.task.run(
        'denoising',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'SIREN',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'w0_initial': 50,
            'num_layers': 4
        },
        reg_p={
            'reg_name': 'AIR',
            'coef': 0.1,
            'mode': 0
        },
        data_p={
            'noise_mode': 'gaussian',
            'noise_parameter': 25
        },
        train_p={'train_epoch': 500}
    )
    result['model'].show()
    return result


def no_reg_denoising():
    """Image denoising without regularization (baseline - early stopping)."""
    result = rss.task.run(
        'denoising',
        data_path='data/img/Barbara.jpg',
        net_p={
            'net_name': 'SIREN',
            'dim_in': 2,
            'dim_hidden': 256,
            'dim_out': 1,
            'w0_initial': 50,
            'num_layers': 4
        },
        data_p={
            'noise_mode': 'gaussian',
            'noise_parameter': 25
        },
        train_p={'train_epoch': 500}
    )
    result['model'].show()
    return result


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("TIP Paper - Minimal Reproducible Example: Image Denoising")
    print("=" * 60)

    results = {}

    print("\n1. No regularization (early stopping)...")
    results['no_reg'] = no_reg_denoising()

    print("\n2. TV regularization...")
    results['tv'] = tv_denoising()

    print("\n3. AIR regularization...")
    results['air'] = air_denoising()

    print("\n4. INRR regularization (proposed)...")
    results['inrr'] = inrr_denoising()

    print("\n5. INRR+ regularization (proposed, enhanced)...")
    results['inrr_plus'] = inrr_plus_denoising()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary (PSNR dB):")
    print("=" * 60)
    for name, res in results.items():
        model = res['model']
        if 'psnr' in model.log_dict and len(model.log_dict['psnr']) > 0:
            print(f"  {name:15s}: {model.log_dict['psnr'][-1]:.2f}")