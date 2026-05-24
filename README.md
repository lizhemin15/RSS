![](./statics/logo.jpg)

# What's RSS?
Using Pytorch to represent both the signal and regularization term, and then solve the inverse problems.

# How to install
```
git clone https://github.com/lizhemin15/RSS.git
cd RSS
pip install -r requirements.txt
python setup.py build
python setup.py install
```

You may face to lake 'libGL.so.1' error, you can install it by:
```
apt-get update
apt install libgl1-mesa-glx libgl1-mesa-dri
apt-get install libglib2.0-dev
```

# Quick Start
## Demo
### Image Completion Demo:
```python
import rss
import os

def basic_completion():
    """Basic image completion example"""
    # Simple one-line usage
    result = rss.task.run('completion',
                         data_path='data/img/example.jpg')
    
    # Show results
    result['model'].show()

def advanced_completion():
    """Advanced image completion with custom parameters"""
    # Custom network architecture using SIREN
    result = rss.task.run(
        'completion',
        data_path='data/img/example.jpg',
        net_p={
            'net_list': [
                {
                    'net_name': 'SIREN',
                    'dim_in': 2,
                    'w0_initial': 50,
                    'dim_hidden': 256,
                    'dim_out': 1
                }
            ]
        },
        train_p={'train_epoch': 100},
        data_p={'random_rate': 0.7}  # 70% pixels missing
    )
    
    # Show intermediate results
    result['model'].show()

def tensor_completion():
    """Image completion using tensor factorization"""
    result = rss.task.run(
        'completion',
        data_path='data/img/example.jpg',
        net_p={
            'net_list': [
                {
                    'net_name': 'TF',
                    'sizes': [256, 256, 1],
                    'dim_cor': [256, 256, 1],
                    'mode': 'tucker'
                }
            ]
        }
    )
    result['model'].show()

if __name__ == '__main__':
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    print("Running basic completion example...")
    basic_completion()
    
    print("\nRunning advanced completion example...")
    advanced_completion()
    
    print("\nRunning tensor completion example...")
    tensor_completion() 
```

### Denoising Demo
```python
import rss
import os

def basic_denoising():
    """Basic image denoising example"""
    result = rss.task.run('denoising',
                         data_path='data/img/noisy.jpg')
    
    result['model'].show()

def advanced_denoising():
    """Advanced denoising with custom architecture"""
    result = rss.task.run(
        'denoising',
        data_path='data/img/noisy.jpg',
        net_p={
            'net_list': [
                # First use tensor factorization
                {
                    'net_name': 'TF',
                    'sizes': [256, 256, 1],
                    'dim_cor': [256, 256, 1],
                    'mode': 'tensor'
                },
                # Then interpolation
                {
                    'net_name': 'Interpolation',
                    'return_type': "feature"
                },
                # Finally SIREN
                {
                    'net_name': "SIREN",
                    'dim_in': 1,
                    'w0_initial': 50,
                    'dim_hidden': 256,
                    'dim_out': 1
                }
            ]
        },
        train_p={'train_epoch': 200}
    )
    
    result['model'].show()

def knn_denoising():
    """Denoising using KNN-based approach"""
    result = rss.task.run(
        'denoising',
        data_path='data/img/noisy.jpg',
        net_p={
            'net_list': [
                {
                    'net_name': 'KNN',
                    'sizes': [256, 256],
                    'dim_cor': [64, 64],
                    'mode': 'tensor',
                    'weights': 'distance'
                }
            ]
        }
    )
    result['model'].show()

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    print("Running basic denoising example...")
    basic_denoising()
    
    print("\nRunning advanced denoising example...")
    advanced_denoising()
    
    print("\nRunning KNN denoising example...")
    knn_denoising() 
```

### Super resolution demo
```python
import rss
import os

def basic_super_resolution():
    """Basic super resolution example"""
    result = rss.task.run('super_resolution',
                         data_path='data/img/low_res.jpg',
                         scale_factor=4)  # 4x upscaling
    
    result['model'].show()

def advanced_super_resolution():
    """Advanced super resolution with custom settings"""
    result = rss.task.run(
        'super_resolution',
        data_path='data/img/low_res.jpg',
        scale_factor=4,
        net_p={
            'net_list': [
                {
                    'net_name': 'SIREN',
                    'dim_in': 2,
                    'w0_initial': 30,
                    'dim_hidden': 256,
                    'dim_out': 3  # RGB output
                }
            ]
        },
        train_p={'train_epoch': 500}
    )
    
    result['model'].show()

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    print("Running basic super resolution example...")
    basic_super_resolution()
    
    print("\nRunning advanced super resolution example...")
    advanced_super_resolution() 
```

### Nerf Demo
```python
import rss
import os

def basic_nerf():
    """Basic NeRF example"""
    result = rss.task.run('nerf',
                         data_path='data/nerf/lego')
    
    # Render novel views
    result['model'].render_path()

def advanced_nerf():
    """Advanced NeRF with custom settings"""
    result = rss.task.run(
        'nerf',
        data_path='data/nerf/lego',
        net_p={
            'net_list': [
                {
                    'net_name': 'HashINR',  # Using hash encoding
                    'n_levels': 16,
                    'n_features_per_level': 2,
                    'log2_hashmap_size': 19
                }
            ]
        },
        train_p={
            'train_epoch': 50000,
            'batch_size': 4096
        }
    )
    
    # Render novel views
    result['model'].render_path()

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    print("Running basic NeRF example...")
    basic_nerf()
    
    print("\nRunning advanced NeRF example...")
    advanced_nerf() 
```



## Get a representation and apply it to your own applications
```python
from rss import get_nn
net = get_nn({'net_name':'SIREN','dim_in':2,'dim_hidden':100,'dim_out':1,'num_layers':4,'w0':1,'w0_initial':30.,'use_bias':True, 'final_activation':None, 'asi_if':False}) # 

all_net_name_list = [
    'composition', 'MLP', 'SIREN', 'WIRE', 'BACON', 'FourierNet', 'GaborNet', 
    'DMF', 'TF', 'Interpolation', 'UNet', 'ResNet', 'skip', 'KNN', 'TDKNN', 
    'FourierFeature', 'HashEmbedder', 'EFF_KAN', 'KAN', 'ChebyKAN', 'FastKAN', 
    'RecurrentINR', 'HashINR', 'DINER', 'SIMINER', 'FFINR', 'KATE', 'TIP', 
    'GAUSS', 'FINER', 'CHEBYFINER', 'FRINR'
]
```

### FRINR: Fourier Reparameterized INR

FRINR replaces standard linear layers with Fourier-reparameterized layers, where the weight matrix is decomposed as `W = lambda @ B` (`B` = fixed Fourier bases, `lambda` = learnable coefficients). This provides implicit spectral bias control.

**Four modes:**
- `'relu'`: Standard ReLU network (baseline)
- `'relu+fr'`: ReLU + Fourier reparameterized hidden layers
- `'sin'`: SIREN network (baseline)
- `'sin+fr'`: SIREN + Fourier reparameterized hidden layers

```python
from rss import get_nn

# ReLU + Fourier reparameterization (recommended for image tasks)
net = get_nn({
    'net_name': 'FRINR',
    'mode': 'relu+fr',
    'dim_in': 2,
    'dim_hidden': 256,
    'dim_out': 1,
    'num_layers': 4,
    'high_freq_num': 128,    # number of high-frequency bases
    'low_freq_num': 128,     # number of low-frequency bases
    'phi_num': 32,           # phase shifts per frequency
    'alpha': 0.05            # basis scaling (0.05 for relu, 0.01 for sin)
})

# SIREN + Fourier reparameterization
net = get_nn({
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
    'first_omega_0': 30.0,   # omega for first layer
    'hidden_omega_0': 30.0   # omega for hidden layers
})

# Use in image completion task
result = rss.task.run(
    'completion',
    data_path='data/img/example.jpg',
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
    data_p={'random_rate': 0.7},
    train_p={'train_epoch': 100}
)
```


## Get a regularization and apply it to your own applications
```python
from rss import get_reg
reg = get_reg({'reg_name', 'AIR'})

# forward
reg_loss = reg(x) # x is the signal need to be regularized

all_reg_name_list = [
    'TV', 'LAP', 'WTV', 'NLTV', 'STV', 'DE', 'AIR', 'INRR', 'RUBI', 
    'MultiReg', 'GroupReg', 'Nuclear'
]
```

# Reproducing TIP Paper Results

This repository contains the source code for the TIP paper **"INRR: Implicit Neural Representation Regularization"**. The `minim_example/` directory provides minimal reproducible scripts for image completion and denoising tasks with all regularization variants.

```
minim_example/
├── completion/run_completion.py    # Image completion (all reg variants)
└── denoising/run_denoising.py      # Image denoising (all reg variants)
└── img/                            # Test images (Barbara, Baboon, etc.)
```

## Image Completion

```python
import rss

size = 256
inrr_alpha = 0.2
nabla_matrix_order_k = 1
lap_k = 1
huber_delta = 0.2

parameters = {
    'net_p': {
        'gpu_id': 0,
        'net_name': 'composition',
        'net_list': [{'net_name': 'SIREN', 'dim_in': 2, 'w0_initial': 30, 'dim_hidden': size}]
    },
    'data_p': {
        'data_shape': (size, size), 'random_rate': 0.5, 'pre_full': True,
        'mask_type': 'random', 'data_path': 'data/img/Barbara.jpg',
        'data_type': 'gray_img', 'ymode': 'completion'
    },
    'train_p': {'train_epoch': 2000},
    'opt_p': {
        'net': {'opt_name': 'Adam', 'lr': 1e-3, 'weight_decay': 0},
        'reg': {'opt_name': 'Adam', 'lr': 1e-4, 'weight_decay': 0}
    }
}

# --- Regularization variants (uncomment one) ---

# TV
# parameters['reg_p'] = {'reg_name': 'TV', 'coef': 1e-2, 'p_norm': 1}

# STV
# parameters['reg_p'] = {'reg_name': 'STV', 'coef': 2e-2, 'p_norm': 1}

# AIR (row + col)
# parameters['reg_p'] = {'reg_name': 'MultiReg', 'reg_list': [
#     {'reg_name': 'AIR', 'coef': 1e-3, 'n': 256, 'mode': 0},
#     {'reg_name': 'AIR', 'coef': 1e-3, 'n': 256, 'mode': 1}]}

# INRR+ (row + col, INRR with TV nabla blend)
# parameters['reg_p'] = {'reg_name': 'MultiReg', 'reg_list': [
#     {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 0, 'w0_initial': 1.,
#      'lap_k': lap_k, 'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k},
#     {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 1, 'w0_initial': 1.,
#      'lap_k': lap_k, 'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k}]}

# INRR+ with Huber (row + col) — recommended
parameters['reg_p'] = {'reg_name': 'MultiReg', 'reg_list': [
    {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 0, 'w0_initial': 1.,
     'lap_k': lap_k, 'lap_mode': 'Huber', 'huber_delta': huber_delta,
     'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k,
     'inr_parameter': {'dim_in': 1, 'dim_out': 100, 'w0_initial': 20.}},
    {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 1, 'w0_initial': 1.,
     'lap_k': lap_k, 'lap_mode': 'Huber', 'huber_delta': huber_delta,
     'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k,
     'inr_parameter': {'dim_in': 1, 'dim_out': 100, 'w0_initial': 20.}}]}

# INRR+ with logcosh (row + col)
# parameters['reg_p'] = {'reg_name': 'MultiReg', 'reg_list': [
#     {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 0, 'w0_initial': 1.,
#      'lap_k': lap_k, 'lap_mode': 'logcosh', 'huber_delta': 0.3,
#      'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k},
#     {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 1, 'w0_initial': 1.,
#      'lap_k': lap_k, 'lap_mode': 'logcosh', 'huber_delta': 0.3,
#      'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k}]}

# GroupReg (patch-level INRR with kmeans grouping)
# parameters['reg_p'] = {'reg_name': 'GroupReg', 'coef': 1e-2,
#     'group_para': {'n_clusters': 22, 'metric': 'cosine', 'reg_mode': 'single'},
#     'each_reg_name': 'INRR', 'start_epoch': 100, 'gpu_id': 0, 'w0_initial': 1.,
#     'x_trans': 'patch', 'stride': 13, 'patch_size': 16,
#     'search_epoch': 100, 'filter_type': None, 'sigma': 1.0, 'lap_k': 3}

rssnet = rss.rssnet(parameters, verbose=False)
rssnet.show_p['show_content'] = 'recovered'
for i in range(10):
    rssnet.train(verbose=False)
    psnr = max(rssnet.log_dict['psnr'])
    nmae = min(rssnet.log_dict['nmae'])
    print(f'PSNR: {psnr:.2f}, NMAE: {nmae:.4f}')
```

## Image Denoising

```python
import rss

size = 256
noise_parameter = 15  # sigma = 5, 10, 15, 20
inrr_alpha = 0.2
nabla_matrix_order_k = 1
lap_k = 1
huber_delta = 0.2

parameters = {
    'net_p': {
        'gpu_id': 0,
        'net_name': 'composition',
        'net_list': [{'net_name': 'SIREN', 'dim_in': 2, 'w0_initial': 30, 'dim_hidden': size}]
    },
    'data_p': {
        'data_shape': (size, size), 'random_rate': 0., 'pre_full': True,
        'mask_type': 'random', 'data_path': 'data/img/Barbara.jpg',
        'data_type': 'gray_img', 'ymode': 'denoising',
        'noise_mode': 'gaussian', 'noise_parameter': noise_parameter
    },
    'train_p': {'train_epoch': 30000},
    'opt_p': {
        'net': {'opt_name': 'Adam', 'lr': 1e-3, 'weight_decay': 0},
        'reg': {'opt_name': 'Adam', 'lr': 1e-4, 'weight_decay': 0}
    }
}

# Same regularization variants as completion (see above)
# INRR+ with Huber (row + col) — recommended
parameters['reg_p'] = {'reg_name': 'MultiReg', 'reg_list': [
    {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 0, 'w0_initial': 1.,
     'lap_k': lap_k, 'lap_mode': 'Huber', 'huber_delta': huber_delta,
     'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k,
     'inr_parameter': {'dim_in': 1, 'dim_out': 100, 'w0_initial': 20.}},
    {'reg_name': 'INRR', 'coef': 1e-2, 'n': size, 'mode': 1, 'w0_initial': 1.,
     'lap_k': lap_k, 'lap_mode': 'Huber', 'huber_delta': huber_delta,
     'inrr_alpha': inrr_alpha, 'nabla_matrix_order_k': nabla_matrix_order_k,
     'inr_parameter': {'dim_in': 1, 'dim_out': 100, 'w0_initial': 20.}}]}

rssnet = rss.rssnet(parameters, verbose=False)
rssnet.train(verbose=False)
psnr = max(rssnet.log_dict['psnr'])
print(f'PSNR: {psnr:.2f} dB')
```

## Using FRINR with INRR

FRINR can be used as a drop-in replacement for SIREN. Simply change the `net_list` parameter:

```python
# ReLU + Fourier reparameterization (recommended)
net_list = [{'net_name': 'FRINR', 'mode': 'relu+fr', 'dim_in': 2,
             'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
             'high_freq_num': 128, 'low_freq_num': 128, 'phi_num': 32, 'alpha': 0.05}]

# SIREN + Fourier reparameterization
net_list = [{'net_name': 'FRINR', 'mode': 'sin+fr', 'dim_in': 2,
             'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
             'high_freq_num': 128, 'low_freq_num': 128, 'phi_num': 32,
             'alpha': 0.01, 'first_omega_0': 30.0, 'hidden_omega_0': 30.0}]

# Low-frequency bases only (stronger implicit smoothing)
net_list = [{'net_name': 'FRINR', 'mode': 'relu+fr', 'dim_in': 2,
             'dim_hidden': 256, 'dim_out': 1, 'num_layers': 4,
             'high_freq_num': 0, 'low_freq_num': 128, 'phi_num': 32, 'alpha': 0.05}]
```

## Available INRR Laplacian Modes

| lap_mode | Description | Key Parameter |
|----------|-------------|---------------|
| `nuclear` (default) | Nuclear norm | — |
| `Huber` | Huber norm | `huber_delta` (default 0.2) |
| `logcosh` | LogCosh norm | `huber_delta` (default 0.3) |
| `quantile` | Quantile norm | `quantile_q` (default 0.5) |
| `lp` | Lp norm | `norm_lap_lp` (default 2) |


