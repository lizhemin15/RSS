![](./statics/logo.jpg)

# What's RSS?
Using Pytorch to represent both the signal and regularization term, and then solve the inverse problems.

# How to install
```
git clone https://gitee.com/lizhemin15/RSS.git
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


