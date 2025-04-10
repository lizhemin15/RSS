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