import rss
import os

def basic_completion():
    """Basic image completion example"""
    # Simple one-line usage
    result = rss.task.run('completion',
                         data_path='data/img/example.jpg',
                         output_path='results/completed.jpg')
    
    # Show results
    result['model'].show()

def advanced_completion():
    """Advanced image completion with custom parameters"""
    # Custom network architecture using SIREN
    result = rss.task.run(
        'completion',
        data_path='data/img/example.jpg',
        output_path='results/completed_siren.jpg',
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
        output_path='results/completed_tensor.jpg',
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