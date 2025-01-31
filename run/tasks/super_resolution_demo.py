import rss
import os

def basic_super_resolution():
    """Basic super resolution example"""
    result = rss.task.run('super_resolution',
                         data_path='data/img/low_res.jpg',
                         output_path='results/high_res.jpg',
                         scale_factor=4)  # 4x upscaling
    
    result['model'].show()

def advanced_super_resolution():
    """Advanced super resolution with custom settings"""
    result = rss.task.run(
        'super_resolution',
        data_path='data/img/low_res.jpg',
        output_path='results/high_res_advanced.jpg',
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