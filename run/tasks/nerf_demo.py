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