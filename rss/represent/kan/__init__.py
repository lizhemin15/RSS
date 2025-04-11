from rss.represent.kan.eff_kan import KAN as EFF_KAN
from rss.represent.kan.fast_kan import FastKAN

__all__ = ["EFF_KAN","FastKAN"]


def get_kan(parameter):
    de_para_dict = {'dim_in':2,'dim_hidden':100,'dim_out':1,'num_layers':4,
                    'spline_type':'spline','grid_size':5,'layer_norm':False}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    # print('KAN : ',parameter)
    if parameter.get('net_name','SIREN') == "EFF_KAN":
        return EFF_KAN([parameter['dim_in'],*[parameter['dim_hidden']]*parameter['num_layers'],
                      parameter['dim_out']],spline_type=parameter['spline_type'],grid_size=parameter['grid_size'],
                      layer_norm = parameter['layer_norm'])
    elif parameter.get('net_name','SIREN') == "FastKAN":
        return FastKAN(layers_hidden=[parameter['dim_in'],*[parameter['dim_hidden']]*parameter['num_layers']],
                       grid_min=-1,grid_max=1,
                       num_grids=parameter['grid_size'])
    else:
        raise NotImplementedError('net_name must be EFF_KAN or FastKAN')