from rss.represent.unn.models.resnet import ResNet
from rss.represent.unn.models.unet import UNet
from rss.represent.unn.models.skip import skip
import torch


def UNN(parameter):
    de_para_dict = {'net_name':'UNet',
                    'input_depth': 2,
                    'pad' : 'reflection',
                    'num_output_channels':3}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    input_depth = parameter.get('input_depth', 2)
    pad = parameter.get('pad', 'reflection')
    num_output_channels = parameter.get('num_output_channels', 3)
    net_name = parameter.get('net_name', 'UNet')
    if net_name == 'UNet':
        return UNet(num_input_channels=input_depth, num_output_channels=num_output_channels, 
                   feature_scale=8, more_layers=1, 
                   concat_x=False, upsample_mode='deconv', 
                   pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)
    elif net_name == 'ResNet':
        return ResNet(input_depth, num_output_channels, 8, 32, need_sigmoid=True, act_fun='LeakyReLU')
    elif net_name == 'skip':
        return skip(input_depth, num_output_channels, 
                    num_channels_down = [128] * 5,
                    num_channels_up   = [128] * 5,
                    num_channels_skip = [0] * 5,  
                    upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    else:
        raise ValueError('net_name is not in the options.')
    

