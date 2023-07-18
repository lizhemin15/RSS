import rss
import torch

import configargparse
parser = configargparse.ArgumentParser()
parser.add_argument("--expname", type=str,default='chair',
                    help='experiment name')
args = parser.parse_args()

nerf_parameter_dict = rss.toolbox.load_json('./json/nerf/'+args.expname+'.json')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
rss.nerf(nerf_parameter_dict)