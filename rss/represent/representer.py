from rss.represent.inr import MLP,SIREN,WIRE,BACONS,FourierNets,GaborNets
import torch.nn as nn
import torch as t
from rss.represent.tensor import DMF,TF
from rss.represent.utils import reshape2
from rss.represent.interpolation import Interpolation
from rss.represent.unn import UNN
from rss.represent.kernel import KNN,TDKNN
from rss.represent.feature import FeatureMap
from rss.represent.kan import get_kan

def get_nn(parameter={}):
    net_name = parameter.get('net_name','SIREN')
    clip_if = parameter.get('clip_if',False)
    if net_name == None:
        net_name = 'None'
    if net_name == 'composition':
        net = Composition(parameter)
    elif net_name == 'MLP':
        net = MLP(parameter)
    elif net_name == 'SIREN':
        net = SIREN(parameter)
    elif net_name == 'WIRE':
        net = WIRE(parameter)
    elif net_name == 'BACON':
        net = BACONS(parameter)
    elif net_name == 'FourierNet':
        net = FourierNets(parameter)
    elif net_name == 'GaborNet':
        net = GaborNets(parameter)
    elif net_name == 'DMF':
        net = DMF(parameter)
    elif net_name == 'TF':
        net = TF(parameter)
    elif net_name == 'Interpolation':
        net = Interpolation(parameter)
    elif net_name in ['UNet','ResNet','skip']:
        net = UNN(parameter)
    elif net_name == 'KNN':
        net = KNN(parameter)
    elif net_name == 'TDKNN':
        net = TDKNN(parameter)
    elif net_name == 'FourierFeature':
        net = FeatureMap(parameter)
    elif net_name in ['EFF_KAN','KAN', 'ChebyKAN']:
        net = get_kan(parameter)
    elif net_name == 'RecurrentINR':
        net = RecurrentINR(parameter)
    else:
        raise ValueError(f'Wrong net_name = {net_name}')
    if clip_if==False:
        return net
    else:
        clip_min = parameter.get('clip_min',0.0)
        clip_max = parameter.get('clip_max',1.0)
        clip_mode = parameter.get('clip_mode','hard')
        return nn.Sequential(
                net,
                ClipLayer(clip_min, clip_max, clip_mode)
            )

class ClipLayer(nn.Module):
    def __init__(self, min_val, max_val, clip_mode='hard'):
        super(ClipLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.clip_mode = clip_mode

    def forward(self, x):
        if self.clip_mode == 'hard':
            return t.clamp(x, min=self.min_val, max=self.max_val)
        elif self.clip_mode == 'tanh':
            # 将输入值归一化到 [-1, 1]
            x_norm = 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1
            # 通过 tanh 函数进行非线性变换
            x_tanh = t.tanh(x_norm)
            # 将 tanh 的输出缩放到 [min_val, max_val]
            x_scaled = (x_tanh + 1) / 2 * (self.max_val - self.min_val) + self.min_val
            return x_scaled
        else:
            raise ValueError(f"Unsupported clip_mode: {self.clip_mode}")

class Composition(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.net_list_para = parameter.get('net_list',[{'net_name':'SIREN'}])
        net_list = []
        for _,net_para in enumerate(self.net_list_para):
            net_para['gpu_id'] = None if 'gpu_id' not in parameter.keys() else parameter['gpu_id']
            net_list.append(get_nn(net_para))
        self.net_list = nn.ModuleList(net_list)

    def forward(self, x_in):
        for i,net in enumerate(self.net_list):
            if i == 0:
                x = net(x_in)
                continue
            if self.net_list_para[i]['net_name'] == 'Interpolation':
                x = net(x=x_in,tau=x)
            else:
                x = net(x)
        return x
        
class Contenate(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.net_list_para = parameter.get('net_list',[{'net_name':'SIREN'}])
        net_list = []
        for _,net_para in enumerate(self.net_list_para):
            net_list.append(get_nn(net_para))
        self.net_list = nn.ModuleList(net_list)


    def forward(self,x_list):
        # Contenate multiple input together to a single net
        pass






class RecurrentINR(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.net = get_nn(parameter.get('subnet_name',{'net_name':'SIREN'}))
        self.recurrent_num = parameter.get('recurrent_num',1)
        dim_in = parameter.get('dim_in',2)
        dim_out = parameter.get('dim_out',1)
        self.dim_in = dim_in
        self.transform_matrix = nn.Parameter(t.randn(dim_in+dim_out,dim_in))


    def transform_xin(self, x_in):
        # 执行变换
        self.transform_matrix[:self.dim_in] /= 10
        result = x_in @ self.transform_matrix
        
        # 计算最大值和最小值
        result_min = t.min(result)
        result_max = t.max(result)
        
        # 归一化到 -1 到 1
        normalized_result = 2 * (result - result_min) / (result_max - result_min) - 1
        
        return normalized_result

    def forward(self, x_in):
        for _ in range(self.recurrent_num):
            x = self.net(x_in)
            x_in = self.transform_xin(t.cat([x_in,x],dim=-1))
        return x



















