import torch.nn as nn
import torch as t
import torch.nn.functional as F
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
from scipy.ndimage import gaussian_filter

import numpy as np


from rss.represent.inr import MLP,SIREN,WIRE,BACONS,FourierNets,GaborNets
from rss.represent.tensor import DMF,TF
from rss.represent.utils import reshape2
from rss.represent.interpolation import Interpolation
from rss.represent.unn import UNN
from rss.represent.kernel import KNN,TDKNN
from rss.represent.feature import FeatureMap,HashEmbedder,KATE_Embedder
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
    elif net_name == 'HashEmbedder':
        net = HashEmbedder(parameter)
    elif net_name == 'KATEEmbedder':
        net = KATE_Embedder(parameter)
    elif net_name in ['EFF_KAN','KAN', 'ChebyKAN']:
        net = get_kan(parameter)
    elif net_name == 'RecurrentINR':
        net = RecurrentINR(parameter)
    elif net_name == 'HashINR':
        net = HashINR(parameter)
    elif net_name == 'DINER':
        net = DINER(parameter)
    elif net_name == 'SIMINER':
        net = SIMINER(parameter)
    elif net_name == 'FFINR':
        net = FFINR(parameter)
    elif net_name == 'KATE':
        net = KATE(parameter)
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


class FFINR(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        ffm_para = parameter.get('FourierFeature_para',{'net_name':'FourierFeature','dim_out':100})
        ffm_para['dim_in'] = parameter.get('dim_in',2)
        ffm_para['gpu_id'] = None if 'gpu_id' not in parameter.keys() else parameter['gpu_id']
        dim_feature = ffm_para['dim_out']
        self.ffm_net = get_nn(ffm_para)

        inr_para = parameter.get('inr_para',{'net_name':'MLP'})
        inr_para['dim_out'] = parameter.get('dim_out',1)
        inr_para['dim_in'] = dim_feature
        self.net = get_nn(inr_para)

    def forward(self,x):
        x = self.ffm_net(x)
        x = self.net(x)
        return x
    
    def to(self, device):
        # Move the model to the specified device
        super().to(device)  # Call the parent's to() method
        self.ffm_net.to(device)  # Move ffm_net to device
        self.net.to(device)      # Move net to device
        return self  # Return self for chaining

class KATE(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        ffm_para = parameter.get('KATEEmbedder_para',{'net_name':'KATEEmbedder','order':0})
        ffm_para['dim_in'] = parameter.get('dim_in',2)
        ffm_para['gpu_id'] = None if 'gpu_id' not in parameter.keys() else parameter['gpu_id']
        dim_feature = ffm_para['dim_out']+1
        self.ffm_net = get_nn(ffm_para)

        inr_para = parameter.get('inr_para',{'net_name':'MLP'})
        inr_para['dim_out'] = parameter.get('dim_out',1)
        inr_para['dim_in'] = dim_feature
        self.net = get_nn(inr_para)

    def forward(self,x):
        x = self.ffm_net(x)
        x = self.net(x)
        return x
    
    def to(self, device):
        # Move the model to the specified device
        super().to(device)  # Call the parent's to() method
        self.ffm_net.to(device)  # Move ffm_net to device
        self.net.to(device)      # Move net to device
        return self  # Return self for chaining


class RecurrentINR(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.net = get_nn(parameter.get('subnet_name',{'net_name':'SIREN'}))
        self.recurrent_num = parameter.get('recurrent_num',1)
        dim_in = parameter.get('dim_in',2)
        dim_out = parameter.get('dim_out',1)
        self.dim_in = dim_in
        self.transform_matrix = nn.Parameter(t.randn(dim_in+dim_out,dim_in))
        # 定义权重向量
        self.weights = self.create_weights(dim_in, dim_out).to(parameter.get('gpu_id',None))

    def create_weights(self, dim_in, dim_out):
        # 创建一个权重向量，前 dim_in 行为 1，后 dim_out 行为 1/10
        weights = t.ones(dim_in + dim_out)
        weights[dim_in:] = 1e-3  # 后 dim_out 行设置为 1/10
        return weights

    def transform_xin(self, x_in):
        # 执行变换
        result = x_in @ (self.transform_matrix * self.weights.view(-1, 1))
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





class HashINR(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.encode_cor_if = parameter.get('encode_cor_if',True)
        self.hash_mode = parameter.get('hash_mode','vanilla')
        self.neighbor_num = parameter.get('neighbor_num',1)
        hash_para = parameter.get('hash_para',{'net_name':'HashEmbedder'})
        self.hash_func = get_nn(hash_para)
        n_levels = hash_para.get('n_levels', 16)  # 默认值为 16
        n_features_per_level = hash_para.get('n_features_per_level', 2)  # 默认值为 2

        inr_para = parameter.get('inr_para', {'net_name':'MLP','num_layers':2,'dim_hidden':16,'activation':'relu'})
        inr_para['dim_out'] = parameter.get('dim_out',1)
        if self.encode_cor_if:
            inr_para['dim_in'] = parameter.get('dim_in',2)
        else:
            inr_para['dim_in'] = 0
        if self.hash_mode == 'vanilla':
            inr_para['dim_in'] = n_levels*n_features_per_level+parameter.get('dim_in',2)
        elif self.hash_mode == 'patch':
            inr_para['dim_in'] = (self.neighbor_num*2+1)**2*n_levels*n_features_per_level+parameter.get('dim_in',2)
        self.net = get_nn(inr_para)


    def forward(self, x):
        # 检查 x 的维度
        if x.dim() == 3:
            x = x.squeeze(0)  # batchsize采样所得，去掉第一维
        if self.hash_mode == 'vanilla':
            if self.encode_cor_if:
                return self.net(t.cat([x,self.hash_func(x)],dim=-1))
            else:
                return self.net(self.hash_func(x))
        elif self.hash_mode == 'patch':
            x_now = t.clone(x)
            for i in range(self.neighbor_num*2+1):
                for j in range(self.neighbor_num*2+1):
                    delta_x = t.tensor([i-self.neighbor_num,j-self.neighbor_num]).view(1,2).to(x.device)/100
                    if self.encode_cor_if:
                        x_now = t.cat([x_now,self.hash_func(x+delta_x)],dim=-1)
                    else:
                        x_now = self.hash_func(x+delta_x)
            return self.net(x_now)



class DINER(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.border = parameter.get('border', 1)
        self.feature_dim = parameter.get('feature_dim', 1)
        self.resolution = parameter.get('resolution', 256)
        self.dim_in = parameter.get('dim_in', 2)


        # G的形状
        G_shape = [self.resolution] * self.dim_in + [self.feature_dim]
        self.G = nn.Parameter(t.randn(G_shape) * 1e-3)

        # 神经网络部分
        inr_para = parameter.get('inr_para', {'net_name': 'MLP'})
        inr_para['dim_out'] = parameter.get('dim_out', 1)
        inr_para['dim_in'] = self.feature_dim
        self.net = get_nn(inr_para)

    def trilinear_interp(self, lower_idx, upper_idx, weight):
        # 三线性插值
        return (
            self.G[lower_idx[:, 0], lower_idx[:, 1], lower_idx[:, 2]] * (1 - weight[:, 0][:, None]) * (1 - weight[:, 1][:, None]) * (1 - weight[:, 2][:, None]) +
            self.G[lower_idx[:, 0], lower_idx[:, 1], upper_idx[:, 2]] * (1 - weight[:, 0][:, None]) * (1 - weight[:, 1][:, None]) * weight[:, 2][:, None] +
            self.G[lower_idx[:, 0], upper_idx[:, 1], lower_idx[:, 2]] * (1 - weight[:, 0][:, None]) * weight[:, 1][:, None] * (1 - weight[:, 2][:, None]) +
            self.G[lower_idx[:, 0], upper_idx[:, 1], upper_idx[:, 2]] * (1 - weight[:, 0][:, None]) * weight[:, 1][:, None] * weight[:, 2][:, None] +
            self.G[upper_idx[:, 0], lower_idx[:, 1], lower_idx[:, 2]] * weight[:, 0][:, None] * (1 - weight[:, 1][:, None]) * (1 - weight[:, 2][:, None]) +
            self.G[upper_idx[:, 0], lower_idx[:, 1], upper_idx[:, 2]] * weight[:, 0][:, None] * (1 - weight[:, 1][:, None]) * weight[:, 2][:, None] +
            self.G[upper_idx[:, 0], upper_idx[:, 1], lower_idx[:, 2]] * weight[:, 0][:, None] * weight[:, 1][:, None] * (1 - weight[:, 2][:, None]) +
            self.G[upper_idx[:, 0], upper_idx[:, 1], upper_idx[:, 2]] * weight[:, 0][:, None] * weight[:, 1][:, None] * weight[:, 2][:, None]
        )

    def bilinear_interp(self, lower_idx, upper_idx, weight):
        # 双线性插值
        return (
            self.G[lower_idx[:, 0], lower_idx[:, 1]] * (1 - weight[:, 0][:, None]) * (1 - weight[:, 1][:, None]) +
            self.G[lower_idx[:, 0], upper_idx[:, 1]] * (1 - weight[:, 0][:, None]) * weight[:, 1][:, None] +
            self.G[upper_idx[:, 0], lower_idx[:, 1]] * weight[:, 0][:, None] * (1 - weight[:, 1][:, None]) +
            self.G[upper_idx[:, 0], upper_idx[:, 1]] * weight[:, 0][:, None] * weight[:, 1][:, None]
        )

    def interpolate(self, lower_idx, upper_idx, weight):
        if self.dim_in == 2:
            return self.bilinear_interp(lower_idx, upper_idx, weight)
        elif self.dim_in == 3:
            return self.trilinear_interp(lower_idx, upper_idx, weight)
        else:
            raise ValueError("dim_in must be either 2 or 3.")

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)  # 去掉第一维

        batch_size = x.shape[0]

        # 将坐标归一化到 [0, resolution]
        normalized_x = (x + self.border) / (2 * self.border) * (self.resolution - 1)

        # 计算下取整和上取整
        lower_idx = t.clamp(t.floor(normalized_x).long(), 0, self.resolution - 1)
        upper_idx = t.clamp(t.ceil(normalized_x).long(), 0, self.resolution - 1)

        # 计算权重
        weight = normalized_x - lower_idx.float()

        # 调用插值函数
        output = self.interpolate(lower_idx, upper_idx, weight)
        self.output = output
        return self.net(output)



class SIMINER(DINER):
    def __init__(self, parameter):
        super().__init__(parameter)
        # 计数参数
        self.forward_count = 0

        self.border = parameter.get('border', 1)
        self.resolution = parameter.get('resolution', 300)
        self.dim_in = parameter.get('dim_in', 2)
        self.feature_dim = parameter.get('feature_dim', 3)
        self.similar_method = parameter.get('similar_method', 'nlm')
        self.inr_input = parameter.get('inr_input', 'concat')

        # G的形状
        G_shape = [self.resolution] * self.dim_in + [self.feature_dim]
        self.G = nn.Parameter(t.randn(G_shape) * 1e-3)

        # 神经网络部分
        inr_para = parameter.get('inr_para', {'net_name':'MLP','num_layers':2,'dim_hidden':16,'activation':'relu'})
        inr_para['dim_out'] = parameter.get('dim_out', 1)
        if self.inr_input == 'concat':
            inr_para['dim_in'] = self.feature_dim+self.dim_in
        else:
            inr_para['dim_in'] = self.feature_dim
        self.net = get_nn(inr_para)

    def update_G(self):
        with t.no_grad():  # 禁用梯度计算
            # 1. 读取数据并转换为numpy数组
            G_numpy = self.G.detach().cpu().numpy()  # detach()用于避免梯度计算
            if self.similar_method == 'nlm':
                sigma_est = np.mean(estimate_sigma(G_numpy, channel_axis=-1))
                patch_kw = dict(
                    patch_size=5, patch_distance=6, channel_axis=-1  # 5x5 patches  # 13x13 search area
                )

                # 2. 在numpy中进行处理
                G_processed = denoise_nl_means(G_numpy, h=8 * sigma_est, fast_mode=True, **patch_kw)
                G_processed = gaussian_filter(G_processed, sigma=8 * sigma_est)
            elif self.similar_method == 'wavelet':
                G_processed = denoise_wavelet(
                                                G_numpy,
                                                channel_axis=-1,
                                                convert2ycbcr=True,
                                                method='BayesShrink',
                                                mode='soft',
                                                rescale_sigma=True,
                                            )


            # 3. 将处理后的numpy数组转换为PyTorch张量
            new_G = t.from_numpy(G_processed).float().to(self.G.device)

            # 4. 使用copy_来更新self.G的值
            self.G.data.copy_(new_G)

            # 5. 清除未使用的临时变量
            del new_G  # 如果不再需要，可以显式删除

            # 6. 清除未使用的缓存
            t.cuda.empty_cache()

    def forward(self, x):
        self.forward_count += 1
        if self.forward_count % 10 == 0 and self.forward_count < 500:
            self.update_G()
        if x.dim() == 3:
            x = x.squeeze(0)  # 去掉第一维

        batch_size = x.shape[0]

        # 将坐标归一化到 [0, resolution]
        normalized_x = (x + self.border) / (2 * self.border) * (self.resolution - 1)

        # 计算下取整和上取整
        lower_idx = t.clamp(t.floor(normalized_x).long(), 0, self.resolution - 1)
        upper_idx = t.clamp(t.ceil(normalized_x).long(), 0, self.resolution - 1)

        # 计算权重
        weight = normalized_x - lower_idx.float()

        # 调用插值函数
        output = self.interpolate(lower_idx, upper_idx, weight)
        self.output = output
        if self.inr_input == 'concat':
            return self.net(t.cat((output, x), dim=-1))
        else:
            return self.net(output)





