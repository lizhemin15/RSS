import torch as t
import math
from rss.represent.utils import to_device

def gaussian_kernel(size, sigma):
    """生成一个高斯卷积核."""
    kernel = t.tensor([math.exp(-((x - size//2)**2) / (2*sigma**2)) for x in range(size)])
    kernel = kernel / kernel.sum()
    return kernel

def create_2d_gaussian_kernel(kernel_size, sigma):
    """生成一个2D高斯卷积核."""
    kernel_1d = gaussian_kernel(kernel_size, sigma)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d



def extract_patches(input_tensor, patch_size, stride, return_type = 'patch',
                    down_sample = False, filter_type = None, sigma = 1):
    # 获取输入张量的形状
    h, w = input_tensor.shape
    # 使用unfold函数分解为补丁
    patches = input_tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    # 获取补丁的形状
    ph, pw = patches.shape[0], patches.shape[1]
    # 将补丁展开为2D矩阵
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    if down_sample:
        # 如果为真，则沿着第二个维度和第三个维度的patch进行平均降采样四倍
        scale = 2
        patches = patches.view(ph, pw, patch_size // scale, scale, patch_size // scale, scale).mean(dim=(3, 5)).contiguous().view(-1, patch_size // scale, patch_size // scale)
    if filter_type == None:
        pass
    elif filter_type == 'gaussian':
        # 如果为'gaussian'，则对每个补丁进行高斯滤波
        filter = create_2d_gaussian_kernel(kernel_size=patches.shape[1], sigma=sigma).unsqueeze(0)
        filter = to_device(filter, patches.device)
        patches = filter*patches
    else:
        raise ValueError('Invalid filter type = ', filter_type)
    if return_type == 'patch':
        return patches
    else:
        return patches.view(patches.shape[0],patches.shape[1]*patches.shape[2])

def downsample_tensor(input_tensor, factor):
    # 获取输入张量的大小
    input_size = input_tensor.size()

    # 计算每个块的大小
    block_size = input_size[0] // factor

    # 重新塑造张量以便进行平均降采样
    input_tensor = input_tensor.view(block_size, factor, block_size, factor)

    # 对张量进行平均降采样
    output_tensor = input_tensor.mean(dim=(1, 3))

    return output_tensor


def A2lap(A_0):
    n = A_0.shape[0]
    device = A_0.device
    Ones = t.ones(n,1)
    I_n = t.from_numpy(np.eye(n)).to(t.float32)
    Ones = to_device(Ones,device)
    I_n = to_device(I_n,device)
    A_1 = A_0 * (t.mm(Ones,Ones.T)-I_n) # A_1 将中间的元素都归零，作为邻接矩阵
    L = -A_1+t.mm(A_1,t.mm(Ones,Ones.T))*I_n # A_2 将邻接矩阵转化为拉普拉斯矩阵
    return L