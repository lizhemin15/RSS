import torch
import torch.nn.functional as F

def extract_patches(input_tensor, patch_size, stride, return_type = 'patch', conv_mode = False):
    # 获取输入张量的形状
    h, w = input_tensor.shape

    # 使用unfold函数分解为补丁
    patches = input_tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)

    # 获取补丁的形状
    ph, pw = patches.shape[0], patches.shape[1]

    # 将补丁展开为2D矩阵
    patches = patches.contiguous().view(-1, patch_size, patch_size)

    if conv_mode:
        # 生成随机卷积核，确保输入输出通道数相同
        kernel_size = 4
        random_kernel = torch.randn(1, 1, kernel_size, kernel_size)
        
        # 在下采样前进行卷积操作，填充使输出形状与输入形状一致
        padding = kernel_size // 2
        patches = patches.view(-1, 1, patch_size, patch_size)  # 展开为2D卷积输入形状
        patches = F.conv2d(patches, random_kernel, padding=padding)
        patches = patches.view(1, 1, ph, pw, patch_size, patch_size)  # 恢复形状

        # 进行平均降采样四倍
        scale = 4
        patches = patches.view(ph, pw, patch_size // scale, scale, patch_size // scale, scale).mean(dim=(3, 5)).contiguous().view(-1, patch_size // scale, patch_size // scale)

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

def conv_tensor(input_tensor, kernel, padding = 0, stride = 1):
    pass