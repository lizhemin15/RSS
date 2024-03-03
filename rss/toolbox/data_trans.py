def extract_patches(input_tensor, patch_size, stride, return_type = 'patch'):
    # 获取输入张量的形状
    h, w = input_tensor.shape

    # 使用unfold函数分解为补丁
    patches = input_tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)

    # 获取补丁的形状
    ph, pw = patches.shape[0], patches.shape[1]

    # 将补丁展开为2D矩阵
    patches = patches.contiguous().view(-1, patch_size, patch_size)
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

