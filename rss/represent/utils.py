import torch.nn.functional as F
import torch as t
from torch import nn
from einops import rearrange
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve
from bm3d import gaussian_kernel

abc_str = 'abcdefghijklmnopqrstuvwxyz'


def get_act(act,coef_cos=1,coef_exp=1):
    act_dict = {
        'relu': F.relu,
        'sigmoid': F.sigmoid,
        'tanh': t.tanh,
        'softmax': F.softmax,
        'threshold': F.threshold,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'relu6': F.relu6,
        'leaky_relu': F.leaky_relu,
        'prelu': F.prelu,
        'rrelu': F.rrelu,
        'logsigmoid': F.logsigmoid,
        'hardshrink': F.hardshrink,
        'tanhshrink': F.tanhshrink,
        'softsign': F.softsign,
        'softplus': F.softplus,
        'softmin': F.softmin,
        'log_softmax': F.log_softmax,
        'softshrink': F.softshrink,
        'sin': t.sin,
        'identity': nn.Identity(),
        'exp': t.exp,
        'gabor': lambda x: t.cos(x*coef_cos) * t.exp(x*coef_exp)
    }
    
    if act in act_dict:
        return act_dict[act]
    else:
        print('Wrong act name:', act)
        return None



def get_opt(opt_type='Adam', parameters=None, lr=1e-3, weight_decay=0):
    # 初始化网络参数的优化器

    optimizer_dict = {
        'Adadelta': t.optim.Adadelta,
        'Adagrad': t.optim.Adagrad,
        'Adam': t.optim.Adam,
        'RegAdam': t.optim.Adam,
        'AdamW': t.optim.AdamW,
        'SparseAdam': t.optim.SparseAdam,
        'Adamax': t.optim.Adamax,
        'ASGD': t.optim.ASGD,
        'LBFGS': t.optim.LBFGS,
        'SGD': t.optim.SGD,
        'NAdam': t.optim.NAdam,
        'RAdam': t.optim.RAdam,
        'RMSprop': t.optim.RMSprop,
        'Rprop': t.optim.Rprop,
    }

    if opt_type == 'Lion':
        from lion_pytorch import Lion
        optimizer_dict['Lion'] = Lion

    if opt_type in optimizer_dict:
        optimizer_class = optimizer_dict[opt_type]
        optimizer = optimizer_class(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Wrong optimization type')
    
    return optimizer


def to_device(obj,device):
    if t.cuda.is_available() and device != 'cpu':
        obj = obj.cuda(device)
    return obj

def reshape2(data):
    # 
    xshape = data.shape
    einstr = add_space(abc_str[:len(xshape)])+' -> ('+add_space(abc_str[:len(xshape)])+') ()'
    return rearrange(data,einstr)

def add_space(oristr):
    addstr = ''
    for i in range(len(oristr)):
        addstr += oristr[i]
        addstr += ' '
    return addstr


def patchify(data, patch_size=32, stride=1):
    patches = data.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    # 将patches展平，并移动维度顺序为 (num_patches, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches


def gaussian_kernel(k, sigma=1.0):
    # 创建一个高斯核
    ax = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def pad_with_zeros(matrix, pad_width):
    # 用0填充矩阵
    return np.pad(matrix, pad_width, mode='constant', constant_values=0)


def extract_patches(M_conv, n, k):
    patches = []
    for i in range(2 * k):
        for j in range(2 * k):
            patch = M_conv[i:i + n, j:j + n]
            patches.append(patch)
    return np.stack(patches)



"""
Define functions needed for the demos.
bm3d
"""




def get_psnr(y_est: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Return PSNR value for y_est and y_ref presuming the noise-free maximum is 1.
    :param y_est: Estimate array
    :param y_ref: Noise-free reference
    :return: PSNR value
    """
    return 10 * np.log10(1 / np.mean(((y_est - y_ref).ravel()) ** 2))


def get_cropped_psnr(y_est: np.ndarray, y_ref: np.ndarray, crop: tuple) -> float:
    """
    Return PSNR value for y_est and y_ref presuming the noise-free maximum is 1.
    Crop the images before calculating the value by crop.
    :param y_est: Estimate array
    :param y_ref: Noise-free reference
    :param crop: Tuple of crop-x and crop-y from both stides
    :return: PSNR value
    """
    return get_psnr(np.atleast_3d(y_est)[crop[0]:-crop[0], crop[1]:-crop[1], :],
                    np.atleast_3d(y_ref)[crop[0]:-crop[0], crop[1]:-crop[1], :])


def get_experiment_kernel(noise_type: str, noise_var: float, sz: tuple = np.array((101, 101))):
    """
    Get kernel for generating noise from specific experiment from the paper.
    :param noise_type: Noise type string, g[0-4](w|)
    :param noise_var: noise variance
    :param sz: size of image, used only for g4 and g4w
    :return: experiment kernel with the l2-norm equal to variance
    """
    # if noiseType == gw / g0
    kernel = np.array([[1]])
    noise_types = ['gw', 'g0', 'g1', 'g2', 'g3', 'g4', 'g1w', 'g2w', 'g3w', 'g4w']
    if noise_type not in noise_types:
        raise ValueError("Noise type must be one of " + str(noise_types))

    if noise_type != "g4" and noise_type != "g4w":
        # Crop this size of kernel when generating,
        # unless pink noise, in which
        # if noiseType == we want to use the full image size
        sz = np.array([101, 101])
    else:
        sz = np.array(sz)

    # Sizes for meshgrids
    sz2 = -(1 - (sz % 2)) * 1 + np.floor(sz / 2)
    sz1 = np.floor(sz / 2)
    uu, vv = np.meshgrid([i for i in range(-int(sz1[0]), int(sz2[0]) + 1)],
                         [i for i in range(-int(sz1[1]), int(sz2[1]) + 1)])

    beta = 0.8

    if noise_type[0:2] == 'g1':
        # Horizontal line
        kernel = np.atleast_2d(16 - abs(np.linspace(1, 31, 31) - 16))

    elif noise_type[0:2] == 'g2':
        # Circular repeating pattern
        scale = 1
        dist = uu ** 2 + vv ** 2
        kernel = np.cos(np.sqrt(dist) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == 'g3':
        # Diagonal line pattern kernel
        scale = 1
        kernel = np.cos((uu + vv) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == 'g4':
        # Pink noise
        dist = uu ** 2 + vv ** 2
        n = sz[0] * sz[1]
        spec = (np.sqrt((np.sqrt(n) * 1e-2) / (np.sqrt(dist) + np.sqrt(n) * 1e-2)))
        kernel = fftshift(ifft2(ifftshift(spec)))

    else:  # gw and g0 are white
        beta = 0

    # -- Noise with additional white component --

    if len(noise_type) > 2 and noise_type[2] == 'w':
        kernel = kernel / np.sqrt(np.sum(kernel ** 2))
        kalpha = np.sqrt((1 - beta) + beta * abs(fft2(kernel, (sz[0], sz[1]))) ** 2)
        kernel = fftshift(ifft2(kalpha))

    kernel = np.real(kernel)
    # Correct variance
    kernel = kernel / np.sqrt(np.sum(kernel ** 2)) * np.sqrt(noise_var)

    return kernel


def get_experiment_noise(noise_type: str, noise_var: float, realization: int, sz: tuple)\
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generate noise for experiment with specified kernel, variance, seed and size.
    Return noise and relevant parameters.
    The generated noise is non-circular.
    :param noise_type: Noise type, see get_experiment_kernel for list of accepted types.
    :param noise_var: Noise variance of the resulting noise
    :param realization: Seed for the noise realization
    :param sz: image size -> size of resulting noise
    :return: noise, PSD, and kernel
    """
    np.random.seed(realization)

    # Get pre-specified kernel
    kernel = get_experiment_kernel(noise_type, noise_var, sz)

    # Create noisy image
    half_kernel = np.ceil(np.array(kernel.shape) / 2)

    if len(sz) == 3 and half_kernel.size == 2:
        half_kernel = [half_kernel[0], half_kernel[1], 0]
        kernel = np.atleast_3d(kernel)

    half_kernel = np.array(half_kernel, dtype=int)

    # Crop edges
    noise = fftconvolve(np.random.normal(size=(sz + 2 * half_kernel)), kernel, mode='same')
    noise = np.atleast_3d(noise)[half_kernel[0]:-half_kernel[0], half_kernel[1]:-half_kernel[1], :]

    psd = abs(fft2(kernel, (sz[0], sz[1]), axes=(0, 1))) ** 2 * sz[0] * sz[1]

    return noise, psd, kernel
