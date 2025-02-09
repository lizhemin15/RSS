import torch 
import torch.nn as nn
import torchvision
import sys
from PIL import Image
import PIL
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def convert(img):
    result = img * 255
    result = result * (result > 0)
    result = result * (result <= 255) + 255 * (result > 255)
    result = result.astype(np.uint8)
    return result

def get_coords(H, W):
    def linspace_func(nx): return torch.linspace(-1.0, 1.0, nx)
    linspaces = map(linspace_func, (H, W))
    coordinates = torch.meshgrid(*linspaces, indexing='ij')
    coords = torch.stack(coordinates, dim=-1)
    return coords.flatten(0, -2)

def apply_f(x, m):
    d = x.shape[2]
    if x.shape[1] == 3:
        (r, g, b) = torch.split(x, [1, 1, 1], dim = 1)

        r_, g_, b_ = torch.fft.fftn(F.pad(r, (0, m - d, 0, m - d), "constant", 0)),\
                     torch.fft.fftn(F.pad(g, (0, m - d, 0, m - d), "constant", 0)), \
                     torch.fft.fftn(F.pad(b, (0, m - d, 0, m - d), "constant", 0))
        y = torch.cat((torch.abs(r_), torch.abs(g_), torch.abs(b_)), 1)
    else:
        y = torch.fft.fftn(F.pad(x, (0, m - d, 0, m - d), "constant", 0))
        y = torch.abs(y)
    return y

def apply_f_(Ameas, x, m):
    set_random_seed(42)
    d = x.shape[2]
    if x.shape[1] == 3:
        (r, g, b) = torch.split(x, [1, 1, 1], dim = 1)
        r_, g_, b_ = torch.matmul(Ameas, r.reshape(x.numel(), 1)),\
                     torch.matmul(Ameas, g.reshape(x.numel(), 1)), \
                     torch.matmul(Ameas, b.reshape(x.numel(), 1))
        y = torch.cat((torch.abs(r_), torch.abs(g_), torch.abs(b_)), 1)
    else:
        y = torch.matmul(Ameas, x.reshape(x.numel(), 1))
        y = torch.abs(y)
    return y

def fftn(x, m):
    d = x.shape[2]
    if x.shape[1] == 3:
        (r, g, b) = torch.split(x, [1, 1, 1], dim=1)
        r_, g_, b_ = torch.fft.fftn(F.pad(r, (0, m - d, 0, m - d), "constant", 0)), \
                     torch.fft.fftn(F.pad(g, (0, m - d, 0, m - d), "constant", 0)), \
                     torch.fft.fftn(F.pad(b, (0, m - d, 0, m - d), "constant", 0))
        y = torch.cat((r_, g_, b_), 1)
    else:
        y = torch.fft.fftn(F.pad(x, (0, m - d, 0, m - d), "constant", 0))
    return y

def ifftn(x):
    if x.shape[1] == 1:
        y = torch.fft.ifftn(x).real
    elif x.shape[1] == 3:
        (r, g, b) = torch.split(x, [1, 1, 1], dim = 1)
        r_, g_, b_ = torch.fft.ifftn(r).real, \
                     torch.fft.ifftn(g).real, \
                     torch.fft.ifftn(b).real
        y = torch.cat((r_, g_, b_), 1)
    return y

def np_to_tensor(img_np):
    #Converts image in numpy.array to torch.Tensor from C x W x H [0..1] to  C x W x H [0..1]
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    #Converts image in numpy.array to torch.Variable from C x W x H [0..1] to  1 x C x W x H [0..1]
    return Variable(np_to_tensor(img_np)[None, :])

def pil_to_np(img_PIL):
    #Converts image in PIL format to np.array from W x H x C [0...255] to C x W x H [0..1]
    ar = np.array(img_PIL)
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]
    ar = ar.astype(np.float32)
    return ar / 255.