import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os  # 添加这行导入

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        # 修改padding计算以确保更好的重叠
        self.padding = patch_size // 2
        
        # 计算输出特征图大小
        self.H = self.W = (img_size + 2 * self.padding - patch_size) // stride + 1
        self.num_patches = self.H * self.W
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, 
                             stride=stride,
                             padding=self.padding)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, img_size, patch_size, stride, embed_dim, out_chans=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        self.padding = patch_size // 2
        self.H = self.W = (img_size + 2 * self.padding - patch_size) // stride + 1
        
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_chans)
        self.upsample = nn.Upsample(size=(img_size, img_size), 
                                  mode='bicubic', 
                                  align_corners=False)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                     h=self.H, w=self.W, p1=self.patch_size, p2=self.patch_size)
        
        if x.size(-1) != self.img_size:
            x = self.upsample(x)
        return x

class TransformerDIP(nn.Module):
    def __init__(self, img_size=256, patch_size=16, stride=8, in_chans=3, 
                 embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.):
        super().__init__()
        
        self.img_size = img_size
        self.in_chans = in_chans
        
        # 从4x4开始
        init_size = img_size // 64  # 4x4
        self.noise = nn.Parameter(torch.randn(1, in_chans, init_size, init_size))
        
        # 使用5x5的高斯核
        gaussian_kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32) / 256.0
        
        self.gaussian_kernel = nn.Parameter(
            gaussian_kernel.view(1, 1, 5, 5).repeat(in_chans, 1, 1, 1),
            requires_grad=False
        )
        
        # 更细致的上采样步骤
        self.target_sizes = [
            img_size // 32,  # 8x8
            img_size // 16,  # 16x16
            img_size // 8,   # 32x32
            img_size // 4,   # 64x64
            img_size // 2,   # 128x128
            img_size        # 256x256
        ]
        
        # 每个尺度的高斯模糊次数
        self.blur_times = 2
        
        # 其他层保持不变
        self.patch_embed = PatchEmbed(img_size, patch_size, stride, in_chans, embed_dim)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.patch_merge = PatchMerging(img_size, patch_size, stride, embed_dim, in_chans)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def apply_gaussian_blur(self, x):
        """多次应用高斯模糊"""
        for _ in range(self.blur_times):
            x = F.conv2d(
                x,
                self.gaussian_kernel,
                padding=2,  # 5x5核需要padding=2
                groups=self.in_chans
            )
        return x
    
    def forward(self,x_in, noise_scale=0.1):
        x = self.noise
        
        # 逐步上采样和多次平滑
        for target_size in self.target_sizes:
            # 上采样
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
            
            # 多次应用高斯模糊
            x = self.apply_gaussian_blur(x)
        
        # 应用noise scale
        x = x * noise_scale
        
        # 后续处理保持不变
        x = self.patch_embed(x)
        for blk in self.transformer:
            x = blk(x)
        x = self.patch_merge(x)
        return x
    

def TIP(parameter):
    de_para_dict = {'img_size': 256, 'patch_size': 16,'stride': 8, 'in_chans': 3, 'embed_dim': 256, 'depth': 12, 'num_heads': 8,'mlp_ratio': 4.}
    parameter.update(de_para_dict)
    return TransformerDIP(img_size=parameter['img_size'], patch_size=parameter['patch_size'], stride=parameter['stride'], in_chans=parameter['in_chans'],
                          embed_dim=parameter['embed_dim'], depth=parameter['depth'], num_heads=parameter['num_heads'], mlp_ratio=parameter['mlp_ratio'])
    