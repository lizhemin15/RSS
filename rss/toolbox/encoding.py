import torch as t
import torch.nn as nn
import torch.nn.functional as F

def linear_interp(voxel_embedds=None, weights=None, x=None, voxel_min_vertex=None, voxel_max_vertex=None):
    """
    Performs linear interpolation.

    Args:
        x: Input tensor of shape B x d.
        voxel_min_vertex: Tensor of shape B x d representing the minimum vertex of each voxel.
        voxel_max_vertex: Tensor of shape B x d representing the maximum vertex of each voxel.
        voxel_embedds: Tensor of shape B x 2^d x d' representing the voxel embeddings.

    Returns:
        x: Tensor of shape B x (d-1) after interpolation.
        voxel_min_vertex: Tensor of shape B x (d-1) after interpolation.
        voxel_max_vertex: Tensor of shape B x (d-1) after interpolation.
        voxel_embedds: Tensor of shape B x 2^(d-1) x d' after interpolation.
    """

    # 获取输入张量的维度
    if x is None:
        d = weights.shape[1]
    else:
        d = x.shape[1]


    # 递归终止条件，当维度d为1时，直接返回结果
    if d == 1:
        return x, voxel_min_vertex, voxel_max_vertex, voxel_embedds
    
    weights = ((x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex))[:,-1][:,None] # B,
    # voxel_embedds B,2^(d-1),d'
    # index二分加权求和
    new_voxel_embedds = voxel_embedds[:,:2^(d-1)]*(1-weights) + voxel_embedds[:,2^(d-1):]*weights # B,2^(d-1),d'

    # step 1
    # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
    c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
    c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
    c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
    c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

    # step 2
    c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
    c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

    # step 3
    c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]
    return new_x, new_voxel_min_vertex, new_voxel_max_vertex, new_voxel_embedds
    




