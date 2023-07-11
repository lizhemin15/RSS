import torch as t
import torch.nn as nn
import torch.nn.functional as F

def linear_interp(x=None, voxel_min_vertex=None, voxel_max_vertex=None,voxel_embedds=None):
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
        
    Final return:
        voxel_embedds: Tensor of shape B x d' after interpolation.
    """
    # 获取输入张量的维度
    d = x.shape[1]
    # 递归终止条件，当维度d为1时，直接返回结果
    if d == 0:
        return t.squeeze(voxel_embedds, dim=1)
    weights = ((x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex))[:,-1][:,None].reshape(-1,1,1) # B,
    # voxel_embedds B,2^(d-1),d'
    # index二分加权求和
    new_voxel_embedds = voxel_embedds[:,:2**(d-1)]*(1-weights) + voxel_embedds[:,2**(d-1):]*weights # B,2^(d-1),d'
    new_voxel_min_vertex = voxel_min_vertex[:,:-1] # B x (d-1)
    new_voxel_max_vertex = voxel_max_vertex[:,:-1] # B x (d-1)
    new_x = x[:,:-1] # B x (d-1)

    return linear_interp(x=new_x, voxel_min_vertex=new_voxel_min_vertex, 
                         voxel_max_vertex=new_voxel_max_vertex,voxel_embedds=new_voxel_embedds)
    




