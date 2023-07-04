import torch as t
import torch.nn as nn
import torch.nn.functional as F

def trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: B x d
    voxel_min_vertex: B x d
    voxel_max_vertex: B x d
    voxel_embedds: B x 2^d x d'
    '''
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x d

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

    return c

