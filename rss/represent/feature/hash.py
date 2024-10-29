import torch
import torch.nn as nn

# 设置全局变量 device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用全局 device 创建张量
BOX_OFFSETS_3D = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], 
                               device=device)

BOX_OFFSETS_2D = torch.tensor([[[i, j] for i in [0, 1] for j in [0, 1]]], 
                               device=device)


class HashEmbedder(nn.Module):
    def __init__(self, bounding_box=[-1, 1], n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution, device=device)
        self.finest_resolution = torch.tensor(finest_resolution, device=device)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size,
                                                      self.n_features_per_level).to(device) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]

        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def bilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 2

        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 2] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 3] * weights[:, 0][:, None]

        c = c00 * (1 - weights[:, 1][:, None]) + c01 * weights[:, 1][:, None]

        return c

    def forward(self, x):
        if x.shape[1] == 3:
            x_embedded_all = []
            for i in range(self.n_levels):
                resolution = torch.floor(self.base_resolution * self.b ** i)
                voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
                    x, self.bounding_box,
                    resolution, self.log2_hashmap_size, is_3d=True)

                voxel_embedds = self.embeddings[i](hashed_voxel_indices)

                x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
                x_embedded_all.append(x_embedded)

            keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
            return torch.cat(x_embedded_all, dim=-1)#, keep_mask

        elif x.shape[1] == 2:
            x_embedded_all = []
            for i in range(self.n_levels):
                resolution = torch.floor(self.base_resolution * self.b ** i)
                voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
                    x, self.bounding_box,
                    resolution, self.log2_hashmap_size, is_3d=False)

                voxel_embedds = self.embeddings[i](hashed_voxel_indices)

                x_embedded = self.bilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
                x_embedded_all.append(x_embedded)

            keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
            return torch.cat(x_embedded_all, dim=-1)#, keep_mask

        else:
            raise ValueError("Unsupported input dimension: {}".format(x.shape[1]))


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size, is_3d=True):
    print('bounding_box:',bounding_box)
    box_min, box_max = torch.tensor(bounding_box, device=xyz.device)

    keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    if is_3d:
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0], device=xyz.device) * grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_3D
        hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    else:
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0], device=xyz.device) * grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_2D
        hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


def hash(coords, log2_hashmap_size):
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1, device=xor_result.device) & xor_result