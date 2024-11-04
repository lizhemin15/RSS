import torch
import torch.nn as nn
import itertools

# 设置全局变量 device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_box_offsets(d):
    return torch.tensor(list(itertools.product(*[[i for i in range(2)] for _ in range(d)])), device=device).float()

def HashEmbedder(parameter):
    de_para_dict = {
        'bounding_box': [-1, 1],
        'n_levels': 16,
        'n_features_per_level': 2,
        'log2_hashmap_size': 19,
        'base_resolution': 16,
        'finest_resolution': 512,
    }
    
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
    
    return HashEmbedders(
        bounding_box=parameter['bounding_box'],
        n_levels=parameter['n_levels'],
        n_features_per_level=parameter['n_features_per_level'],
        log2_hashmap_size=parameter['log2_hashmap_size'],
        base_resolution=parameter['base_resolution'],
        finest_resolution=parameter['finest_resolution']
    )

class HashEmbedders(nn.Module):
    def __init__(self, bounding_box=[-1, 1], n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512, non_local_if=False):
        super(HashEmbedders, self).__init__()
        self.forward_count = 0
        self.non_local_if = non_local_if
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution, device=device)
        self.finest_resolution = torch.tensor(finest_resolution, device=device)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        # Create box offsets for arbitrary dimensions
        self.d = len(bounding_box) // 2  # Assuming bounding_box is of the format [min1, max1, min2, max2, ...]
        self.box_offsets = create_box_offsets(self.d)

        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size,
                                                      self.n_features_per_level).to(device) for _ in range(n_levels)])
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def interpolate(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embeds):
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x d
        num_vertices = 2 ** self.d

        # Calculate the corner weighted contributions
        output = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)

        for i in range(num_vertices):
            weight_contribution = torch.prod(weights ** (self.box_offsets[i].type_as(weights)), dim=-1)
            output += voxel_embeds[:, i] * weight_contribution[:, None]

        return output

    def forward(self, x):
        if self.forward_count == 0:
            self.box_offsets = self.box_offsets.to(x.device)
        self.forward_count += 1
        if self.non_local_if == False or self.forward_count < 100:
            x_embedded_all = []
            for i in range(self.n_levels):
                resolution = torch.floor(self.base_resolution * self.b ** i).to(x.device)
                voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
                    x, self.bounding_box,
                    resolution, self.log2_hashmap_size, is_d=self.d)

                voxel_embeds = self.embeddings[i](hashed_voxel_indices)

                x_embedded = self.interpolate(x, voxel_min_vertex, voxel_max_vertex, voxel_embeds)
                x_embedded_all.append(x_embedded)

            keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
            return torch.cat(x_embedded_all, dim=-1)

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size, is_d):
    box_min, box_max = torch.tensor(bounding_box, device=xyz.device).view(-1, 2).T

    keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + create_box_offsets(is_d).to(xyz.device)
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

def hash(coords, log2_hashmap_size):
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    
    # Convert coords to long type to perform bitwise operations
    coords = coords.long()  # Change this line to convert to integer type

    xor_result = torch.zeros_like(coords[..., 0], dtype=torch.long)  # Ensure xor_result is of long type
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i % len(primes)]

    return (1 << log2_hashmap_size) - 1 & xor_result


# import torch
# import torch.nn as nn

# # 设置全局变量 device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # 使用全局 device 创建张量
# BOX_OFFSETS_3D = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], 
#                                device=device)

# BOX_OFFSETS_2D = torch.tensor([[[i, j] for i in [0, 1] for j in [0, 1]]], 
#                                device=device)


# def HashEmbedder(parameter):
#     de_para_dict = {
#         'bounding_box': [-1, 1],
#         'n_levels': 16,
#         'n_features_per_level': 2,
#         'log2_hashmap_size': 19,
#         'base_resolution': 16,
#         'finest_resolution': 512,
#     }
    
#     for key in de_para_dict.keys():
#         param_now = parameter.get(key, de_para_dict.get(key))
#         parameter[key] = param_now
    
#     # print('HashEmbedder : ', parameter)
#     return HashEmbedders(
#         bounding_box=parameter['bounding_box'],
#         n_levels=parameter['n_levels'],
#         n_features_per_level=parameter['n_features_per_level'],
#         log2_hashmap_size=parameter['log2_hashmap_size'],
#         base_resolution=parameter['base_resolution'],
#         finest_resolution=parameter['finest_resolution']
#     )


# class HashEmbedders(nn.Module):
#     def __init__(self, bounding_box=[-1, 1], n_levels=16, n_features_per_level=2,
#                  log2_hashmap_size=19, base_resolution=16, finest_resolution=512, non_local_if=False):
#         super(HashEmbedders, self).__init__()
#         # ('bounding_box:',bounding_box)
#         self.forward_count = 0
#         self.non_local_if = non_local_if
#         self.bounding_box = bounding_box
#         self.n_levels = n_levels
#         self.n_features_per_level = n_features_per_level
#         self.log2_hashmap_size = log2_hashmap_size
#         self.base_resolution = torch.tensor(base_resolution, device=device)
#         self.finest_resolution = torch.tensor(finest_resolution, device=device)
#         self.out_dim = self.n_levels * self.n_features_per_level

#         self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

#         self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size,
#                                                       self.n_features_per_level).to(device) for i in range(n_levels)])
#         # custom uniform initialization
#         for i in range(n_levels):
#             nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

#     def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
#         weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

#         c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
#         c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
#         c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
#         c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]

#         c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
#         c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

#         c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

#         return c

#     def bilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
#         weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 2

#         c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 2] * weights[:, 0][:, None]
#         c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 3] * weights[:, 0][:, None]

#         c = c00 * (1 - weights[:, 1][:, None]) + c01 * weights[:, 1][:, None]

#         return c

#     def forward(self, x):
#         self.forward_count += 1
#         if self.non_local_if == False or self.forward_count < 100:
#             if x.shape[1] == 3:
#                 x_embedded_all = []
#                 for i in range(self.n_levels):
#                     resolution = torch.floor(self.base_resolution * self.b ** i).to(x.device)
#                     voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
#                         x, self.bounding_box,
#                         resolution, self.log2_hashmap_size, is_3d=True)

#                     voxel_embedds = self.embeddings[i](hashed_voxel_indices)

#                     x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
#                     x_embedded_all.append(x_embedded)

#                 keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
#                 return torch.cat(x_embedded_all, dim=-1)#, keep_mask

#             elif x.shape[1] == 2:
#                 x_embedded_all = []
#                 for i in range(self.n_levels):
#                     resolution = torch.floor(self.base_resolution * self.b ** i).to(x.device)
#                     voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
#                         x, self.bounding_box,
#                         resolution, self.log2_hashmap_size, is_3d=False)

#                     voxel_embedds = self.embeddings[i](hashed_voxel_indices)

#                     x_embedded = self.bilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
#                     x_embedded_all.append(x_embedded)

#                 keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
#                 return torch.cat(x_embedded_all, dim=-1)#, keep_mask

#             else:
#                 raise ValueError("Unsupported input dimension: {}".format(x.shape[1]))
#         elif self.non_local_if == True and self.forward_count >= 100:
#             if x.shape[1] == 3:
#                 pass
#             elif x.shape[1] == 2:
#                 pass
#             else:
#                 raise ValueError("Unsupported input dimension: {}".format(x.shape[1]))



# def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size, is_3d=True):
    
#     box_min, box_max = torch.tensor(bounding_box, device=xyz.device)

#     keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
#     if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
#         xyz = torch.clamp(xyz, min=box_min, max=box_max)

#     grid_size = (box_max - box_min) / resolution

#     if is_3d:
#         bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
#         voxel_min_vertex = bottom_left_idx * grid_size + box_min
#         voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0], device=xyz.device) * grid_size

#         voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_3D.to(xyz.device)
#         hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

#     else:
#         bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
#         voxel_min_vertex = bottom_left_idx * grid_size + box_min
#         voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0], device=xyz.device) * grid_size

#         voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_2D.to(xyz.device)
#         hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

#     return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


# def hash(coords, log2_hashmap_size):
#     primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

#     xor_result = torch.zeros_like(coords)[..., 0]
#     for i in range(coords.shape[-1]):
#         xor_result ^= coords[..., i] * primes[i]

#     return torch.tensor((1 << log2_hashmap_size) - 1, device=xor_result.device) & xor_result

