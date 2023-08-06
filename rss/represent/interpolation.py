import torch as t


def bina_index(i,d,weights):
    """
    Input:
    i: 1-dim int, index of the voxel
    d: int, dimension of the voxel
    weights: [B,d], weight of the voxel

    Output:
    add_index: [B,d], 0 or 1, 0 means lower voxel, 1 means higher voxel
    weight: [B,1], weight of the voxel
    """
    add_index = t.zeros((1,d))
    index_i = i
    weight = t.ones((weights.shape[0],1))
    for j in range(d):
        add_index[:,j] = index_i%2 # lower (0) or higher(1) voxel at j-th dimension
        index_i = index_i//2
        if add_index[:,j] == 0:
            weight *= 1-weights[:,j:j+1]
        else:
            weight *= weights[:,j:j+1]
    add_index = add_index.long()
    return add_index,weight


class Interpolation_cls(t.nn.Module):
    def __init__(self,tau_range='default'):
        super().__init__()
        self.tau_range = tau_range
    
    def forward(self,x,tau):
        """
        Differentiable to tau

        Input:
        x: [B,d]
        tau: [n_1,n_2,...,n_d,F], Assuming tau is a uniformly sampled grid
        
        Output:
        y: [B,F]
        """
        B,d = x.shape
        F = tau.shape[-1]
        dim_tensor = t.tensor(tau.shape[:-1]).unsqueeze(0)-1 # (1,d)
        if self.tau_range == 'default':
            tau_range = t.tensor([[0,1]]*d).float().T.to(x)
        x_rerange = (x-tau_range[0,:])/(tau_range[1,:]-tau_range[0,:]) # [B,d], rerange x into [0,1] to calculate the index of x
        x_index = t.floor(dim_tensor*x_rerange).long() # [B,d], calculate the index of x in tau, first floor then clip, only need to calculate the lower index
        for j in range(d):
            x_index[:,j] = t.minimum(x_index[:,j], t.tensor(tau.shape[j]-2)) # clip the last vox
        weights = dim_tensor*x_rerange-x_index # [B,d]
        tau_vox = t.rand((2**d,B,F)).to(tau) # [2**d,B,F]
        for i in range(2**d):
            add_index,weight = bina_index(i,d,weights) # [B,d], [B,1]
            new_index = x_index+add_index
            tau_vox[i,:,:] = tau[list(new_index.T)]*weight
        y = t.sum(tau_vox,dim=0)
        return y

def Interpolation(parameter):
    de_para_dict = {'tau_range':'default'}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return Interpolation_cls(parameter)