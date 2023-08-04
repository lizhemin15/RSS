import torch as t





def Interpolation(x,tau_range,tau):
    """
    Input:
    data_type : pytorch tensor
    x: [B,d]
    tau_range: [2,d]
    tau: [n_1,n_2,...,n_d,F], Assuming tau is a uniformly sampled grid
    
    Output:
    y: [B,F]
    """
    B,d = x.shape
    F = tau.shape[-1]
    dim_tensor = t.tensor(tau.shape[:-1]).unsqueeze(0)-1 # (1,d)
    x_rerange = (x-tau_range[0,:])/(tau_range[1,:]-tau_range[0,:]) # [B,d], rerange x into [0,1] to calculate the index of x
    x_index = None # [B,d], calculate the index of x in tau, first floor then clip, only need to calculate the lower index


    pass