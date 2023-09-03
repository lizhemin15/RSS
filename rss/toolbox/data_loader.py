import numpy as np
import torch as t
import torch.nn as nn
from einops import rearrange
from rss.represent.utils import to_device
abc_str = 'abcdefghijklmnopqrstuvwxyz'

#TODO all the data are 1 dimension output, which is not suitable for general

def get_dataloader(x_mode='inr',batch_size=128,shuffle_if=False,
                   data=None,mask=None,xrange=1,noisy_data=None,
                   ymode='completion',return_data_type='loader',gpu_id=0,out_dim_one=True):
    # Given x_mode
    # Return a pytorch dataloader generator or generator list
    # Principle: process data on numpy untill the last step
    cor_list,inrarr = get_cor(data.shape,xrange)

    def get_data_feature(data,mask=None,kernel_size=3):
        """
        # Random conv feature
        Args:
        data: ndarray, input data
        Returns:
        data_feature: ndarray, random feature
        """
        data_tensor = t.tensor(data*mask.astype(data.dtype))
        conv = nn.Conv2d(1, 16, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        nn.init.normal_(conv.weight,mean=1,std=1e0)
        data_feature = rearrange(conv(data_tensor.unsqueeze(0).unsqueeze(0)),'a b c d -> (a c d) (b)')
        # t.squeeze(conv(data_tensor.unsqueeze(0).permute(0,3,1,2))).permute(1,2,0)
        data_feature /= data_feature.max()
        return data_feature.detach().numpy().astype(data.dtype)
    
    if x_mode == 'inr_feature':
        # the input of inr is feature
        inrarr = get_data_feature(data,mask=mask,kernel_size=7)
    elif x_mode == 'inr_combine':
        # combine the cor and feature as the input of inr
        inrarr = np.concatenate((get_data_feature(data,mask=mask,kernel_size=7), inrarr), axis=1)

    def get_data_loader(xin,data,mask,batch_size,shuffle,ymode='completion',noisy_data=None):
        xin = t.tensor(xin).to(t.float32)
        mask = t.tensor(mask).to(t.float32)
        data = t.tensor(data).to(t.float32)
        # print(xin.shape,(mask==1).reshape(-1).shape,data.shape)
        if ymode == 'completion':
            data_train_set = t.utils.data.TensorDataset(xin[(mask==1).reshape(-1)],data[mask==1])
            data_train_loader = t.utils.data.DataLoader(data_train_set,batch_size = batch_size,shuffle=shuffle)
            data_val_set = t.utils.data.TensorDataset(xin[(mask==0).reshape(-1)],data[mask==0])
            data_val_loader = t.utils.data.DataLoader(data_val_set,batch_size = batch_size,shuffle=shuffle)
            data_test_set = t.utils.data.TensorDataset(xin,data.reshape(-1,1))
            data_test_loader = t.utils.data.DataLoader(data_test_set,batch_size = batch_size,shuffle=False)
        elif ymode == 'denoising':
            noisy_data = reshape2(noisy_data)
            noisy_data = t.tensor(noisy_data).to(t.float32)
            data_train_set = t.utils.data.TensorDataset(xin[(mask==1).reshape(-1)],noisy_data[mask==1])
            data_train_loader = t.utils.data.DataLoader(data_train_set,batch_size = batch_size,shuffle=shuffle)
            data_val_set = t.utils.data.TensorDataset(xin[(mask==0).reshape(-1)],data[mask==0])
            data_val_loader = t.utils.data.DataLoader(data_val_set,batch_size = batch_size,shuffle=shuffle)
            data_test_set = t.utils.data.TensorDataset(xin,data)
            data_test_loader = t.utils.data.DataLoader(data_test_set,batch_size = batch_size,shuffle=False)
        else:
            raise('Wrong ymode = ',ymode)
        return [data_train_loader,data_val_loader,data_test_loader]

    def get_data_tensor(xin,data,ymode='completion',noisy_data=None,random_if=False):
        xin = t.tensor(xin).to(t.float32)
        # mask = t.tensor(mask).to(t.float32)
        data = t.tensor(data).to(t.float32)
        xin = to_device(xin,gpu_id)
        # mask = to_device(mask,gpu_id)
        data = to_device(data,gpu_id)
        # print(xin.shape,(mask==1).reshape(-1).shape,data.shape)
        if random_if:
            xin = to_device(t.randn(data.shape),gpu_id)
        if ymode == 'completion':
            if out_dim_one:
                data_train_loader = (xin,data.reshape(-1,1))
                # data_val_loader = (xin[(mask==0).reshape(-1)],data[mask==0])
                data_test_loader = (xin,data.reshape(-1,1))
        elif ymode == 'denoising':
            noisy_data = reshape2(noisy_data)
            noisy_data = t.tensor(noisy_data).to(t.float32)
            noisy_data = to_device(noisy_data,gpu_id)
            if out_dim_one:
                data_train_loader = (xin,noisy_data.reshape(-1,1))
                # data_val_loader = (xin[(mask==0).reshape(-1)],data[mask==0])
                data_test_loader = (xin,data.reshape(-1,1))
        else:
            raise('Wrong ymode = ',ymode)
        return [data_train_loader,data_test_loader]


    if return_data_type == 'tensor':
        data_train_loader,data_test_loader = get_data_tensor(xin=inrarr,data=data,
                                                            noisy_data=noisy_data,ymode=ymode)
        return {'obs_tensor':data_train_loader,'real_tensor':data_test_loader}
    elif return_data_type == 'random':
        data_train_loader,data_test_loader = get_data_tensor(xin=inrarr,data=data,
                                                            noisy_data=noisy_data,ymode=ymode,random_if=True)
        return {'obs_tensor':data_train_loader,'real_tensor':data_test_loader}




    # if x_mode in ['inr','inr_feature','inr_combine']:
    #     # return a generator
    #     # train: used to train, val: the remaind data, test: all data by sequence
    #     # print(data.shape,mask.shape)
    #     data_train_loader,data_val_loader,data_test_loader = get_data_loader(xin=inrarr,data=data,
    #                                                         mask=mask,batch_size=batch_size,shuffle=shuffle_if,
    #                                                         noisy_data=noisy_data,ymode=ymode)
    #     return {'train_loader':data_train_loader,'val_loader':data_val_loader,'test_loader':data_test_loader}
    

    # elif x_mode in ['splitinr','dmf','sparse','tf','dip']:
    #     # return a list
    #     reshape_cor_list = []
    #     for cor in cor_list:
    #         reshape_cor_list.append(t.tensor(cor.reshape(-1,1)).to(t.float32))
    #     return_list = [reshape_cor_list]
    #     if ymode == 'completion':
    #         return_list.append(t.tensor(data).to(t.float32))
    #     else:
    #         return_list.append(t.tensor(noisy_data).to(t.float32))
    #     return_list.append(t.tensor(data).to(t.float32))
    #     return_list.append(mask)
    #     return return_list

    # else:
    #     return None
    #     # raise('Wrong x_mode= ',str(x_mode))

def reshape2(data):
    xshape = data.shape
    einstr = add_space(abc_str[:len(xshape)])+' -> ('+add_space(abc_str[:len(xshape)])+') ()'
    return rearrange(data,einstr)

def add_space(oristr):
    addstr = ''
    for i in range(len(oristr)):
        addstr += oristr[i]
        addstr += ' '
    return addstr

def get_cor(xshape,xrange):  
    cor_list = []
    for i,point_num in enumerate(xshape):
        cor_list.append(np.linspace(-xrange,xrange,point_num))
        # if i == 0:
        #     if len(xshape) == 1:
        #         cor_list.append(np.linspace(-xrange,xrange,point_num))
        #     else:
        #         cor_list.append(np.linspace(-xrange,xrange,xshape[1]))
        # elif i == 1:
        #     cor_list.append(np.linspace(-xrange,xrange,xshape[0]))
        # else:
        #     cor_list.append(np.linspace(-xrange,xrange,point_num))
    corv_list = np.meshgrid(*cor_list)
    coor = np.stack(corv_list,axis=len(xshape))
    einstr = add_space(abc_str[:len(xshape)])+' '+abc_str[len(xshape)]+' -> ('+add_space(abc_str[:len(xshape)])+') '+abc_str[len(xshape)]
    return cor_list,rearrange(coor,einstr)
