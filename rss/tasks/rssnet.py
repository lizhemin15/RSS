from rss import toolbox,represent
from rss.represent.utils import to_device
import torch.nn as nn
import matplotlib.pyplot as plt
import torch as t
import numpy as np
import time
t.backends.cudnn.enabled = True
t.backends.cudnn.benchmark = True 

class rssnet(object):
    def __init__(self,parameters,verbose=True) -> None:
        parameter_list = ['net_p','reg_p','data_p','opt_p','train_p','show_p','save_p']
        self.init_parameter(parameters,parameter_list)
        self.init_net()
        self.init_reg()
        self.init_data()
        self.init_opt()
        self.init_train()
        self.init_save()
        self.init_show()
        self.update_parameter(parameter_list,verbose)
        
    def init_parameter(self,parameters,parameter_list):
        """
        Initialize a parameter.

        Args:
            key (str): The key of the parameter to initialize. Options are:
                - net_p: The parameter used for the network.
                - reg_p: The parameter used for the regularization.
                - data_p: The parameter used for the data.
                - opt_p: The parameter used for the optimizer.
                - train_p: The parameter used for the training.
                - show_p: The parameter used for showing information.
                - save_p: The parameter used for saving.

        Returns:
            parameter (dict): The initialization key
        Raises:
            ValueError: If the input key is not one of the options.
        """
        for key in parameter_list:
            param_now = parameters.get(key,{})
            setattr(self,key,param_now)
    
    def update_parameter(self,parameter_list,verbose=True):
        self.parameter_all = {}
        for key in parameter_list:
            self.parameter_all[key] = getattr(self,key)
            if verbose:
                print(key,self.parameter_all[key])

    def init_net(self):
        de_para_dict = {'net_name':'SIREN','gpu_id':0}
        for key in de_para_dict.keys():
            param_now = self.net_p.get(key,de_para_dict.get(key))
            self.net_p[key] = param_now
        # print('net_p : ',self.net_p)
        self.net = represent.get_nn(self.net_p)
        self.net = to_device(self.net,self.net_p['gpu_id'])


    def init_reg(self):
        de_para_dict = {'reg_name':None}
        for key in de_para_dict.keys():
            param_now = self.reg_p.get(key,de_para_dict.get(key))
            self.reg_p[key] = param_now
        self.reg_p['gpu_id'] = self.net_p['gpu_id']
        # print('net_p : ',self.net_p)
        if self.reg_p['reg_name'] != None:
            self.reg = represent.get_reg(self.reg_p)
            self.reg = to_device(self.reg,self.reg_p['gpu_id'])

    def init_data(self):
        de_para_dict = {'data_path':None,'data_type':'syn','data_shape':(10,10),'down_sample':[1,1,1],
                        'mask_type':'random','random_rate':0.0,'mask_path':None,'mask_shape':'same','seeds':88,'down_sample_rate':2,
                        'noise_mode':None,'noise_parameter':0.0,
                        'x_mode':'inr','batch_size':128,'shuffle_if':False,'xrange':1,'ymode':'completion','return_data_type':'tensor',
                        'pre_full':False,'out_dim_one':True}
        for key in de_para_dict.keys():
            param_now = self.data_p.get(key,de_para_dict.get(key))
            self.data_p[key] = param_now
        # print('data_p : ',self.data_p)
        self.data = toolbox.load_data(data_path=self.data_p['data_path'],data_type=self.data_p['data_type'],
                                      data_shape=self.data_p['data_shape'],down_sample=self.data_p['down_sample'])
        self.mask = toolbox.load_mask(mask_type=self.data_p['mask_type'],random_rate=self.data_p['random_rate'],mask_path=self.data_p['mask_path'],
                                      data_shape=self.data.shape,mask_shape=self.data_p['mask_shape'],seeds=self.data_p['seeds'],
                                      down_sample_rate=self.data_p['down_sample_rate'],gpu_id=self.net_p['gpu_id'])
        self.mask = to_device(t.tensor(self.mask).to(t.float32),self.net_p['gpu_id'])
        self.data_noise = toolbox.add_noise(self.data,mode=self.data_p['noise_mode'],parameter=self.data_p['noise_parameter'],seeds=self.data_p['seeds'])
        self.data_train = toolbox.get_dataloader(x_mode=self.data_p['x_mode'],batch_size=self.data_p['batch_size'],
                                                 shuffle_if=self.data_p['shuffle_if'],
                                                data=self.data,mask=self.mask,xrange=self.data_p['xrange'],noisy_data=self.data_noise,
                                                ymode=self.data_p['ymode'],return_data_type=self.data_p['return_data_type'],
                                                gpu_id=self.net_p['gpu_id'],out_dim_one=self.data_p['out_dim_one'])
        
        

    def init_opt(self):
        de_para_dict = {'net':{'opt_name':'Adam','lr':1e-4,'weight_decay':0},'reg':{'opt_name':'Adam','lr':1e-4,'weight_decay':0}}
        for key in de_para_dict.keys():
            param_now = self.opt_p.get(key,de_para_dict.get(key))
            self.opt_p[key] = param_now
        # print('opt_p : ',self.opt_p)
        self.net_opt = represent.get_opt(opt_type=self.opt_p['net']['opt_name'],
                                         parameters=self.net.parameters(),lr=self.opt_p['net']['lr'],
                                         weight_decay=self.opt_p['net']['weight_decay'])
        if self.reg_p['reg_name'] != None and len(list(self.reg.parameters()))>0:
            self.train_reg_if = True
        else:
            self.train_reg_if = False
        if self.train_reg_if:
            self.reg_opt = represent.get_opt(opt_type=self.opt_p['reg']['opt_name'],
                                            parameters=self.reg.parameters(),lr=self.opt_p['reg']['lr'],
                                            weight_decay=self.opt_p['reg']['weight_decay'])


    def init_train(self):
        de_para_dict = {'task_name':self.data_p['ymode'],'train_epoch':10,'loss_fn':'mse'}
        for key in de_para_dict.keys():
            param_now = self.train_p.get(key,de_para_dict.get(key))
            self.train_p[key] = param_now
        if self.train_p['loss_fn'] == 'mse':
            self.loss_fn = nn.MSELoss()
        # print('train_p : ',self.train_p)

    def init_save(self):
        de_para_dict = {'save_if':False}
        for key in de_para_dict.keys():
            param_now = self.save_p.get(key,de_para_dict.get(key))
            self.save_p[key] = param_now

    def init_show(self):
        de_para_dict = {'show_type':'gray_img','show_content':'original'}
        for key in de_para_dict.keys():
            param_now = self.show_p.get(key,de_para_dict.get(key))
            self.show_p[key] = param_now

    def train(self,verbose=True):
        # Construct loss function
        if self.data_p['return_data_type'] == 'random':
            unn_index = 0
        else:
            unn_index = 1
        if self.data_p['return_data_type'] in ['tensor','random']:
            if (not hasattr(self, 'log_dict')) or ('time' not in self.log_dict):
                self.start_time = time.time()
            for ite in range(self.train_p['train_epoch']):
                time_now = time.time()
                self.log('time',time_now-self.start_time)
                if (self.net_p['net_name'] in ['UNet','ResNet','skip']) or (self.net_p['net_name']=='KNN' and self.net_p['mode'] in ['UNet','ResNet','skip']):
                    pre = self.net(self.data_train['obs_tensor'][unn_index].reshape(1,-1,self.data_p['data_shape'][0],self.data_p['data_shape'][1]))
                    pre = pre.reshape(self.data_p['data_shape'])
                else:
                    pre = self.net(self.data_train['obs_tensor'][0][(self.mask==1).reshape(-1)])
                
                loss = 0

                if self.reg_p['reg_name'] != None:
                    reg_tensor = pre.reshape(self.data_p['data_shape'])
                    reg_loss = self.reg(reg_tensor)
                    loss += reg_loss

                if self.data_p['pre_full'] == True:
                    pre = pre[self.mask==1]
                target = self.data_train['obs_tensor'][1][(self.mask==1).reshape(-1)].reshape(pre.shape)
                loss += self.loss_fn(pre,target)

                self.log('fid_loss',loss.item())
                self.net_opt.zero_grad()
                if self.train_reg_if:
                    self.reg_opt.zero_grad()
                loss.backward()
                self.net_opt.step()
                if self.train_reg_if:
                    self.reg_opt.step()
                # test and val loss
                with t.no_grad():
                    if self.net_p['net_name'] in ['UNet','ResNet','skip'] or (self.net_p['net_name']=='KNN' and self.net_p['mode'] in ['UNet','ResNet','skip']):
                        pre = self.net(self.data_train['obs_tensor'][unn_index].reshape(1,-1,self.data_p['data_shape'][0],self.data_p['data_shape'][1]))
                        pre = pre.reshape(self.data_p['data_shape'])
                    else:
                        pre = self.net(self.data_train['obs_tensor'][0][(self.mask==0).reshape(-1)])
                    if self.data_p['pre_full'] == True:
                        pre = pre[self.mask==0]
                    target = self.data_train['obs_tensor'][1][(self.mask==0).reshape(-1)].reshape(pre.shape)
                    loss = self.loss_fn(pre,target)
                    self.log('val_loss',loss.item())

                    if self.net_p['net_name'] in ['UNet','ResNet','skip']:
                        pre = self.net(self.data_train['real_tensor'][unn_index].reshape(1,-1,self.data_p['data_shape'][0],self.data_p['data_shape'][1]))
                        pre = pre.reshape(self.data_p['data_shape'])
                    else:
                        pre = self.net(self.data_train['real_tensor'][0])
                    target = self.data_train['real_tensor'][1].reshape(pre.shape)
                    self.pre = pre
                    self.target = target
                    loss = self.loss_fn(pre,target)
                    self.log('test_loss',loss.item())
                    self.log('psnr',self.cal_psnr(pre,target).item())
                    self.log('nmae',self.cal_nmae(pre,target))
                    if self.reg_p['reg_name'] != None:
                        self.log('reg_loss',reg_loss)

            if verbose == True:    
                print('loss on test set',self.log_dict['test_loss'][-1])
                print('PSNR=',self.log_dict['psnr'][-1],'dB')
                print('NMAE=',self.log_dict['nmae'][-1])
                if self.reg_p['reg_name'] != None:
                    print('loss of regularizer',self.log_dict['reg_loss'][-1])
            


    def log(self,name,content):
        if 'log_dict' not in self.__dict__:
            self.log_dict = {}
        if name not in self.log_dict:
            self.log_dict[name] = [content]
        else:
            self.log_dict[name].append(content)



    def show(self):
        de_para_dict = {'show_type':'gray_img','show_content':'recovered','show_axis':False}
        for key in de_para_dict.keys():
            param_now = self.show_p.get(key,de_para_dict.get(key))
            self.show_p[key] = param_now
        if self.show_p['show_content'] == 'recovered':
            if self.net_p['net_name'] in ['UNet','ResNet','skip']:
                if self.data_p['return_data_type'] == 'random':
                    unn_index = 0
                else:
                    unn_index = 1
                pre_img = self.net(self.data_train['obs_tensor'][unn_index].reshape(1,-1,self.data_p['data_shape'][0],self.data_p['data_shape'][1]))
                pre_img = pre_img.reshape(self.data_p['data_shape'])
            else:
                pre_img = self.net(self.data_train['obs_tensor'][0])
            show_img = pre_img.reshape(self.data_p['data_shape']).detach().cpu().numpy()
            #print('PSNR=',self.cal_psnr(show_img,self.data_train['obs_tensor'][1].reshape(self.data_p['data_shape']).detach().cpu().numpy()),'dB')
        elif self.show_p['show_content'] == 'original':
            show_img = self.data_train['obs_tensor'][1].reshape(self.data_p['data_shape']).detach().cpu().numpy()
        if self.show_p['show_type'] == 'gray_img':
            plt.imshow(show_img,'gray',vmin=0,vmax=1)
        elif self.show_p['show_type'] == 'red_img':
            import seaborn as sns
            sns.set()
            plt.imshow(show_img)
        else:
            raise('Wrong show_type in show_p:',self.show_p['show_type'])
        if self.show_p['show_axis'] == False:
            plt.axis('off')
        if self.save_p['save_if'] == True:
            plt.savefig(self.save_p['save_path'], bbox_inches='tight', pad_inches=0)
        plt.show()
        

    def save(self):
        de_para_dict = {'save_if':False,'save_path':None}
        for key in de_para_dict.keys():
            param_now = self.save_p.get(key,de_para_dict.get(key))
            self.save_p[key] = param_now

    def cal_psnr(self, pre, target):
        def mse(pre, target):
            err = t.sum((pre.float() - target.float()) ** 2)
            err /= float(pre.shape[0] * pre.shape[1])
            return err
        
        def psnr(pre, target):
            max_pixel = t.max(target)
            mse_value = mse(pre, target)
            if mse_value == 0:
                return 100
            return 20 * t.log10(max_pixel / t.sqrt(mse_value))
        
        return psnr(pre, target)
    
    def cal_nmae(self,pre, target):
        max_pixel,min_pixel = t.max(target),t.min(target)
        unseen_num = t.sum(1-self.mask)
        if unseen_num<1e-3:
            return 0
        else:
            return t.sum(t.abs((pre-target)*(1-self.mask).reshape(pre.shape)))/unseen_num/(max_pixel-min_pixel)
        pass
    # def cal_psnr(self,imageA, imageB):
    #     def mse(imageA, imageB):
    #         err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    #         err /= float(imageA.shape[0] * imageA.shape[1])
    #         return err
        
    #     def psnr(imageA, imageB):
    #         max_pixel = np.max(imageB)
    #         mse_value = mse(imageA, imageB)
    #         if mse_value == 0:
    #             return 100
    #         return 20 * np.log10(max_pixel / np.sqrt(mse_value))
    #     return psnr(imageA, imageB)