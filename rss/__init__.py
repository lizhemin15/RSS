""" rss
"""
__title__ = 'rss'
__version__ = '0.0.1'
#__build__ = 0x021300
__author__ = 'Zhemin Li'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023 Zhemin Li'


## Top Level Modules
from rss import toolbox,represent
from rss.represent.utils import to_device
import torch.nn as nn
import matplotlib.pyplot as plt
import torch as t
t.backends.cudnn.enabled = True
t.backends.cudnn.benchmark = True 
# from rss.represent import get_nn
__all__ = []

class rssnet(object):
    def __init__(self,parameters) -> None:
        parameter_list = ['net_p','reg_p','data_p','opt_p','train_p','show_p','save_p']
        self.init_parameter(parameters,parameter_list)
        self.init_net()
        self.init_reg()
        self.init_data()
        self.init_opt()
        self.init_train()
        self.update_parameter(parameter_list)
        



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
    
    def update_parameter(self,parameter_list):
        self.parameter_all = {}
        for key in parameter_list:
            self.parameter_all[key] = getattr(self,key)
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
        pass

    def init_data(self):
        de_para_dict = {'data_path':None,'data_type':'syn','data_shape':(10,10),'down_sample':[1,1,1],
                        'mask_type':'random','random_rate':0.0,'mask_path':None,'mask_shape':'same','seeds':88,'down_sample_rate':2,
                        'noise_mode':None,'noise_parameter':0.0,
                        'x_mode':'inr','batch_size':128,'shuffle_if':False,'xrange':1,'ymode':'completion','return_data_type':'tensor',
                        'pre_full':False}
        for key in de_para_dict.keys():
            param_now = self.data_p.get(key,de_para_dict.get(key))
            self.data_p[key] = param_now
        # print('data_p : ',self.data_p)
        self.data = toolbox.load_data(data_path=self.data_p['data_path'],data_type=self.data_p['data_type'],
                                      data_shape=self.data_p['data_shape'],down_sample=self.data_p['down_sample'])
        self.mask = toolbox.load_mask(mask_type=self.data_p['mask_type'],random_rate=self.data_p['random_rate'],mask_path=self.data_p['mask_path'],
                                      data_shape=self.data.shape,mask_shape=self.data_p['mask_shape'],seeds=self.data_p['seeds'],
                                      down_sample_rate=self.data_p['down_sample_rate'])
        self.data_noise = toolbox.add_noise(self.data,mode=self.data_p['noise_mode'],parameter=self.data_p['noise_parameter'],seeds=self.data_p['seeds'])
        self.data_train = toolbox.get_dataloader(x_mode=self.data_p['x_mode'],batch_size=self.data_p['batch_size'],
                                                 shuffle_if=self.data_p['shuffle_if'],
                                                data=self.data,mask=self.mask,xrange=self.data_p['xrange'],noisy_data=self.data_noise,
                                                ymode=self.data_p['ymode'],return_data_type=self.data_p['return_data_type'],
                                                gpu_id=self.net_p['gpu_id'])
        
        

    def init_opt(self):
        de_para_dict = {'net':{'opt_name':'Adam','lr':1e-4,'weight_decay':0},'reg':{'opt_name':'Adam','lr':1e-4,'weight_decay':0}}
        for key in de_para_dict.keys():
            param_now = self.opt_p.get(key,de_para_dict.get(key))
            self.opt_p[key] = param_now
        # print('opt_p : ',self.opt_p)
        self.net_opt = represent.get_opt(opt_type=self.opt_p['net']['opt_name'],parameters=self.net.parameters(),lr=self.opt_p['net']['lr'],weight_decay=self.opt_p['net']['weight_decay'])

    def init_train(self):
        de_para_dict = {'task_name':self.data_p['ymode'],'train_epoch':10,'loss_fn':'mse'}
        for key in de_para_dict.keys():
            param_now = self.train_p.get(key,de_para_dict.get(key))
            self.train_p[key] = param_now
        if self.train_p['loss_fn'] == 'mse':
            self.loss_fn = nn.MSELoss()
        # print('train_p : ',self.train_p)

    def train(self):
        # Construct loss function
        if self.data_p['return_data_type'] == 'tensor':
            for ite in range(self.train_p['train_epoch']):
                pre = self.net(self.data_train['obs_tensor'][0][(self.mask==1).reshape(-1)])
                if self.data_p['pre_full'] == True:
                    print((self.mask==1).shape,pre.shape)
                    pre = pre[self.mask==1]
                target = self.data_train['obs_tensor'][1][self.mask==1].reshape(pre.shape)
                loss = self.loss_fn(pre,target)
                self.log('fid_loss',loss.item())
                self.net_opt.zero_grad()
                loss.backward()
                self.net_opt.step()
                # test and val loss
                with t.no_grad():
                    pre = self.net(self.data_train['obs_tensor'][0][(self.mask==0).reshape(-1)])
                    if self.data_p['pre_full'] == True:
                        pre = pre[self.mask==0]
                    target = self.data_train['obs_tensor'][1][self.mask==0].reshape(pre.shape)
                    loss = self.loss_fn(pre,target)
                    self.log('val_loss',loss.item())
                    pre = self.net(self.data_train['real_tensor'][0])
                    target = self.data_train['real_tensor'][1].reshape(pre.shape)
                    loss = self.loss_fn(pre,target)
                    self.log('test_loss',loss.item())
            print('loss on test set',self.log_dict['test_loss'][-1])
            
    def log(self,name,content):
        if 'log_dict' not in self.__dict__:
            self.log_dict = {}
        if name not in self.log_dict:
            self.log_dict[name] = [content]
        else:
            self.log_dict[name].append(content)



    def show(self):
        de_para_dict = {'show_type':'gray_img','show_content':'recovered'}
        for key in de_para_dict.keys():
            param_now = self.show_p.get(key,de_para_dict.get(key))
            self.show_p[key] = param_now
        if self.show_p['show_content'] == 'recovered':
            pre_img = self.net(self.data_train['obs_tensor'][0])
            show_img = pre_img.reshape(self.data_p['data_shape']).detach().cpu().numpy()
        elif self.show_p['show_content'] == 'original':
            show_img = self.data_train['obs_tensor'][1].reshape(self.data_p['data_shape']).detach().cpu().numpy()
        if self.show_p['show_type'] == 'gray_img':
            plt.imshow(show_img,'gray')
        elif self.show_p['show_type'] == 'red_img':
            import seaborn as sns
            sns.set()
            plt.imshow(show_img)
        plt.axis('off')
        plt.show()

    def save(self):
        de_para_dict = {}
        for key in de_para_dict.keys():
            param_now = self.save_p.get(key,de_para_dict.get(key))
            self.save_p[key] = param_now
        



