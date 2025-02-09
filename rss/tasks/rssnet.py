from rss import toolbox,represent
from rss.represent.utils import to_device
import torch.nn as nn
import matplotlib.pyplot as plt
import torch as t
import numpy as np
import time
import dill as pkl
import os
import imageio
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.special import expit as sigmoid
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import mean_squared_error as ski_mse
from skimage.metrics import normalized_root_mse as ski_nrmse
t.backends.cudnn.enabled = True
t.backends.cudnn.benchmark = True 

class rssnet(object):
    def __init__(self,parameters,verbose=True) -> None:
        parameter_list = ['net_p','reg_p','data_p', 'noise_p', 'opt_p','train_p','show_p','save_p','task_p']
        self.init_parameter(parameters,parameter_list)
        self.init_task()
        self.init_net()
        self.init_reg()
        self.init_data()
        self.init_noise()
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
                - noise_p: The parameter used for the represent noise.
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

    def init_task(self):
        """Initialize task-specific parameters and settings."""
        de_para_dict = {
            'task_type': 'completion',  # 支持 'completion', 'denoising', 'fpr', 'gpr'
            'loss_type': 'mse',  # 默认使用MSE loss
            'metrics': ['psnr', 'nmae', 'rmse'],  # 默认评估指标
            'hyper_params': {}, # 超参数
            'gpu_id': 0  # 添加gpu_id参数,默认为0
        }
        
        for key in de_para_dict.keys():
            param_now = self.task_p.get(key, de_para_dict.get(key))
            self.task_p[key] = param_now
        
        # 如果需要使用lpips,提前检查依赖
        if 'lpips' in self.task_p['metrics']:
            try:
                self._has_lpips = True
            except ImportError:
                print("Warning: LPIPS metric is requested but 'torchmetrics' package is not installed. "
                      "Please install it with 'pip install torchmetrics'")
                self._has_lpips = False
        else:
            self._has_lpips = False

        # 根据任务不同初始化一些变量
        if self.task_p['task_type'] in ['fpr','gpr']:
            self.var_pr_target = self.data_train['obs_tensor'].reshape(self.data_p['data_shape'])
            self.var_pr_d = self.data_p['mask_shape'][1]
            self.var_pr_r = self.task_p['hyper_params']['r']
            self.var_pr_m = int(self.var_pr_r*self.var_pr_d)
            self.var_pr_I = t.ones((1,1,self.var_pr_m,self.var_pr_m))

    def init_net(self):
        de_para_dict = {'net_name':'SIREN','clip_if':False,'clip_min':0.0,'clip_max':1.0,'clip_mode':'hard'}
        for key in de_para_dict.keys():
            param_now = self.net_p.get(key,de_para_dict.get(key))
            self.net_p[key] = param_now
        # 使用task_p中的gpu_id
        self.net_p['gpu_id'] = self.task_p['gpu_id']
        self.net = represent.get_nn(self.net_p)
        self.net = to_device(self.net,self.net_p['gpu_id'])

    def init_reg(self):
        de_para_dict = {'reg_name':None}
        for key in de_para_dict.keys():
            param_now = self.reg_p.get(key,de_para_dict.get(key))
            self.reg_p[key] = param_now
        # 使用task_p中的gpu_id
        self.reg_p['gpu_id'] = self.task_p['gpu_id']
        if self.reg_p['reg_name'] != None:
            self.reg = represent.get_reg(self.reg_p)
            self.reg = to_device(self.reg,self.reg_p['gpu_id'])

    def init_data(self):
        # 保留所有默认参数
        de_para_dict = {
            'data_path': None,
            'data_type': 'syn',
            'data_shape': (10,10),
            'down_sample': [1,1,1],
            'mask_type': 'random',
            'random_rate': 0.0,
            'mask_path': None,
            'mask_shape': 'same',
            'seeds': 88,
            'down_sample_rate': 2,
            'mask_unobs_path': None,
            'noise_mode': None,
            'noise_parameter': 0.0,
            'x_mode': 'inr',
            'batch_size': 128,
            'shuffle_if': False,
            'xrange': 1,
            'ymode': 'completion',  # 保留默认值
            'return_data_type': 'tensor',
            'pre_full': False,
            'out_dim_one': True,
            'mat_get_func': lambda x: x
        }
        
        # 如果task_p中有task_type,则覆盖ymode
        if hasattr(self, 'task_p') and 'task_type' in self.task_p:
            de_para_dict['ymode'] = self.task_p['task_type']
        
        for key in de_para_dict.keys():
            param_now = self.data_p.get(key, de_para_dict.get(key))
            self.data_p[key] = param_now

        # print('data_p : ',self.data_p)
        self.data = toolbox.load_data(data_path=self.data_p['data_path'],data_type=self.data_p['data_type'],
                                      data_shape=self.data_p['data_shape'],down_sample=self.data_p['down_sample'],
                                      mat_get_func=self.data_p['mat_get_func'])
        if self.data_p['data_shape'] == None:
            self.data_p['data_shape'] = self.data.shape
        
        # 根据task_type决定mask生成方式
        if self.task_p['task_type'] in ['fpr', 'gpr']:
            # 对于fpr和gpr任务,使用全1的mask
            self.mask = t.ones(self.data.shape)
        else:
            # 其他任务使用原有的mask生成逻辑
            self.mask = toolbox.load_mask(mask_type=self.data_p['mask_type'],
                                        random_rate=self.data_p['random_rate'],
                                        mask_path=self.data_p['mask_path'],
                                        data_shape=self.data.shape,
                                        mask_shape=self.data_p['mask_shape'],
                                        seeds=self.data_p['seeds'],
                                        down_sample_rate=self.data_p['down_sample_rate'],
                                        gpu_id=self.task_p['gpu_id'])
        
        self.mask = to_device(t.tensor(self.mask>0.5).to(t.float32),self.net_p['gpu_id'])
        
        # mask_unobs的生成保持不变
        if self.data_p['mask_unobs_path'] == None:
            self.mask_unobs = 1-self.mask
        else:
            self.mask_unobs = toolbox.load_mask(mask_type=self.data_p['mask_type'],
                                              random_rate=self.data_p['random_rate'],
                                              mask_path=self.data_p['mask_unobs_path'],
                                              data_shape=self.data.shape,
                                              mask_shape=self.data_p['mask_shape'],
                                              seeds=self.data_p['seeds'],
                                              down_sample_rate=self.data_p['down_sample_rate'],
                                              gpu_id=self.task_p['gpu_id'])
            self.mask_unobs = to_device(t.tensor(self.mask_unobs>0.5).to(t.float32),self.net_p['gpu_id'])
        
        self.data_noise = toolbox.add_noise(self.data,mode=self.data_p['noise_mode'],parameter=self.data_p['noise_parameter'],
                                            seeds=self.data_p['seeds'])
        self.data_train = toolbox.get_dataloader(
            x_mode=self.data_p['x_mode'],
            batch_size=self.data_p['batch_size'],
            shuffle_if=self.data_p['shuffle_if'],
            data=self.data,
            mask=self.mask,
            xrange=self.data_p['xrange'],
            noisy_data=self.data_noise,
            ymode=self.data_p['ymode'],
            return_data_type=self.data_p['return_data_type'],
            gpu_id=self.net_p['gpu_id'],
            out_dim_one=self.data_p['out_dim_one']
        )

    def init_noise(self):
        de_para_dict = {'noise_term':False,'sparse_coef':1, 'parameter_type': 'matrix', 'init_std': 1e-3}
        for key in de_para_dict.keys():
            param_now = self.noise_p.get(key,de_para_dict.get(key))
            self.noise_p[key] = param_now
        if self.noise_p['noise_term'] == True:
            if self.noise_p['parameter_type'] =='matrix':
                # 使用 to_device 替换 .to()
                noise_data = t.randn(self.data_p['data_shape']).to(t.float32)*self.noise_p['init_std']
                noise_data = to_device(noise_data, self.net_p['gpu_id'])
                self.noise = nn.Parameter(noise_data, requires_grad=True)
            elif self.noise_p['parameter_type'] == 'implicit':
                # 使用 to_device 替换 .to()
                noise1_data = t.randn(self.data_p['data_shape']).to(t.float32)*self.noise_p['init_std']
                noise1_data = to_device(noise1_data, self.net_p['gpu_id'])
                self.noise1 = nn.Parameter(noise1_data, requires_grad=True)
                
                noise2_data = t.randn(self.data_p['data_shape']).to(t.float32)*self.noise_p['init_std']
                noise2_data = to_device(noise2_data, self.net_p['gpu_id'])
                self.noise2 = nn.Parameter(noise2_data, requires_grad=True)
                self.noise = self.noise1**2 - self.noise2**2
            else:
                raise ValueError('parameter_type should be matrix or implicit')

    def init_opt(self):
        de_para_dict = {'net':{'opt_name':'Adam','lr':1e-4,'weight_decay':0},
                        'reg':{'opt_name':'Adam','lr':1e-4,'weight_decay':0},
                        'noise':{'opt_name':'Adam','lr':1e-4,'weight_decay':0}}
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
        if self.noise_p['noise_term'] == True:
            if self.noise_p['parameter_type'] =='matrix':
                self.noise_opt = represent.get_opt(opt_type=self.opt_p['noise']['opt_name'],
                                                parameters=[self.noise],lr=self.opt_p['noise']['lr'],
                                                weight_decay=self.opt_p['noise']['weight_decay'])
            elif self.noise_p['parameter_type'] == 'implicit':
                self.noise_opt = represent.get_opt(opt_type=self.opt_p['noise']['opt_name'],
                                                parameters=[self.noise1,self.noise2],lr=self.opt_p['noise']['lr'],
                                                weight_decay=self.opt_p['noise']['weight_decay'])
            else:
                raise ValueError('parameter_type should be matrix or implicit')

    def init_train(self):
        # 保留所有默认参数
        de_para_dict = {
            'task_name': 'completion',  # 保留默认值
            'train_epoch': 10,
            'loss_fn': 'mse',  # 保留默认值
            'rmse_round': False
        }
        
        # 如果task_p中有相应设置,则覆盖默认值
        if hasattr(self, 'task_p'):
            if 'task_type' in self.task_p:
                de_para_dict['task_name'] = self.task_p['task_type']
            if 'loss_type' in self.task_p:
                de_para_dict['loss_fn'] = self.task_p['loss_type']
        
        for key in de_para_dict.keys():
            param_now = self.train_p.get(key, de_para_dict.get(key))
            self.train_p[key] = param_now

        # 根据train_p['loss_fn']设置loss function
        # 这样保持了向后兼容性
        if self.train_p['loss_fn'] == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.train_p['loss_fn'] == 'mae':
            self.loss_fn = nn.L1Loss()
        # 可以添加其他 loss types...

    def init_save(self):
        de_para_dict = {'save_if':False,'save_path':''}
        for key in de_para_dict.keys():
            param_now = self.save_p.get(key,de_para_dict.get(key))
            self.save_p[key] = param_now

    def init_show(self):
        de_para_dict = {'show_type':'gray_img','show_content':'original'}
        for key in de_para_dict.keys():
            param_now = self.show_p.get(key,de_para_dict.get(key))
            self.show_p[key] = param_now

    def train(self, verbose=True):
        """Train the model."""
        self._prepare_training()
        
        for ite in range(self.train_p['train_epoch']):
            # Log training time
            time_now = time.time()
            self.log('time', time_now-self.start_time)
            
            # Forward pass and loss computation
            self._forward_backward_optimize()
            
            # Evaluate and log metrics
            if ite % (self.train_p['train_epoch']//10) == 0:
                self._evaluate_and_log(ite)
                
        # Finalize training
        self._finalize_training(verbose)

    def _forward_backward_optimize(self):
        """Forward pass, loss computation, backward pass, and optimization step."""
        if self.task_p['task_type'] in ['completion','denoising']:
            pre, reg_tensor = self._forward_pass()
            loss = self._compute_loss(pre, reg_tensor)
            self._backward_and_optimize(loss)
        elif self.task_p['task_type'] in ['fpr','gpr']:
            numit_inner = self.task_p.get('hyper_params', {}).get('numit_inner', 5)

            if not hasattr(self, 'var_pr_v'):
                self.var_pr_v = 0
            for _ in range(numit_inner):
                pre, reg_tensor = self._forward_pass()
                loss = self._compute_loss(pre, reg_tensor)
                self._backward_and_optimize(loss)

    def _prepare_training(self):
        """Prepare for training by initializing necessary variables."""
        if self.data_p['return_data_type'] == 'random':
            self.unn_index = 0
        else:
            self.unn_index = 1
        
        if (not hasattr(self, 'log_dict')) or ('time' not in self.log_dict):
            self.start_time = time.time()
        
        full_nets_list = ['UNet', 'ResNet', 'skip', 'DMF', 'TF', 'KNN', 'TIP']
        self.full_pre_if = (self.net_p['net_name'] in full_nets_list) or \
                           (self.net_p['net_name']=='KNN' and self.net_p['mode'] in full_nets_list) or \
                           (self.net_p['net_name']=='composition' and self.net_p['net_list'][0]['net_name'] in full_nets_list)

    def _forward_pass(self):
        """Execute forward pass and return predictions."""
        if self.full_pre_if:
            pre = self.net(self.data_train['obs_tensor'][self.unn_index].reshape(1,-1,self.data_p['data_shape'][0],
                                                                                 self.data_p['data_shape'][1]))
            pre = pre.reshape(self.data_p['data_shape'])
            reg_tensor = pre.reshape(self.data_p['data_shape'])
            pre = pre[self.mask==1]
        elif self.reg_p['reg_name'] != None:
            pre = self.net(self.data_train['obs_tensor'][0])
            reg_tensor = pre.reshape(self.data_p['data_shape'])
        else:
            pre = self.net(self.data_train['obs_tensor'][0][(self.mask==1).reshape(-1)])
            reg_tensor = None
        if self.task_p['task_type'] in ['fpr','gpr']:
            pre = pre.reshape(self.data_p['data_shape'])
        return pre, reg_tensor

    def _compute_loss(self, pre, reg_tensor):
        """Compute training loss including regularization if applicable."""
        loss = 0
        
        # Add regularization loss if applicable
        if self.reg_p['reg_name'] != None:
            reg_loss = self.reg(reg_tensor)
            loss += reg_loss
            self.log('reg_loss', reg_loss)
            if not self.full_pre_if:
                pre = pre[(self.mask).reshape(pre.shape)==1]
        
        # Compute main loss
        target = self.data_train['obs_tensor'][1][(self.mask==1).reshape(-1)].reshape(pre.shape)
        if self.noise_p['noise_term']:
            if self.noise_p['parameter_type'] == 'implicit':
                self.noise = self.noise1**2 - self.noise2**2
            loss += self.loss_fn(pre+self.noise[(self.mask==1)].reshape(pre.shape), target)
            if self.noise_p['parameter_type'] == 'matrix':
                loss += self.noise_p['sparse_coef']*t.mean(t.abs(self.noise))
        else:
            loss += self.loss_fn(pre, target)
        
        return loss

    def _backward_and_optimize(self, loss):
        """Execute backward pass and optimization step."""
        self.net_opt.zero_grad()
        if self.train_reg_if:
            self.reg_opt.zero_grad()
        if self.noise_p['noise_term']:
            self.noise_opt.zero_grad()
        
        loss.backward()
        
        self.net_opt.step()
        if self.train_reg_if:
            self.reg_opt.step()
        if self.noise_p['noise_term']:
            self.noise_opt.step()

    def _evaluate_and_log(self, iteration):
        """Evaluate model and log metrics."""
        with t.no_grad():
            # Get predictions
            if self.net_p['net_name'] in ['UNet','ResNet','skip'] or \
               (self.net_p['net_name']=='KNN' and self.net_p['mode'] in ['UNet','ResNet','skip']):
                pre = self.net(self.data_train['obs_tensor'][self.unn_index].reshape(1,-1,self.data_p['data_shape'][0],self.data_p['data_shape'][1]))
            else:
                pre = self.net(self.data_train['obs_tensor'][0])
            
            # Calculate validation loss
            if self.task_p['task_type'] == 'completion':
                pre_val = pre.reshape(self.data_p['data_shape'])[self.mask_unobs==1]
            else:
                pre_val = pre
            target = self.data_train['real_tensor'][1]
            if self.task_p['task_type'] == 'completion':
                target = target[(self.mask_unobs==1).reshape(-1)].reshape(pre_val.shape)
            else:
                target = target.reshape(pre_val.shape)
            
            val_loss = self.loss_fn(pre_val, target)
            self.log('val_loss', val_loss.item())
            
            # Calculate test metrics
            self.pre = pre
            self.target = target
            if self.task_p['task_type'] == 'completion':
                test_loss = self.loss_fn(pre_val, target)
                # For metrics calculation, reshape pre to match target
                pre_metrics = pre.reshape(self.data_p['data_shape'])
                target_metrics = self.data_train['real_tensor'][1].reshape(self.data_p['data_shape'])
            else:
                test_loss = self.loss_fn(pre, target.reshape(pre.shape))
                pre_metrics = pre
                target_metrics = target.reshape(pre.shape)
            
            # Log test loss
            self.log('test_loss', test_loss.item())
            
            # 更新指标计算方法映射
            metric_funcs = {
                'psnr': self.cal_psnr,
                'nmae': self.cal_nmae,
                'rmse': self.cal_rmse,
                'auc': self.cal_auc,
                'aupr': self.cal_aupr,
                'ssim': self.cal_ssim,
                'mse': self.cal_mse,
                'mae': self.cal_mae,
                'nrmse': self.cal_nrmse,
                'lpips': self.cal_lpips
            }
            
            # 根据metrics列表计算并记录指标
            for metric in self.task_p['metrics']:
                if metric in metric_funcs:
                    metric_value = metric_funcs[metric](pre_metrics, target_metrics)
                    # 只记录非None的指标值
                    if metric_value is not None:
                        self.log(metric, metric_value)
                else:
                    print(f"Warning: Metric '{metric}' is not supported")
            
            if (iteration+1)%(self.train_p['train_epoch']//10) == 0:
                self.log('img')

    def _finalize_training(self, verbose):
        """Finalize training by saving logs and printing results if requested."""
        if self.save_p['save_if']:
            self.save_logs(verbose=verbose)
        
        if verbose:
            print('loss on test set', self.log_dict['test_loss'][-1])
            print('PSNR=', self.log_dict['psnr'][-1], 'dB')
            print('NMAE=', self.log_dict['nmae'][-1])
            print('RMSE=', self.log_dict['rmse'][-1])
            if self.reg_p['reg_name'] != None:
                print('loss of regularizer', self.log_dict['reg_loss'][-1])

    def log(self,name,content=None):
        if 'log_dict' not in self.__dict__:
            self.log_dict = {'parameter_all':self.parameter_all}
        if name == 'img':
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
            content = show_img
        if name not in self.log_dict:
            self.log_dict[name] = [content]
        else:
            self.log_dict[name].append(content)

    def show(self):
        de_para_dict = {'show_type':'gray_img','show_content':'original','show_axis':False, 'show_info_on_img':False}
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
            if self.task_p['task_type'] == 'completion':
                show_img = show_img*self.mask.reshape(self.data_p['data_shape']).detach().cpu().numpy()

        else:
            raise('Wrong show_content in show_p:',self.show_p['show_content'])
        if self.show_p['show_type'] == 'gray_img':
            plt.imshow(show_img,'gray',vmin=0,vmax=1)
        elif self.show_p['show_type'] == 'red_img':
            import seaborn as sns
            sns.set()
            plt.imshow(show_img,vmin=0,vmax=1)
        else:
            raise('Wrong show_type in show_p:',self.show_p['show_type'])
        if self.show_p['show_axis'] == False:
            plt.axis('off')
        if self.show_p['show_info_on_img']:
            try:
                epoch = len(self.log_dict['psnr'])
                psnr = self.log_dict['psnr'][-1]
                psnr = round(psnr,2)
            except:
                epoch = 0
                psnr = 0
            plt.text(20, 40, 'Epoch='+str(epoch)+'\nPSNR='+str(psnr)+'dB', color='white', fontsize=12, backgroundcolor='black')
        if self.save_p['save_if'] == True:
            if self.save_p['save_path'].split('.')[-1] in ['png','jpg','jpeg']:
                save_img_path = self.save_p['save_path']
            else:
                try:
                    epoch = len(self.log_dict['psnr'])
                    psnr = self.log_dict['psnr'][-1]
                    psnr = round(psnr,2)
                except:
                    epoch = 0
                    psnr = 0
                save_img_path = self.save_p['save_path']+'epoch_'+str(epoch)+'.png'
                if not os.path.exists(self.save_p['save_path']):
                    os.makedirs(self.save_p['save_path'])
            plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0)
        if self.noise_p['noise_term'] == True:
            print('noise_mean',t.abs(self.noise.mean()).item())
        plt.show()
        
    def save_logs(self,verbose=True):
        # 检测文件夹是否存在，不存在则创建
        if not os.path.exists(self.save_p['save_path']):
            os.makedirs(self.save_p['save_path'])
        with open(self.save_p['save_path']+"logs.pkl", "wb") as f:
            # print('self.log_dict',self.log_dict)
            pkl.dump(self.log_dict, f)
            if verbose == True:
                print('save logs to',self.save_p['save_path']+"logs.pkl")

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
        
        return psnr(pre, target).item()
    
    def cal_nmae(self,pre, target):
        max_pixel,min_pixel = t.max(target),t.min(target)
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num<1e-3:
            return 0
        else:
            result = t.sum(t.abs((pre-target)*(self.mask_unobs).reshape(pre.shape))/unseen_num/(max_pixel-min_pixel))
            return result.item()

    def cal_rmse(self, pre, target):
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0
        else:
            target_max = target.max()
            if target_max > 2:
                target_min = 1
            else:
                target_min = 0
            if self.train_p['rmse_round']:
                squared_diff = (t.clamp(t.round(pre),target_min,target_max) - target) ** 2
            else:
                squared_diff = (t.clamp(pre,target_min,target_max) - target) ** 2
            masked_squared_diff = squared_diff * (self.mask_unobs).reshape(pre.shape)
            mse = t.sum(masked_squared_diff) / unseen_num
            rmse = t.sqrt(mse)
            return rmse.item()

    def cal_auc(self, pre, target):
        """Calculate AUC score for binary classification.
        
        Args:
            pre: Model predictions
            target: Ground truth labels
            
        Returns:
            float: AUC score
        """
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0
        else:
            # 获取未观测区域的预测和真实值
            masked_pre = pre * (self.mask_unobs).reshape(pre.shape)
            masked_target = target * (self.mask_unobs).reshape(target.shape)
            
            # 转换为numpy数组
            masked_pre_np = masked_pre.cpu().detach().numpy().flatten()
            masked_target_np = masked_target.cpu().detach().numpy().flatten()
            
            # 检查是否为有效的二分类问题
            unique_labels = np.unique(masked_target_np)
            if len(unique_labels) < 2:
                return 0.5  # 当只有一个类别时返回0.5
            
            # 将预测值转换为概率
            masked_pre_np = sigmoid(masked_pre_np)
            
            try:
                # 计算AUC
                auc_score = roc_auc_score(masked_target_np, masked_pre_np)
                return auc_score
            except ValueError:
                return 0.5  # 处理异常情况

    def cal_aupr(self, pre, target):
        """Calculate Area Under Precision-Recall Curve (AUPR) for binary classification.
        
        Args:
            pre: Model predictions
            target: Ground truth labels
            
        Returns:
            float: AUPR score
        """
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0
        else:
            # 获取未观测区域的预测和真实值
            masked_pre = pre * (self.mask_unobs).reshape(pre.shape)
            masked_target = target * (self.mask_unobs).reshape(target.shape)
            
            # 转换为numpy数组
            masked_pre_np = masked_pre.cpu().detach().numpy().flatten()
            masked_target_np = masked_target.cpu().detach().numpy().flatten()
            
            # 检查是否为有效的二分类问题
            unique_labels = np.unique(masked_target_np)
            if len(unique_labels) < 2:
                return 0.5  # 当只有一个类别时返回0.5
            
            # 将预测值转换为概率
            masked_pre_np = sigmoid(masked_pre_np)
            
            try:
                # 计算Precision-Recall曲线
                precision, recall, _ = precision_recall_curve(masked_target_np, masked_pre_np)
                # 计算AUPR
                aupr_score = auc(recall, precision)
                return aupr_score
            except ValueError:
                return 0.5  # 处理异常情况

    def gen_gif(self, fps=10, save_type = 'gif', start_frame=0, end_frame=None):
        # 获取文件夹中的所有文件
        files = os.listdir(self.save_p['save_path'])
        # 过滤出.png文件并排序
        png_files = sorted([f for f in files if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))
        # 生成完整的文件路径
        images = [imageio.imread(os.path.join(self.save_p['save_path'], file)) for file in png_files]
        if end_frame is None:
            images = images[start_frame:end_frame]
        # 生成GIF
        imageio.mimsave(self.save_p['save_path']+'result.'+save_type, images, fps=fps)

    def cal_ssim(self, pre, target):
        """Calculate SSIM (Structural Similarity Index) between prediction and target.
        
        Args:
            pre: Predicted image tensor
            target: Target image tensor
            
        Returns:
            float: SSIM value
        """
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 1.0
        else:
            # 转换为numpy数组
            pre_np = pre.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            
            # 确保数值范围在[0,1]之间
            pre_np = np.clip(pre_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            
            try:
                # 计算SSIM
                ssim_value = ssim(target_np, pre_np, 
                                data_range=1.0,
                                multichannel=False)
                return ssim_value
            except ValueError:
                return 0.0

    def cal_mse(self, pre, target):
        """Calculate MSE (Mean Squared Error) between prediction and target.
        
        Args:
            pre: Predicted image tensor
            target: Target image tensor
            
        Returns:
            float: MSE value
        """
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0.0
        else:
            # 获取未观测区域的预测和真实值
            masked_pre = pre * (self.mask_unobs).reshape(pre.shape)
            masked_target = target * (self.mask_unobs).reshape(target.shape)
            
            # 计算MSE
            mse_value = t.mean((masked_pre - masked_target) ** 2)
            return mse_value.item()

    def cal_mae(self, pre, target):
        """Calculate MAE (Mean Absolute Error) between prediction and target.
        
        Args:
            pre: Predicted image tensor
            target: Target image tensor
            
        Returns:
            float: MAE value
        """
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0.0
        else:
            # 获取未观测区域的预测和真实值
            masked_pre = pre * (self.mask_unobs).reshape(pre.shape)
            masked_target = target * (self.mask_unobs).reshape(target.shape)
            
            # 计算MAE
            mae_value = t.mean(t.abs(masked_pre - masked_target))
            return mae_value.item()

    def cal_nrmse(self, pre, target):
        """Calculate NRMSE (Normalized Root Mean Square Error) between prediction and target.
        
        Args:
            pre: Predicted image tensor
            target: Target image tensor
            
        Returns:
            float: NRMSE value
        """
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0.0
        else:
            # 转换为numpy数组
            pre_np = pre.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            
            try:
                # 计算NRMSE
                nrmse_value = ski_nrmse(target_np, pre_np, normalization='euclidean')
                return nrmse_value
            except ValueError:
                return 0.0

    def cal_lpips(self, pre, target):
        """Calculate LPIPS between prediction and target."""
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0.0
        
        if not self._has_lpips:
            return None
        
        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
            
            if not hasattr(self, '_lpips_model'):
                # 使用 to_device 替换 .to()
                lpips_model = LPIPS(net_type='alex')
                self._lpips_model = to_device(lpips_model, self.task_p['gpu_id'])
            
            pre_np = pre.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            
            pre_np = np.clip(pre_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            
            pre_t = t.from_numpy(pre_np).unsqueeze(0).unsqueeze(0) * 2 - 1
            target_t = t.from_numpy(target_np).unsqueeze(0).unsqueeze(0) * 2 - 1
            
            # 使用 to_device 替换 .to()
            pre_t = to_device(pre_t, self.task_p['gpu_id'])
            target_t = to_device(target_t, self.task_p['gpu_id'])
            
            with t.no_grad():
                lpips_value = self._lpips_model(pre_t, target_t).item()
            
            return lpips_value
            
        except Exception as e:
            print(f"Warning: Error calculating LPIPS: {str(e)}")
            return None