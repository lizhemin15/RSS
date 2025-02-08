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
from sklearn.metrics import roc_auc_score
t.backends.cudnn.enabled = True
t.backends.cudnn.benchmark = True 

class rssnet(object):
    def __init__(self,parameters,verbose=True) -> None:
        parameter_list = ['net_p','reg_p','data_p', 'noise_p', 'opt_p','train_p','show_p','save_p']
        self.init_parameter(parameters,parameter_list)
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

    def init_net(self):
        de_para_dict = {'net_name':'SIREN','gpu_id':0,'clip_if':False,'clip_min':0.0,'clip_max':1.0,'clip_mode':'hard'}
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
                        'mask_unobs_path':None,
                        'noise_mode':None,'noise_parameter':0.0,
                        'x_mode':'inr','batch_size':128,'shuffle_if':False,'xrange':1,'ymode':'completion','return_data_type':'tensor',
                        'pre_full':False,'out_dim_one':True,'mat_get_func':lambda x: x}
        for key in de_para_dict.keys():
            param_now = self.data_p.get(key,de_para_dict.get(key))
            self.data_p[key] = param_now
        # print('data_p : ',self.data_p)
        self.data = toolbox.load_data(data_path=self.data_p['data_path'],data_type=self.data_p['data_type'],
                                      data_shape=self.data_p['data_shape'],down_sample=self.data_p['down_sample'],
                                      mat_get_func=self.data_p['mat_get_func'])
        if self.data_p['data_shape'] == None:
            self.data_p['data_shape'] = self.data.shape
        self.mask = toolbox.load_mask(mask_type=self.data_p['mask_type'],random_rate=self.data_p['random_rate'],mask_path=self.data_p['mask_path'],
                                      data_shape=self.data.shape,mask_shape=self.data_p['mask_shape'],seeds=self.data_p['seeds'],
                                      down_sample_rate=self.data_p['down_sample_rate'],gpu_id=self.net_p['gpu_id'])
        self.mask = to_device(t.tensor(self.mask>0.5).to(t.float32),self.net_p['gpu_id'])
        if self.data_p['mask_unobs_path'] == None:
            self.mask_unobs = 1-self.mask
        else:
            self.mask_unobs = toolbox.load_mask(mask_type=self.data_p['mask_type'],random_rate=self.data_p['random_rate'],mask_path=self.data_p['mask_unobs_path'],
                                      data_shape=self.data.shape,mask_shape=self.data_p['mask_shape'],seeds=self.data_p['seeds'],
                                      down_sample_rate=self.data_p['down_sample_rate'],gpu_id=self.net_p['gpu_id'])
            self.mask_unobs = to_device(t.tensor(self.mask_unobs>0.5).to(t.float32),self.net_p['gpu_id'])
        
        self.data_noise = toolbox.add_noise(self.data,mode=self.data_p['noise_mode'],parameter=self.data_p['noise_parameter'],
                                            seeds=self.data_p['seeds'])
        self.data_train = toolbox.get_dataloader(x_mode=self.data_p['x_mode'],batch_size=self.data_p['batch_size'],
                                                 shuffle_if=self.data_p['shuffle_if'],
                                                data=self.data,mask=self.mask,xrange=self.data_p['xrange'],noisy_data=self.data_noise,
                                                ymode=self.data_p['ymode'],return_data_type=self.data_p['return_data_type'],
                                                gpu_id=self.net_p['gpu_id'],out_dim_one=self.data_p['out_dim_one'])

    def init_noise(self):
        de_para_dict = {'noise_term':False,'sparse_coef':1, 'parameter_type': 'matrix', 'init_std': 1e-3}
        for key in de_para_dict.keys():
            param_now = self.noise_p.get(key,de_para_dict.get(key))
            self.noise_p[key] = param_now
        if self.noise_p['noise_term'] == True:
            if self.noise_p['parameter_type'] =='matrix':
                self.noise = nn.Parameter(t.randn(self.data_p['data_shape']).to(t.float32)*self.noise_p['init_std'], requires_grad=True)
                self.noise.data = to_device(self.noise.data,self.net_p['gpu_id'])
            elif self.noise_p['parameter_type'] == 'implicit':
                self.noise1 = nn.Parameter(t.randn(self.data_p['data_shape']).to(t.float32)*self.noise_p['init_std'], requires_grad=True)
                self.noise1.data = to_device(self.noise1.data,self.net_p['gpu_id'])
                self.noise2 = nn.Parameter(t.randn(self.data_p['data_shape']).to(t.float32)*self.noise_p['init_std'], requires_grad=True)
                self.noise2.data = to_device(self.noise2.data,self.net_p['gpu_id'])
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
        de_para_dict = {'task_name':self.data_p['ymode'],'train_epoch':10,'loss_fn':'mse'}
        for key in de_para_dict.keys():
            param_now = self.train_p.get(key,de_para_dict.get(key))
            self.train_p[key] = param_now
        if self.train_p['loss_fn'] == 'mse':
            self.loss_fn = nn.MSELoss()
        self.rmse_round = self.train_p.get('rmse_round',False)


        # print('train_p : ',self.train_p)

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
            pre, reg_tensor = self._forward_pass() 
            loss = self._compute_loss(pre, reg_tensor)
            
            # Backward and optimize
            self._backward_and_optimize(loss)
            
            # Evaluate and log metrics
            if ite % (self.train_p['train_epoch']//10) == 0:
                self._evaluate_and_log(ite)
                
        # Finalize training
        self._finalize_training(verbose)

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
            pre = self.net(self.data_train['obs_tensor'][self.unn_index].reshape(1,-1,self.data_p['data_shape'][0],self.data_p['data_shape'][1]))
            pre = pre.reshape(self.data_p['data_shape'])
            reg_tensor = pre.reshape(self.data_p['data_shape'])
            pre = pre[self.mask==1]
        elif self.reg_p['reg_name'] != None:
            pre = self.net(self.data_train['obs_tensor'][0])
            reg_tensor = pre.reshape(self.data_p['data_shape'])
        else:
            pre = self.net(self.data_train['obs_tensor'][0][(self.mask==1).reshape(-1)])
            reg_tensor = None
        
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
            if self.data_p['ymode'] == 'completion':
                pre_val = pre.reshape(self.data_p['data_shape'])[self.mask_unobs==1]
            else:
                pre_val = pre
            target = self.data_train['real_tensor'][1]
            if self.data_p['ymode'] == 'completion':
                target = target[(self.mask_unobs==1).reshape(-1)].reshape(pre_val.shape)
            else:
                target = target.reshape(pre_val.shape)
            
            val_loss = self.loss_fn(pre_val, target)
            self.log('val_loss', val_loss.item())
            
            # Calculate test metrics
            self.pre = pre
            self.target = target
            if self.data_p['ymode'] == 'completion':
                test_loss = self.loss_fn(pre_val, target)
                # For metrics calculation, reshape pre to match target
                pre_metrics = pre.reshape(self.data_p['data_shape'])
                target_metrics = self.data_train['real_tensor'][1].reshape(self.data_p['data_shape'])
            else:
                test_loss = self.loss_fn(pre, target.reshape(pre.shape))
                pre_metrics = pre
                target_metrics = target.reshape(pre.shape)
            
            # Log metrics
            self.log('test_loss', test_loss.item())
            self.log('psnr', self.cal_psnr(pre_metrics, target_metrics))
            self.log('nmae', self.cal_nmae(pre_metrics, target_metrics))
            self.log('rmse', self.cal_rmse(pre_metrics, target_metrics))
            
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
            if self.data_p['ymode'] == 'completion':
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
            if self.rmse_round:
                squared_diff = (t.clamp(t.round(pre),target_min,target_max) - target) ** 2
            else:
                squared_diff = (t.clamp(pre,target_min,target_max) - target) ** 2
            masked_squared_diff = squared_diff * (self.mask_unobs).reshape(pre.shape)
            mse = t.sum(masked_squared_diff) / unseen_num
            rmse = t.sqrt(mse)
            return rmse.item()

    def cal_auc(self, pre, target):
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0
        else:
            masked_pre = pre * (self.mask_unobs).reshape(pre.shape)
            masked_target = target * (self.mask_unobs).reshape(target.shape)
            # 将predictions和targets转换成numpy数组以便使用sklearn的roc_auc_score
            masked_pre_np = masked_pre.cpu().detach().numpy()
            masked_target_np = masked_target.cpu().detach().numpy()
            # 计算AUC
            auc = roc_auc_score(masked_target_np, masked_pre_np)
            return auc

    def cal_aupr(self, pre, target):
        unseen_num = t.sum(self.mask_unobs)
        if unseen_num < 1e-3:
            return 0
        else:
            masked_pre = pre * (self.mask_unobs).reshape(pre.shape)
            masked_target = target * (self.mask_unobs).reshape(target.shape)
            # 将predictions和targets转换成numpy数组以便使用sklearn的precision_recall_curve
            masked_pre_np = masked_pre.cpu().detach().numpy()
            masked_target_np = masked_target.cpu().detach().numpy()
            # 计算Precision-Recall曲线
            precision, recall, _ = precision_recall_curve(masked_target_np, masked_pre_np)
            # 计算AUPR
            aupr = auc(recall, precision)
            return aupr

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