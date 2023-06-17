""" rss
"""
__title__ = 'rss'
__version__ = '0.0.1'
#__build__ = 0x021300
__author__ = 'Zhemin Li'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023 Zhemin Li'


## Top Level Modules
from rss import toolbox,represent,tasks
# from rss.represent import get_nn
__all__ = []

class rssnet(object):
    def __init__(self,parameters) -> None:
        self.init_parameter(parameters)
        self.init_net()
        self.init_reg()
        self.init_data()
        self.init_opt()



    def init_parameter(self,parameters):
        """
        Initialize a parameter.

        Args:
            key (str): The key of the parameter to initialize. Options are:
                - net_p: The parameter used for the network.
                - reg_p: The parameter used for the regularization.
                - data_p: The parameter used for the data.
                - opt_p: The parameter used for the task, loss is constructed here, and device is selected.
                - train_p: The parameter used for the training.
                - show_p: The parameter used for showing information.
                - save_p: The parameter used for saving.

        Returns:
            parameter (dict): The initialization key
        Raises:
            ValueError: If the input key is not one of the options.
        """
        parameter_list = ['net_p','reg_p','data_p','task_p','train_p','show_p','save_p']
        for key in parameter_list:
            param_now = parameters.get(key,{})
            setattr(self,key,param_now)
            

    def init_net(self):
        de_para_dict = {'net_name':'SIREN'}
        for key in de_para_dict.keys():
            param_now = self.net_p.get(key,de_para_dict.get(key))
            self.net_p[key] = param_now
        self.net = represent.get_nn(self.net_p)

    def init_reg(self):
        pass

    def init_data(self):
        de_para_dict = {'data_path':None,'data_type':'syn','data_shape':(256,256),'down_sample':[1,1,1],
                        'mask_type':'random','random_rate':0.0,'mask_path':None,'mask_shape':'same','seeds':88,'down_sample_rate':2,
                        'noise_mode':None,'noise_parameter':0.0,
                        'x_mode':'inr','batch_size':128,'shuffle_if':False,'xrange':1,'ymode':'completion'}
        for key in de_para_dict.keys():
            param_now = self.data_p.get(key,de_para_dict.get(key))
            self.data_p[key] = param_now
        print('data_p : ',self.data_p)
        # all the data are numpy on cpu
        self.data = toolbox.load_data(data_path=self.data_p['data_path'],data_type=self.data_p['data_type'],
                                      data_shape=self.data_p['data_shape'],down_sample=self.data_p['down_sample'])
        self.mask = toolbox.load_mask(mask_type=self.data_p['mask_type'],random_rate=self.data_p['random_rate'],mask_path=self.data_p['mask_path'],
                                      data_shape=self.data.shape,mask_shape=self.data_p['mask_shape'],seeds=self.data_p['seeds'],down_sample_rate=self.data_p['down_sample_rate'])
        self.data_noise = toolbox.add_noise(self.data,mode=self.data_p['noise_mode'],parameter=self.data_p['noise_parameter'],seeds=self.data_p['seeds'])
        self.data_train = toolbox.get_dataloader(x_mode=self.data_p['x_mode'],batch_size=self.data_p['batch_size'],
                                                 shuffle_if=self.data_p['shuffle_if'],
                                                data=self.data,mask=self.mask,xrange=self.data_p['xrange'],noisy_data=self.data_noise,
                                                ymode=self.data_p['ymode'])
        

    def init_opt(self):
        pass

    def train(self):
        pass

    def show(self):
        pass

    def save(self):
        pass



