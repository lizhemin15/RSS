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
        self.init_noise()
        self.init_reg()
        self.init_data()
        self.init_task()



    def init_parameter(self,parameters):
        """
        Initialize a parameter.

        Args:
            key (str): The key of the parameter to initialize. Options are:
                - net_p: The parameter used for the network.
                - noise_p: The parameter used for the noise.
                - reg_p: The parameter used for the regularization.
                - data_p: The parameter used for the data.
                - task_p: The parameter used for the task.
                - train_p: The parameter used for the training.
                - show_p: The parameter used for showing information.
                - save_p: The parameter used for saving.

        Returns:
            parameter (dict): The initialization key
        Raises:
            ValueError: If the input key is not one of the options.
        """
        parameter_list = ['net_p','noise_p','reg_p','data_p','task_p','train_p','show_p','save_p']
        for key in parameter_list:
            if key == 'net_p':
                default_param = {'net_name':'SIREN'}
            elif key == 'noise_p':
                default_param = 'b'
            elif key == 'reg_p':
                default_param = {'s':1}
            elif key == 'data_p':
                default_param = None
            elif key == 'task_p':
                default_param = None
            elif key == 'train_p':
                default_param = None
            elif key == 'show_p':
                default_param = None
            elif key == 'save_p':
                default_param = None
            else:
                raise('Wrong key = ',key)
            param_now = parameters.get(key,default_param)
            setattr(self,key,param_now)
            print(key,':',param_now)
        return None

    def init_net(self):
        self.net = represent.get_nn(self.net_p)

    def init_noise(self):
        pass

    def init_reg(self):
        pass

    def init_data(self):
        pass

    def init_task(self):
        pass

    def train(self):
        pass

    def show(self):
        pass

    def save(self):
        pass



def go():
    print('gogogo')
    pass
