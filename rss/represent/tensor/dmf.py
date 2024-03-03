import torch.nn as nn
from rss.represent.inr import SIREN
import torch

class DMF_net(nn.Module):
    def __init__(self,params):
        # Initial the parameter (Deep linear network)
        super().__init__()
        hidden_sizes = params.get('sizes',[])
        hidden_sizes.reverse()
        self.mode = params.get('mode','vanilla')
        if self.mode == 'vanilla':
            std_w = params.get('std_w',1e-3)
            layers = zip(hidden_sizes, hidden_sizes[1:])
            nn_list = []
            for (f_in,f_out) in layers:
                nn_list.append(nn.Linear(f_in, f_out, bias=False))
            self.model = nn.Sequential(*nn_list)
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight,mean=0,std=std_w)
        elif self.mode == 'inr':
            net_list = []
            self.input_list = []
            net_list.append(SIREN({'dim_in':1,'dim_hidden':256,'dim_out':hidden_sizes[0],'num_layers':2,'w0':1,'w0_initial':30.,'use_bias':True}))
            self.input_list.append(torch.linspace(-1,1,hidden_sizes[0]).reshape(-1,1))
            net_list.append(SIREN({'dim_in':1,'dim_hidden':256,'dim_out':hidden_sizes[-1],'num_layers':2,'w0':1,'w0_initial':30.,'use_bias':True}))
            self.input_list.append(torch.linspace(-1,1,hidden_sizes[-1]).reshape(-1,1))
            self.net_list = nn.ModuleList(net_list)
        else:
            raise('Do not support mode named ',self.mode)

    def forward(self,*args):
        if self.mode == 'vanilla':
            def get_e2e(model):
                #获取预测矩阵
                weight = None
                for fc in model.children():
                    assert isinstance(fc, nn.Linear) and fc.bias is None
                    if weight is None:
                        weight = fc.weight.t()
                    else:
                        weight = fc(weight)
                return weight
            return get_e2e(self.model)
        elif self.mode == 'inr':
            pre = []
            for i in range(len(self.net_list)):
                net_now = self.net_list[i]
                input_now = self.input_list[i].to(net_now.layers[0].weight.device)
                pre.append(net_now(input_now))
            self.pre = pre
            return self.pre[0]@self.pre[1].T
        else:
            raise('Do not support mode named ',self.mode)



def DMF(parameter):
    de_para_dict = {'sizes':[],'std_w':1e-3}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return DMF_net(parameter)

