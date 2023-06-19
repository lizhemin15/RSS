import torch.nn as nn

class DMF(nn.Module):
    def __init__(self,params):
        # Initial the parameter (Deep linear network)
        super().__init__()
        hidden_sizes = params.get('sizes',[])
        hidden_sizes.reverse()
        std_w = params.get('std_w',1e-3)
        layers = zip(hidden_sizes, hidden_sizes[1:])
        nn_list = []
        for (f_in,f_out) in layers:
            nn_list.append(nn.Linear(f_in, f_out, bias=False))
        self.model = nn.Sequential(*nn_list)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=std_w)

    def forward(self,*args):
        # Initial data
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





