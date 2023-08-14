import torch.nn as nn
import torch as t
import numpy as np
from einops import rearrange
from rss.represent import get_nn

def to_device(obj,device):
    if t.cuda.is_available() and device != 'cpu':
        obj = obj.cuda(device)
    return obj

def add_space(sent):
    sent_new = ''
    for i in range(2*len(sent)):
        if i%2 == 0:
            sent_new += sent[i//2]
        else:
            sent_new += ' '
    return sent_new

def get_opstr(mode=0,shape=(100,100)):
    abc_str = 'abcdefghijklmnopqrstuvwxyz'
    all_str = add_space(abc_str[:len(shape)])
    change_str = add_space(abc_str[mode]+'('+abc_str[:mode]+abc_str[mode+1:len(shape)]+')')
    return all_str+'-> '+change_str

def get_reg(parameter):
    reg_name = parameter.get('reg_name', 'TV')
    if reg_name in ['TV', 'LAP']:
        de_para_dict = {'coef': 1, 'p_norm': 2, "mode":0}
    elif reg_name == 'AIR':
        de_para_dict = {'n': 100, 'coef': 1, 'mode': 0}
    elif reg_name == 'INRR':
        de_para_dict = {'coef': 1, 'mode': 0, 'inr_parameter': {'dim_in': 1,'dim_out':100}}
    elif reg_name == 'MultiReg':
        de_para_dict = {'reg_list':[{'reg_name':'TV'}]}
    else:
        pass
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
    if reg_name != 'MultiReg':
        return regularizer(parameter)
    else:
        return MultiReg(parameter)

class MultiReg(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.reg_list_para = parameter.get('reg_list',[{'reg_name':'TV'}])
        reg_list = []
        for _,reg_para in enumerate(self.reg_list_para):
            reg_list.append(get_reg(reg_para))
        self.reg_list = nn.ModuleList(reg_list)

    def forward(self,x):
        reg_loss = 0
        for _,reg in enumerate(self.reg_list):
            reg_loss += reg(x)
        return reg_loss


class regularizer(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.reg_parameter = parameter
        self.reg_name = parameter['reg_name']
        # init opt parameters
        self.mode = self.reg_parameter['mode']
        if self.reg_name == 'AIR':
            self.n = self.reg_parameter['n']
            self.A_0 = nn.Linear(self.n,self.n,bias=False)
            self.softmin = nn.Softmin(1)
            
        elif self.reg_name == 'INRR':
            net = get_nn(self.reg_parameter['inr_parameter'])
            self.net = nn.Sequential(net,nn.Softmax())

    

    def forward(self,x):
        if self.reg_name == 'TV':
            return self.tv(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'LAP':
            return self.lap(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'AIR':
            return self.air(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'INRR':
            return self.inrr(x)*self.reg_parameter["coef"]


    def tv(self,M):
        """
        M: torch tensor type
        p: p-norm
        """
        p = self.reg_parameter['p_norm']
        center = M[1:M.shape[0]-1,1:M.shape[1]-1]
        up = M[1:M.shape[0]-1,0:M.shape[1]-2]
        down = M[1:M.shape[0]-1,2:M.shape[1]]
        left = M[0:M.shape[0]-2,1:M.shape[1]-1]
        right = M[2:M.shape[0],1:M.shape[1]-1]
        Var1 = 2*center-up-down
        Var2 = 2*center-left-right
        return (t.norm(Var1,p=p)+t.norm(Var2,p=p))/M.shape[0]

    def lap(self,M):
        """
        M: torch tensor type
        p: p-norm
        """
        p = self.reg_parameter['p_norm']
        center = M[1:M.shape[0]-1,1:M.shape[1]-1]
        up = M[1:M.shape[0]-1,0:M.shape[1]-2]
        down = M[1:M.shape[0]-1,2:M.shape[1]]
        left = M[0:M.shape[0]-2,1:M.shape[1]-1]
        right = M[2:M.shape[0],1:M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=p)/M.shape[0]

    def air(self,W):
        device = W.device
        Ones = t.ones(self.n,1)
        I_n = t.from_numpy(np.eye(self.n)).to(t.float32)
        Ones = to_device(Ones,device)
        I_n = to_device(I_n,device)
        A_0 = self.A_0.weight # A_0 \in \mathbb{R}^{n \times n}
        A_1 = self.softmin(A_0) # A_1 中的元素的取值 \in (0,1) 和为1
        A_2 = (A_1+A_1.T)/2 # A_2 一定是对称的
        A_3 = A_2 * (t.mm(Ones,Ones.T)-I_n) # A_3 将中间的元素都归零，作为邻接矩阵
        A_4 = -A_3+t.mm(A_3,t.mm(Ones,Ones.T))*I_n # A_4 将邻接矩阵转化为拉普拉斯矩阵
        self.lap = A_4
        opstr = get_opstr(mode=self.mode,shape=W.shape)
        W = rearrange(W,opstr)
        return t.trace(t.mm(W.T,t.mm(A_4,W)))/(W.shape[0]*W.shape[1])#+l1 #行关系

    def inrr(self,W):
        self.device = W.device
        opstr = get_opstr(mode=self.mode,shape=W.shape)
        img = rearrange(W,opstr)
        n = img.shape[0]
        coor = t.linspace(-1,1,n).reshape(-1,1)
        coor = to_device(coor,self.device)
        self.A_0 = self.net(coor)@self.net(coor).T
        self.L = self.A2lap(self.A_0)
        return t.trace(img.T@self.L@img)/(img.shape[0]*img.shape[1])

    def A2lap(self,A_0):
        n = A_0.shape[0]
        Ones = t.ones(n,1)
        I_n = t.from_numpy(np.eye(n)).to(t.float32)
        Ones = to_device(Ones,self.device)
        I_n = to_device(I_n,self.device)
        A_1 = A_0 * (t.mm(Ones,Ones.T)-I_n) # A_1 将中间的元素都归零，作为邻接矩阵
        L = -A_1+t.mm(A_1,t.mm(Ones,Ones.T))*I_n # A_2 将邻接矩阵转化为拉普拉斯矩阵
        return L
