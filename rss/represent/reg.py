import torch.nn as nn
import torch as t
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from rss.represent import get_nn
from rss import toolbox
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import scipy.ndimage
from rss.represent.utils import gaussian_kernel, pad_with_zeros, extract_patches
from scipy.spatial.distance import cdist

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
    if reg_name in ['TV', 'LAP', 'WTV', 'NLTV']:
        de_para_dict = {'coef': 1, 'p_norm': 2, "mode":0, 'topk': 10, 'sigma': 1, 'patch_size': 4, 'search_epoch':100}
    elif reg_name == 'DE':
        de_para_dict = {'coef': 1, "mode":0, "lap_k":1}
    elif reg_name == 'AIR':
        de_para_dict = {'n': 100, 'coef': 1, 'mode': 0, "lap_k":1}
    elif reg_name == 'INRR':
        de_para_dict = {'coef': 1, 'mode': 0, "lap_k":1, 'inr_parameter': {'dim_in': 1,'dim_out':100}}
    elif reg_name == 'RUBI':
        de_para_dict = {'coef': 1, "mode":None}
    elif reg_name == 'MultiReg':
        de_para_dict = {'reg_list':[{'reg_name':'TV'}]}
    elif reg_name == 'GroupReg':
        de_para_dict = {'group_para':{'n_clusters':10,'metric':'cosine'},'each_reg_name':'AIR','start_epoch':100,'search_epoch':1000}
    else:
        de_para_dict = {"mode":None}
    #if reg_name not in MultiRegDict.keys():
    de_para_dict["x_trans"] = "ori"
    de_para_dict["factor"] = 1
    de_para_dict["patch_size"] = 16
    de_para_dict["stride"] = 16
    de_para_dict['sparse_index'] = None
        
    for key in de_para_dict.keys():
        param_now = parameter.get(key, de_para_dict.get(key))
        parameter[key] = param_now
    if reg_name not in MultiRegDict.keys():
        return regularizer(parameter)
    else:
        return MultiRegDict[reg_name](parameter)
        # return MultiReg(parameter)

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

class GroupReg(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.reg_parameter = parameter
        self.epoch_now = 0
        self.group_para = parameter.get('group_para',{'group_mode':'kmeans','n_clusters':10,
                                                      'metric':'cosine','reg_mode':'single'})
        self.x_trans = parameter.get("x_trans","ori")
        self.reg_mode = self.group_para.get('reg_mode','single')

    def init_reg(self,x):
        device = x.device
        # x: (sample_num,feature_num)
        if 'patch' in self.x_trans:
            # for patch-based regularization
            x = toolbox.extract_patches(input_tensor=x, patch_size=self.reg_parameter.get("patch_size",16),
                                         stride=self.reg_parameter.get("stride",16), return_type = 'vector')
        elif self.x_trans == 'ori':
            opstr = get_opstr(mode=self.reg_parameter.get('mode',0),shape=x.shape)
            x = rearrange(x,opstr)
        else:
            raise ValueError("x_trans should be 'patch' or 'ori', but got {}".format(self.x_trans))
        x = x.detach().cpu().numpy()
        # calculate the group of regularization, with k-means algorithm
        if self.group_para.get('group_mode', 'kmeans') == 'kmeans':
            # calculate the group of regularization, with k-means algorithm
            D = pairwise_distances(x, metric=self.group_para.get('metric', 'cosine'))
            kmeans = KMeans(n_clusters=self.group_para.get('n_clusters', 10))
            kmeans.fit(x, D)
            labels = kmeans.labels_
            sparse_index_list = []
            for i in range(kmeans.n_clusters):
                sparse_index_list.append(np.where(labels == i)[0])
        elif self.group_para.get('group_mode', 'kmeans') == 'knn':
            # calculate the group of regularization, with K-Neighbors algorithm
            n_neighbors = self.group_para.get('n_clusters', 10)  # 使用n_clusters作为近邻数
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.group_para.get('metric', 'cosine'))
            knn.fit(x)
            distances, indices = knn.kneighbors(x)
            sparse_index_list = []
            for idx in indices:
                sparse_index_list.append(idx)
        else:
            raise ValueError("group_mode should be 'kmeans' or 'knn', but got {}".format(self.group_para['group_mode']))
        reg_name = self.reg_parameter.get('each_reg_name','AIR')
        reg_list = []
        if self.reg_mode =='multi':
            for sparse_index in sparse_index_list:
                new_parameter = self.reg_parameter.copy()
                new_parameter['sparse_index'] = sparse_index
                new_parameter['n'] = len(sparse_index)
                # print(len(sparse_index))
                new_parameter['reg_name'] = reg_name
                new_parameter['reg_mode'] = self.reg_mode
                reg_list.append(to_device(get_reg(new_parameter),device))
        elif self.reg_mode == 'single':
            # check the reg_name, must be INRR
            if reg_name != 'INRR':
                raise ValueError("reg_mode is 'single', but each_reg_name should be 'INRR', but got {}".format(reg_name))
            # Then we only need to initialize one INRR and use it in differetn groups with different sparse_index
            new_parameter = self.reg_parameter.copy()
            new_parameter['reg_name'] = reg_name
            new_parameter['reg_mode'] = self.reg_mode
            new_parameter['n'] = x.shape[0]
            reg_list.append(to_device(get_reg(new_parameter),device))
            self.sparse_index_list = sparse_index_list
        else:
            raise ValueError("reg_mode should be'multi' or'single', but got {}".format(self.reg_mode))
        self.reg_list = nn.ModuleList(reg_list)

    def forward(self,x):
        reg_loss = 0
        if self.epoch_now >= self.reg_parameter.get('start_epoch',100):
            if (self.epoch_now-self.reg_parameter.get('start_epoch',100)) % self.reg_parameter.get('search_epoch',1000) == 0:
                self.init_reg(x)
            else:
                if self.reg_mode =='multi':
                    for _,reg in enumerate(self.reg_list):
                        reg_loss += reg(x)
                elif self.reg_mode =='single':
                    for sparse_index in self.sparse_index_list:
                        reg_loss += self.reg_list[0](x,sparse_index)
                else:
                    raise ValueError("reg_mode should be'multi' or'single', but got {}".format(self.reg_mode))
        self.epoch_now += 1
        return reg_loss

MultiRegDict = {"MultiReg":MultiReg,"GroupReg":GroupReg}

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, error):
        abs_error = t.abs(error)
        quadratic = t.min(abs_error, t.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + t.tensor(self.delta) * linear
        return t.mean(loss)

class regularizer(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        self.reg_parameter = parameter
        self.reg_name = parameter['reg_name']
        self.reg_mode = parameter.get('reg_mode','single')
        self.x_trans = parameter["x_trans"]
        self.sparse_index = parameter.get('sparse_index',None)
        self.n = self.reg_parameter.get('n',100)
        self.epoch_now = 0
        if self.x_trans != 'ori':
            self.factor = parameter["factor"]
            self.patch_size = parameter["patch_size"]
            self.stride = parameter["stride"]
            self.filter_type = parameter.get('filter_type', None)
            self.sigma = parameter.get('sigma',1)
            # 计算图像块数量 
            self.num_blocks_h = (self.n - self.patch_size) // self.stride + 1
            self.num_blocks_w = (self.n - self.patch_size) // self.stride + 1
        # init opt parameters 
        self.mode = self.reg_parameter['mode']
        if self.reg_name in ['AIR','INRR','TV']:
            self.lap_mode = self.reg_parameter.get('lap_mode','vanilla')
            self.huber_delta = self.reg_parameter.get('huber_delta',0.1)
            if self.lap_mode == 'Huber':
                self.huber_loss = HuberLoss(delta=self.huber_delta)
            self.norm_lap_lp = self.reg_parameter.get('norm_lap_lp',1)
            self.quantile_q = self.reg_parameter.get('quantile_q',0.5)
        if self.reg_name == 'AIR':
            self.A_0 = nn.Linear(self.n,self.n,bias=False)
            self.softmin = nn.Softmin(1)
        elif self.reg_name == 'INRR':
            if 'patch' in self.x_trans and self.reg_mode =='single':
                self.reg_parameter['inr_parameter']['dim_in'] = 2
            net = get_nn(self.reg_parameter['inr_parameter'])
            self.net = nn.Sequential(net,nn.Softmax(dim=-1))
            self.inrr_alpha = self.reg_parameter.get('inrr_alpha',1.0)
            self.nabla_matrix_order_k = self.reg_parameter.get('nabla_matrix_order_k',1)
        elif self.reg_name == 'DE':
            self.A_0 = parameter['A_0']
            self.temperature = parameter.get('temperature',1)
            self.softmax = nn.Softmax(dim=-1)
        elif self.reg_name == 'RUBI':
            self.ite_num = 0

    def forward(self,x,sparse_index=None):
        self.epoch_now += 1
        if 'down_sample' == self.x_trans:
            x = toolbox.downsample_tensor(input_tensor=x, factor=self.factor)
        if 'patch' == self.x_trans:
            x = toolbox.extract_patches(input_tensor=x, patch_size=self.patch_size,
                                         stride=self.stride, return_type = 'vector',
                                         filter_type=self.filter_type, sigma = self.sigma)
        if 'patch_down' == self.x_trans:
            x = toolbox.extract_patches(input_tensor=x, patch_size=self.patch_size, stride=self.stride, return_type = 'patch', down_sample = True)

        if self.reg_mode == 'single':
            # single 的情况下，意味着不同的正则计算的时候会共享同一个INRR参数，但是每次会传入 sparse_index 来标识到底去计算哪一部分
            self.sparse_index = sparse_index
        if self.sparse_index is not None:
            x = x[self.sparse_index]
        if self.reg_name == 'TV':
            return self.tv(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'WTV':
            return self.wtv(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'NLTV':
            return self.nltv(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'LAP':
            return self.lap_reg(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'DE':
            return self.air(x,mode='fix')*self.reg_parameter["coef"]
        elif self.reg_name == 'AIR':
            return self.air(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'INRR':
            return self.inrr(x)*self.reg_parameter["coef"]
        elif self.reg_name == 'RUBI':
            return self.rubi(x)*self.reg_parameter["coef"]
        else:
            raise('Not support regularizer named ',self.reg_name,'please check the regularizer name in TV, LAP, AIR, INRR, RUBI')



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
        if self.lap_mode == 'vanilla':
            return (t.norm(Var1,p=p)+t.norm(Var2,p=p))/M.shape[0]/M.shape[1]
        elif self.lap_mode == 'Huber':
            return self.huber_loss(t.abs(Var1))+self.huber_loss(t.abs(Var2))
        elif self.lap_mode == 'quantile':
            return t.norm(t.clamp(Var1-self.quantile_q*t.abs(Var1),min=0,max=self.quantile_q*t.abs(Var1))+t.clamp(Var2-self.quantile_q*t.abs(Var2),min=0,max=self.quantile_q*t.abs(Var2)),p=p)/M.shape[0]
        else:
            raise ValueError("lap_mode should be 'vanilla', 'Huber', or 'quantile', but got {}".format(self.lap_mode))

    def wtv(self, M):
        """
        M: torch tensor type, shaped as (n,n)
        p: p-norm
        """
        p = self.reg_parameter['p_norm']
        center = M[1:M.shape[0]-1,1:M.shape[1]-1]
        up = M[1:M.shape[0]-1,0:M.shape[1]-2]
        down = M[1:M.shape[0]-1,2:M.shape[1]]
        left = M[0:M.shape[0]-2,1:M.shape[1]-1]
        right = M[2:M.shape[0],1:M.shape[1]-1]
        Var = t.abs(2*center-up-down)+t.abs(2*center-left-right) # shape: (n-2,n-2)
        weight = (1  / (t.abs(Var)+1e-2)).detach().clone() # shape: (n-2,n-2)
        return t.norm(weight*Var,p=p)/M.shape[0]

    def nltv(self, M):
        """
        M: torch tensor type, shaped as (n, n)
        p: p-norm
        """
        if self.epoch_now % self.reg_parameter.get('search_epoch',100) == 0:
            p = self.reg_parameter['p_norm']
            M_np = M.detach().cpu().numpy()  # M_np的形状为 (n, n)
            n = M_np.shape[0]
            patch_size = self.reg_parameter.get('patch_size', 4)  # 获取patch_size
            sigma = self.reg_parameter.get('sigma', 1.0)  # 默认sigma为1.0
            self.k = self.reg_parameter.get('topk', 10)  # 默认topk为10
            kernel = gaussian_kernel(patch_size, sigma)  # kernel的形状为 (2*patch_size+1, 2*patch_size+1)
            M_padded = pad_with_zeros(M_np, patch_size)  # M_padded的形状为 (n+2*patch_size, n+2*patch_size)
            M_conv = scipy.ndimage.convolve(M_padded, kernel, mode='constant', cval=0.0)  # M_conv的形状为 (n+2*patch_size, n+2*patch_size)
            # 提取所有 (n, n) 的框并堆叠成新的张量
            patches = extract_patches(M_conv, n, patch_size)  # patches的形状为 ((2*patch_size)**2, n, n)
            # 计算第一个通道（0通道）的向量作为特征
            features = patches.reshape((n*n, -1))  # 特征的形状为 (n*n, (2*patch_size+1)**2)
            # 计算所有向量之间的距离
            self.distances = cdist(features, features, metric='minkowski', p=p)  # 距离矩阵的形状为 (n*n, n*n)
            # 找到k个最近的邻居
            indices = np.argsort(self.distances, axis=1)[:, :self.k]  # 形状为 (n*n, k)
            # 将indices整理成(n,n,k)的形状
            self.indices = indices.reshape((n, n, self.k))
            return 0
        if self.epoch_now < self.reg_parameter.get('start_epoch',100):
            return 0
        else:
            n = M.shape[0]
            # 将self.indices转换为int64类型
            indices = t.tensor(self.indices, dtype=t.int64).to(M.device)
            
            # 输出调试信息
            print(f"M.shape: {M.shape}")
            print(f"indices.shape: {indices.shape}")
            print(f"Indices range: {indices.min().item()} to {indices.max().item()}")
            
            # 使用转换后的indices进行gather操作
            k_nearest_values = t.gather(M.unsqueeze(-1).expand(-1, -1, self.k), dim=-1, index=indices)  # 形状为 (n, n, k)
            
            # 输出调试信息
            print(f"k_nearest_values.shape: {k_nearest_values.shape}")
            
            k_nearest_distances = np.take_along_axis(self.distances.reshape(n, n, n*n), self.indices, axis=-1)  # 形状为 (n, n, k)
            k_nearest_distances = t.exp(-t.tensor(k_nearest_distances).to(M.device))
            
            # 输出调试信息
            print(f"k_nearest_distances.shape: {k_nearest_distances.shape}")
            
            return t.mean(k_nearest_distances * (M.unsqueeze(-1) - k_nearest_values) ** 2)



    def lap_reg(self,M):
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


    def air(self,W,mode='learn'):
        lap_k = self.reg_parameter.get('lap_k',1)
        device = W.device
        Ones = t.ones(self.n,1)
        I_n = t.from_numpy(np.eye(self.n)).to(t.float32)
        Ones = to_device(Ones,device)
        I_n = to_device(I_n,device)
        if mode == 'learn':
            A_0 = self.A_0.weight # A_0 \in \mathbb{R}^{n \times n}
            A_1 = self.softmin(A_0) # A_1 中的元素的取值 \in (0,1) 和为1
            A_2 = (A_1+A_1.T)/2 # A_2 一定是对称的
        else:
            A_0 = self.A_0 # A_0 \in \mathbb{R}^{n \times n}
            A_1 = self.softmax(A_0/self.temperature)
            A_2 = (A_1+A_1.T)/2
        A_3 = A_2 * (t.mm(Ones,Ones.T)-I_n) # A_3 将中间的元素都归零，作为邻接矩阵
        A_4 = -A_3+t.mm(A_3,t.mm(Ones,Ones.T))*I_n # A_4 将邻接矩阵转化为拉普拉斯矩阵
        self.lap = A_4.clone()
        for _ in range(lap_k-1):
            self.lap = self.lap@A_4
        opstr = get_opstr(mode=self.mode,shape=W.shape)
        W = rearrange(W,opstr)
        return self.lap_loss(W,self.lap,lap_mode=self.lap_mode,norm_lap_lp=self.norm_lap_lp,huber_delta=self.huber_delta,q=self.quantile_q)
        # t.trace(t.mm(W.T,t.mm(self.lap,W)))/(W.shape[0]*W.shape[1])#+l1 #行关系



    def inrr(self,W):
        # GroupReg 中，multi和single模式下传入的W都是已经取了 sparse_index 的
        self.device = W.device
        lap_k = self.reg_parameter.get('lap_k',1)
        opstr = get_opstr(mode=self.mode,shape=W.shape)
        img = rearrange(W,opstr)
        n = img.shape[0]
        if self.reg_mode == 'single':
            # 共享参数时，需要考虑到inr中的相对位置
            if 'patch' in self.x_trans:
                if self.mode == 0:
                    num_blocks_h,num_blocks_w = self.num_blocks_h,self.num_blocks_w
                else:
                    num_blocks_h,num_blocks_w = self.patch_size,self.patch_size
                x = t.linspace(-1, 1, num_blocks_h)
                y = t.linspace(-1, 1, num_blocks_w)
                grid_x, grid_y = t.meshgrid(x, y, indexing='ij')
                coor = t.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            elif self.x_trans == 'ori':
                coor = t.linspace(-1,1,self.n).reshape(-1,1)
            else:
                raise ValueError("x_trans should be 'patch' or 'ori', but got {}".format(self.x_trans))
        elif self.reg_mode =='multi' or self.reg_mode == 'original':
            # 当不共享参数时，patch或ori均使用单独的根据目前的形状n来计算的坐标，用于捕获连续性
            coor = t.linspace(-1,1,n).reshape(-1,1)
        else:
            raise ValueError("reg_mode should be'single' or'multi', but got {}".format(self.reg_mode))
           
        if self.reg_mode == 'single' and self.sparse_index is not None:
            # 这是因为当 self.reg_mode 为 single时，共享同一个inr，所以要在同一个坐标下取相应的子坐标。
            coor = coor[self.sparse_index]

        coor = to_device(coor,self.device)
        self.A_0 = self.net(coor)
        # print(self.A_0.shape)
        self.A_0 = self.A_0@(self.A_0.T)
        A_4 = self.A2lap(self.A_0)
        self.lap = A_4.clone()
        for _ in range(lap_k-1):
            self.lap = self.lap@A_4
        # print('lap shape:',self.lap.shape)
        return self.lap_loss(img,self.lap,lap_mode=self.lap_mode,norm_lap_lp=self.norm_lap_lp,huber_delta=self.huber_delta,q=self.quantile_q)
        # t.trace(img.T@self.lap@img)/(img.shape[0]*img.shape[1])

    def A2lap(self,A_0):
        n = A_0.shape[0]
        Ones = t.ones(n,1)
        I_n = t.from_numpy(np.eye(n)).to(t.float32)
        Ones = to_device(Ones,self.device)
        I_n = to_device(I_n,self.device)
        A_1 = A_0 * (t.mm(Ones,Ones.T)-I_n) # A_1 将中间的元素都归零，作为邻接矩阵
        L = -A_1+t.mm(A_1,t.mm(Ones,Ones.T))*I_n # A_2 将邻接矩阵转化为拉普拉斯矩阵
        nabla_matrix = self.create_nabla_matrix(n, order_k=self.nabla_matrix_order_k)
        # 最终的 L 矩阵
        if sefl.inrr_alpha >= 0:
            L = self.inrr_alpha*L + (1-self.inrr_alpha)*nabla_matrix
        else:
            # 当 inrr_alpha 为负时，使用相乘的形式进行融合
            L = nabla_matrix@L
        return L
    
    def create_nabla_matrix(self,n, order_k=1):
        I_n = t.from_numpy(np.eye(n)).to(t.float32)
        I_n = to_device(I_n,self.device)
        # 创建矩阵 J，其 (i, i+1) 处为1，其他地方为0
        J = t.diag(t.ones(n-1), 1)  # 只需要这一行就可以生成 J 矩阵
        J[-1,0] = 1
        J = to_device(J, self.device)
        nabla_matrix = I_n-(J+J.T)/2
        final_nabla_matrix = t.eye(n)
        final_nabla_matrix = to_device(final_nabla_matrix,self.device)
        for k in range(order_k):
            final_nabla_matrix = final_nabla_matrix@nabla_matrix
            # final_nabla_matrix = final_nabla_matrix/final_nabla_matrix[0,0]
        return final_nabla_matrix
 
    def lap_loss(self,W,lap,lap_mode='vanilla',norm_lap_lp=1,huber_delta=0.3,q=0.5):
        # Given laplacian matrix lap and the regularized matrix W, compute the loss
        if lap_mode == 'vanilla':
            return t.trace(W.T@lap@W)/(W.shape[0]*W.shape[1])
        elif lap_mode == 'lp':
            return t.norm(lap@W,norm_lap_lp)/(W.shape[0]*W.shape[1])
        elif lap_mode == 'Huber':
            err = lap @ W
            return self.huber_loss(err)
        elif lap_mode == 'logcosh':
            return t.log(t.cosh(lap@W)).mean()
        elif lap_mode == 'quantile':
            err = lap @ W
            return (q*t.relu(err)+t.relu(-err)*(1-q)).mean()
        else:
            raise ValueError("lap_mode should be 'vanilla', 'lp', or 'Huber', but got {}".format(lap_mode))


    def rubi(self,M):
        if self.ite_num == 0:
            self.M_old = M.detach().clone()
        else:
            self.M_old = M.detach().clone()*0.0001+0.9999*self.M_old
        self.ite_num += 1
        result = t.mean(M*(M-self.M_old))
        if self.ite_num%100==0:
            print(result)
        return result

