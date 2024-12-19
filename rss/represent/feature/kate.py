import torch
import torch.nn as nn


class KATE_Feature(nn.Module):
    def __init__(self, order=0, **kwargs):
        super().__init__()
        self.order = order
        self.a = nn.Parameter(torch.randn(order + 1))

    def forward(self, x):
        # 初始化 y 的形状，dim=1的维度为0
        y_shape = list(x.size())  # 获取 x 的形状
        y_shape[1] = 0  # 在 dim=1 处设置为 0
        y = torch.empty(y_shape, device=x.device)  # 在同一设备上创建空张量

        for i in range(self.order + 1):  # 包括 order
            # 将 x 的 i 次幂乘以参数 a[i] 并在 dim=1 上拼接
            y = torch.cat((y, self.a[i] * x**i), dim=1)

        return y
    


def KATE_Embedder(parameter):
    de_para_dict = {'order':0}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    return KATE_Feature(**parameter)