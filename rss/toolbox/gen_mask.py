
import numpy as np
import cv2


def load_mask(mask_type='random',random_rate=0.0,mask_path=None,data_shape=None,
              mask_shape='same',seeds=88,down_sample_rate=2,gpu_id=0):
    np.random.seed(seeds)
    # random_rate is the rate of dropped pixels
    if mask_shape == 'same':
        mask_shape = data_shape
    if mask_type == 'random':
        mask_mask = np.random.random(mask_shape)
        mask = np.ones(mask_shape)
        mask[mask_mask<=random_rate] = 0
        return np.zeros(data_shape)+mask
    elif mask_type == 'img':
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask,(data_shape[1],data_shape[0]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        mask = np.around(mask).astype(np.uint8)
        if len(data_shape) == 3:
            mask = np.expand_dims(mask,axis=2)
        elif len(data_shape) == 4:
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=3)
        return np.zeros(data_shape)+mask
    elif mask_type == 'patch':
        mask = np.ones((data_shape[0],data_shape[1]))
        mask[70:100,150:190] = 0
        mask[200:230,200:230] = 0
        if len(data_shape) == 3:
            mask = np.expand_dims(mask,axis=2)
        elif len(data_shape) == 4:
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=3)
        return np.zeros(data_shape)+mask
    elif mask_type == 'numpy':
        return np.load(mask_path)
    elif mask_type == 'down_sample':
        mask = np.zeros(mask_shape)
        if isinstance(down_sample_rate,int):
            if len(mask_shape) == 1:
                mask[::down_sample_rate] = 1
            elif len(mask_shape) == 2:
                mask[::down_sample_rate,::down_sample_rate] = 1
            elif len(mask_shape) == 3:
                mask[::down_sample_rate,::down_sample_rate,::down_sample_rate] = 1
            elif len(mask_shape) == 4:
                mask[::down_sample_rate,::down_sample_rate,::down_sample_rate,::down_sample_rate] = 1
            else:
                raise('Do not support the dim of tensor > 4')
        else:
            if len(mask_shape) == 1:
                mask[::down_sample_rate[0]] = 1
            elif len(mask_shape) == 2:
                mask[::down_sample_rate[0],::down_sample_rate[1]] = 1
            elif len(mask_shape) == 3:
                mask[::down_sample_rate[0],::down_sample_rate[1],::down_sample_rate[2]] = 1
            elif len(mask_shape) == 4:
                mask[::down_sample_rate[0],::down_sample_rate[1],::down_sample_rate[2],::down_sample_rate[3]] = 1
            else:
                raise('Do not support the dim of tensor > 4')
        return mask
    elif mask_type == 'extend':
        mask = np.zeros((data_shape[0],data_shape[1]))
        mask[data_shape[0]//8:data_shape[0]*7//8,data_shape[0]//8:data_shape[0]*7//8] = 1
        if len(data_shape) == 3:
            mask = np.expand_dims(mask,axis=2)
        elif len(data_shape) == 4:
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=3)
        return np.zeros(data_shape)+mask
    elif mask_type == 'random_col':
        # 生成一个与数据列数相同的随机掩码，用于确定哪些列将被缺失
        cols = data_shape[1]
        random_mask = np.random.random(cols)
        # 创建一个全1的掩码
        mask = np.ones(data_shape)
        # 将随机掩码中小于等于random_rate的列置为0
        mask[:, random_mask <= random_rate] = 0
        return np.zeros(data_shape) + mask
    elif mask_type == 'random_row':
        # 生成一个与数据行数相同的随机掩码，用于确定哪些行将被缺失
        rows = data_shape[0]
        random_mask = np.random.random(rows)
        # 创建一个全1的掩码
        mask = np.ones(mask_shape)
        # 将随机掩码中小于等于random_rate的行置为0
        mask[random_mask <= random_rate, :] = 0
        return np.zeros(data_shape) + mask
    elif mask_type == 'diagonal':
        mask = np.ones((data_shape[0],data_shape[1]))
        # 生成一个下标矩阵
        indices = np.arange(data_shape[0])[:, None] - np.arange(data_shape[1])
        # 将满足 |i-j| < k 的元素设为0
        mask[np.abs(indices) < 5] = 0
        return np.zeros(data_shape)+mask
    else:
        raise('Wrong mask type = ',mask_type)