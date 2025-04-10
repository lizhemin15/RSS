import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py

def load_data(data_path,data_type='gray_img',data_shape=None,down_sample=[1,1,1],mat_get_func=lambda x:x):
    # load data from disk
    # return numpy array
    if data_type == 'gray_img' or data_type == 'rgb_img':
        # rescale to [0,1]
        img = cv2.imread(data_path)
        if data_shape != None:
            img = cv2.resize(img,(data_shape[1],data_shape[0]))
        if data_type == 'gray_img':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
            img = img[::down_sample[0],::down_sample[1]]
        else:
            img = img.astype(np.float32)/255.0
            img = img[::down_sample[0],::down_sample[1],:]
        return img
    elif data_type == 'numpy':
        return np.load(data_path)
    elif data_type == 'syn':
        if data_path == 'circle':
            return syn_circle(data_shape)
        elif data_path == 'low_rank':
            return syn_low_rank(data_shape)
        else:
            return syn_circle(data_shape)
    elif data_type == 'rgb_video' or data_type == 'gray_video':
        cap = cv2.VideoCapture(data_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, frame = cap.read()
        if data_type == 'rgb_video':
            frame = frame[::down_sample[0],::down_sample[1],:]
            vd_np = np.zeros((frame.shape[0],frame.shape[1],3,frame_count))
            cap = cv2.VideoCapture(data_path)
            for i in range(vd_np.shape[-1]):
                _, frame = cap.read()
                frame = frame.astype(np.float32)/255.0
                frame = frame[::down_sample[0],::down_sample[1],:]
                vd_np[:,:,:,i] = frame[:,:,(2,1,0)]
            return vd_np[:,:,:,::down_sample[2]]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[::down_sample[0],::down_sample[1]]
            vd_np = np.zeros((frame.shape[0],frame.shape[1],frame_count))
            cap = cv2.VideoCapture(data_path)
            for i in range(vd_np.shape[-1]):
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
                frame = frame[::down_sample[0],::down_sample[1]]
                vd_np[:,:,i] = frame
            return vd_np[:,:,::down_sample[2]]
    elif data_type == 'mat':
        try:
            # print('Loading mat file from ',data_path)
            return mat_get_func(loadmat(data_path))
        except:
            db = h5py.File(data_path, 'r')
            ds = mat_get_func(db)
            try:
                if 'ir' in ds.keys():
                    # print('Solvable ds: ',ds.keys())
                    data = np.asarray(ds['data'])
                    ir   = np.asarray(ds['ir'])
                    jc   = np.asarray(ds['jc'])
                    out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
                else:
                    # print('unsolvable ds: ',ds.keys())
                    for key in ds.keys():
                        print("    %s: %s" % (key, ds[key]))
                    out = ds
            except AttributeError:
                # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
                out = np.asarray(ds).astype(np.float32).T
            db.close()
            return out
    else:
        raise('Wrong data type = ',data_type)

def save_data(data_path,data_type='numpy',data=None):
    if data_type == 'img':
        if data.ndim == 2:
            plt.imsave(data_path, np.clip(data,0,1))
        else:
            plt.imsave(data_path, np.clip(data[:,:,(2,1,0)],0,1))
    elif data_type == 'numpy':
        np.save(data_path,data)
    else:
        raise('Wrong data type = ',data_type)



def syn_circle(data_shape):
    x = np.squeeze(np.linspace(-1, 1, data_shape[0]))
    y = np.squeeze(np.linspace(-1, 1, data_shape[1]))
    x1,y1 = np.meshgrid(x,y)
    z = np.sin(100*np.pi*np.sin(np.pi/3*np.sqrt(x1**2+y1**2)))
    z = z.astype('float32')/z.max()
    return z


def syn_low_rank(data_shape):
    x = np.random.randn(data_shape[0],5)
    y = np.random.randn(data_shape[1],5)
    return np.dot(x,y.T)