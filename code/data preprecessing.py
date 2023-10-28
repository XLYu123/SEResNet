#西储大学数据
import scipy.io as io
import numpy as np
import pickle

Train_sample = []
# "E:\200-研究生\230-实验\C/12k DriveEndBearingFault Data/130.mat"
data = io.loadmat(r"C:\Users\du\Desktop\")
x20 = data['X144_DE_time']  #数据长度

xx = len(x20) #数据长度
length = 3072
sample_n = 300 #样本数

for j in range(sample_n):
    random_start = np.random.randint(low=0, high=(xx - 2 * length))
    X1 = x20[random_start: random_start + length]
    Train_sample.append(X1)

b = np.array(Train_sample)
a = b[:,:,0]

def save_pickle_v2(path,name,x):

    with open(path+name, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
    print('save to path:',path)
    print('Save successfully!')

import os
path_out='./dataset/'
os.makedirs(path_out,exist_ok=True) #如果没有该文件夹，则创建此文件夹
save_pickle_v2(path_out,name='C4_test.pkl',x=a)

# AUV数据