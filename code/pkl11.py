import pickle
import tensorflow as tf
import pandas as pd
import numpy as np

path =r"C:\Users\du\Desktop\C8/"
# path1 = "C:/Users/du/Desktop/dataset/"
faults0 = ['C8']
for fault0 in faults0:
    with open(path+ fault0+ '.pkl', 'rb') as f0:
        data_0 = pickle.load(f0)

label_0 = pd.DataFrame(300*[6])
label_0 = np.array(label_0)
data_0 = (data_0,label_0)
import pickle
import numpy as np

def save_pickle_v1(path,name,x):#
    with open(path+name, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)  #pickle.dump () 封装是一个将Python数据对象转化为字节流的过程
        print('save to path:', path)
        print('Save successfully!')


path_out= r"C:\Users\du\Desktop\C8/"
save_pickle_v1(path_out,name='C6_test.pkl',x=data_0)


# faults0 = ['C0']
# for fault0 in faults0:
#     with open(path1+ fault0+ '.pkl', 'rb') as f0:
#         data_1 = pickle.load(f0)