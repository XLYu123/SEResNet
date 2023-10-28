import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon

def boxplot(data_list,label,dir):
    fig, ax = plt.subplots()
    ax.boxplot(data_list,
                notch=False, labels = label,patch_artist = False, medianprops={'color':'red'},boxprops = {'color':'blue','linewidth':'1.0'},
                capprops={'color':'black','linewidth':'1.0'})
    color = ['blue', 'orange', 'green','red','purple']  # 有多少box就对应设置多少颜色
    # ax.legend(['RF-GAN'], loc='upper right')
    # plt.ylim(ymin=0,ymax=100)
    fig.savefig(dir)
    plt.show()

def post_hoc(x1,x2):
    stat_post, p_post = wilcoxon(x1,x2)
    print('wilcoxon Post hoc: ',stat_post, p_post)
    # interpret
    alpha = 0.05
    if p_post > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

#以下为非batchnorm 操作后的生成数据
# normal (with acc metric)
cnn = np.array([0.948819048,	0.92257619,	0.949042857	,0.88537619	,0.9538,0.935566667,
                0.95472381,0.930214286,0.949195238,0.951519048,0.939566667,
                0.949852381,0.961585714	,0.956928571,0.960752381,0.956038095,0.951347619,
                0.966571429,0.962904762,0.960638095,0.963261905,0.925080952,0.938819048,
                0.946785714,0.962914286	,0.925138095,0.949738095,0.956133333,0.951461905,0.9312
])

cnn_eca = np.array([0.947166667,0.954328571,0.951285714,0.951309524,0.95497619,	0.954695238	,0.948295238,0.96342381	,
                    0.947947619,0.923009524,0.9341,0.941628571,0.946752381,0.932971429,0.962209524,0.951933333,0.949209524,
                    0.955566667	,0.95412381,0.927433333,0.92902381,	0.954866667,0.962028571	,0.950666667,0.929985714,0.935709524,
                    0.937990476	,0.955519048,0.962290476,0.960142857

])

cnn_resnet = np.array([0.950604762,0.982714286,0.969009524,0.958033333,0.961319048,0.948261905,0.957152381,0.969480952,
                       0.97312381,0.976228571,0.924652381,0.924652381,0.947266667,0.946942857,0.965547619,0.965895238,
                       0.966447619,0.961971429,0.97482381,0.96282381,0.962795238,0.954528571,0.957957143,0.960385714,
                       0.948104762,0.963057143,0.958933333,0.961214286,0.950809524,0.920780952

])

cnn_res_se = np.array([0.9899,9.75E-01,9.82E-01	,
                       9.83E-01,9.87E-01,9.81E-01,
                       0.975519048,0.981938095,0.983690476,
                       0.981428571,0.983966667,0.983371429,
                       0.980790476,0.974733333,0.986638095,
                       0.987814286,0.988485714,0.980752381,
                       0.982642857,0.985761905,0.984533333,
                       0.9848,0.981838095,0.978247619,0.97787619,
                       0.982880952,0.983928571,0.987361905,0.987114286,0.982920197

])
cnn_se = np.array([0.954633333,0.953238095,0.959547619,0.962095238,0.958552381,
                   0.937638095,0.958728571,0.956857143,0.962838095,0.943928571,
                   0.968138095,0.952404762,0.964738095,0.949785714,0.93727619,
                   0.942052381,0.954833333,0.951657143,0.957757143,0.959066667,
                   0.964128571,0.959409524,0.947380952,0.973561905,0.961638095,
                   0.950466667,0.958380952,0.968790476,0.945533333,0.954790476


])

# def ModelSensity test
stat, p = friedmanchisquare(cnn,cnn_eca,cnn_resnet,cnn_res_se,cnn_se)
print('Friedman Test: ',stat, p)
alpha = 0.05
if p > alpha:
     print('There is no difference between them (fail to reject H0)')
else:
    print('There is difference between them (reject H0) and Jump to the Post hoc!')
    print('*'*52)
    print('-'*20+'pos hoc test'+'-'*20)
    post_hoc(cnn,cnn_eca)
    print('*' * 52)

# mean = np.mean(cnn_res_eca)
# print(mean)

# #BoxPlot for Fig_8 (average accuracy)
# plt.figure(1)
save_path='./fig/boxplot.png'
labels='1DCNN','1DECA','1DRNN','MSA-1DCNN'
data_lists=[cnn,cnn_eca,cnn_resnet,cnn_res_se]

boxplot(data_list=data_lists,label=labels,dir=save_path)
plt.ylim(ymin=0,ymax=1)
plt.savefig('./fig/boxplot.png')
plt.show()