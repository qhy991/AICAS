# -*- coding: utf-8 -*-#
 
#-------------------------------------------------------------------------------
# Name:         TestModel
# Description:  
# Author:       Administrator
# Date:         2021/3/8
#-------------------------------------------------------------------------------
'''
模块简介：该模块完成的模型的测试，通过加载相应的函数，测试模型分类的结果，绘制AUC图和混淆矩阵
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import objectclass
from  MakeDate import  *
from ConcludeAccuracy import  *
 
def reload_net():
    trainednet = torch.load('net.pkl')
    return trainednet
 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))#改变每个轴对应的数值
    plt.show()
 
 
def test(path):
    testloader = loadtestdata(path)
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    #imshow(torchvision.utils.make_grid(images,nrow=5))  # nrow是每行显示的图片数量，缺省值为8
    #%5s中的5表示占位5个字符
    #print('GroundTruth: ' , " ".join('%5s' % objectclass.class_names[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    #获取分类的预测分数
    sum_score=outputs.data.numpy()
    row,col=sum_score.shape
    score=[]
    for i in range(row):
        score.append(np.max(sum_score[i],axis=0))
    true_label=labels.tolist()#张量转换为列表
    #print('Predicted: ', " ".join('%5s' % objectclass.class_names[predicted[j]] for j in range(25)))
 
    pre_value=predicted.tolist()
    score_result=concludescore(pre_value,true_label)
    print('准确率 精确率 召回率：\n',score_result)
 
    #绘制ROC曲线
    drawAUC_TwoClass(true_label,  score)
 
    #绘制混淆矩阵
    cm = confusion_matrix(true_label,  pre_value)
    print('混淆矩阵：\n',cm)
    labels_name=['cat','dog']
    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")
  # 打印前25个预测值
 
 
if __name__=="__main__":
    path='F:\\PytorchTest\\torchdeeplearnmodel\\classcatanddog\\data\\test\\'
    test(path)