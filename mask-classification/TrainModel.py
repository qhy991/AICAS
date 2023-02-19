# -*- coding: utf-8 -*-#
 
#-------------------------------------------------------------------------------
# Name:         TrainModel
# Description:  
# Author:       Administrator
# Date:         2021/3/8
#-------------------------------------------------------------------------------
 
'''
模块简介：完成模型的训练并保存模型
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
 
from MakeDate import *
from Dmodel import  Net
from tensorboardX import SummaryWriter
 
def trainandsave(TRAINPATH,SAVEPATH,EPOCHS):
    trainloader = loadtraindata(TRAINPATH)
    # 神经网络结构
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   # 学习率为0.001
    criterion = nn.CrossEntropyLoss()   # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
 
    writer=SummaryWriter('./runs')
    # 训练部分
    for epoch in range(EPOCHS):    # 训练的数据量为5个epoch，每个epoch为一个循环
                            # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable
            optimizer.zero_grad()        # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            # forward + backward + optimize
            outputs = net(inputs)        # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            #创建可视化图
            writer.add_scalar("Loss/train",loss)
            loss.backward()                    # loss反向传播
            optimizer.step()                   # 反向传播后参数更新
            #print(loss.data.item())
            losssum=round(loss.data.item(),2)
            running_loss += losssum   # loss累加
            #加入参数
            writer.add_scalar("Loss/train_losssum",losssum)
            # writer.add_histogram("Param/weight",outputs.weight,epoch)
            if i % 50 == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
    print('Finished Training')
 
    # 保存神经网络,一种类型是保存参数，一种类型为保存模型
    savemodelname='net.pkl'
    savemodelname_params='net_params.pkl'
    savepath1=SAVEPATH+savemodelname
    savepath2=SAVEPATH+savemodelname_params
 
    torch.save(net,savepath1)                      # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), savepath2)  # 只保存神经网络的模型参数
 
if __name__=="__main__":
 
    #设置训练和模型保存路径
    trainpath='/home/qhy/Reserach/AICAS/mask-classification/log'
    savepath='/home/qhy/Reserach/AICAS/mask-classification/log'
 
    #设置超参数
    EPOCHS=10    #训练的轮数
 
    trainandsave(trainpath,savepath,EPOCHS)