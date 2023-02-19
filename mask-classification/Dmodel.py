# -*- coding: utf-8 -*-#
 
#-------------------------------------------------------------------------------
# Name:         Dmodel
# Description:  
# Author:       Administrator
# Date:         2021/3/8
#-------------------------------------------------------------------------------
'''
模块简介:该模块完成了深度学习模型的搭建
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
 
class Net(nn.Module):                 # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)       # 卷积层
        self.pool = nn.MaxPool2d(2, 2)        # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)      # 卷积层
 
 
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)      # 10个输出
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
         # 从卷基层到全连接层的维度转换
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
if __name__=="__main__":
    net=Net()
    print(net)