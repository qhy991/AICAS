# -*- coding: utf-8 -*-#
 
#-------------------------------------------------------------------------------
# Name:         testgraph
# Description:  
# Author:       Administrator
# Date:         2021/3/17
#-------------------------------------------------------------------------------
 
#导入依赖包
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import  copy
import os
 
'''
文件简介：
该模块主要评估神经网络的计算结果，包含如下：
(1)计算分类的准确率、精确率、召回率；
(2)绘制ROC曲线；
reference data:
(1)https://blog.csdn.net/hfutdog/article/details/88085878
(2)https://blog.csdn.net/hfutdog/article/details/88079934
(3)https://www.jianshu.com/p/5df19746daf9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~精确率~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(1)Macro Average:宏平均是指在计算均值时使每个类别具有相同的权重，最后结果是每个类别的指标的算术平均值。
(2)Micro Average:微平均是指计算多分类指标时赋予所有类别的每个样本相同的权重，将所有样本合在一起计算各个指标
(3)如果每个类别的样本数量差不多，那么宏平均和微平均没有太大差异
(4)如果每个类别的样本数量差异很大，那么注重样本量多的类时使用微平均，注重样本量少的类时使用宏平均
(5)如果微平均大大低于宏平均，那么检查样本量多的类来确定指标表现差的原因
(6)如果宏平均大大低于微平均，那么检查样本量少的类来确定指标表现差的原因
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
 
y_pred = [0, 1, 1, 1]
y_true = [0, 1, 0,1]
 
#计算真阳性 假阳性
def  concludescore(y_pred,y_true):
    score_result=[]
 
    #计算准确率
    Accuracy=accuracy_score(y_true, y_pred)
 
    #计算精确率
    Precision=precision_score(y_true, y_pred, average='macro')
    #precision_score(y_true, y_pred, average='micro')
    # precision_score(y_true, y_pred, average='weighted')
    # precision_score(y_true, y_pred, average=None)
 
    #计算召回率
    Recall=recall_score(y_true, y_pred, average='macro')
    # recall_score(y_true, y_pred, average='micro')
    # recall_score(y_true, y_pred, average='weighted')
    # recall_score(y_true, y_pred, average=None)
 
    score_result.append(Accuracy)
    score_result.append(Precision)
    score_result.append(Recall)
    return  score_result
 
#y_true:真实值，y_score：预测值的得分，比如实际值为1，预测值为0.5
 
 
#二分类AUC曲线
def drawAUC_TwoClass(y_true,y_score):
    fpr, tpr, thresholds =roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')#设定图例的位置，右下角
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    if os.path.exists('./resultphoto')==False:
        os.makedirs('./resultphoto')
    plt.savefig('resultphoto/AUC_TwoClass.png', format='png')
    plt.show()
 
 
#多分类AUC曲线
def drawAUC_ManyClass(y_true,y_score,num_class):
    #将每类单独看作0/1二分类器，分别描述每类auc；
    plt.figure()
    for cls in range(num_class):
        #计算每一个类别的auc
        on_y_true=copy.deepcopy(y_true)
        for on_class in range(len(on_y_true)):
            if on_y_true[on_class]==cls:
                on_y_true[on_class]=1
            else:
                on_y_true[on_class]=0
        fpr, tpr, thresholds =roc_curve(on_y_true,y_score)
        roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
        labelauc="AUC--"+str(cls)
        #开始画ROC曲线
        if cls/2==0:
            color='g'
        elif cls/2==1:
            color='b'
        else:
            color='cyan'
        plt.plot(fpr, tpr, color,label= labelauc+'= %0.2f'% roc_auc)
        on_y_true.clear()
 
    plt.legend(loc='lower right')#设定图例的位置，右下角
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    if os.path.exists('./resultphoto')==False:
        os.makedirs('./resultphoto')
    plt.savefig('resultphoto/AUC_ManyClass.png', format='png')
    plt.show()
 
 
#绘制混淆矩阵
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if os.path.exists('./resultphoto')==False:
        os.makedirs('./resultphoto')
    plt.savefig('resultphoto/HAR_cm.png', format='png')
    plt.show()
 
 
 
 
if __name__=="__main__":
    #测试二分类
    # y = np.array([0, 0, 0, 1])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])
    # drawAUC_TwoClass(y,scores)
 
    #测试多分类
    # y_true=[0,1,2,3,0,1,2,3,1,2,0]
    # y_score=[0.7,0.8,0.9,0.4,0.5,0.5,0.9,0.6,0.7,0.3,0.8]
    # num_class=4
    # drawAUC_ManyClass(y_true,y_score,num_class)
 
    #绘制混淆矩阵
    test_y=[0,1,0,0,0,1,1,1]
    pred_y=[0,0,1,0,1,0,1,1]
    cm = confusion_matrix(test_y, pred_y)
    labels_name=[0,1]
    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")