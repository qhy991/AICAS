# -*- coding: utf-8 -*-#
 
#-------------------------------------------------------------------------------
# Name:         testfunctional
# Description:  
# Author:       Administrator
# Date:         2021/3/11
#-------------------------------------------------------------------------------
import cv2 as cv
import  numpy as np
import matplotlib.pyplot as plt
 
img=cv.imread("01.jpg")
npimg =np.array(img)
 
npimg[200:330][200:330]=[255,0,0]
print(npimg.shape)
plt.imshow(npimg)
plt.show()