# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:31:01 2019

@author: karth
"""
import cv2
import glob
import scipy.misc
count = 1
for img in glob.glob("D:/MobWorx/VideoSupRes/Check_Finala/Images/*.png"):
    img_lr= cv2.imread(img)
    low_h, low_w = 256,256
    img_lr = scipy.misc.imresize(img_lr, (low_h, low_w))
    cv2.imwrite('D:/MobWorx/VideoSupRes/Check_Finala/Images/'+'low'+str(count)+'.jpeg',img_lr)
    count = count+1