# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:07:55 2019

@author: karth
"""

from tensorflow import keras
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import scipy 
import os
import re

model = keras.models.load_model('MobileSRNet.h5')
input = keras.Input((None, None, 3))
output = model(input)
model = keras.models.Model(input, output)
count = 1

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]    
filenames = sorted(glob.glob("D:/MobWorx/VideoSupRes/Check_Finala/Images/*.jpg"))
filenames.sort(key=natural_sort_key)
images = [cv2.imread(img) for img in filenames]
for img in images:
    img_lr= img
    imgs_hr = []
    imgs_lr = []
    #img_lr = scipy.misc.imresize(img_lr, (240, 320))
    imgs_lr.append(img_lr)
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    
    #generated image of 4x
    fake_hr = np.squeeze(model.predict(imgs_lr),axis = 0)
    
    #unscaling
    fin_hr = ((fake_hr+ 1.)* 127.5).astype(np.uint8)
    cv2.imwrite('D:/MobWorx/VideoSupRes/Check_Finala/Res/'+str(count)+'.jpg',fin_hr)
    count=count+1
