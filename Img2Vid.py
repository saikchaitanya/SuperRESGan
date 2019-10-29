import cv2
import numpy as np
import os
import re
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)] 
def frames_to_video(inputpath,outputpath,fps):
   image_array = []
   files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
   files.sort(key=natural_sort_key)
   for i in range(len(files)):
       img = cv2.imread(os.path.join(inputpath, files[i]))
       size =  (img.shape[1],img.shape[0])
       img = cv2.resize(img,size)
       image_array.append(img)
   fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   out = cv2.VideoWriter(outputpath,fourcc, fps, size)
   for i in range(len(image_array)):
       out.write(image_array[i])
   out.release()


inputpath = 'ImgVid'
outpath =  'supres1.mp4'
fps = 23
frames_to_video(inputpath,outpath,fps)