#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:40:16 2019

@author: Di Shen
"""

import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import os
import re
img_root="/Users/stephaniexia/Downloads/postiveboxes"
#Edit each frame's appearing time!
fps=2
fourcc=VideoWriter_fourcc(*"MJPG")
videoWriter=cv2.VideoWriter("MaskVSAssoLSTM.mp4",fourcc,fps,(1200,1200))

im_names=os.listdir(img_root)
im_names.sort(key= lambda x:float(re.findall(r'\d+',x)[0]))

for im_name in im_names:
    print(im_name)
    frame=cv2.imread(os.path.join(img_root,im_name))
    videoWriter.write(frame)
	
videoWriter.release()
