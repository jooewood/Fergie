#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

#folder = '/home/tensorflow/Downloads/picture_title'
## add image path by your order
# paths_imgs = glob.glob('/home/tensorflow/Downloads/picture_title/*.jpg')

paths_imgs = []
for i in range(len(paths)):
    tmp = '%s/%d.jpg' % (folder, i)
    if os.path.exists(tmp):
        paths_imgs.append(tmp)
        
#count = 0 
#for filename in paths_imgs:
#    tmp = '%s/%d.jpg' % (folder, count)
#    os.system('mv {} {}'.format(filename, tmp))
#    count += 1
data_folder = '/home/tensorflow/deepdrug_gan/gentrl/hpk1_virtual_117wan_codesize_50_batchsize_1000_epoch_20'
input_folder = '%s/picture_rdkit_draw' % data_folder
paths_imgs = glob.glob('%s/*.jpg' % input_folder)
out_folder= '%s/picture_title' % data_folder
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
img_array = []
count = 7
for i, filename in enumerate(paths_imgs):
    tmp = '%s/%d.jpg' % (input_folder, i)
    img = Image.open(tmp)
    plt.figure(figsize=(15, 15))
    plt.subplot(1,1,1)
    plt.title('Iteration: {}'.format(count), fontsize=30)
    plt.axis('off')
    plt.imshow(img)
    img_path = '{}/{}.jpg'.format(out_folder, i)
    plt.savefig(img_path, quality=95, optimize=True)
    plt.close()
    count += 1

## get each image's height and width
img_array = []
for i in range(1000):
    filename = '%s/%d.jpg' % (input_folder, i)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

fps = 24
out = cv2.VideoWriter("/home/tensorflow/Downloads/gen_mol.avi",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()