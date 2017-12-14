#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:42:33 2017

@author: lijunyi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from densecrf import DenseCRF
from compatibility import PottsCompatibility
from PIL import Image

def pascal_palette_inv():
  palette = {0:(  0,   0,   0),
             1:(128,   0,   0),
             2:(  0, 128,   0),
             3:(128, 128,   0),
             4:(  0,   0, 128),
             5:(128,   0, 128),
             6:(  0, 128, 128),
             7:(128, 128, 128),
             8:( 64,   0,   0),
             9:(192,   0,   0),
             10:( 64, 128,   0),
             11:(192, 128,   0),
             12:( 64,   0, 128),
             13:(192,   0, 128),
             14:( 64, 128, 128),
             15:(192, 128, 128),
             16:(  0,  64,   0),
             17:(128,  64,   0),
             18:(  0, 192,   0),
             19:(128, 192,   0),
             20:(  0,  64, 128) }
  return palette

def convert_to_color_segmentation(arr_2d):
  arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1],3), dtype=np.uint8)
  palette = pascal_palette_inv()

  
  for c, i in palette.items():
      m = (arr_2d == np.array(c).reshape(1,1))
      arr_3d[m]=i

  return arr_3d

img = plt.imread("../../data/Ground_truth/2007_000032.png")
unary = io.loadmat("../../data/Unary_images/2007_000032.mat")
unary_energy = unary["energy"].reshape(21,img.shape[0]*img.shape[1])
crf = DenseCRF(img.shape[0], img.shape[1], 21)
crf.set_unary(unary_energy)
crf.add_pairwise_gaussian(3 ,3, PottsCompatibility(3))
crf.add_pairwise_bilateral(80, 80, 13, 13, 13, img, PottsCompatibility(10))
Q = crf.inference(5)
Prediction = np.argmax(Q,0).reshape(img.shape[0],img.shape[1]).astype(np.uint8)
Prediction_image = convert_to_color_segmentation(Prediction)
im = Image.fromarray(Prediction_image)
im.save("../../data/Prediction/2007_000032.png")
