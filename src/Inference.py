#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:00:37 2017

@author: lijunyi
"""

import numpy as np
import scipy.io as io
from densecrf import DenseCRF
from compatibility import PottsCompatibility
from PIL import Image
import utils
import os.path as path
from skimage.io import imread

def inference(theta,line,Image_dir,Unary_dir,Prediction_dir,if_save):
    
    Image_name = str(line) +".jpg"
    Image_path = path.join(Image_dir,Image_name)
    Unary_name = str(line) +".mat"
    Unary_path = path.join(Unary_dir,Unary_name)
    Prediction_name = str(line) +".png"
    Prediction_path = path.join(Prediction_dir,Prediction_name)
    img = imread(Image_path)
    unary = io.loadmat(Unary_path)
    unary_energy = unary["energy"].reshape(21,img.shape[0]*img.shape[1])
    crf = DenseCRF(img.shape[0], img.shape[1], 21)
    crf.set_unary(-unary_energy)
    crf.add_pairwise_gaussian(1 ,1, PottsCompatibility(3))
    crf.add_pairwise_bilateral(theta[0], theta[0], theta[1], theta[1], theta[1], img, PottsCompatibility(10))
    Q = crf.inference(5)
    Prediction = np.argmax(Q,0).reshape(img.shape[0],img.shape[1]).astype(np.uint8)
    if(if_save):
        Prediction_image = utils.convert_to_color_segmentation(Prediction)
        im = Image.fromarray(Prediction_image)
        im.save(Prediction_path)
    return Prediction