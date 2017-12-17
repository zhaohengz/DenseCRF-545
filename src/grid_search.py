#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:13:06 2017

@author: lijunyi
"""

import numpy as np
from Inference import inference
import os
from skimage.io import imread
import utils
import evaluation as eval
import scipy

Image_dir = "../../data/Images"
Unary_dir = "/media/lijunyi/Seagate JIONG/results"
Prediction_dir = "../../data/Prediction"
Name_list = "../../data/namelist/train.txt"
Ground_truth_dir = "../../data/Ground_truth"
output_path = "../../data/evaluation_results"

with open(Name_list,"r")  as f:
     lines = f.read().splitlines()

lines_shuffled = np.random.permutation(lines)
Sample_size = 500
lines_sampled = lines_shuffled[0:Sample_size]

iou_vec = np.zeros((21,3,3))

for i,theta1 in enumerate(np.linspace(1,121,3)):
    for j,theta2 in enumerate(np.linspace(1,21,3)):
        int_vec = np.zeros((21,Sample_size))
        union_vec = np.zeros((21,Sample_size))
        theta = np.array([theta1,theta2])
        for ptr,line in enumerate(lines_sampled):
            Q = inference(theta,line,Image_dir,Unary_dir,Prediction_dir,False)
            img_truth_name = os.path.join(Ground_truth_dir, line) + '.png'
            img_truth = imread(img_truth_name)
            truth_data = utils.convert_from_color_segmentation(img_truth)
            for class_idx in range(0,21):
                int_i, uni_i = eval.int_uni_cls(truth_data, Q, class_idx)
                int_vec[class_idx,ptr] = int_i
                union_vec[class_idx,ptr] = uni_i
        iou_vec[:,i,j] = [np.sum(int_vec[i])/np.sum(union_vec[i]) for i in range(21)]
        scipy.io.savemat(os.path.join(output_path,"grid_search_"+str(i*3+j)),{"grid_search":iou_vec[:,i,j]})
scipy.io.savemat(os.path.join(output_path,"grid_search"),{"grid_search":iou_vec})
        