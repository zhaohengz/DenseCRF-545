#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:14:47 2017

@author: lijunyi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import os
import utils
from skimage.io import imread


def outgenf(img_name):
    Prediction_dir = "../../data/Prediction"
    Prediction_name = str(img_name) +".png"
    Prediction_path = os.path.join(Prediction_dir,Prediction_name)
    prediction_img = imread(Prediction_path)
    prediction = utils.convert_from_color_segmentation(prediction_img)
    return prediction

Ground_truth_dir = "../../data/Ground_truth"
Name_list_dir = "../../data/namelist/val.txt"
output_path = "../../data/evaluation_results"
iou_cls,iou_img_dict=eval.evaluate_IoU_class_general(Ground_truth_dir,outgenf,Name_list_dir)
scipy.io.savemat(os.path.join(output_path,"iou_cls"),{"iou_cls":iou_cls})
scipy.io.savemat(os.path.join(output_path,"iou_img_dict"),iou_img_dict)