#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Compute the KL-divergence between ground_truth images and predictions made by CRF
Author: Junyi Li(junyili@umich.edu)
'''

import numpy as np
from Inference import inference
import os
from skimage.io import imread
import utils
import evaluation as eval
import scipy
from densecrf import DenseCRF
from compatibility import PottsCompatibility


Image_dir = "../../data/Images"
Unary_dir = "/media/lijunyi/Seagate JIONG/results"
Prediction_dir = "../../data/Prediction"
Name_list = "../../data/namelist/val.txt"
Ground_truth_dir = "../../data/Ground_truth"
output_path = "../../data/evaluation_results"

with open(Name_list,"r")  as f:
     lines = f.read().splitlines()

np.random.seed(42)
lines_shuffled = np.random.permutation(lines)
Sample_size = 94
lines_sampled = lines_shuffled[0:Sample_size]
Max_iteration = 20
kl_divergence = np.zeros((Sample_size,Max_iteration))

for i,line in enumerate(lines_sampled):
    Image_name = str(line) +".jpg"
    Image_path = os.path.join(Image_dir,Image_name)
    Unary_name = str(line) +".mat"
    Unary_path = os.path.join(Unary_dir,Unary_name)
    img_truth_name = os.path.join(Ground_truth_dir, str(line)) + '.png'
    img_truth = imread(img_truth_name)
    truth_data = utils.convert_from_color_segmentation(img_truth)
    img = imread(Image_path)
    unary = scipy.io.loadmat(Unary_path)
    unary_energy = unary["energy"].reshape(21,img.shape[0]*img.shape[1])
    crf = DenseCRF(img.shape[0], img.shape[1], 21,Ground_truth=truth_data)
    crf.set_unary(-unary_energy)
    crf.add_pairwise_gaussian(1 ,1, PottsCompatibility(3))
    crf.add_pairwise_bilateral(61, 61, 11, 11, 11, img, PottsCompatibility(10))
    Q = crf.inference(Max_iteration)
    kl_divergence[i,:] = crf.KL_divergence
    print(kl_divergence[i,:])
scipy.io.savemat(os.path.join(output_path,"kl_divergence"),{"kl_divergence":kl_divergence})
