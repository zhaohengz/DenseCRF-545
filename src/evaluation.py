#!/usr/bin/env python
#from measure import predict
import numpy as np
import os
import sys
from skimage.io import imread,imsave
import utils

#Calculate IoU for matrix A and B with a given label
def int_uni_cls(A, B, label):
    A_mask = (A==label)
    B_mask = (B==label)
    intersection = np.sum(np.bitwise_and(A_mask,B_mask))
    union = np.sum(np.bitwise_or(A_mask,B_mask))
    return intersection, union

# evaluate_label : function to calculate the accuracy of image segmentation and labeling.
# Inputs         : truth_dir  : path of groud truth image
#                  outgenf    : a function with input argument as a specific image name 
#                               and output as matrix for perdicted label for each pixel
#                  txt_dir    : path of .txt file that stores name of image
#                  class_idx  : equals to the 1~20 label, (0 for background)
# Outputs        : pixel-wise accuracy, camputed through
def evaluate_IoU_class_general(truth_dir, outgenf, imglist_file):
    class_count = len(utils.pascal_classes())+1 #include background as a class
    int_vec = [list() for i in range(class_count)]
    union_vec = [list() for i in range(class_count)]
    with open(imglist_file,"r") as imf:
        im_names = imf.read().splitlines()

    im_counts = len(im_names)
    #Store the IoU for each specific image
    iou_img_dict = {}
    int_vec = np.zeros((class_count,im_counts))
    union_vec = np.zeros((class_count,im_counts))

    ptr = 0;
    for img_name in im_names:
        img_truth_name = os.path.join(truth_dir, img_name) + '.png'
        img_truth = imread(img_truth_name)
        truth_data = utils.convert_from_color_segmentation(img_truth)
        output_data = outgenf(img_name)
        #Calculate IoU for each class
        iou_img = np.zeros(class_count)
        for class_idx in range(0,class_count):
            int_i, uni_i = int_uni_cls(truth_data, output_data, class_idx)
            if (int_i == 0 and uni_i == 0):
                iou_img[class_idx] = np.NaN
            else:
                iou_img[class_idx] = int_i/uni_i
            int_vec[class_idx,ptr] = int_i
            union_vec[class_idx,ptr] = uni_i
        iou_img_dict["img_"+img_name] = iou_img #add "img_" here to ensure variable name is valid in matlab
        ptr = ptr + 1
    iou_cls = [np.sum(int_vec[i])/np.sum(union_vec[i]) for i in range(class_count)]
    return iou_cls, iou_img_dict