#!/usr/bin/env python
# from measure import predict
import numpy as np
import os
import sys
from skimage.io import imread,imsave
from utils import convert_from_color_segmentation

# gen_image     : function to generate output image (via img = predict(img) function)
# Inputs        : image_dir  : path of original image
#                 output_dir : path to store the output image
#                 txt_dir    : path of .txt file that stores name of original image
def gen_image(image_dir, output_dir, txt_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(txt_dir, 'rb') as f:
        for img_name in f:
            img_base_name = img_name.strip()
            img_name = os.path.join(image_dir, img_base_name.decode()) + '.png'
            img = imread(img_name)
            img = predict(img)
            imsave(os.path.join(output_dir, img_base_name.decode()) + '.png', img)

    exit()


# gen_namelist  : Generate the list of image name and store as .txt file in a
#                 target path
# Inputs        :          txt_dir : path of .txt file to be operated
#                       ourput_dir : path of .txt file to be generated
#                 output_file_name : name of the output file
def gen_namelist(txt_dir, output_dir, output_file_name):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    target = open(output_dir + output_file_name,'w')
    with open(txt_dir, 'r') as f:
        for string in f:
            sp = string.split()
            if sp[1] == '1':
                target.write(sp[0]+'\n')
    target.close()
    exit()

# gen_mask :  Generate the mask matrix of A. For elements in A equals to val,
#             keep it, and replace other elements with 0
# Inputs   :   A : matrix to be operated
#            val : target value
# Outputs  : masked matrix A
def gen_mask(A, val):
    m = np.size(A,0); n = np.size(A,1)
    temA = np.absolute(val*np.ones((m,n)) - A)
    maskA = np.ones((m,n)) - np.sign(temA)
    return val*maskA.astype(int)


# int_uni    : Compute the interacton over union accuracy of labeling matrix.
# Inputs     :      A,B : matrix of two image labels(consisting of 1~20).
#                 label : label of evaluated class
# Outputs    : err : interaction over union error
#              num : number of union pixels
def int_uni(A, B, label):
    mask_truth = gen_mask(A, label)
    mask_output = gen_mask(B, label)
    inter = gen_mask(mask_truth + mask_output, 2*label)
    union = gen_mask(mask_truth + mask_output, label)
    err = 1 - np.sum(inter)/(np.sum(inter)+2*np.sum(union))
    num = (np.sum(inter)+2*np.sum(union))/(2*label)
    return err, num


# evaluate_label : function to calculate the accuracy of image segmentation and labeling.
# Inputs         : truth_dir  : path of groud truth image
#                  output_dir : path of algorithm outputs image
#                  txt_dir    : path of .txt file that stores name of image
#                  class_idx  : equals to the 1~20 label, (0 for background)
# Outputs        : pixel-wise accuracy, camputed through
def evaluate_class(truth_dir, output_dir, txt_dir, class_idx):
    error = [];
    numunion = [];
    with open(txt_dir, 'rb') as f:
        for img_name in f:
            img_base_name = img_name.strip()
            img_truth_name = os.path.join(truth_dir, img_base_name.decode()) + '.png'
            img_output_name = os.path.join(output_dir, img_base_name.decode()) + '.png'
            img_truth = imread(img_truth_name)
            img_output = imread(img_output_name)
            truth_data = convert_from_color_segmentation(img_truth)
            output_data = convert_from_color_segmentation(img_output)
            err, num = int_uni(truth_data, output_data, class_idx)
            error = np.append(error, err)
            numunion = np.append(numunion, num)

    return 1 - np.sum(np.dot(error,numunion))/(np.sum(numunion))
