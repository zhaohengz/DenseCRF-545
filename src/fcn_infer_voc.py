'''
Calculating and saving the unary potential of each pixel for each image.
All unary potential will be save to the path specified in the variable "result_path" within a .mat file (MATLAB compatible)
Each file would be around 30MB, so please set "result_path" to a large disk. 

Please download the caffe model first. See the instructions in fcn.berkeleyvision.org-master/voc-fcn8s

Based on code provided by Jonathan Long et al. on https://github.com/shelhamer/fcn.berkeleyvision.org (infer.py)
Modified by Jiong Zhu (jiongzhu@umich.edu)
'''
import numpy as np
from PIL import Image

import caffe
import scipy.io
import os

os.chdir("fcn.berkeleyvision.org-master")

#List of image names to calculate (no extension needed just as VOC imageset list)
img_list_file = "imglist.txt"

#Path to images
img_path = "../data/pascal/VOC2012/JPEGImages/"

#Path to save the unary results
result_path = "result/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
with open(img_list_file,"r") as imf:
    im_names_base = imf.readlines()
im_names = [fn.strip() for fn in im_names_base]

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)

for im_name in im_names:
    filepath = os.path.join(img_path,im_name+".jpg")
    if os.path.isfile(filepath):
	im = Image.open(filepath)
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	in_ = in_.transpose((2,0,1))

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	score = net.blobs['score'].data[0]
	label = score.argmax(axis=0)
	
	energy = np.zeros(score.shape)
	for i in range(score.shape[1]):
	    for j in range(score.shape[2]):
                energy[:,i,j] = np.log(np.exp(score[:,i,j])/np.sum(np.exp(score[:,i,j])))
	#Both the unary potential named "energy", and the raw output of the FCN net named "raw_score" is saved.
	scipy.io.savemat(os.path.join(result_path, im_name),{"raw_score":score, "energy":energy},do_compression=True)
    else:
        print("File not exist!")
        print(filepath)

