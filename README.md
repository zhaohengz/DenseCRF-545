## Synopsis

This  is for the final project of EECS545.

## Description

The project implements Fully Connected Conditional Random Field based image segmentation.


## Getting Started

When doing inference, Fully Connected CRF needs the output of other classifiers as its starting point,such as TexonBoost and Fully Convolutional Network(FCN) and so on. In our approach, we use FCN. The FCN Installation section shows how to setup FCN. Besides a basis classifier, you also need to link "libpermutohedral.so" to perform permutohedral bilateral filtering, which is an important part of inference process,you can find instructions in bilateral filtering section. with all these prerequisites, you can define a densecrf object to perform inference on single image, to perform inference on  a group of images, you can use Inference method; To compute IoU over a set of images, use evaluation method.

### DenceCRF Object

DenseCRF is the main class to perform inference, to define a DenseCRF Object:
```
import densecrf
# Inputs         : height : height of an image
#                  width  : width of an image
#                  number_labels: the number of classes to classify
crf = DenseCRF(height,width,number_labels)
```
To perform single image inference, use 'crf.inference' method:
```
import densecrf
# Input         : num_ite : the number of iterations you want inference algorithms perform, typically, you can set num_ite=10 and get good  segmentaion results
# Output          prediction_map: a 2D map with same width and height as the image,and each pixel represents the label of
#  
Q = crf.infernce(num_ite)
```

### Inference
To perform inference on a group of images, use 'inference.Inference' method:
```
import inference
# Inputs         : theta : this parameter is used to set parameters for densecrf object,
#                  line:  an array of image names
#                  Image_dir : the directory saving images
#                  Prediction_dir: the directory to save the output of inference
#                  if_save: option to choose if save inference results as PNG images
inference.Inference(theta,line,Image_dir,Unary_dir,Prediction_dir,if_save)
```
To compute IoU over a group of inference results, use 'evaluation.evaluate_IoU_class_general' mehod:

```
import evaluation
# evaluate_label : function to calculate the accuracy of image segmentation and labeling.
# Inputs         : truth_dir  : path of groud truth image
#                  outgenf    : a function with input argument as a specific image name
#                               and output as matrix for perdicted label for each pixel
#                  imglist_file   : path of .txt file that stores name of image
# Outputs        : iou_cls: IoU accuracy over each class
#                : iou_img_dict: IoU accuracy over each image
evaluation.evaluate_IoU_class_general(truth_dir, outgenf, imglist_file):
```
## Bilateral filtering

## Install FCN


## Dataset

Our algorithms are tested on PASCAL VOC 2012 dataset, you can acquire this dataset from [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/]

## Group Members

* Junyi Li,
* Chen Wang,
* Jiong Zhu,
* Zhaoheng Zheng,
