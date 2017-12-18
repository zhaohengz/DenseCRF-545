'''
Evalutaion of FCN model
author: Jiong Zhu(jiongzhu@umich.edu)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import evaluation as eval
import os
import utils

output_path = "../data/fcn_eval_result/"
fcn_results_path = "../data/fcn_result"
img_list_file = "../data/val.txt"
truth_dir = "../data/VOCdevkit/VOC2012/SegmentationClass"

if not os.path.exists(output_path):
	os.makedirs(output_path)

#Switch of whether save the labeled image
save_predict = True

if save_predict:
	#Generate legend for plotting the inference result
	cls = utils.pascal_classes()
	palette = utils.pascal_palette_inv()
	legend = []
	for (clsname,clsindex) in cls.items():
		legend.append(mpatches.Patch(color=np.array(palette[clsindex])/255, label=clsname))

#fcn_predict:	Give prediction result by fcn model
#input:			im_name:	name of the image
#output:		prediction: predicted label for each pixel as a matrix
def fcn_predict(im_name):
	fcn_result = scipy.io.loadmat(os.path.join(fcn_results_path,im_name))
	score = fcn_result["raw_score"]
	prediction = score.argmax(axis=0)

	if save_predict:
		prediction_image = utils.convert_to_color_segmentation(prediction)
		plt.imshow(prediction_image)
		plt.legend(handles=legend,loc=3,bbox_to_anchor=(0., 1.02, 1., .102),
			   ncol=5, mode="expand", borderaxespad=0.)
		gcf = plt.gcf()
		resolution = (3840,2160)
		dpi = 250
		gcf.set_size_inches(resolution[0]/dpi, resolution[1]/dpi)
		plt.savefig(os.path.join(output_path,im_name+".png"))

	return prediction


#Call evaluation function, use the prediction function "fcn_predic" defined
fcn_iou_cls, fcn_iou_img = eval.evaluate_IoU_class_general(truth_dir,fcn_predict,img_list_file)

#Save results to mat file
scipy.io.savemat(os.path.join(output_path,"fcn_iou_cls"),{"iou_cls":fcn_iou_cls})
scipy.io.savemat(os.path.join(output_path,"fcn_iou_img"),fcn_iou_img)