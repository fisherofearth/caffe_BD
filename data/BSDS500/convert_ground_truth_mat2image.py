#!/usr/bin/env python

##################################
# configuration

# select which group of ground truth to use
group = 0 # 0 ~ 4

source_path_gdt = './BSDS500/data/groundTruth/'
target_path_gdt = './BSDS500_BD/groundTruth/'

source_path_img = './BSDS500/data/images/'
target_path_img = './BSDS500_BD/images/'

box_size = (160, 160)
stride = 100

##################################

import os
import numpy as np
import scipy.io as sio
from scipy import misc
from PIL import Image
from PIL import ImageFilter

a = 1

def generate_boxes(img_size):
	boxes = []
	for y in xrange(((img_size[1]-box_size[1])/stride) + 1):
		by = y * stride
		for x in xrange(((img_size[0]-box_size[0])//stride) + 1):
			bx = x * stride
			boxes.append((bx, by, bx + box_size[0], by+ box_size[1]))
	return boxes


def grab_groundtruth(filename):
	return sio.loadmat(filename)['groundTruth'][0,group]['Boundaries'][0,0] 
    

def convert_groundtruth(folderName):
	spath = source_path_gdt + folderName + '/'
	tPath = target_path_gdt + folderName + '/'

	if not os.path.exists(tPath):
		os.makedirs(tPath)

	counter = 0
	for file in os.listdir(spath):
		if file.endswith('.mat'):
			gt = grab_groundtruth(spath+file)

			img = Image.fromarray(np.uint8(gt)*255)#.filter(ImageFilter.SMOOTH)
			#img = img.point(lambda p: p >50 and 255 )  
			
			#m = 255/np.max(np.max(img))
			#img = img.point(lambda p: p * m )  

			boxes = generate_boxes(img.size)
			
			for i in xrange(len(boxes)):
				img.crop(boxes[i]).save(tPath+str(file)[0:-4]+str(i)+'.png', 'PNG')
				counter += 1
			#print str(file)[0:-4]
	return counter

def convert_images(folderName):

	spath = source_path_img + folderName + '/'
	tPath = target_path_img + folderName + '/'

	if not os.path.exists(tPath):
		os.makedirs(tPath)

	counter = 0
	for file in os.listdir(spath):
		if file.endswith('.jpg'):

			img = Image.open(spath+file)#.filter(ImageFilter.SMOOTH_MORE)
			boxes = generate_boxes(img.size)
			
			for i in xrange(len(boxes)):
				img.crop(boxes[i]).save(tPath+str(file)[0:-4]+str(i)+'.jpg', 'JPEG')
				counter += 1

	return counter


if __name__ == '__main__':
	print 'converting ground truth ...'
	num_Y_train = convert_groundtruth('train')
	num_Y_val  	= convert_groundtruth('val')
	#num_Y_test 	= convert_groundtruth('test')

	num_X_train = convert_images('train')
	num_X_val  	= convert_images('val')
	#num_X_test 	= convert_images('test')

