#!/usr/bin/env python

##################################
# configuration

source_path_gdt = './BSDS500_BD/groundTruth/'
source_path_img = './BSDS500_BD/images/'

type_img = '.jpg'
type_gdt = '.png'

##################################

import os


def generate_pair_lst(group):
	xPath = source_path_img + group + '/'
	yPath = source_path_gdt + group + '/'

	filename_x = []
	filename_y = []
	for file in os.listdir(xPath):
		if file.endswith(type_img):
			filename_x.append(int(str(file)[0:-4]))

	for file in os.listdir(yPath):
		if file.endswith(type_gdt):
			filename_y.append(int(str(file)[0:-4]))

	filename_x.sort()
	filename_y.sort()

	# unit test - check if image matches ground truth
	for x,y in zip(filename_x, filename_y):
		if x != y:
			print 'error' 
			return 

	with open(group+'_pair.lst', 'w') as f:
		for x,y in zip(filename_x, filename_y):
			f.write('{0}{1}{2} {3}{4}{5}\n'.format(
				xPath[2:], x, type_img, 
				yPath[2:], y, type_gdt))


def generate_single_lst(group):

	xPath = source_path_img + group + '/'

	filename_x = []
	filename_y = []
	for file in os.listdir(xPath):
		if file.endswith(type_img):
			filename_x.append(int(str(file)[0:-4]))

	
	filename_x.sort()

	with open(group+'_single.lst', 'w') as f:
		for x in filename_x:
			f.write('{0}{1}{2}\n'.format(xPath[2:], x, type_img))

		
if __name__ == "__main__":
	generate_pair_lst('train')
	generate_pair_lst('val')
	#generate_pair_lst('test')

	generate_single_lst('train')
	generate_single_lst('val')
	#generate_single_lst('test')
