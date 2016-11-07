#! /usr/bin/env python3

import sys
import os
import argparse


# values taken from inception examples
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


categories = []
image_names = {}


class Image():

	TRAINING = "training"
	TESTING = "testing"
	VALIDATION = "validation"

	def __init__(self, name, t):
		self.name = name
		self.im_type = t		
		

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("images_dir", help="Directory containing image folders")
	parser.add_argument("precalc_dir", help="Directory to store the feature values for images run through the network")
	parser.add_argument("training_steps", help="The number of training steps to take")

	return parser.parse_args()




# return a dictionary of training, testing, and validation images for this category
def get_image_names(category, full_path):
	count = 0
	training = []
	testing = []
	validation = []

	for image in os.listdir(full_path):
		if image.lower().endswith(('.jpg', '.JPG', '.jpeg', '.JPEG')):
			
			# 5% to validation set
			if (count % 20 == 19):
				validation.append(image)
			# 10% to testing set
			elif (count % 20 >= 17):
				testing.append(image)
			# 85% to training
			else:
				training.append(image)
	return {"training" : training,
		"testing" : testing,
		"validation" : validation }


def precalc_image(image, category, images_dir, precalc_dir):
	# if os.path.join(

def precalc_categories(images, categories, images_dir, precalc_dir):
	for category in categories:
		image_dict = images[category]

		for image in image_dict["training"]:
			precalc_image(image, category, images_dir, precalc_dir)

if __name__ == "__main__":
	args = get_args()
	
	current_dir = os.getcwd()

	images_dir = args.images_dir
	training_steps = args.training_steps
	precalc_dir = args.precalc_dir

	# for each image category name (item)
	for item in os.listdir(args.images_dir):
		full_path = os.path.join(current_dir, images_dir, item)
		
		if os.path.isdir(full_path):
			categories.append(item)
			image_names[item] = add_image_names(full_path)
			
	

	precalc_categories(image_names, categories, images_dir, precalc_dir)


