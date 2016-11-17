#! /usr/bin/env python2

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import random

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

LEARNING_RATE = 0.01

categories = []
image_names = {}


class Image():

	TRAINING = "training"
	TESTING = "testing"
	VALIDATION = "validation"

	def __init__(self, name, t):
		self.name = name
		self.im_type = t		
		


# from inception graph creation on GitHub
def create_inception_graph(inception_location):
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        inception_location, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      precalc_tensor, image_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, precalc_tensor, image_data_tensor, resized_input_tensor


# taken from visualization examples
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.histogram_summary('histogram', var)

# taken from retrain example
def add_evaluation_step(result_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(result_tensor, 1), \
        tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step

# taken from layer construction tutorial
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      #tf.histogram_summary('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  #tf.histogram_summary('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, ground_truth_input)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("inception_dir", help="Directory where the inception graph should be located")
	parser.add_argument("images_dir", help="Directory containing image folders")
	parser.add_argument("precalc_dir", help="Directory to store the feature values for images run through the network")
	parser.add_argument("training_steps", help="The number of training steps to take")

	return parser.parse_args()




# return a dictionary of training, testing, and validation images for this category
def add_image_names(category, full_path):
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
			count += 1

	return {"training" : training,
		"testing" : testing,
		"validation" : validation }


def precalc_image(image_contents, session, image_data_tensor, precalc_tensor):
	
	values = session.run(precalc_tensor, {image_data_tensor: image_contents})
	return np.squeeze(values)

	
def precalc_categories(session, image_data_tensor, precalc_tensor, images, categories, images_dir, precalc_dir):
	for category in categories:
		
		# create the subdirectory for this category in precalcs
		category_dir = os.path.join(precalc_dir, category)
		if not os.path.isdir(category_dir):
			os.makedirs(category_dir)

		image_dict = images[category]

		for key in image_dict.keys():

			for image in image_dict[key]:

				full_image_path = os.path.join(images_dir, category, image)
				
				# use tensorflow's image opening library
				image_contents = gfile.FastGFile(full_image_path, 'rb').read()
				
				precalc_name = os.path.join(precalc_dir, category, image) + ".txt"
					
				if os.path.isfile(precalc_name):
					print("Precalculated file: {} exists".format(image))
				else:
					precalc_file =  open(precalc_name, 'w')
					values = precalc_image(image_contents, session, image_data_tensor, precalc_tensor)
					precalc_file.write(','.join(str(x) for x in values))
					print("Created precalculation for file: {}".format(image))

if __name__ == "__main__":
	args = get_args()
	
	current_dir = os.getcwd()

	images_dir = os.path.join(current_dir, args.images_dir)
	training_steps = int(args.training_steps)
	precalc_dir = os.path.join(current_dir, args.precalc_dir)
	inception_dir = os.path.join(current_dir, args.inception_dir)

	# for each image category name (item)
	for item in os.listdir(args.images_dir):
		full_path = os.path.join(images_dir, item)
		
		if os.path.isdir(full_path):
			categories.append(item)
			image_names[item] = add_image_names(item, full_path)
			
	
	graph, precalc_tensor, image_data_tensor, resized_image_tensor = (create_inception_graph(inception_dir))
	
	session = tf.Session()	

	# create all precalculated values
	precalc_categories(session, image_data_tensor, precalc_tensor, image_names, categories, images_dir, precalc_dir)

	class_count = len(image_names.keys())

	#consider applying distortionsa

	
	# new classification layer for meals!
	(train_step, cross_entropy, bottleneck_input, ground_truth_input, 
		final_tensor) = add_final_training_ops(class_count, "meal_tensor",precalc_tensor)

	

	evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

	#merged = tf.summary.merge_all()
	merged = tf.merge_all_summaries()

	init = tf.initialize_all_variables() # global_variables_initializer()
	session.run(init)

	batch_size = 10
	for i in range (training_steps):
		
		# generate a set of 10 images to test
		precalcs = []
		ground_truths = []
		for x in range(batch_size):
			category_index = random.randrange(class_count)
			category = categories[category_index]
			training_set = image_names[category]["training"]
			file_index = random.randrange(len(training_set))
			filename = training_set[file_index]
			
			full_path = os.path.join(precalc_dir, category, filename) + '.txt'
			precalc_file = open(full_path, 'r')
			precalc_string = precalc_file.read()
			precalc_values = [float(x) for x in precalc_string.split(',')]
			precalcs.append(precalc_values)

			ground_truth = np.zeros(class_count, dtype=np.float32)
			ground_truth[category_index] = 1.0
			ground_truths.append(ground_truth)

		
		train_summary, _ = session.run([merged, train_step],
			feed_dict={bottleneck_input: precalcs,
			ground_truth_input: ground_truths})
			
		sys.stdout.write("\rTrained step {}".format(i))
		sys.stdout.flush()


	print('\n')

	# test accuracy with 500 tests on previously unseen images
	for x in range(500):
		category_index = random.randrange(class_count)
		category = categories[category_index]
		testing_set = image_names[category]["testing"]
		file_index = random.randrange(len(testing_set))
		filename = testing_set[file_index]

		full_path = os.path.join(precalc_dir, category, filename) + '.txt'
		precalc_file = open(full_path, 'r')
		precalc_string = precalc_file.read()
		precalc_values = [float(x) for x in precalc_string.split(',')]
		precalcs.append(precalc_values)

		ground_truth = np.zeros(class_count, dtype=np.float32)
		ground_truth[category_index] = 1.0
		ground_truths.append(ground_truth)


	test_accuracy = session.run(evaluation_step,
		feed_dict={bottleneck_input: precalcs,
		ground_truth_input: ground_truths})
	print('Testing final accuracy with images outside of training set:\
		Accuracy: {}'.format(test_accuracy*100))
	


	output_graph = graph_util.convert_variables_to_constants(session, graph.as_graph_def(), ["meal_tensor"])
	graph_file = gfile.FastGFile("meal_classifier", 'wb')
	labels_file = gfile.FastGFile("meal_labels", 'w')
	
	graph_file.write(output_graph.SerializeToString())
	labels_file.write('\n'.join(categories) + '\n')	


