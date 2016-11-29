To run the labeling:

To run the testing on a folder of images, the command is:

'''
python3 label_image.py -t <image_directory>
'''

The folder <image_directory> must contain jpg images and one file named "label.txt", which contains a list of image names and corresponding labels

The default graph file is "meal_classifier" and the default label list is "meal_labels". These are the 2 files output by the training script.  To change the files used, the command line arguments can be provided:

