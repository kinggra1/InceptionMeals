To run the labeling:

To run the testing on a folder of images, the command is:

```
python3 label_image.py -t <image_directory>
```

The folder <image_directory> must contain jpg images and one file named "label.txt", which contains a list of image names and corresponding labels

The default graph file is "meal_classifier" and the default label list is "meal_labels". These are the 2 files output by the training script.  To change the files used, the command line arguments `-l <label_file>` and `-g <graph_file>` can be used.

It is also possible to run the classifier on a single image.  Instead of the '-t' option, use `-i <filename>`

By default, the output for a test folder ('-t') will contain the summary results as specified in the project guidelines, and the output for a single image ('-i') will output the labels determined for that image. To see a more detailed output containing the confidence scores, predicted, and actual classes on every image, use the `-v True` option.

Here is the complete list of command line arguments:

```
usage: label_image.py [-h] [-i IMAGE_NAME] [-g GRAPH_FILE] [-l LABEL_FILE]                                                                                                                                                         
                      [-t TEST_DIR] [-v VERBOSE]                                                                                                                                                                                   
                                                                                                                                                                                                                                   
optional arguments:                                                                                                                                                                                                                
  -h, --help     show this help message and exit                                                                                                                                                                                   
  -i IMAGE_NAME  If provided, the classifier will run on this image file only                                                                                                                                                      
  -g GRAPH_FILE  The inception graph file, default: 'meal_classifier'                                                                                                                                                              
  -l LABEL_FILE  The labels for the inception graph, default: 'meal_labels'                                                                                                                                                        
  -t TEST_DIR    The folder containing the test images and 'label.txt',                                                                                                                                                            
                 default: 'imaged_tryout'                                                                                                                                                                                          
  -v VERBOSE     'True' or 'False', enable verbose output
  ```