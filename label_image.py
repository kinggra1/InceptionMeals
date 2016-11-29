import tensorflow as tf
import sys
import os
import argparse

verbose = True

labels = [ 'salad', 'pasta', 'hotdog', 'frenchfry', 'burger', 'apple', 'banana', 'broccoli', 'pizza', 'egg', 'tomato', 'rice', 'strawberry', 'cookie' ]



<<<<<<< HEAD
def get_final_predictions(sess, label_lines, meal_tensor, image_filename):
    image_data = tf.gfile.FastGFile(image_filename, 'rb').read()         
            
    predictions = sess.run(meal_tensor, {'DecodeJpeg/contents:0': image_data})
=======
def get_final_predictions(sess, label_lines, softmax_tensor, image_filename):
    image_data = tf.gfile.FastGFile(image_filename, 'rb').read()         
            
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a


    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    if verbose:
<<<<<<< HEAD
        print(image_filename)
=======
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    final_pred = []
<<<<<<< HEAD
    if predictions[0][top_k[0]] < 0.6 and predictions[0][top_k[1]] > 0.1:
        final_pred = top_k[0:2]
    elif predictions[0][top_k[0]] < 0.8 and predictions[0][top_k[1]] > 0.2:
        final_pred = top_k[0:2]
    elif predictions[0][top_k[0]] * 0.8 < predictions[0][top_k[1]]:
        final_pred = top_k[0:2]
    else:
=======
    if predictions[0][top_k[0]] > 0.2:
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
        final_pred = top_k[0:1]

    return final_pred

def test_folder(test_dir, test_labels_filename, graph_file, label_file):
    test_images_file = open( os.path.join(test_dir, test_labels_filename) )
    test_images = [line.rstrip().split(' ') for line in test_images_file]

    label_lines = [line.rstrip() for line in tf.gfile.GFile(os.path.join(current_dir, label_file))]

    count_detected = {}
    count_misdetected = {}
    count_true = {}
    count_images = 0

    for label in labels:
        count_detected[label] = 0
        count_misdetected[label] = 0
        count_true[label] = 0

<<<<<<< HEAD
=======
    # Unpersists graph from file
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
    with tf.gfile.FastGFile(os.path.join(current_dir, graph_file), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
<<<<<<< HEAD
        meal_tensor = sess.graph.get_tensor_by_name('meal_tensor:0')
        for current_image in test_images:
            count_images += 1

            final_predictions = get_final_predictions(sess, label_lines, meal_tensor, os.path.join(test_dir, current_image[0]))

            if verbose:
                print("Pred:", end=' ')
            for node_id in final_predictions:
                if verbose:
                    print(label_lines[node_id], end=' ')
=======
        softmax_tensor = sess.graph.get_tensor_by_name('meal_tensor:0')
        for current_image in test_images:
            count_images += 1

            final_predictions = get_final_predictions(sess, label_lines, softmax_tensor, os.path.join(test_dir, current_image[0]))

            for node_id in final_predictions:
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
                if label_lines[node_id] in current_image[1:]:
                    count_detected[label_lines[node_id]] += 1
                else:
                    count_misdetected[label_lines[node_id]] += 1

            for label in current_image[1:]:
                count_true[label] += 1
            if verbose:
<<<<<<< HEAD
                print()
                print(current_image[1:])
                print()
                    
=======
                print(current_image[1:])
                print()
                    
            
            
            # print(current_image)
            

            # print(label_lines[top_k[0]])
            # print(current_image[1])
            # print('status=%s' % (label_lines[top_k[0]] == current_image[1]))
            # print()
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a

    if verbose:
        print(count_detected)
        print(count_misdetected)
        print(count_true)
        print(count_images)
        print()

<<<<<<< HEAD
    avg_result = 0
    avg_det = 0
    avg_rej = 0
    if verbose:
        print("Class        Det. Rej. Avg.\n---------------------------")
=======
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
    for label in labels:
        if label in label_lines:
            true_detection_rate = count_detected[label] / count_true[label]
            true_rejection_rate = 1.0 - (count_misdetected[label] / (count_images - count_true[label]))
<<<<<<< HEAD
            avg_result += (true_detection_rate + true_rejection_rate) / 2.0
            avg_det += true_detection_rate
            avg_rej += true_rejection_rate
            if verbose:
                print('%-12s %4.2f %4.2f %4.2f' % (label, true_detection_rate, true_rejection_rate, (true_detection_rate + true_rejection_rate) / 2.0))
                
=======
            if verbose:
                print('%-12s %4.2f %4.2f %4.2f' % (label, true_detection_rate, true_rejection_rate, (true_detection_rate + true_rejection_rate) / 2.0))
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
            
            else:
                print('%.2f' % ((true_detection_rate + true_rejection_rate) / 2.0), end=' ')
        else:
            if verbose:
                print('%s 0 not trained' % (label))
            
            else:
                print(0, end = ' ')
<<<<<<< HEAD
    avg_result = avg_result / len(labels)
    avg_det = avg_det / len(labels)
    avg_rej = avg_rej / len(labels)
    if verbose:
        print('%-12s %4.2f %4.2f %4.2f' % ("Avg:", avg_det, avg_rej, avg_result))
=======

>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
    print()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="image_name", default=None, help="If provided, the classifier will run on this image file only")
    parser.add_argument("-g", dest="graph_file", default="meal_classifier", help="The inception graph file, default: 'meal_classifier'")
    parser.add_argument("-l", dest="label_file", default="meal_labels", help="The labels for the inception graph, default: 'meal_labels'")
    parser.add_argument("-t", dest="test_dir", default="imaged_tryout", help="The folder containing the test images and 'label.txt', default: 'imaged_tryout'")
    parser.add_argument("-v", dest="verbose", type=bool, default=False, help="'True' or 'False', enable verbose output")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    graph_file = args.graph_file
    test_labels_filename = "label.txt"
    current_dir = os.getcwd()
    test_dir = os.path.join(current_dir, args.test_dir)
    label_file = args.label_file
    verbose = args.verbose

    if args.image_name != None:
        # Unpersists graph from file
        with tf.gfile.FastGFile(os.path.join(current_dir, graph_file), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
<<<<<<< HEAD
            meal_tensor = sess.graph.get_tensor_by_name('meal_tensor:0')
            label_lines = [line.rstrip() for line in tf.gfile.GFile(os.path.join(current_dir, label_file))]
            final_predictions = get_final_predictions(sess, label_lines, meal_tensor, os.path.join(current_dir, args.image_name))

            print("\nPREDICTIONS: ")
            for node_id in final_predictions:
                print(label_lines[node_id], end=' ')
            print()

=======
            softmax_tensor = sess.graph.get_tensor_by_name('meal_tensor:0')
            label_lines = [line.rstrip() for line in tf.gfile.GFile(os.path.join(current_dir, label_file))]
            final_predictions = get_final_predictions(sess, label_lines, softmax_tensor, os.path.join(current_dir, args.image_name))

            print("output: ")
            for node_id in final_predictions:
                print(label_lines[node_id])
>>>>>>> 05f9bdfe509f9ac4708b84947fcc205f98bc576a
    else:
        test_folder(test_dir, test_labels_filename, graph_file, label_file)