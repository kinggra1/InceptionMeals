import tensorflow as tf
import sys
import os

verbose = False

test_labels_filename = "label.txt"
current_dir = os.getcwd()

test_dir = os.path.join(current_dir, "imaged_tryout")
test_images_file = open( os.path.join(test_dir, test_labels_filename) )

test_images = [line.rstrip().split(' ') for line in test_images_file]

labels = [ 'salad', 'pasta', 'hotdog', 'frenchfry', 'burger', 'apple', 'banana', 'broccoli', 'pizza', 'egg', 'tomato', 'rice', 'strawberry', 'cookie' ]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(os.path.join(current_dir, "meal_labels"))]

count_detected = {}
count_misdetected = {}
count_true = {}
count_images = 0


for label in labels:
    count_detected[label] = 0
    count_misdetected[label] = 0
    count_true[label] = 0

def get_final_predictions(predictions):
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    if verbose:
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    final_pred = []
    if predictions[0][top_k[0]] > 0.75:
        final_pred = top_k[0:1]

    return final_pred


# Unpersists graph from file
with tf.gfile.FastGFile(os.path.join(current_dir, "meal_classifier"), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    for current_image in test_images:
        count_images += 1
        # Feed the image_data as input to the graph and get first prediction
        image_data = tf.gfile.FastGFile(os.path.join(test_dir, current_image[0]), 'rb').read()
        softmax_tensor = sess.graph.get_tensor_by_name('meal_tensor:0')
        
        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # print(predictions)
        
        final_predictions = get_final_predictions(predictions)

        for node_id in final_predictions:
            if label_lines[node_id] in current_image[1:]:
                count_detected[label_lines[node_id]] += 1
            else:
                count_misdetected[label_lines[node_id]] += 1

        for label in current_image[1:]:
            count_true[label] += 1
        if verbose:
            print(current_image[1:])
            print()
                
        
        
        # print(current_image)
        

        # print(label_lines[top_k[0]])
        # print(current_image[1])
        # print('status=%s' % (label_lines[top_k[0]] == current_image[1]))
        # print()

if verbose:
    print(count_detected)
    print(count_misdetected)
    print(count_true)
    print(count_images)
    print()

for label in labels:
    if label in label_lines:
        true_detection_rate = count_detected[label] / count_true[label]
        true_rejection_rate = 1.0 - (count_misdetected[label] / (count_images - count_true[label]))
        if verbose:
            print('%s %.2f %.2f %.2f' % (label, true_detection_rate, true_rejection_rate, (true_detection_rate + true_rejection_rate) / 2.0))
        
        else:
            print('%.2f' % ((true_detection_rate + true_rejection_rate) / 2.0), end=' ')
    else:
        if verbose:
            print('%s 0 not trained' % (label))
        
        else:
            print(0, end = ' ')

print()