import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from flags import parse_args
FLAGS, unparsed = parse_args()


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 90

'''
loading model data
'''
detection_graph = tf.Graph()

PATH_TO_CKPT = os.path.join(FLAGS.path_to_frozen_model, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(FLAGS.path_to_label_items, 'labels_items.pbtxt')

def loading_model_data():
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

'''
List of the strings that is used to add correct label for each box.
'''

def loading_label_index():

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index

'''
load image and convert image to numpy array
'''
def load_image_into_numpy_array(PATH_TO_IMAGE):
    image = Image.open(PATH_TO_IMAGE)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
    category_index = loading_label_index()
    loading_model_data()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            PATH_TO_IMAGE = os.path.join(FLAGS.path_to_storage_image, 'test.jpg')
            image_np = load_image_into_numpy_array(PATH_TO_IMAGE)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            # Visualization of the results of a detection.将识别结果标记在图片上
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            # output result输出
            for i in range(3):
                if classes[0][i] in category_index.keys():
                    class_name = category_index[classes[0][i]]['name']
                else:
                    class_name = 'N/A'
            
            plt.imsave(os.path.join(FLAGS.path_to_output_image, 'output.png'), image_np)
