import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import collections
import math

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
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def find_object_dection_box(boxes,
                            classes,
                            scores):
    min_score_thresh=.4
    box_to_color_map = collections.defaultdict(str)
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = 'none'
            else:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                box_to_color_map[box] = class_name
    return box_to_color_map

if __name__ == '__main__':
    category_index = loading_label_index()
    loading_model_data()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            PATH_TO_IMAGE = os.path.join(FLAGS.path_to_storage_image, 'test.jpg')
            image = Image.open(PATH_TO_IMAGE)
            (im_width, im_height) = image.size
            image_np = load_image_into_numpy_array(image)
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
            #print (boxes)
            #imageCut = image.crop(boxes)
            # Visualization of the results of a detection.将识别结果标记在图片上
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                min_score_thresh=.4,
                use_normalized_coordinates=True,
                line_thickness=8)

            box_to_color_map = find_object_dection_box(np.squeeze(boxes),
                                                       np.squeeze(classes).astype(np.int32),
                                                       np.squeeze(scores))
            object_num = 0
            object_car_num = 0
            for box, className in box_to_color_map.items():
                ymin, xmin, ymax, xmax = box

                object_num = object_num + 1
                if className == 'car':
                    imageFileName =  'car' + str(object_num) + '.png'
                    print ('imageFileName',imageFileName)
                    boxCut = (int(xmin*im_width),int(ymin*im_height),math.ceil(xmax*im_width),math.ceil(ymax*im_height))
                    print ('boxCut',boxCut)
                    object_car_num = object_car_num + 1
                    imageFile = image.crop(boxCut)
                    plt.imsave(os.path.join(FLAGS.path_to_output_image, imageFileName), imageFile)
            print ('发现的车辆数目为：',object_car_num)
            # output result输出
            # for i in range(3):
            #     if classes[0][i] in category_index.keys():
            #         class_name = category_index[classes[0][i]]['name']
            #     else:
            #         class_name = 'N/A'
            #     print("物体：%s 概率：%s" % (class_name, scores[0][i]))

            plt.imsave(os.path.join(FLAGS.path_to_output_image, 'output.png'), image_np)
