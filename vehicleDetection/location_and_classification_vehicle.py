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

import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw


FLAGS, unparsed = parse_args()

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 90

'''
loading model data
'''
detection_graph = tf.Graph()

PATH_TO_LOCATION_CKPT = os.path.join(FLAGS.path_to_frozen_model, 'frozen_inference_graph.pb')
PATH_TO_LOCATION_LABELS = os.path.join(FLAGS.path_to_label_items, 'labels_items.pbtxt')

def loading_model_data():
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_LOCATION_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

'''
List of the strings that is used to add correct label for each box.
'''

def loading_label_index():

    label_map = label_map_util.load_labelmap(PATH_TO_LOCATION_LABELS)
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

'''
CLASSIFICATION 
'''
class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)

    def load(self, label_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human

        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def create_graph(model_file=None):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    if not model_file:
        model_file = FLAGS.path_to_classification_model_file
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image, model_file=None):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = open(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph(model_file)

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('output:0')
        predictions = sess.run(softmax_tensor,
                               {'input:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup(FLAGS.path_to_classification_label)

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        top_names = []
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            top_names.append(human_string)
            score = predictions[node_id]
            print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
    return predictions, top_k, top_names
'''
load image and convert image to numpy array
'''

def draw_imageInfo_into_image(image,imageName,imageInfo,display_image_info):
    try:
        font = ImageFont.truetype('wqy-microhei.ttc', 8)
    except IOError:
        print ('加载字体失败')
        font = ImageFont.load_default()
    box = imageInfo[imageName]
    ymin, xmin, ymax, xmax = box
    (im_width, im_height) = image.size
    xmin = xmin * im_width
    ymin = ymin * im_height
    xmax = xmax * im_width
    ymax = ymax * im_height
    draw = ImageDraw.Draw(image)
    draw.line([(xmin, ymin), (xmin, ymax),(xmax, ymax),
               (xmax, ymin), (xmin, ymin)], width=5, fill='blue')

    draw.text((xmin, ymin),display_image_info,fill='black',font=font)
    plt.imsave(os.path.join(FLAGS.path_to_output_image, 'output1.png'), image)



if __name__ == '__main__':
    object_car_num = 0
    imageFileNameList = []
    imageInfo = {}
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
            for box, className in box_to_color_map.items():
                ymin, xmin, ymax, xmax = box

                object_num = object_num + 1
                if className == 'car':
                    imageFileName =  'car' + str(object_num) + '.png'
                    imageFileNameList.append(imageFileName)
                    imageInfo[imageFileName] = box
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
    print ('image info:',imageInfo)
    image = Image.open(PATH_TO_IMAGE)
    for i in range(len(imageFileNameList)):
        print("序号：%s   值：%s" % (i + 1, imageFileNameList[i]))
        imageFileName = os.path.join(FLAGS.path_to_output_image, imageFileNameList[i])
        predictions, top_k, top_names = run_inference_on_image(imageFileName)
        #print ('predictions',predictions)
        id = top_k[0]
        display_image_info = top_names[0] + ":" + str(predictions[id])
        draw_imageInfo_into_image(image,imageFileNameList[i],imageInfo,display_image_info)
        print ('id:',id)
        print ('display_image_info',display_image_info)
        print ('top_k',top_k)
        print ('top_names',top_names)
