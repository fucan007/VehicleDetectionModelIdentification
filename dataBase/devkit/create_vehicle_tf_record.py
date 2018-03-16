import hashlib
import io
import logging
import os
import random
import re
import argparse
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
import scipy.io

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='/home/xiaohui/AI/VehicleDetectionModelIdentification/dataBase/cars_train',
                        help='Root directory to import ospet dataset.')

    parser.add_argument('--train_filename_list', type=str, default='/home/xiaohui/AI/VehicleDetectionModelIdentification/dataBase/devkit/train.txt',
                        help='Root directory to import ospet dataset.')

    parser.add_argument('--annotations_file', type=str, default='./cars_train_annos.mat',
                        help='Root directory to import ospet dataset.')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Path to directory to output TFRecords.')
    parser.add_argument('--label_map_path', type=str, default='/home/xiaohui/AI/VehicleDetectionModelIdentification/dataBase/quiz-w8-data/labels_items.txt',
                        help='Path to label map proto.')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

def dict_to_tf_example(example , image_dir,annotations):
    filename = example + '.jpg'
    img_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
        logging.warning('This is warning message')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    image_width = image.size[0]
    image_height = image.size[1]
    print ('image_width',image_width)
    print ('image_height',image_height)
    image_class = []

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    xmin = annotations[0][0][0]
    ymin = annotations[1][0][0]
    xmax = annotations[2][0][0]
    ymax = annotations[3][0][0]
    print ('annotations[0][0][0]',annotations[0][0][0])
    print ('annotations[1][0][0]',annotations[1][0][0])
    print ('annotations[2][0][0]',annotations[2][0][0])
    print ('annotations[3][0][0]',annotations[3][0][0])
    image_class.append(int(annotations[4][0][0]))
    image_filename = annotations[5][0]
    print ('filename',image_filename)
    print ('image_class',image_class)
    print ('the type of image_class',type(image_class))
    xmins.append(xmin / image_width)
    ymins.append(ymin / image_height)
    xmaxs.append(xmax / image_width)
    ymaxs.append(ymax / image_height)
    print ('xmins:',xmins)
    print ('ymins:',ymins)
    print ('xmaxs:',xmaxs)
    print ('ymaxs:',ymaxs)

    feature_dict = {
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(
            image_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            image_filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/label': dataset_util.int64_list_feature(image_class),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return tf_example

def create_tf_record(output_filename,image_dir,examples_list,annotations):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.warning('On image %d of %d', idx, len(examples_list))
        example_index = example.lstrip('0')
        print ('example_index',example_index)
        example_index = int(example_index) - 1
        try:
            tf_example = dict_to_tf_example(example , image_dir,annotations.item(example_index))
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.info('Invalid example: %s, ignoring.', xml_path)

    writer.close()

def main(_):
    FLAGS, unparsed = parse_args()
    train_output_path = os.path.join(FLAGS.output_dir, 'vehicle_train.record')
    test_output_path = os.path.join(FLAGS.output_dir, 'vehicle_test.record')
    image_dir = FLAGS.train_data_dir
    train_filename = FLAGS.train_filename_list
    data = scipy.io.loadmat(FLAGS.annotations_file)
    annotations = data['annotations']
    examples_list = dataset_util.read_examples_list(train_filename)

    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]

    create_tf_record(train_output_path,image_dir,train_examples,annotations)
    create_tf_record(test_output_path,image_dir,val_examples,annotations)
if __name__ == '__main__':

    tf.app.run()
