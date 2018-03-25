#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import pytz


tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)


def parse_args(check=True):

    parser = argparse.ArgumentParser()
    current_path = os.getcwd()
    parent_path = os.path.abspath('..')#获得当前工作目录的父目录

    parser.add_argument('--path_to_frozen_model', type=str, default=current_path + '/vehicleDataBase',
                        help='path to save frozen_inference_graph.pb .')

    parser.add_argument('--path_to_storage_image', type=str, default=parent_path + '/dataBase/image',
                        help='path to save image.')

    parser.add_argument('--path_to_label_items', type=str, default=current_path + '/vehicleDataBase',
                        help='path to save mscoco_label_map.pbtxt.')

    parser.add_argument('--path_to_output_image', type=str, default=current_path + '/output',
                        help='path to save the output.')

    parser.add_argument('--path_to_classification_label', type=str, default=current_path + '/vehicleDataBase/labels.txt',
                        help='path to classification label.')

    parser.add_argument('--path_to_classification_model_file', type=str, default=current_path + '/vehicleDataBase/freeze_pj_vehicle.pb',
                        help='path to classification model file.')

    parser.add_argument('--num_top_predictions',type=int,default=5,help='Display this many predictions.')


    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    for x in dir(FLAGS):
        print(getattr(FLAGS, x))