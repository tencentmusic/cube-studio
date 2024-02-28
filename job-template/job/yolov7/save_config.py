# -*- coding: utf-8 -*-
import os,sys
import shutil

base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)
sys.path.append(os.path.realpath(__file__))

import logging
BASE_LOGGING_CONF = '[%(levelname)s] [%(asctime)s] %(message)s'
logging.basicConfig(level=logging.INFO,format=BASE_LOGGING_CONF)

import argparse
import datetime
import json
import time
import uuid
import re
import subprocess
import sys
yolo_classes = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("obj launcher")
#    # XGBClassifier XGBRegressor
    arg_parser.add_argument('--train', type=str, help="coco训练数据地址", default='')
    arg_parser.add_argument('--val', type=str, help="训练数据地址", default='')
    arg_parser.add_argument('--classes', type=str, help="分类类型，逗号分割", default='')
    arg_parser.add_argument('--weights', type=str, help="预训练权重文件地址", default='weights/yolov7_training.pt')

    args = arg_parser.parse_args()
    logging.info("{} args: {}".format(__file__, args))
    data_config = open('/yolov7/data.yaml').read()
    train_config = open('/yolov7/yolov7.yaml').read()
    classses = args.classes.split(',')
    classses = [x.strip() for x in classses if x.strip()]
    if not classses:
        classses=yolo_classes

    data_config = data_config.replace('TRAIN_DATATSE',args.train).replace('VAL_DATASET',args.val).replace('CLASSES_NUM',str(len(classses))).replace('CLASSES',str(classses))
    train_config = train_config.replace('CLASSES_NUM',str(len(classses)))
    with open('/yolov7/yolov7.yaml','w') as f_train_cfg:
        f_train_cfg.write(train_config)
    with open('/yolov7/data.yaml','w') as f_data_cfg:
        f_data_cfg.write(data_config)



