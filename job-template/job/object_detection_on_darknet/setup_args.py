# -*- coding: utf-8 -*-
import os,sys
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

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("obj launcher")
#    # XGBClassifier XGBRegressor
    arg_parser.add_argument('--train_cfg', type=str, help="模型参数配置、训练配置", default='')
    arg_parser.add_argument('--data_cfg', type=str, help="训练数据配置", default='')
    arg_parser.add_argument('--weights', type=str, help="权重文件", default='')

    args = arg_parser.parse_args()
    logging.info("{} args: {}".format(__file__, args))

    train_cfg = args.train_cfg
    data_cfg = args.data_cfg

    with open('/app/darknet/cfg/train.cfg','w') as f_train_cfg:
        f_train_cfg.write(train_cfg)
    with open('/app/darknet/cfg/data.cfg','w') as f_data_cfg:
        f_data_cfg.write(data_cfg)






