
import sys
import os

import traceback
import argparse
import base64
import logging
import time,datetime
import json
import requests
from flask import redirect
import os
from os.path import splitext, basename
import time
import numpy as np
import datetime
import logging
import flask
import werkzeug
import optparse
import cv2
from flask import jsonify,request
from PIL import Image,ImageFont
from PIL import ImageDraw
import urllib
import torch
from PIL import Image

import pysnooper
from flask import Flask

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

UPLOAD_FOLDER = "UPLOAD_FOLDER"
# myfont = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 20)
# YOLO_DATA_PATH = os.getenv('YOLO_DATA_PATH','yolo/coco.data')
# YOLO_CFG_PATH = os.getenv('YOLO_CFG_PATH','yolo/yolov3.cfg')
# YOLO_WEIGHTS_PATH = os.getenv('YOLO_WEIGHTS_PATH','yolo/yolov3.weights')
#
# class ImageDetector(object):
#     def __init__(self):
#         self.detector = Detector(YOLO_DATA_PATH,YOLO_CFG_PATH,YOLO_WEIGHTS_PATH)
#
#     # @pysnooper.snoop()
#     def classify_image(self, image_path):
#         print("Classfy : ", image_path)
#         res = self.detector.detect(image_path)
#         print(res)
#
#         img = Image.open(image_path)
#         dr = ImageDraw.Draw(img)
#         for data in res:
#             class_name = data['class']
#             x, y, w, h = data['left'],data['top'],data['right'] - data['left'],data['bottom'] - data['top']
#             # 画矩形框
#             dr.rectangle((x, y, x + w, y + h), outline=(46, 254, 46),width=3)
#
#             # 写文字
#             # 设置字体和大小
#             # myfont = ImageFont.truetype("static/glyphicons-halflings-regular.ttf", 100)
#
#             dr.text((data['left'],data['top']),class_name, font=myfont, fill = 'red')
#         out_image_path = image_path[:image_path.rfind('.')] + '_deect' + image_path[image_path.rfind('.'):]
#         img.save(out_image_path)
#         return out_image_path
#
# image_detector = ImageDetector()


@app.route('/api/v1.0/model',methods=['GET','POST'])
# @pysnooper.snoop()
def classify_rest():
    try:
        data = request.json
        image_decode = base64.b64decode(data['image_data'])
        image_path = os.path.join(UPLOAD_FOLDER, str(datetime.datetime.now()) + ".jpg")
        nparr = np.fromstring(image_decode, np.uint8)
        # 从nparr中读取数据，并把数据转换(解码)成图像格式
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(image_path, img_np)

        logging.info('Saving to %s.', image_path)

        out_image_path = image_detector.classify_image(image_path)
        file = open(out_image_path, 'rb')
        base64_str = base64.b64encode(file.read()).decode('utf-8')
        os.remove(out_image_path)
        os.remove(image_path)
        return jsonify({
            "status": 0,
            "result": {
                data.get('image_id','image_id'):base64_str
            },
            "message":""
        })

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return jsonify(val = 'Cannot open uploaded image.')




@app.route('/')
def hello():
    return redirect('/static/index.html')

if __name__=='__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0',debug=True,port='8080')

