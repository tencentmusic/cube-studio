import enum

import flask,os,sys,json,time,random,io,base64
from flask import redirect
from flask import render_template
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
from PIL import Image

import pysnooper
from ..model import Field,Field_type

from flask import Flask

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

class Server():

    web_examples=[]

    def __init__(self,model,docker=None):
        self.model=model
        self.docker=docker

    # 启动服务
    def server(self,port=8080):
        self.model.load_model()
        @app.route(f'/api/model/{self.model.name}/version/{self.model.version}/', methods=['GET', 'POST'])
        # @pysnooper.snoop(watch_explode=())
        def web_inference(self=self):
            try:
                data = request.json
                inputs=self.model.inference_inputs
                inference_kargs={}
                for input in inputs:
                    inference_kargs[input.name] = input.default
                    if input.type==Field_type.text and data.get(input.name,''):
                        inference_kargs[input.name] = data.get(input.name, input.default)

                    if input.type==Field_type.image and data.get(input.name,''):
                        image_decode = base64.b64decode(data[input.name])
                        image_path = os.path.join("upload",self.model.name,self.model.version, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".jpg")
                        if not os.path.exists(os.path.dirname(image_path)):
                            os.makedirs(os.path.dirname(image_path))
                        nparr = np.fromstring(image_decode, np.uint8)
                        # 从nparr中读取数据，并把数据转换(解码)成图像格式
                        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        cv2.imwrite(image_path, img_np)

                        logging.info('Saving to %s.', image_path)
                        inference_kargs[input.name] = image_path

                all_back = self.model.inference(**inference_kargs)
                if type(all_back)!=list:
                    all_back=[all_back]
                for back in all_back:
                    if back.get('image',''):
                        save_file_path = back['image']
                        if os.path.exists(save_file_path):
                            f = open(back['image'], 'rb')
                            image_data = f.read()
                            base64_data = base64.b64encode(image_data)  # base64编码
                            back['image'] = str(base64_data,encoding='utf-8')

                return jsonify({
                    "status": 0,
                    "result": all_back,
                    "message": ""
                })

            except Exception as err:
                logging.info('Uploaded image open error: %s', err)
                return jsonify(val='Cannot open uploaded image.')

        @app.route('/')
        def home(self=self):
            data = {
                "name": self.model.name,
                "label": self.model.label,
                "describe": self.model.describe,
                "doc": self.model.doc,
                "pic":self.model.pic,
                "input":self.model.inference_inputs,
                "example":self.web_examples
            }
            print(data)
            return render_template('vision.html', data=data)

        @app.route('/info')
        # @pysnooper.snoop()
        def info(self=self):
            info = {
                "name": self.model.name,
                "label": self.model.label,
                "describe": self.model.description,
                "field": self.model.field,
                "scenes": self.model.scenes,
                "status": self.model.status,
                "version": self.model.version,
                "doc": self.model.doc,
                "pic": self.model.pic,
                "web_example":self.web_examples,
                "inference_inputs": [input.to_json() for input in self.model.inference_inputs]
            }
            return jsonify(info)

        app.run(host='0.0.0.0', debug=True, port=port)

