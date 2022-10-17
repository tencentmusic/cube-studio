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
from flask import Flask

from flask import Flask

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

Field_type = enum.Enum('Field_type', ('text', 'image', 'video', 'stream', 'text_select', 'image_select', 'text_multi', 'image_multi'))

class Field():
    def __init__(self,type:Field_type,name:str,label:str,describe='',choices=[],default='',validators=[]):
        self.type=type
        self.name=name
        self.label=label
        self.describe=describe
        self.choices=choices
        self.default=default
        self.validators=validators

    def to_json(self):
        return {
            "type":str(self.type),
            "name":self.name,
            "label":self.label,
            "describe":self.describe,
            "choices":self.choices,
            "default":self.default,
            "validators":self.validators
        }

class Server():

    web_inputs=[]
    web_outputs=[]
    web_examples=[]

    def __init__(self,model,docker=None):
        self.model=model
        self.docker=docker

    # 启动服务
    def server(self,docker=''):
        # 使用docker启动
        if docker:
            command='docker run --name photo --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/ai-photo bash init'




        @app.route(f'/api/model/{self.model.name[0]}/version/{self.model.version[0]}/', methods=['GET', 'POST'])
        # @pysnooper.snoop(watch_explode=())
        def web_inference(self=self):
            try:
                data = request.json
                inputs=self.web_inputs
                inference_kargs={}
                for input in inputs:
                    inference_kargs[input.name] = input.default
                    if input.type==Field_type.text and data.get(input.name,''):
                        inference_kargs[input.name] = data.get(input.name, input.default)

                    if input.type==Field_type.image and data.get(input.name,''):
                        image_decode = base64.b64decode(data[input.name])
                        image_path = os.path.join("upload",self.model.name[0],self.model.version[0], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".jpg")
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
                "name": self.model.name[0],
                "label": self.model.label[0],
                "describe": self.model.describe[0],
                "doc": self.model.doc[0],
                "pic":self.model.pic[0],
                "input":self.web_inputs,
                "output":self.web_outputs,
                "example":self.web_examples
            }
            print(data)
            return render_template('vision.html', data=data)

        @app.route('/info')
        # @pysnooper.snoop()
        def info(self=self):
            info = {
                "name": self.model.name[0],
                "label": self.model.label[0],
                "describe": self.model.description[0],
                "field": self.model.field[0],
                "scenes": self.model.scenes[0],
                "status": self.model.status[0],
                "version": self.model.version[0],
                "doc": self.model.doc[0],
                "pic": self.model.pic[0],
                "web_example":self.web_examples,
                "web_inputs": [web_input.to_json() for web_input in self.web_inputs]
            }
            return jsonify(info)


        app.run(host='0.0.0.0', debug=True, port='8080')

