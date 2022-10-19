import enum
import shutil

import flask,os,sys,json,time,random,io,base64
from flask import redirect
from flask import render_template
import sys
import os
from flask import abort, current_app, flash, g, redirect, request, session, url_for
from flask_babel import lazy_gettext
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
from ...util.py_github import get_repo_user

from flask import Flask

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')
user_history={

}
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
        @pysnooper.snoop()
        def info(self=self):
            for example in self.web_examples:
                for input in self.model.inference_inputs:
                    if input.name in example:
                        # 示例图片转为在线图片
                        if input.type.name=='image' and 'http' not in example[input.name]:
                            base_name = os.path.basename(example[input.name])
                            save_path = os.path.dirname(os.path.abspath(__file__))+'/static/example/'+base_name
                            if not os.path.exists(save_path):
                                shutil.copy(os.path.join(os.getcwd(),example[input.name]),save_path)
                            example[input.name]=request.host_url.strip('/')+"/static/example/"+base_name

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
        #
        # @app.route('/login')
        # def login(self=self):
        #     GITHUB_APPKEY = '24c051d2b3ec2def190b'  # ioa登录时申请的appkey
        #     GITHUB_SECRET = 'ae6beda4731b5dfc8dd923502d8b55ac8bc6c3b8'
        #     GITHUB_AUTH_URL = 'https://github.com/login/oauth/authorize?client_id=%s&redirect_uri=%s'
        # 
        #     request_data = request.args.to_dict()
        #     comed_url = request_data.get('login_url', '')
        #     login_url = '%s/login/' % request.host_url.strip('/')
        #     if comed_url:
        #         login_url += "?login_url=" + comed_url
        #     oa_auth_url = GITHUB_AUTH_URL
        #     appkey = GITHUB_APPKEY
        #     g.user = session.get('user', '')
        #     if 'code' in request.args:
        #         # user check first login
        #         data = {
        #             'code': request.args.get('code'),
        #             'client_id': GITHUB_APPKEY,
        #             'client_secret': GITHUB_SECRET
        #         }
        #         r = requests.post("https://github.com/login/oauth/access_token", data=data, timeout=2, headers={
        #             'accept': 'application/json'
        #         })
        #         if r.status_code == 200:
        #             json_data = r.json()
        #             accessToken = json_data.get('access_token')
        #             res = requests.get('https://api.github.com/user', headers={
        #                 'accept': 'application/json',
        #                 'Authorization': 'token ' + accessToken
        #             })
        #             print(res)
        #             print(res.json())
        #             user = res.json().get('login') or None  # name是中文名，login是英文名，不能if user
        #             all_users = get_repo_user(7)
        #             if user in all_users:
        #                 g.user = user
        #             else:
        #                 return 'star cube-studio项目 <a href="https://github.com/tencentmusic/cube-studio">https://github.com/tencentmusic/cube-studio</a>  后重新登录，如果已经star请一分钟后重试'
        #             if g.user: g.user = g.user.replace('.', '')
        # 
        #         else:
        #             message = str(r.content, 'utf-8')
        #             print(message)
        #             g.user = None
        # 
        #     # remember user
        #     if g.user and g.user != '':
        #         session['user'] = g.user
        #     else:
        #         return redirect(oa_auth_url % (str(appkey), login_url,))

        # @app.before_request
        # def check_login():
        #     if '/static' in request.path or '/logout' in request.path or '/login' in request.path or '/health' in request.path:
        #         return
        #     if not g.user:
        #         return redirect('/login')

        app.run(host='0.0.0.0', debug=True, port=port)

