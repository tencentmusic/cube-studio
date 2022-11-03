import enum
import shutil

import flask,os,sys,json,time,random,io,base64
from flask import redirect
from flask import render_template
import sys
import uuid
import os
import re
from flask import abort, current_app, flash, g, redirect, request, session, url_for
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
from ..model import Field,Field_type,Validator
from ...util.py_github import get_repo_user
from ...util.log import AbstractEventLogger

from flask import Flask

user_history={

}
class Server():

    web_examples=[]
    pre_url=''
    def __init__(self,model,docker=None):
        self.model=model
        self.docker=docker
        self.pre_url=self.model.name

    # 启动服务
    def server(self,port=8080):

        app = Flask(__name__,
                    static_url_path=f'/{self.pre_url}/static/',
                    static_folder='static',
                    template_folder='templates')
        app.config['SECRET_KEY'] = os.urandom(24)
        app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

        # 文件转url
        def file2url(file_path):
            base_name = os.path.basename(file_path)
            save_path = os.path.dirname(os.path.abspath(__file__)) + '/static/example/' + base_name
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                shutil.copy(file_path, save_path)
            return request.host_url.strip('/') + f"/{self.pre_url}/static/example/" + base_name

        # 视频转流
        def video_stram(self,video_path):
            vid = cv2.VideoCapture(video_path)
            while True:
                return_value, frame = vid.read()
                image = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

        self.model.load_model()

        @app.route(f'/{self.pre_url}/api/model/{self.model.name}/version/{self.model.version}/', methods=['GET', 'POST'])
        @pysnooper.snoop(watch_explode=('files'))
        def web_inference():
            try:
                # 从json里面读取信息
                data = request.json
                inputs=self.model.inference_inputs
                inference_kargs={}
                for input_field in inputs:
                    inference_kargs[input_field.name] = data.get(input_field.name,input_field.default)

                    # if input_field.type==Field_type.text and data.get(input_field.name,''):
                    #     inference_kargs[input_field.name] = data.get(input_field.name, input_field.default)

                    # 对上传图片进行处理，单上传和多上传
                    if input_field.type==Field_type.image and data.get(input_field.name,''):
                        # 对于图片base64编码
                        input_data=[]
                        if type(data[input_field.name])!=list:
                            data[input_field.name] = [data[input_field.name]]

                        for img_base64_str in data[input_field.name]:
                            img_str = re.sub("^data:.*;base64,",'',img_base64_str)
                            image_decode = base64.b64decode(img_str)

                            image_path = os.path.join("upload",self.model.name,self.model.version, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"-"+str(random.randint(0,100)) + ".jpg")
                            os.makedirs(os.path.dirname(image_path),exist_ok=True)
                            nparr = np.fromstring(image_decode, np.uint8)
                            # 从nparr中读取数据，并把数据转换(解码)成图像格式
                            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            cv2.imwrite(image_path, img_np)

                            logging.info('Saving to %s.', image_path)
                            input_data.append(image_path)

                        if input_field.validators.max==1:
                            inference_kargs[input_field.name] = input_data[0]
                        if input_field.validators.max>1:
                            inference_kargs[input_field.name] = input_data

                    if input_field.type == Field_type.image_select and data.get(input_field.name, ''):
                        # 将选中内容转为
                        input_data=[]
                        if type(data[input_field.name])!=list:
                            data[input_field.name] = [data[input_field.name]]
                        for value in data[input_field.name]:
                            # 单个字符的不合法
                            if len(value)==1:
                                continue
                            if 'http://' in value or "https://" in value:
                                input_data.append(value[value.rindex("/")+1:])
                            else:
                                input_data.append(value)
                        input_data=list(set(input_data))
                        if input_field.validators.max==1:
                            inference_kargs[input_field.name] = input_data[0]
                        if input_field.validators.max>1:
                            inference_kargs[input_field.name] = input_data


                # 从file里面读取文件
                # print(request.files)
                files = request.files
                for key in files:
                    print(files[key])
                files = request.form
                for key in files:
                    print(files[key])
                files = request.args
                for key in files:
                    print(files[key])
                files = request.values
                for key in files:
                    print(files[key])
                for input in inputs:
                    if input.name in request.files:
                        file = request.files.get(input.name)

                        file_name = file.filename
                        file_path = os.path.join("upload", self.model.name, self.model.version,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S-')+ file_name)
                        os.makedirs(os.path.dirname(file_path),exist_ok=True)
                        file.save(file_path)  # 保存文件
                        inference_kargs[input.name] = file_path

                print(inference_kargs)
                # 修正处理结果
                all_back = self.model.inference(**inference_kargs)
                if type(all_back)!=list:
                    all_back=[all_back]
                for back in all_back:
                    # 如果是图片，写的不是http
                    if back.get('image',''):
                        save_file_path = back['image']
                        if os.path.exists(save_file_path):
                            # f = open(back['image'], 'rb')
                            # image_data = f.read()
                            # base64_data = base64.b64encode(image_data)  # base64编码
                            # back['image'] = str(base64_data,encoding='utf-8')
                            back['image'] = file2url(save_file_path)

                    # 如果是视频，写的不是http
                    if back.get('video',''):
                        save_file_path = back['video']
                        if os.path.exists(save_file_path):
                            back['video']=file2url(save_file_path)

                    # 如果是语音，写的不是http
                    if back.get('audio',''):
                        save_file_path = back['audio']
                        if os.path.exists(save_file_path):
                            back['audio']=file2url(save_file_path)

                return jsonify({
                    "status": 0,
                    "result": all_back,
                    "message": ""
                })

            except Exception as err:
                logging.info('Uploaded image open error: %s', err)
                return jsonify(val='Cannot open uploaded image.')

        # @app.route('/')
        # @app.route(f'/{self.pre_url}')
        # def home():
        #     return redirect('/frontend/index.html#/frontend')
        #     data = {
        #         "name": self.model.name,
        #         "label": self.model.label,
        #         "describe": self.model.describe,
        #         "doc": self.model.doc,
        #         "pic":self.model.pic,
        #         "input":self.model.inference_inputs,
        #         "example":self.web_examples
        #     }
        #     print(data)
        #     return render_template('vision.html', data=data)

        # 监控度量
        @app.route(f'/{self.pre_url}/metrics')
        def metrics():
            return jsonify(user_history)

        # 健康检查
        @app.route(f'/{self.pre_url}/health')
        def health():
            return 'ok'

        @app.route(f'/{self.pre_url}/info')
        @pysnooper.snoop()
        def info():
            # example中图片转为在线地址
            for example in self.web_examples:
                for arg_filed in self.model.inference_inputs:
                    if arg_filed.name in example:  # 这个示例提供了这个参数
                        # 示例图片/视频转为在线地址
                        if ("image" in arg_filed.type.name or 'video' in arg_filed.type.name) and 'http' not in example[arg_filed.name]:
                            example[arg_filed.name]=file2url(example[arg_filed.name])

            # 将图片和语音/视频的可选值和默认值，都转为在线网址
            for input in self.model.inference_inputs:
                if 'image' in input.type.name or 'video' in input.type.name or 'audio' in input.type.name:
                    # 对于单选
                    if '_select' in input.type.name and input.validators.max==1 and input.default and 'http' not in input.default:
                        input.default = file2url(input.default)

                    # 对于多选
                    if '_select' in input.type.name and input.validators.max>1 and input.default:
                        for i,default in enumerate(input.default):
                            if 'http' not in default:
                                input.default[i] = file2url(default)
                    # 对于可选值，也转为url
                    if input.choices:
                        for i,choice in enumerate(input.choices):
                            if 'http' not in choice:
                                input.choices[i]=file2url(choice)

                # 对于输入类型做一些纠正
                if input.type.name=='int' and input.validators:
                    input.validators.regex = '[0-9]*'
                if input.type.name=='double' and input.validators:
                    input.validators.regex = '[0-9/.]*'

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
                "web_examples":self.web_examples,
                "inference_inputs": [input.to_json() for input in self.model.inference_inputs],
                'inference_url':f'/{self.pre_url}/api/model/{self.model.name}/version/{self.model.version}/',
                "aihub_url":"http://www.data-master.net/frontend/aihub/model_market/model_all",
                "github_url":"https://github.com/tencentmusic/cube-studio",
                "user":f"/{self.pre_url}/login",
                "rec_apps":[
                    {
                        "pic":"https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f",
                        "label":"图片修复"
                    },
                    {
                        "pic": "https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f",
                        "label": "图片修复"
                    },
                    {
                        "pic": "https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f",
                        "label": "图片修复"
                    }
                ]
            }
            return jsonify(info)

        # 此函数不在应用内，而在中心平台内，但是和应用使用同一个域名
        @app.route('/aihub/login/<app_name>')
        @pysnooper.snoop()
        def app_login(app_name=''):
            GITHUB_APPKEY = '69ee1c07fb4764b7fd34'
            GITHUB_SECRET = '795c023eb495317e86713fa5624ffcee3d00e585'
            GITHUB_AUTH_URL = 'https://github.com/login/oauth/authorize?client_id=%s'
            # 应用内登录才设置跳转地址
            if app_name and app_name!="demo":
                session['login_url'] = request.host_url.strip('/')+f"/{app_name}/info"
            oa_auth_url = GITHUB_AUTH_URL
            appkey = GITHUB_APPKEY
            username = session.get('username', '')
            g.username =''
            if 'anonymous' not in username and username:
                g.username=username

            if 'code' in request.args:
                # user check first login
                data = {
                    'code': request.args.get('code'),
                    'client_id': GITHUB_APPKEY,
                    'client_secret': GITHUB_SECRET
                }
                r = requests.post("https://github.com/login/oauth/access_token", data=data, timeout=2, headers={
                    'accept': 'application/json'
                })
                if r.status_code == 200:
                    json_data = r.json()
                    accessToken = json_data.get('access_token')
                    res = requests.get('https://api.github.com/user', headers={
                        'accept': 'application/json',
                        'Authorization': 'token ' + accessToken
                    })
                    print(res)
                    print(res.json())
                    user = res.json().get('login') or None  # name是中文名，login是英文名，不能if user
                    all_users = get_repo_user(7)
                    if user in all_users:
                        g.username = user
                    else:
                        return 'star cube-studio项目 <a href="https://github.com/tencentmusic/cube-studio">https://github.com/tencentmusic/cube-studio</a>  后重新登录，如果已经star请一分钟后重试'

                else:
                    message = str(r.content, 'utf-8')
                    print(message)
                    g.username = None

            # remember user
            if g.username and g.username != '':
                session['username'] = g.username
                login_url = session.get('login_url','https://github.com/tencentmusic/cube-studio')
                return redirect(login_url)
            else:
                return redirect(oa_auth_url % (str(appkey),))

        @app.before_request
        def check_login():
            req_url = request.path
            print(req_url)
            # 只对后端接口
            if '/frontend' not in req_url:
                username = session.get('username', "anonymous-" + uuid.uuid4().hex[:16])
                session['username']=username

                num = user_history.get(username, {}).get(req_url, 0)
                # 匿名用户对后端的请求次数超过1次就需要登录
                # if num > 1 and self.pre_url in req_url and 'anonymous-' in username:
                #     return jsonify({
                #         "status": 1,
                #         "result": {},
                #         "message": "匿名用户尽可访问一次，获得更多访问次数，需登录并激活用户"
                #     })
                #
                # if num > 10 and self.pre_url in req_url:
                #     return jsonify({
                #         "status": 2,
                #         "result": "https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%E5%BC%80%E6%BA%90%E4%B8%80%E7%AB%99%E5%BC%8F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B9%B3%E5%8F%B0.mp4",
                #         "message": "登录用户仅可访问10次，播放视频获得更多访问次数"
                #     })

        # 配置影响后操作
        @app.after_request
        def apply_http_headers(response):
            req_url = request.path
            if '/frontend' not in req_url:
                username = session['username']
                user_history[username] = {
                    req_url: user_history.get(username, {}).get(req_url, 0) + 1
                }
                print(user_history)
            return response

        app.run(host='0.0.0.0', debug=True, port=port)



