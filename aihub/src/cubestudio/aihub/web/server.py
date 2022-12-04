import copy
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
from celery import Celery
from celery.result import AsyncResult
import pysnooper
from ..model import Field,Field_type,Validator
from ...util.py_github import get_repo_user
from ...util.log import AbstractEventLogger
from ...util.py_shell import exec

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

        self.save_time=datetime.datetime.now()

        if self.model.pic and 'http' not in self.model.pic:
            save_path = os.path.dirname(os.path.abspath(__file__)) + '/static/example/' + self.model.name + "/" + self.model.pic.strip('/')
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copyfile(self.model.pic, save_path)

    # 启动服务
    # @pysnooper.snoop()
    def server(self, port=8080, debug=True):

        app = Flask(__name__,
                    static_url_path=f'/{self.pre_url}/static/',
                    static_folder='static',
                    template_folder='templates')
        app.config['SECRET_KEY'] = os.urandom(24)
        app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)
        redis_url_default = 'redis://:admin@127.0.0.1:6379/0'
        CELERY_BROKER_URL = os.getenv('REDIS_URL', redis_url_default)
        CELERY_RESULT_BACKEND = os.getenv('REDIS_URL', redis_url_default)

        celery_app = Celery(self.model.name, broker=CELERY_BROKER_URL,backend=CELERY_RESULT_BACKEND)
        celery_app.conf.update(app.config)

        # 如果是同步服务，并且是一个celery worker，就多进程启动消费推理
        if os.getenv('REQ_TYPE', 'synchronous') == 'synchronous':
            # 如果同时消费其他异步任务，就启动celery
            if '127.0.0.1' not in CELERY_BROKER_URL:
                command = 'celery --app=cubestudio.aihub.web.celery_app:celery_app worker -Q %s --loglevel=info --pool=prefork -Ofair -c 10'%(self.model.name)
                print(command)
                exec(command)
            # 同步任务要加载模型
            self.model.load_model()
            # 如果有测试用例，就直接推理一遍测试用户，可以预加载模型
            if self.model.web_examples and len(self.model.web_examples)>0:
                input = self.model.web_examples[0].get("input",{})
                try:
                    if input:
                        self.model.inference(**input)
                except Exception as e:
                    print(e)

        # 文件转url。视频转码，音频转码等
        # @pysnooper.snoop()
        def file2url(file_path):
            # base_name = os.path.basename(file_path)
            if '.avi' in file_path:
                from cubestudio.util.py_video import Video2Mp4
                Video2Mp4(file_path,file_path.replace('.avi','.mp4'))
                file_path = file_path.replace('.avi','.mp4')

            save_path = os.path.dirname(os.path.abspath(__file__)) + '/static/example/'+self.model.name+"/" + file_path.strip('/')
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                shutil.copyfile(file_path, save_path)
            return f"/{self.pre_url}/static/example/"+self.model.name+"/" + file_path.strip('/')

        # resize图片
        # @pysnooper.snoop()
        def resize_img(image_path):
            image = cv2.imread(image_path)
            fheight = min(image.shape[0], 1920) / image.shape[0]
            fwidth = min(image.shape[1], 1920) / image.shape[1]
            if fwidth==fwidth==1:
                return
            if fheight < fwidth:
                # 缩放高度
                dst = cv2.resize(image, None, fx=fheight, fy=fheight, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(image_path, dst)
            else:
                dst = cv2.resize(image, None, fx=fwidth, fy=fwidth, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(image_path, dst)


        # 视频转流
        def video_stram(video_path):
            vid = cv2.VideoCapture(video_path)
            while True:
                return_value, frame = vid.read()
                image = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

        # 直接模型推理，并将结果转化为http链接
        @pysnooper.snoop()
        def model_inference(inference_kargs):
            # 处理返回值
            try:
                all_back = self.model.inference(**inference_kargs)
                if type(all_back) != list:
                    all_back = [all_back]
                for back in all_back:
                    # 如果是图片，写的不是http
                    if back.get('image', ''):
                        save_file_path = back['image']
                        if os.path.exists(save_file_path):
                            # resize_img(save_file_path)
                            back['image'] = file2url(save_file_path)

                    # 如果是视频，写的不是http
                    if back.get('video', ''):
                        save_file_path = back['video']
                        if os.path.exists(save_file_path):
                            back['video'] = file2url(save_file_path)

                    # 如果是语音，写的不是http
                    if back.get('audio', ''):
                        save_file_path = back['audio']
                        if os.path.exists(save_file_path):
                            back['audio'] = file2url(save_file_path)
            except Exception as e:
                all_back=[
                    {"text":str(e)}
                ]
            return all_back


        # 用来给本地celery请求使用
        @app.route(f'/{self.pre_url}/celery/inference',methods=['GET', 'POST'])
        @pysnooper.snoop()
        def celery_inference():
            data = request.json
            data.update(request.form.to_dict())
            return jsonify(model_inference(data))


        # 将请求data转化为推理函数参数
        def req2inference_args(data):
            inputs = copy.deepcopy(self.model.inference_inputs)
            inference_kargs = {}
            for input_field in inputs:
                inference_kargs[input_field.name] = data.get(input_field.name, input_field.default)

                # if input_field.type==Field_type.text and data.get(input_field.name,''):
                #     inference_kargs[input_field.name] = data.get(input_field.name, input_field.default)

                # 对上传图片进行处理，单上传和多上传
                if input_field.type == Field_type.image and data.get(input_field.name, ''):

                    # 对于图片base64编码
                    input_data = []
                    if type(data[input_field.name]) != list:
                        data[input_field.name] = [data[input_field.name]]

                    for img_base64_str in data[input_field.name]:
                        if 'http://' in img_base64_str or "https://" in img_base64_str:
                            req = requests.get(img_base64_str)
                            ext = img_base64_str[img_base64_str.rindex(".") + 1:]
                            ext = ext if len(ext) < 6 else 'jpg'
                            image_path = os.path.join(os.getenv('UPLOAD_DIR', 'upload'), self.model.name,
                                                      self.model.version,
                                                      datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "-" + str(
                                                          random.randint(0, 100)) + "." + ext)
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            with open(image_path, "wb") as f:
                                f.write(req.content)
                                f.close()
                            input_data.append(image_path)

                        else:
                            img_str = re.sub("^data:.*;base64,", '', img_base64_str)
                            ext = re.search("^data:(.*);base64,", img_base64_str).group(1)
                            ext = ext[ext.rindex("/") + 1:]

                            image_decode = base64.b64decode(img_str)

                            image_path = os.path.join(os.getenv('UPLOAD_DIR', 'upload'), self.model.name,
                                                      self.model.version,
                                                      datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "-" + str(
                                                          random.randint(0, 100)) + "." + ext)
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            nparr = np.fromstring(image_decode, np.uint8)
                            # 从nparr中读取数据，并把数据转换(解码)成图像格式
                            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            cv2.imwrite(image_path, img_np)

                            logging.info('Saving to %s.', image_path)
                            input_data.append(image_path)

                    # # 图片缩放
                    # for image_path in input_data:
                    #     resize_img(image_path)

                    if input_field.validators.max == 1:
                        inference_kargs[input_field.name] = input_data[0]
                    if input_field.validators.max > 1:
                        inference_kargs[input_field.name] = input_data

                # 单选或者多选图片
                if input_field.type == Field_type.image_select and data.get(input_field.name, ''):
                    # 将选中内容转为
                    input_data = []
                    if type(data[input_field.name]) != list:
                        data[input_field.name] = [data[input_field.name]]
                    for value in data[input_field.name]:
                        # 单个字符的不合法
                        if len(value) == 1:
                            continue
                        if 'http://' in value or "https://" in value:
                            input_data.append(value[value.rindex("/") + 1:])
                        else:
                            input_data.append(value)
                    input_data = list(set(input_data))
                    if input_field.validators.max == 1:
                        inference_kargs[input_field.name] = input_data[0]
                    if input_field.validators.max > 1:
                        inference_kargs[input_field.name] = input_data

                # 音视频文件上传
                if (input_field.type == Field_type.video or input_field.type == Field_type.audio) and data.get(
                        input_field.name, ''):
                    input_data = []
                    if type(data[input_field.name]) != list:
                        data[input_field.name] = [data[input_field.name]]

                    for file_base64_str in data[input_field.name]:
                        if 'http://' in file_base64_str or "https://" in file_base64_str:
                            req = requests.get(file_base64_str)
                            ext = file_base64_str[file_base64_str.rindex(".") + 1:]
                            ext = ext if len(
                                ext) < 6 else 'mp4' if input_field.type == Field_type.video else 'mp3' if input_field.type == Field_type.audio else 'unknow'
                            image_path = os.path.join(os.getenv('UPLOAD_DIR', 'upload'), self.model.name,
                                                      self.model.version,
                                                      datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "-" + str(
                                                          random.randint(0, 100)) + "." + ext)
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            with open(image_path, "wb") as f:
                                f.write(req.content)
                                f.close()
                            input_data.append(image_path)

                        else:
                            file_str = re.sub("^data:.*;base64,", '', file_base64_str)
                            ext = re.search("^data:(.*);base64,", file_base64_str).group(1)
                            ext = ext[ext.rindex("/") + 1:]
                            file_decode = base64.b64decode(file_str)

                            file_path = os.path.join(os.getenv('UPLOAD_DIR', 'upload'), self.model.name,
                                                     self.model.version,
                                                     datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "-" + str(
                                                         random.randint(0, 100)) + "." + ext)
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            f = open(file_path, mode='wb')
                            f.write(file_decode)
                            f.close()

                            logging.info('Saving to %s.', file_path)
                            input_data.append(file_path)

                    if input_field.validators.max == 1:
                        inference_kargs[input_field.name] = input_data[0]
                    if input_field.validators.max > 1:
                        inference_kargs[input_field.name] = input_data

                # 单选或者多选音视频
                if (
                        input_field.type == Field_type.audio_select or input_field.type == Field_type.video_select) and data.get(
                        input_field.name, ''):
                    # 将选中内容转为
                    input_data = []
                    if type(data[input_field.name]) != list:
                        data[input_field.name] = [data[input_field.name]]
                    for value in data[input_field.name]:
                        # 单个字符的不合法
                        if len(value) == 1:
                            continue
                        if 'http://' in value or "https://" in value:
                            input_data.append(value[value.rindex("/") + 1:])
                        else:
                            input_data.append(value)
                    input_data = list(set(input_data))
                    if input_field.validators.max == 1:
                        inference_kargs[input_field.name] = input_data[0]
                    if input_field.validators.max > 1:
                        inference_kargs[input_field.name] = input_data
            return inference_kargs

        # web请求后台
        @app.route(f'/{self.pre_url}/api/model/{self.model.name}/version/{self.model.version}/', methods=['GET', 'POST'])
        @pysnooper.snoop()
        def web_inference():
            # 从json里面读取信息
            data = request.json
            data.update(request.form.to_dict())
            begin_time = datetime.datetime.now()

            # 将输入进行转化为文件，或正确的格式
            try:
                # 异步推理，或者同步推理，有资源的机器启动同步，没有资源的启动异步
                all_back=[]
                status=0
                if os.getenv('REQ_TYPE', 'synchronous') == 'asynchronous':

                    # 直接写入和resize cos会非常耗时
                    # os.env['UPLOAD_DIR'] = f'/src/cubestudio/aihub/web/static/example/{self.pre_url}/upload'
                    inference_data = req2inference_args(data)
                    for arg in inference_data:
                        if inference_data[arg][:7]=='upload/':
                            if os.path.exists(inference_data[arg]):
                                # 要求宽带比较大才行
                                des_path = os.path.join(f'/src/cubestudio/aihub/web/static/example/{self.pre_url}/',inference_data[arg])
                                os.makedirs(os.path.dirname(des_path),exist_ok=True)
                                shutil.copyfile(inference_data[arg],des_path)
                                inference_data[arg] = des_path
                    kwargs = {
                        "data": inference_data
                    }
                    from .celery_app import inference
                    task = inference.apply_async(kwargs=kwargs, expires=120, retry=False)

                    # 原始数据大文件 放redis会比较耗时
                    # kwargs = {
                    #     "data": data,
                    #     "name": self.model.name,
                    #     "version": self.model.version
                    # }
                    # from .celery_app import reqdata_inference
                    # task = reqdata_inference.apply_async(kwargs=kwargs,expires=120,retry=False)

                    begin_time = datetime.datetime.now()
                    for i in range(100):
                        time.sleep(1)
                        async_task = AsyncResult(id=task.id, app=celery_app)
                        print("async_task.id", async_task.id, flush=True)
                        # 判断异步任务是否执行成功
                        if async_task.successful():
                            # 获取异步任务的返回值
                            result = async_task.get()
                            print(result)
                            if type(result) == list:
                                print("执行完成")
                            all_back = result
                            break
                        else:
                            print("任务还未执行完成", flush=True)
                    if not all_back:
                        status=1
                        all_back=[{"text": "耗时过久，未获取到推理结果"}]

                else:
                    inference_kargs = req2inference_args(data)
                    all_back = model_inference(inference_kargs)

                g.second = (datetime.datetime.now() - begin_time).total_seconds()

                # 如果返回的就是最终的响应
                if type(all_back)==dict and 'status' in all_back:
                    return jsonify(all_back)

                return jsonify({
                    "status": status,
                    "result": all_back,
                    "message": ""
                })

            except Exception as err:
                print(str(err))
                return jsonify({
                    "status":0,
                    "result":[],
                    "message":"推理失败了"
                })


        @app.route('/')
        @app.route(f'/{self.pre_url}')
        def home():
            return redirect(f'/aihub/{self.pre_url}')

        @app.route(f'/{self.pre_url}/navbar_right')
        def navbar_right():
            data = [
                {
                    "text": "开源社区",
                    "icon": '',
                    "link": 'https://github.com/tencentmusic/cube-studio'
                },
                {
                    "text": "AIHub",
                    "icon": '',
                    "link": 'http://www.data-master.net:8880/frontend/aihub/model_market/model_all'
                },
                {
                    "text": '',
                    "icon": '<svg t="1669560735821" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2687" width="64" height="64"><path d="M500 128.8c-95.2 5.6-173.6 83.2-180 178.4-7.2 112 80.8 205.6 191.2 205.6 106.4 0 192-86.4 192-192 0.8-110.4-92-198.4-203.2-192zM512 575.2c-128 0-383.2 64-383.2 192v96c0 17.6 14.4 32 32 32h702.4c17.6 0 32-14.4 32-32V766.4c0-127.2-255.2-191.2-383.2-191.2z" p-id="2688"></path></svg>',
                    "link": "http://www.data-master.net:8880/login"
                }
            ]
            # 返回模板
            return jsonify(data)

        # 监控度量
        @app.route(f'/{self.pre_url}/metrics')
        def metrics():
            return jsonify(user_history)

        # 健康检查
        @app.route(f'/{self.pre_url}/health')
        def health():
            return 'ok'

        @app.route(f'/{self.pre_url}/info')
        # @pysnooper.snoop()
        def info():
            inference_inputs = copy.deepcopy(self.model.inference_inputs)
            web_examples = copy.deepcopy(self.model.web_examples)
            # example中图片转为在线地址
            for example in web_examples:
                example_input = example.get('input',{})
                for arg_filed in inference_inputs:
                    if arg_filed.name in example_input:  # 这个示例提供了这个参数
                        # 示例图片/视频转为在线地址
                        if ("image" in arg_filed.type.name or 'video' in arg_filed.type.name or 'audio' in arg_filed.type.name) and 'http' not in example_input[arg_filed.name]:
                            example_input[arg_filed.name]=file2url(example_input[arg_filed.name])

            # 将图片和语音/视频的可选值和默认值，都转为在线网址
            for input in inference_inputs:
                if 'image' in input.type.name or 'video' in input.type.name or 'audio' in input.type.name:

                    # # 对于单选
                    # if '_select' in input.type.name and input.validators.max==1 and input.default and 'http' not in input.default:
                    #     input.default = file2url(input.default)
                    #
                    # # 对于多选
                    # if '_select' in input.type.name and input.validators.max>1 and input.default:
                    #     for i,default in enumerate(input.default):
                    #         if 'http' not in default:
                    #             input.default[i] = file2url(default)

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

            # web界面choice显示
            inference_inputs = [input.to_json() for input in inference_inputs]
            for input in inference_inputs:
                choices = input['values']
                for choice in choices:
                    if '/' in choice['id']:
                        choice['id']=choice['id'][choice['id'].rindex('/')+1:]

            rec_apps = [
                {
                    "pic": f"/{self.pre_url.strip('/')}/static/rec/rec1.jpg",
                    "label": "图片卡通化",
                    "url": "/aihub/gfpgan"
                },
                {
                    "pic": f"/{self.pre_url.strip('/')}/static/rec/rec2.jpg",
                    "label": "文本生成图片",
                    "url": "/aihub/stable-diffusion"
                },
                {
                    "pic": f"/{self.pre_url.strip('/')}/static/rec/rec3.jpg",
                    "label": "图片卡通化",
                    "url": "/aihub/gfpgan"
                },
            ]
            all_info_path = '/src/cubestudio/aihub/web/static/rec/all_info.json'
            all_info=json.load(open(all_info_path,mode='r')) if os.path.exists(all_info_path) else {}
            if all_info:
                all_info = dict(sorted(all_info.items(), key = lambda item: int(item[1].get('hot',1)),reverse=True))
                all_apps_name=list(all_info.keys())
                i=ii=0
                while i < len(all_apps_name):
                    if ii < len(rec_apps):
                        # 不推荐当前正在打开的应用，或者打开过的应用
                        if all_apps_name[i]==self.pre_url:
                            i=i+1
                            continue
                        info =all_info[all_apps_name[i]]
                        rec_apps[i]={
                            "pic": f"/%s/static/example/%s/%s"%(info['name'],info['name'],info['pic']),
                            "label": info['label'],
                            "url": "/aihub/"+info['name']
                        }
                        i=i+1
                        ii=ii+1
            print(rec_apps)
            info = {
                "name": self.model.name,
                "label": self.model.label,
                "describe": self.model.describe,
                "field": self.model.field,
                "scenes": self.model.scenes,
                "status": self.model.status,
                "version": self.model.version,
                "doc": self.model.doc,
                "pic": self.model.pic if 'http' in self.model.pic else file2url(self.model.pic),
                "web_examples":web_examples,
                "inference_inputs": inference_inputs,
                'inference_url':f'/{self.pre_url}/api/model/{self.model.name}/version/{self.model.version}/',
                "aihub_url":"http://www.data-master.net:8880/frontend/aihub/model_market/model_all",
                "github_url":"https://github.com/tencentmusic/cube-studio",
                "user":f"http://www.data-master.net:8880/login",
                "rec_apps":rec_apps
            }
            return jsonify(info)

        @app.before_request
        # @pysnooper.snoop()
        def check_login():
            req_url = request.path
            print(req_url)
            # 分享来自主主平台的cookie
            username = request.cookies.get('myapp_username','')
            # 获取来自上一次的记忆
            if not username:
                username = session.get('username','')
            # 创建新匿名用户
            if not username:
                username = "anonymous-" + uuid.uuid4().hex[:16]

            # res传递给浏览器记录
            session['username']=username

            # 只对后端接口
            if os.getenv('REQ_TYPE', 'synchronous') == 'asynchronous' and '/api/model/' in req_url:

                num = user_history.get(username, {}).get(req_url, {}).get('num',0)

                # 匿名用户对后端的请求次数超过1次就需要登录
                if num > 0 and 'anonymous-' in username:

                    return jsonify(
                        {
                            "message": "",
                            "result": [
                                {
                                    "text": "匿名用户仅可访问一次，扫码进群获取更多访问次数",
                                    "image":"https://luanpeng.oss-cn-qingdao.aliyuncs.com/csdn/ad1.jpg"
                                }
                            ],
                            "status": 0
                        }
                    )

                # if num > 10:
                #     return jsonify(
                #         {
                #             "message": "",
                #             "result": [
                #                 {
                #                     "text": "登录用户仅可访问10次，播放视频获得更多访问次数"
                #                 }
                #             ],
                #             "status": 0
                #         }
                #     )


        # 配置响应后操作：统计
        @app.after_request
        # @pysnooper.snoop()
        def apply_http_headers(response):

            req_url = request.path
            if os.getenv('REQ_TYPE', 'synchronous') == 'asynchronous' and '/api/model/' in req_url:  #  and type(response.get_json())==list
                username = session['username']
                if username not in user_history:
                    user_history[username] = {}
                if req_url not in user_history[username]:
                    user_history[username][req_url]={}
                # 记录热度
                user_history[username][req_url]["num"]=user_history.get(username, {}).get(req_url, {}).get('num',0) + 1
                # 记录总耗时
                if hasattr(g,'second'):
                    user_history[username][req_url]["second"]=user_history.get(username, {}).get(req_url, {}).get('second',10) + g.second

                # 统计信息汇总到all_info_path
                print(user_history)
                if (self.save_time-datetime.datetime.now()).total_seconds()>300:
                    all_info_path = '/src/cubestudio/aihub/web/static/rec/all_info.json'
                    os.makedirs(os.path.dirname(all_info_path), exist_ok=True)
                    all_info=json.load(open(all_info_path,mode='r')) if os.path.exists(all_info_path) else {}
                    if self.pre_url not in all_info:
                        all_info[self.pre_url]=json.load(open("info.json",mode='r'))
                    # 记录hot
                    for username in user_history:
                        for path in user_history[username]:
                            if '/api/model/' in path:
                                all_info[self.pre_url]["hot"] = int(all_info[self.pre_url].get("hot",0))+int(user_history[username][path].get('num',0))

                    json.dump(all_info,open(all_info_path,mode='w'))

            return response

        app.run(host='0.0.0.0', debug=debug, port=port)



