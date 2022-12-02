import time

import requests
from flask import (
    flash,
    g,
    Markup,
    redirect,
    Response,
)
import hashlib
import json
import os
from flask import send_file
from myapp.utils.py.py_k8s import K8s
import datetime
from flask import jsonify
from myapp import conf
from myapp.views.base import BaseMyappView
import pysnooper
from flask_appbuilder import expose
from myapp import appbuilder
from flask import stream_with_context, request
from urllib.parse import urlencode

ACCESS_TOKEN = ''
ACCESS_TOKEN_TIME=datetime.datetime.now()
JSAPI_TICKET=''
JSAPI_TICKET_TIME=datetime.datetime.now()

APPID='wx9a246d09f8aecb12'
APPSECRET='e3393db21b07a06837fe2db619228105'

class Wechat(BaseMyappView):
    route_base='/wechat'
    default_view = 'token'   # 设置进入蓝图的默认访问视图（没有设置网址的情况下）

    @expose('/MP_verify_Z7Sr2bHKVvpIU2v7.txt')
    @pysnooper.snoop()
    def verify(self):
        file = open('MP_verify_Z7Sr2bHKVvpIU2v7.txt',mode='w')
        file.write('Z7Sr2bHKVvpIU2v7')
        file.close()
        return send_file(os.path.join(os.getcwd(),'MP_verify_Z7Sr2bHKVvpIU2v7.txt'))

    @expose('/token')
    @pysnooper.snoop()
    def token(self):

        data = request.json or {}
        data.update(request.args)
        signature = data.get('signature','')
        timestamp = data.get('timestamp','')
        nonce = data.get('nonce','')
        echostr = data.get('echostr','')
        token = "cubestudio"  # 请按照公众平台官网\基本配置中信息填写
        list = [token, timestamp, nonce]
        list.sort()
        sha1 = hashlib.sha1()
        sha1.update(list[0].encode('utf-8'))
        sha1.update(list[1].encode('utf-8'))
        sha1.update(list[2].encode('utf-8'))
        hashcode = sha1.hexdigest()
        print("handle/GET func: hashcode, signature: ", hashcode, signature)
        if hashcode == signature:
            return echostr
        else:
            return ""

    @pysnooper.snoop()
    def refresh_access_token(self):
        global ACCESS_TOKEN,ACCESS_TOKEN_TIME

        url=f'https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={APPID}&secret={APPSECRET}'
        if not ACCESS_TOKEN or (datetime.datetime.now()-ACCESS_TOKEN_TIME).total_seconds()>7000:
            res = requests.get(url)
            data = res.json() or {}
            access_token = data.get('access_token','')
            if access_token:
                ACCESS_TOKEN=access_token
                ACCESS_TOKEN_TIME=datetime.datetime.now()


    @expose('/jsapi',methods=['GET','POST'])
    @pysnooper.snoop()
    def jsapi(self):
        _args = request.json or {}
        _args.update(json.loads(request.args.get('form_data', "{}")))
        _args.update(request.args)
        url =_args.get('url','')
        nonceStr ='cubestudio'
        timestamp = str(int(time.time()))

        global JSAPI_TICKET,JSAPI_TICKET_TIME
        if not JSAPI_TICKET or (datetime.datetime.now()-JSAPI_TICKET_TIME).total_seconds()>7000:
            self.refresh_access_token()
            url=f"https://api.weixin.qq.com/cgi-bin/ticket/getticket?access_token={ACCESS_TOKEN}&type=jsapi"
            res = requests.get(url)
            data = res.json() or {}
            errcode = data.get('errcode',1)
            ticket = data.get('ticket','')
            if ticket:
                JSAPI_TICKET = ticket
                JSAPI_TICKET_TIME = datetime.datetime.now()

        # 按字母顺序排序
        data = {
            "jsapi_ticket": JSAPI_TICKET,
            "noncestr": nonceStr,
            "timestamp": timestamp,
            "url":url
        }
        str1 = "&".join(['%s=%s' % (key.lower(), data[key]) for key in sorted(data)])
        hashcode = hashlib.sha1(str1.encode('utf8')).hexdigest()

        return jsonify({
            "code": 0,
            "message": "success",
            "data": {
                "nonceStr": nonceStr,
                "timestamp": timestamp,
                "signature": hashcode,
                "appId": APPID,
                "jsapi_ticket":JSAPI_TICKET
            }
        })




# add_view_no_menu添加视图，但是没有菜单栏显示
appbuilder.add_view_no_menu(Wechat)





