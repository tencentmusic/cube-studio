
from flask import (
    flash,
    g,
    Markup,
    redirect,
    Response,
)
from myapp.utils.py.py_k8s import K8s
import datetime
from flask import jsonify
from myapp import conf
from myapp.views.base import BaseMyappView
import pysnooper
from flask_appbuilder import expose
from myapp import appbuilder
from flask import stream_with_context, request

class Wechat(BaseMyappView):
    route_base='/wechat'
    default_view = 'token'   # 设置进入蓝图的默认访问视图（没有设置网址的情况下）

    @expose('/token')
    @pysnooper.snoop()
    def token(self):
        import hashlib
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


# add_view_no_menu添加视图，但是没有菜单栏显示
appbuilder.add_view_no_menu(Wechat)





