
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from flask import Flask, jsonify
import pysnooper
from apispec import yaml_utils
from flask import Blueprint, current_app, jsonify, make_response, request
from flask_babel import lazy_gettext as _
from myapp import conf, db, get_feature_flags, security_manager,event_logger
import yaml
from myapp.views.base import BaseMyappView

from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from myapp import app, appbuilder

class Myapp(BaseMyappView):
    route_base='/myapp'
    default_view = 'welcome'   # 设置进入蓝图的默认访问视图（没有设置网址的情况下）

    @expose('/welcome')
    @expose('/profile/<username>/')
    def welcome(self,username=None):
        if not g.user or not g.user.get_id():
            return redirect(appbuilder.get_url_for_login)
        if username:
            msg = 'Hello ' + username + " !"
        else:
            msg = 'Hello '+g.user.username+" !"

        # 返回模板
        return self.render_template('hello.html', msg=msg)

    @expose('/home')
    def home(self):
        from myapp.project import HOME_CONFIG
        data = HOME_CONFIG
        # 返回模板
        return self.render_template('home.html', data=data)


    @expose("/web/log/<cluster_name>/<namespace>/<pod_name>", methods=["GET",])
    def web_log(self,cluster_name,namespace,pod_name):
        from myapp.utils.py.py_k8s import K8s
        all_clusters = conf.get('CLUSTERS',{})
        if cluster_name in all_clusters:
            kubeconfig = all_clusters[cluster_name]['KUBECONFIG']
            pod_url = all_clusters[cluster_name].get('K8S_DASHBOARD_CLUSTER') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, pod_name)
        else:
            kubeconfig = None
            pod_url = conf.get('K8S_DASHBOARD_CLUSTER') + "#/log/%s/%s/pod?namespace=%s&container=%s" % (namespace, pod_name, namespace, pod_name)

        k8s = K8s(kubeconfig)
        pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        if pod:
            pod = pod[0]
            flash('当前pod状态：%s'%pod['status'],category='warning')
        data = {
            "url": pod_url,
            "target": 'div.kd-logs-container',
            "delay": 1000,
            "loading":True,
            "currentHeight": 128
        }
        # 返回模板
        if cluster_name == 'local':
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    @expose('/feature/check')
    def featureCheck(self):
        url = request.values.get("url", type=str, default=None)
        if '/myapp/home' in url:
            pass
            username=g.user.username

            # 数据格式说明 dict:
            # 'delay': Integer 延时隐藏 单位: 毫秒 0为不隐藏
            # 'hit': Boolean 是否命中
            # 'target': String 当前目标
            # 'type': String 类型 目前仅支持html类型
            # 'title': String 标题
            # 'content': String 内容html内容
            # /static/appbuilder/mnt/make_pipeline.mp4
            # data = {
            #     'content': '<video poster="/static/assets/images/ad/video-cover2.png" width="100%" height="auto" controls >\
            #                     <source src="https://xx.xx.xx/make_job_template.mp4" type="video/mp4">\
            #                 </video>',
            #     'delay': 5000,
            #     'hit': True,
            #     'target': url,
            #     'title': '开发定制一个任务模板',
            #     'type': 'html',
            # }
            # # 返回模板
            # return jsonify(data)
        return jsonify({})


# add_view_no_menu添加视图，但是没有菜单栏显示
appbuilder.add_view_no_menu(Myapp)

