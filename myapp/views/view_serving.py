from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask import Blueprint, current_app, jsonify, make_response, request
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_serving import Service
from myapp.models.model_team import Project,Project_User
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import action
from myapp import app, appbuilder,db,event_logger
import logging
import re
import copy
import uuid
import requests
from myapp.exceptions import MyappException
from flask_appbuilder.security.decorators import has_access
from myapp.models.model_job import Repository
from flask_wtf.file import FileAllowed, FileField, FileRequired
from werkzeug.datastructures import FileStorage
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from myapp import security_manager
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from wtforms import BooleanField, IntegerField, SelectField, StringField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from myapp.utils.py import py_k8s
import os, zipfile
import shutil
from myapp.views.view_team import filter_join_org_project
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
from .base import (
    DeleteMixin,
    api,
    BaseMyappView,
    check_ownership,
    data_payload_response,
    DeleteMixin,
    generate_download_headers,
    get_error_msg,
    get_user_roles,
    handle_api_exception,
    json_error_response,
    json_success,
    MyappFilter,
    MyappModelView,

)
from sqlalchemy import and_, or_, select
from .baseApi import (
    MyappModelRestApi
)

from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config



class Service_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # public_project_id =
        # logging.info(join_projects_id)
        return query.filter(self.model.project_id.in_(join_projects_id))


class Service_ModelView_base():
    datamodel = SQLAInterface(Service)
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    show_columns = ['name', 'label','images','volume_mount','working_dir','command','env','resource_memory','resource_cpu','resource_gpu','replicas','ports','host_url','link']
    add_columns = ['project','name', 'label','images','working_dir','command','env','resource_memory','resource_cpu','resource_gpu','replicas','ports','host']
    list_columns = ['project','name_url','host_url','ip','creator','modified','deploy']
    edit_columns = ['project','name', 'label','images','working_dir','command','env','resource_memory','resource_cpu','resource_gpu','replicas','ports','volume_mount','host',]
    base_order = ('id','desc')
    order_columns = ['id']
    label_title = '云原生服务'
    base_filters = [["id", Service_Filter, lambda: []]]  # 设置权限过滤器


    add_form_extra_fields={
        "project": QuerySelectField(
            _(datamodel.obj.lab('project')),
            query_factory=filter_join_org_project,
            allow_blank=True,
            widget=Select2Widget()
        ),
        "name":StringField(_(datamodel.obj.lab('name')), description='英文名(字母、数字、- 组成)，最长50个字符',widget=BS3TextFieldWidget(), validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54)]),
        "label":StringField(_(datamodel.obj.lab('label')), description='中文名', widget=BS3TextFieldWidget(),validators=[DataRequired()]),
        "images": StringField(_(datamodel.obj.lab('images')), description='镜像全称', widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "volume_mount":StringField(_(datamodel.obj.lab('volume_mount')),description='外部挂载，格式:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,注意pvc会自动挂载对应目录下的个人rtx子目录',widget=BS3TextFieldWidget(),default=''),
        "working_dir": StringField(_(datamodel.obj.lab('working_dir')),description='工作目录，容器启动的初始所在目录，不填默认使用Dockerfile内定义的工作目录',widget=BS3TextFieldWidget()),
        "command":StringField(_(datamodel.obj.lab('command')), description='启动命令，支持多行命令',widget=MyBS3TextAreaFieldWidget(rows=3)),
        "node_selector":StringField(_(datamodel.obj.lab('node_selector')), description='运行当前服务所在的机器',widget=BS3TextFieldWidget(),default='cpu=true,serving=true'),
        "resource_memory":StringField(_(datamodel.obj.lab('resource_memory')),default=Service.resource_memory.default.arg,description='内存的资源使用限制，示例1G，10G， 最大100G，如需更多联系管路员',widget=BS3TextFieldWidget(),validators=[DataRequired()]),
        "resource_cpu":StringField(_(datamodel.obj.lab('resource_cpu')), default=Service.resource_cpu.default.arg,description='cpu的资源使用限制(单位核)，示例 0.4，10，最大50核，如需更多联系管路员',widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "replicas": StringField(_(datamodel.obj.lab('replicas')), default=Service.replicas.default.arg,description='pod副本数，用来配置高可用',widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "ports": StringField(_(datamodel.obj.lab('ports')), default=Service.ports.default.arg,description='进程端口号，逗号分隔',widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "env": StringField(_(datamodel.obj.lab('env')), default=Service.env.default.arg, description='使用模板的task自动添加的环境变量，支持模板变量。书写格式:每行一个环境变量env_key=env_value',widget=MyBS3TextAreaFieldWidget()),
        "host": StringField(_(datamodel.obj.lab('host')), default=Service.host.default.arg,
                           description='访问域名，http://xx.service.%s'%conf.get('ISTIO_INGRESS_DOMAIN',''),
                           widget=BS3TextFieldWidget()),
    }

    gpu_type = conf.get('GPU_TYPE')
    if gpu_type == 'TENCENT':
        add_form_extra_fields['resource_gpu'] = StringField(_(datamodel.obj.lab('resource_gpu')),
                                                                  default='0,0',
                                                                  description='gpu的资源使用限制(core,memory)，示例:10,2（10%的单卡核数和2*256M的显存），其中core为小于100的整数或100的整数倍，表示占用的单卡的百分比例，memory为整数，表示n*256M的显存',
                                                                  widget=BS3TextFieldWidget())
    if gpu_type == 'NVIDIA':
        add_form_extra_fields['resource_gpu'] = StringField(_(datamodel.obj.lab('resource_gpu')), default='0',
                                                                  description='gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡',
                                                                  widget=BS3TextFieldWidget())

    edit_form_extra_fields = add_form_extra_fields
    # edit_form_extra_fields['name']=StringField(_(datamodel.obj.lab('name')), description='英文名(字母、数字、- 组成)，最长50个字符',widget=MyBS3TextFieldWidget(readonly=True), validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54)]),


    def pre_add(self, item):
        if not item.volume_mount:
            item.volume_mount=item.project.volume_mount

    def delete_old_service(self,service_name,cluster):
        service_external_name = (service_name + "-external").lower()[:60].strip('-')
        from myapp.utils.py.py_k8s import K8s
        k8s = K8s(cluster.get('KUBECONFIG',''))
        namespace = conf.get('SERVICE_NAMESPACE')
        k8s.delete_deployment(namespace=namespace, name=service_name)
        k8s.delete_service(namespace=namespace, name=service_name)
        k8s.delete_service(namespace=namespace, name=service_external_name)
        k8s.delete_istio_ingress(namespace=namespace, name=service_name)


    def pre_update(self, item):
        # 修改了名称的话，要把之前的删掉
        if self.src_item_json.get('name','')!=item.name:
            self.delete_old_service(self.src_item_json.get('name',''),item.project.cluster)
            flash('检测到修改名称，旧服务已清理完成', category='warning')

    def pre_delete(self, item):
        self.delete_old_service(item.name,item.project.cluster)
        flash('服务已清理完成', category='warning')

    @expose('/clear/<service_id>', methods=['POST', "GET"])
    def clear(self, service_id):
        service = db.session.query(Service).filter_by(id=service_id).first()
        self.delete_old_service(service.name,service.project.cluster)
        flash('服务清理完成', category='warning')
        return redirect('/service_modelview/list/')

    @expose('/deploy/<service_id>',methods=['POST',"GET"])
    def deploy(self,service_id):
        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])

        service = db.session.query(Service).filter_by(id=service_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service.project.cluster.get('KUBECONFIG',''))
        namespace = conf.get('SERVICE_NAMESPACE')

        volume_mount = service.volume_mount

        k8s_client.create_deployment(namespace=namespace,
                              name=service.name,
                              replicas=service.replicas,
                              labels={"app":service.name,"username":service.created_by.username},
                              command=['bash','-c',service.command] if service.command else None,
                              args=None,
                              volume_mount=volume_mount,
                              working_dir=service.working_dir,
                              node_selector=service.get_node_selector(),
                              resource_memory=service.resource_memory,
                              resource_cpu=service.resource_cpu,
                              resource_gpu=service.resource_gpu if service.resource_gpu else '',
                              image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
                              image_pull_secrets=image_secrets,
                              image=service.images,
                              hostAliases=conf.get('HOSTALIASES',''),
                              env=service.env,
                              privileged=False,
                              accounts=None,
                              username=service.created_by.username,
                              ports=[int(port) for port in service.ports.split(',')]
                              )


        ports = [int(port) for port in service.ports.split(',')]

        k8s_client.create_service(
            namespace=namespace,
            name=service.name,
            username=service.created_by.username,
            ports=ports
        )
        # 如果域名配置的gateway，就用这个
        host = service.name+"."+conf.get('SERVICE_DOMAIN')
        if service.host:
            host=service.host.replace('http://','').replace('https://','').strip()
            if "/" in host:
                host = host[:host.index("/")]
            if ":" in host:
                host = host[:host.index(":")]
        k8s_client.create_istio_ingress(namespace=namespace,
                           name=service.name,
                           host = host,
                           ports=service.ports.split(',')
                           )

        # 以ip形式访问的话，使用的代理ip。不然不好处理机器服务化机器扩容和缩容时ip变化
        # 创建EXTERNAL_IP的服务
        SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', None)
        if not SERVICE_EXTERNAL_IP and service.project.expand:
            SERVICE_EXTERNAL_IP = json.loads(service.project.expand).get('SERVICE_EXTERNAL_IP', SERVICE_EXTERNAL_IP)
            if type(SERVICE_EXTERNAL_IP)==str:
                SERVICE_EXTERNAL_IP = [SERVICE_EXTERNAL_IP]

        if SERVICE_EXTERNAL_IP:
            service_ports = [[30000+10*service.id+index,port] for index,port in enumerate(ports)]
            service_external_name = (service.name + "-external").lower()[:60].strip('-')
            k8s_client.create_service(
                namespace=namespace,
                name=service_external_name,
                username=service.created_by.username,
                ports=service_ports,
                selector={"app": service.name, 'user': service.created_by.username},
                externalIPs=SERVICE_EXTERNAL_IP
            )


        # # 创建虚拟服务做代理
        # crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})
        # crd_name =  "service-%s"%service.name
        # crd_list = k8s.get_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, return_dict=None)
        # for vs_obj in crd_list:
        #     if vs_obj['name'] == crd_name:
        #         k8s.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
        #         time.sleep(1)
        # crd_json = {
        #     "apiVersion": "networking.istio.io/v1alpha3",
        #     "kind": "VirtualService",
        #     "metadata": {
        #         "name": crd_name,
        #         "namespace": namespace
        #     },
        #     "spec": {
        #         "gateways": [
        #             "kubeflow/kubeflow-gateway"
        #         ],
        #         "hosts": [
        #             "*"
        #         ],
        #         "http": [
        #             {
        #                 "match": [
        #                     {
        #                         "uri": {
        #                             "prefix": "/service/%s/"%service.name
        #                         }
        #                     }
        #                 ],
        #                 "rewrite": {
        #                     "uri": "/"
        #                 },
        #                 "route": [
        #                     {
        #                         "destination": {
        #                             "host": "%s.service.svc.cluster.local"%service.name,
        #                             "port": {
        #                                 "number": int(service.ports.split(',')[0])
        #                             }
        #                         }
        #                     }
        #                 ],
        #                 "timeout": "300s"
        #             }
        #         ]
        #     }
        # }
        #
        # # print(crd_json)
        # crd = k8s.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=crd_json)
        # # return crd



        flash('服务部署完成',category='warning')
        return redirect('/service_modelview/list/')



    @expose('/link/<service_id>')
    def link(self, service_id):
        service = db.session.query(Service).filter_by(id=service_id).first()
        url = "http://" + service.name + "." + conf.get('SERVICE_DOMAIN')
        if service.host:
            if 'http://' in service.host or 'https://' in service.host:
                url = service.host
            else:
                url = "http://"+service.host
        data={
            "url":url   # 'http://127.0.0.1:8080/video_sample/' #
        }

        # 返回模板
        return self.render_template('link.html', data=data)


class Service_ModelView(Service_ModelView_base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Service)
appbuilder.add_view(Service_ModelView,"内部服务",icon = 'fa-internet-explorer',category = '服务化')


class Service_ModelView_Api(Service_ModelView_base,MyappModelRestApi):
    datamodel = SQLAInterface(Service)
    route_base = '/service_modelview/api'

appbuilder.add_api(Service_ModelView_Api)