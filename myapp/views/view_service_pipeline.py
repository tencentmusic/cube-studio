from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __

from flask_babel import lazy_gettext as _
import uuid
import re
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_service_pipeline import Service_Pipeline
from myapp.models.model_job import Repository
from myapp.models.model_team import Project,Project_User
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.actions import action
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith,FilterEqual,FilterNotEqual
from wtforms.validators import EqualTo,Length
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db,event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from jinja2 import Template
from jinja2 import contextfilter
from jinja2 import Environment, BaseLoader, DebugUndefined, StrictUndefined
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from myapp.forms import JsonValidator
from myapp.views.view_task import Task_ModelView
from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from myapp.project import push_message,push_admin
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget,BS3TextAreaFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from myapp.utils.py import py_k8s
from flask_wtf.file import FileField
import shlex
import re,copy
from kubernetes.client.models import (
    V1Container, V1EnvVar, V1EnvFromSource, V1SecurityContext, V1Probe,
    V1ResourceRequirements, V1VolumeDevice, V1VolumeMount, V1ContainerPort,
    V1Lifecycle, V1Volume,V1SecurityContext
)
from .baseApi import (
    MyappModelRestApi
)
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
from myapp import security_manager
from myapp.views.view_team import filter_join_org_project
import kfp    # 使用自定义的就要把pip安装的删除了
from werkzeug.datastructures import FileStorage
from kubernetes import client as k8s_client
from .base import (
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
    json_response
)

from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config
logging = app.logger


class Service_Pipeline_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # public_project_id =
        # logging.info(join_projects_id)
        return query.filter(
            or_(
                self.model.project_id.in_(join_projects_id),
                # self.model.project.name.in_(['public'])
            )
        )







class Service_Pipeline_ModelView_Base():

    label_title='任务流'
    datamodel = SQLAInterface(Service_Pipeline)
    check_redirect_list_url = '/service_pipeline_modelview/list/'

    base_permissions = ['can_show','can_edit','can_list','can_delete','can_add']
    base_order = ("changed_on", "desc")
    # order_columns = ['id','changed_on']
    order_columns = ['id']

    list_columns = ['project','service_pipeline_url','creator','modified','operate_html']
    add_columns = ['project','name','describe','namespace','images','env','resource_memory', 'resource_cpu','resource_gpu', 'replicas','dag_json','alert_status','alert_user','parameter']
    show_columns = ['project','name','describe','namespace','run_id','created_by','changed_by','created_on','changed_on','expand_html','parameter_html']
    edit_columns = add_columns


    base_filters = [["id", Service_Pipeline_Filter, lambda: []]]  # 设置权限过滤器
    conv = GeneralModelConverter(datamodel)


    add_form_extra_fields = {

        "name": StringField(
            _(datamodel.obj.lab('name')),
            description="英文名(字母、数字、- 组成)，最长50个字符",
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54),DataRequired()]
        ),
        "project":QuerySelectField(
            _(datamodel.obj.lab('project')),
            query_factory=filter_join_org_project,
            allow_blank=True,
            widget=Select2Widget()
        ),
        "dag_json": StringField(
            _(datamodel.obj.lab('dag_json')),
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "namespace": StringField(
            _(datamodel.obj.lab('namespace')),
            description="部署task所在的命名空间(目前无需填写)",
            default='service',
            widget=BS3TextFieldWidget()
        ),
        "images":StringField(
                _(datamodel.obj.lab('images')),
                default='ccr.ccs.tencentyun.com/cube-studio/service-pipeline',
                description="推理服务镜像",
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            ),
        "node_selector": StringField(
            _(datamodel.obj.lab('node_selector')),
            description="部署task所在的机器(目前无需填写)",
            widget=BS3TextFieldWidget(),
            default=datamodel.obj.node_selector.default.arg
        ),
        "image_pull_policy": SelectField(
            _(datamodel.obj.lab('image_pull_policy')),
            description="镜像拉取策略(always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)",
            widget=Select2Widget(),
            choices=[['Always','Always'],['IfNotPresent','IfNotPresent']]
        ),

        "alert_status":MySelectMultipleField(
            label=_(datamodel.obj.lab('alert_status')),
            widget=Select2ManyWidget(),
            choices=[[x, x] for x in
                     ['Created', 'Pending', 'Running', 'Succeeded', 'Failed', 'Unknown', 'Waiting', 'Terminated']],
            description="选择通知状态"
        ),
        "alert_user": StringField(
            label=_(datamodel.obj.lab('alert_user')),
            widget=BS3TextFieldWidget(),
            description="选择通知用户，每个用户使用逗号分隔"
        ),


        "label": StringField(_(datamodel.obj.lab('label')), description='中文名', widget=BS3TextFieldWidget(),
                             validators=[DataRequired()]),
        "resource_memory": StringField(_(datamodel.obj.lab('resource_memory')),
                                       default=Service_Pipeline.resource_memory.default.arg,
                                       description='内存的资源使用限制，示例1G，10G， 最大100G，如需更多联系管路员', widget=BS3TextFieldWidget(),
                                       validators=[DataRequired()]),
        "resource_cpu": StringField(_(datamodel.obj.lab('resource_cpu')), default=Service_Pipeline.resource_cpu.default.arg,
                                    description='cpu的资源使用限制(单位核)，示例 0.4，10，最大50核，如需更多联系管路员',
                                    widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "resource_gpu": StringField(_(datamodel.obj.lab('resource_gpu')), default='0',
                                    description='gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡。申请具体的卡型号，可以类似 1(V100),目前支持T4/V100/A100/VGPU',
                                    widget=BS3TextFieldWidget()),
        "replicas": StringField(_(datamodel.obj.lab('replicas')), default=Service_Pipeline.replicas.default.arg,
                                description='pod副本数，用来配置高可用', widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "env": StringField(_(datamodel.obj.lab('env')), default=Service_Pipeline.env.default.arg,
                           description='使用模板的task自动添加的环境变量，支持模板变量。书写格式:每行一个环境变量env_key=env_value',
                           widget=MyBS3TextAreaFieldWidget()),


    }


    edit_form_extra_fields = add_form_extra_fields


    # 检测是否具有编辑权限，只有creator和admin可以编辑
    def check_edit_permission(self, item):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return True
        if g.user and g.user.username and hasattr(item,'created_by'):
            if g.user.username==item.created_by.username:
                return True
        flash('just creator can edit/delete ', 'warning')
        return False


    # 验证args参数
    @pysnooper.snoop(watch_explode=('item'))
    def service_pipeline_args_check(self, item):
        core.validate_str(item.name,'name')
        if not item.dag_json:
            item.dag_json='{}'
        core.validate_json(item.dag_json)
        # 校验task的关系，没有闭环，并且顺序要对。没有配置的，自动没有上游，独立
        # @pysnooper.snoop()
        def order_by_upstream(dag_json):
            order_dag={}
            tasks_name = list(dag_json.keys())  # 如果没有配全的话，可能只有局部的task
            i=0
            while tasks_name:
                i+=1
                if i>100:  # 不会有100个依赖关系
                    break
                for task_name in tasks_name:
                    # 没有上游的情况
                    if not dag_json[task_name]:
                        order_dag[task_name]=dag_json[task_name]
                        tasks_name.remove(task_name)
                        continue
                    # 没有上游的情况
                    elif 'upstream' not in dag_json[task_name] or not dag_json[task_name]['upstream']:
                        order_dag[task_name] = dag_json[task_name]
                        tasks_name.remove(task_name)
                        continue
                    # 如果有上游依赖的话，先看上游任务是否已经加到里面了。
                    upstream_all_ready=True
                    for upstream_task_name in dag_json[task_name]['upstream']:
                        if upstream_task_name not in order_dag:
                            upstream_all_ready=False
                    if upstream_all_ready:
                        order_dag[task_name]=dag_json[task_name]
                        tasks_name.remove(task_name)
                    else:
                        dag_json[task_name]['upstream']=[]
                        order_dag[task_name] = dag_json[task_name]
                        tasks_name.remove(task_name)

            if list(dag_json.keys()).sort()!=list(order_dag.keys()).sort():
                flash('dag service pipeline 存在循环或未知上游',category='warning')
                raise MyappException('dag service pipeline 存在循环或未知上游')
            return order_dag

        # 配置上缺少的默认上游
        dag_json = json.loads(item.dag_json)
        item.dag_json = json.dumps(order_by_upstream(copy.deepcopy(dag_json)),ensure_ascii=False,indent=4)

        # raise Exception('args is not valid')



    # @pysnooper.snoop()
    def pre_add(self, item):
        item.name = item.name.replace('_', '-')[0:54].lower().strip('-')
        # item.alert_status = ','.join(item.alert_status)
        # self.service_pipeline_args_check(item)
        item.create_datetime=datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()
        item.parameter = json.dumps({}, indent=4, ensure_ascii=False)
        item.volume_mount = item.project.volume_mount + ",%s(configmap):/config/" % item.name

    # @pysnooper.snoop()
    def pre_update(self, item):
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        else:
            item.expand='{}'
        item.name = item.name.replace('_', '-')[0:54].lower()
        item.alert_status = ','.join(item.alert_status)
        # self.service_pipeline_args_check(item)
        item.change_datetime = datetime.datetime.now()
        item.parameter = json.dumps(json.loads(item.parameter),indent=4,ensure_ascii=False) if item.parameter else '{}'
        item.dag_json = json.dumps(json.loads(item.dag_json), indent=4,ensure_ascii=False) if item.dag_json else '{}'
        item.volume_mount = item.project.volume_mount + ",%s(configmap):/config/" % item.name



    @expose("/my/list/")
    def my(self):
        try:
            user_id=g.user.id
            if user_id:
                service_pipelines = db.session.query(Service_Pipeline).filter_by(created_by_fk=user_id).all()
                back=[]
                for service_pipeline in service_pipelines:
                    back.append(service_pipeline.to_json())
                return json_response(message='success',status=0,result=back)
        except Exception as e:
            print(e)
            return json_response(message=str(e),status=-1,result={})




    def check_service_pipeline_perms(user_fun):
        # @pysnooper.snoop()
        def wraps(*args, **kwargs):
            service_pipeline_id = int(kwargs.get('service_pipeline_id','0'))
            if not service_pipeline_id:
                response = make_response("service_pipeline_id not exist")
                response.status_code = 404
                return response

            user_roles = [role.name.lower() for role in g.user.roles]
            if "admin" in user_roles:
                return user_fun(*args, **kwargs)

            join_projects_id = security_manager.get_join_projects_id(db.session)
            service_pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()
            if service_pipeline.project.id in join_projects_id:
                return user_fun(*args, **kwargs)

            response = make_response("no perms to run pipeline %s"%service_pipeline_id)
            response.status_code = 403
            return response

        return wraps


    # 构建同步服务
    def build_http(self,service_pipeline):
        pass


    # 构建异步服务
    @pysnooper.snoop()
    def build_mq_consumer(self,service_pipeline):
        namespace = conf.get('SERVICE_PIPELINE_NAMESPACE')
        name = service_pipeline.name
        command = service_pipeline.command
        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])


        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service_pipeline.project.cluster.get('KUBECONFIG',''))
        dag_json=service_pipeline.dag_json if service_pipeline.dag_json else '{}'

        # 生成服务使用的configmap

        config_data = {
            "dag.json":dag_json
        }
        k8s_client.create_configmap(namespace=namespace,name=name,data=config_data,labels={'app':name})
        env = service_pipeline.env
        if conf.get('SERVICE_PIPELINE_JAEGER',''):
            env['JAEGER_HOST']=conf.get('SERVICE_PIPELINE_JAEGER','')
            env['SERVICE_NAME'] = name

        labels = {"app": name, "user": service_pipeline.created_by.username,"pod-type":"service-pipeline"}
        k8s_client.create_deployment(
            namespace=namespace,
            name=name,
            replicas=service_pipeline.replicas,
            labels=labels,
            # command=['sh','-c',command] if command else None,
            command=['bash', '-c', "python mq-pipeline/cube_kafka.py"],
            args=None,
            volume_mount=service_pipeline.volume_mount,
            working_dir=service_pipeline.working_dir,
            node_selector=service_pipeline.get_node_selector(),
            resource_memory=service_pipeline.resource_memory,
            resource_cpu=service_pipeline.resource_cpu,
            resource_gpu=service_pipeline.resource_gpu if service_pipeline.resource_gpu else '',
            image_pull_policy=conf.get('IMAGE_PULL_POLICY','Always'),
            image_pull_secrets=image_secrets,
            image=service_pipeline.images,
            hostAliases=conf.get('HOSTALIASES',''),
            env=env,
            privileged=False,
            accounts=None,
            username=service_pipeline.created_by.username,
            ports=None
        )

        pass

    # 只能有一个入口。不能同时接口两个队列
    # # @event_logger.log_this
    @expose("/run_service_pipeline/<service_pipeline_id>", methods=["GET", "POST"])
    @check_service_pipeline_perms
    def run_service_pipeline(self,service_pipeline_id):
        service_pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()
        dag_json = json.loads(service_pipeline.dag_json)
        root_nodes_name = service_pipeline.get_root_node_name()
        self.clear(service_pipeline_id)
        if root_nodes_name:
            root_node_name=root_nodes_name[0]
            root_node=dag_json[root_node_name]
            # 构建异步
            if root_node['template-group']=='endpoint' and root_node['template']=='mq':
                self.build_mq_consumer(service_pipeline)

            # 构建同步
            if root_node['template-group'] == 'endpoint' and root_node['template'] == 'gateway':
                self.build_http(service_pipeline)


        return redirect("/service_pipeline_modelview/web/log/%s"%service_pipeline_id)
        # return redirect(run_url)


    # # @event_logger.log_this
    @expose("/web/<service_pipeline_id>", methods=["GET"])
    def web(self,service_pipeline_id):
        service_pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()

        # service_pipeline.dag_json = service_pipeline.fix_dag_json()
        # service_pipeline.expand = json.dumps(service_pipeline.fix_position(), indent=4, ensure_ascii=False)

        db.session.commit()
        print(service_pipeline_id)
        data = {
            "url": '/static/appbuilder/vison/index.html?pipeline_id=%s'%service_pipeline_id  # 前后端集成完毕，这里需要修改掉
        }
        # 返回模板
        return self.render_template('link.html', data=data)

    # # @event_logger.log_this
    @expose("/web/log/<service_pipeline_id>", methods=["GET"])
    def web_log(self,service_pipeline_id):
        service_pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()
        if service_pipeline.run_id:
            data = {
                "url": service_pipeline.project.cluster.get('PIPELINE_URL') + "runs/details/" + service_pipeline.run_id,
                "target": "div.page_f1flacxk:nth-of-type(0)",   # "div.page_f1flacxk:nth-of-type(0)",
                "delay":500,
                "loading": True
            }
            # 返回模板
            if service_pipeline.project.cluster['NAME']==conf.get('ENVIRONMENT'):
                return self.render_template('link.html', data=data)
            else:
                return self.render_template('external_link.html', data=data)
        else:
            flash('no running instance','warning')
            return redirect('/service_pipeline_modelview/web/%s'%service_pipeline.id)

    # 链路跟踪
    @expose("/web/monitoring/<service_pipeline_id>", methods=["GET"])
    def web_monitoring(self,service_pipeline_id):
        service_pipeline = db.session.query(Service_Pipeline).filter_by(id=int(service_pipeline_id)).first()
        if service_pipeline.run_id:
            url = service_pipeline.project.cluster.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_TASK_PATH')+ service_pipeline.name
            return redirect(url)
        else:
            flash('no running instance','warning')
            return redirect('/service_pipeline_modelview/web/%s'%service_pipeline.id)

    # # @event_logger.log_this
    @expose("/web/pod/<service_pipeline_id>", methods=["GET"])
    def web_pod(self,service_pipeline_id):
        service_pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()
        data = {
            "url": service_pipeline.project.cluster.get('K8S_DASHBOARD_CLUSTER', '') + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_PIPELINE_NAMESPACE'), service_pipeline.name.replace('_', '-')),
            "target":"div.kd-chrome-container.kd-bg-background",
            "delay":500,
            "loading": True
        }
        # 返回模板
        if service_pipeline.project.cluster['NAME']==conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    @expose('/clear/<service_id>', methods=['POST', "GET"])
    def clear(self, service_pipeline_id):
        service_pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()

        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service_pipeline.project.cluster.get('KUBECONFIG',''))
        namespace = conf.get('SERVICE_PIPELINE_NAMESPACE')
        k8s_client.delete_deployment(namespace=namespace, name=service_pipeline.name)

        flash('服务清理完成', category='warning')
        return redirect('/service_pipeline_modelview/list/')



    @expose("/config/<service_pipeline_id>",methods=("GET",'POST'))
    def pipeline_config(self,service_pipeline_id):
        print(service_pipeline_id)
        pipeline = db.session.query(Service_Pipeline).filter_by(id=service_pipeline_id).first()
        if not pipeline:
            return jsonify({
                "status":1,
                "message":"服务流不存在",
                "result":{}
            })
        if request.method.lower()=='post':
            data = request.get_json()
            request_config =data.get('config',{})
            request_dag = data.get('dag_json', {})
            if request_config:
                pipeline.config = json.dumps(request_config,indent=4,ensure_ascii=False)
            if request_dag:
                pipeline.dag_json = json.dumps(request_dag, indent=4, ensure_ascii=False)
            db.session.commit()
        config = {
            "id":pipeline.id,
            "name":pipeline.name,
            "label":pipeline.describe,
            "project":pipeline.project.describe,
            "pipeline_ui_config":{
                "alert":{
                    "alert_user":{
                        "type": "str",
                        "item_type": "str",
                        "label": "报警用户",
                        "require": 1,
                        "choice": [],
                        "range": "",
                        "default": "",
                        "placeholder": "报警用户名，逗号分隔",
                        "describe": "报警用户，逗号分隔",
                        "editable": 1,
                        "condition": "",
                        "sub_args": {}
                    }
                }
            },
            "pipeline_jump_button": [
                {
                    "name":"资源查看",
                    "action_url":"",
                    "icon_svg":'<svg t="1644980982636" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2611" width="128" height="128"><path d="M913.937279 113.328092c-32.94432-32.946366-76.898391-51.089585-123.763768-51.089585s-90.819448 18.143219-123.763768 51.089585L416.737356 362.999454c-32.946366 32.94432-51.089585 76.898391-51.089585 123.763768s18.143219 90.819448 51.087539 123.763768c25.406646 25.40767 57.58451 42.144866 93.053326 48.403406 1.76418 0.312108 3.51915 0.463558 5.249561 0.463558 14.288424 0 26.951839-10.244318 29.519314-24.802896 2.879584-16.322757-8.016581-31.889291-24.339338-34.768875-23.278169-4.106528-44.38386-15.081487-61.039191-31.736818-21.61018-21.61018-33.509185-50.489928-33.509185-81.322144s11.899004-59.711963 33.509185-81.322144l15.864316-15.864316c-0.267083 1.121544-0.478907 2.267647-0.6191 3.440355-1.955538 16.45988 9.800203 31.386848 26.260084 33.344432 25.863041 3.072989 49.213865 14.378475 67.527976 32.692586 21.608134 21.608134 33.509185 50.489928 33.509185 81.322144s-11.901051 59.71401-33.509185 81.322144L318.53987 871.368764c-21.61018 21.61018-50.489928 33.511231-81.322144 33.511231-30.832216 0-59.711963-11.901051-81.322144-33.511231-21.61018-21.61018-33.509185-50.489928-33.509185-81.322144s11.899004-59.711963 33.509185-81.322144l169.43597-169.438017c11.720949-11.718903 11.720949-30.722722 0-42.441625-11.718903-11.718903-30.722722-11.718903-42.441625 0L113.452935 666.282852c-32.946366 32.94432-51.089585 76.898391-51.089585 123.763768 0 46.865377 18.143219 90.819448 51.089585 123.763768 32.94432 32.946366 76.898391 51.091632 123.763768 51.091632s90.819448-18.145266 123.763768-51.091632l249.673409-249.671363c32.946366-32.94432 51.089585-76.898391 51.089585-123.763768-0.002047-46.865377-18.145266-90.819448-51.089585-123.763768-27.5341-27.536146-64.073294-45.240367-102.885252-49.854455-3.618411-0.428765-7.161097-0.196475-10.508331 0.601704l211.589023-211.589023c21.61018-21.61018 50.489928-33.509185 81.322144-33.509185s59.711963 11.899004 81.322144 33.509185c21.61018 21.61018 33.509185 50.489928 33.509185 81.322144s-11.899004 59.711963-33.509185 81.322144l-150.180418 150.182464c-11.720949 11.718903-11.720949 30.722722 0 42.441625 11.718903 11.718903 30.722722 11.718903 42.441625 0l150.180418-150.182464c32.946366-32.94432 51.089585-76.898391 51.089585-123.763768C965.026864 190.226482 946.882622 146.272411 913.937279 113.328092z" p-id="2612" fill="#225ed2"></path></svg>'
                },
                {
                    "name": "链路追踪",
                    "action_url": "http://swallow.music.woa.com/myapp/swallow#/?pathUrl=%2Fswallow%2FdispatchOps%2FTaskListManager",
                    "icon_svg": '<svg t="1644980982636" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2611" width="128" height="128"><path d="M913.937279 113.328092c-32.94432-32.946366-76.898391-51.089585-123.763768-51.089585s-90.819448 18.143219-123.763768 51.089585L416.737356 362.999454c-32.946366 32.94432-51.089585 76.898391-51.089585 123.763768s18.143219 90.819448 51.087539 123.763768c25.406646 25.40767 57.58451 42.144866 93.053326 48.403406 1.76418 0.312108 3.51915 0.463558 5.249561 0.463558 14.288424 0 26.951839-10.244318 29.519314-24.802896 2.879584-16.322757-8.016581-31.889291-24.339338-34.768875-23.278169-4.106528-44.38386-15.081487-61.039191-31.736818-21.61018-21.61018-33.509185-50.489928-33.509185-81.322144s11.899004-59.711963 33.509185-81.322144l15.864316-15.864316c-0.267083 1.121544-0.478907 2.267647-0.6191 3.440355-1.955538 16.45988 9.800203 31.386848 26.260084 33.344432 25.863041 3.072989 49.213865 14.378475 67.527976 32.692586 21.608134 21.608134 33.509185 50.489928 33.509185 81.322144s-11.901051 59.71401-33.509185 81.322144L318.53987 871.368764c-21.61018 21.61018-50.489928 33.511231-81.322144 33.511231-30.832216 0-59.711963-11.901051-81.322144-33.511231-21.61018-21.61018-33.509185-50.489928-33.509185-81.322144s11.899004-59.711963 33.509185-81.322144l169.43597-169.438017c11.720949-11.718903 11.720949-30.722722 0-42.441625-11.718903-11.718903-30.722722-11.718903-42.441625 0L113.452935 666.282852c-32.946366 32.94432-51.089585 76.898391-51.089585 123.763768 0 46.865377 18.143219 90.819448 51.089585 123.763768 32.94432 32.946366 76.898391 51.091632 123.763768 51.091632s90.819448-18.145266 123.763768-51.091632l249.673409-249.671363c32.946366-32.94432 51.089585-76.898391 51.089585-123.763768-0.002047-46.865377-18.145266-90.819448-51.089585-123.763768-27.5341-27.536146-64.073294-45.240367-102.885252-49.854455-3.618411-0.428765-7.161097-0.196475-10.508331 0.601704l211.589023-211.589023c21.61018-21.61018 50.489928-33.509185 81.322144-33.509185s59.711963 11.899004 81.322144 33.509185c21.61018 21.61018 33.509185 50.489928 33.509185 81.322144s-11.899004 59.711963-33.509185 81.322144l-150.180418 150.182464c-11.720949 11.718903-11.720949 30.722722 0 42.441625 11.718903 11.718903 30.722722 11.718903 42.441625 0l150.180418-150.182464c32.946366-32.94432 51.089585-76.898391 51.089585-123.763768C965.026864 190.226482 946.882622 146.272411 913.937279 113.328092z" p-id="2612" fill="#225ed2"></path></svg>'
                }
            ],
            "pipeline_run_button": [
            ],
            "task_jump_button": [],
            "dag_json":json.loads(pipeline.dag_json),
            "config": json.loads(pipeline.config),
            "message": "success",
            "status": 0
        }
        return jsonify(config)



    @expose("/template/list/")
    def template_list(self):

        all_template={
            "message": "success",
            "templte_common_ui_config":{
            },
            "template_group_order": ["入口", "逻辑节点", "功能节点"],
            "templte_list": {
                "入口":[
                    {
                        "template_name": "kafka",
                        "template_id": 1,
                        "templte_ui_config": {
                            "shell": {
                                "topic": {
                                    "type": "str",
                                    "item_type": "str",
                                    "label": "topic",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "predict",
                                    "placeholder": "",
                                    "describe": "kafka topic",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "consumer_num": {
                                    "type": "str",
                                    "item_type": "str",
                                    "label": "消费者数目",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "4",
                                    "placeholder": "",
                                    "describe": "消费者数目",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "bootstrap_servers": {
                                    "type": "str",
                                    "item_type": "str",
                                    "label": "地址",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "127.0.0.1:9092",
                                    "placeholder": "",
                                    "describe": "xx.xx.xx.xx:9092,xx.xx.xx.xx:9092",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "group": {
                                    "type": "str",
                                    "item_type": "str",
                                    "label": "分组",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "predict",
                                    "placeholder": "",
                                    "describe": "消费者分组",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                }
                            }
                        },
                        "username": g.user.username,
                        "changed_on": datetime.datetime.now(),
                        "created_on": datetime.datetime.now(),
                        "label": "kafka",
                        "describe": "消费kafka数据",
                        "help_url": "",
                        "pass_through": {
                            # 无论什么内容  通过task的字段透传回来
                        }
                    }
                ],
                "逻辑节点":[
                    {
                        "template_name": "switch",
                        "template_id": 2,
                        "templte_ui_config": {
                            "shell": {
                                "case": {
                                    "type": "text",
                                    "item_type": "str",
                                    "label": "表达式",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "int(input['node2'])<3:node4,node5:'3'\ndefault:node6:'0'",
                                    "placeholder": "",
                                    "describe": "条件:下游节点:输出     其中input为节点输入",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                }
                            }
                        },
                        "username": g.user.username,
                        "changed_on": datetime.datetime.now(),
                        "created_on": datetime.datetime.now(),
                        "label": "switch-case逻辑节点",
                        "describe": "控制数据的流量",
                        "help_url": "",
                        "pass_through": {
                            # 无论什么内容  通过task的字段透传回来
                        }

                    },
                ],
                "功能节点": [
                    {
                        "template_name": "http",
                        "template_id": 3,
                        "templte_ui_config": {
                            "shell": {
                                "method": {
                                    "type": "choice",
                                    "item_type": "str",
                                    "label": "请求方式",
                                    "require": 1,
                                    "choice": ["GET","POST"],
                                    "range": "",
                                    "default": "POST",
                                    "placeholder": "",
                                    "describe": "请求方式",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "url": {
                                    "type": "str",
                                    "item_type": "str",
                                    "label": "请求地址",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "http://127.0.0.1:8080/api",
                                    "placeholder": "",
                                    "describe": "请求地址",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "header": {
                                    "type": "text",
                                    "item_type": "str",
                                    "label": "请求头",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "{}",
                                    "placeholder": "",
                                    "describe": "请求头",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "timeout": {
                                    "type": "int",
                                    "item_type": "str",
                                    "label": "请求超时",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "300",
                                    "placeholder": "",
                                    "describe": "请求超时",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                                "date": {
                                    "type": "text",
                                    "item_type": "str",
                                    "label": "请求内容",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "{}",
                                    "placeholder": "",
                                    "describe": "请求内容",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                }
                            }
                        },
                        "username": g.user.username,
                        "changed_on": datetime.datetime.now(),
                        "created_on": datetime.datetime.now(),
                        "label": "http请求",
                        "describe": "http请求",
                        "help_url": "",
                        "pass_through": {
                            # 无论什么内容  通过task的字段透传回来
                        }
                    },
                    {
                        "template_name": "自定义方法",
                        "template_id": 4,
                        "templte_ui_config": {
                            "shell": {
                                "sdk_path": {
                                    "type": "str",
                                    "item_type": "str",
                                    "label": "函数文件地址",
                                    "require": 1,
                                    "choice": [],
                                    "range": "",
                                    "default": "",
                                    "placeholder": "",
                                    "describe": "函数文件地址，文件名和python类型要相同",
                                    "editable": 1,
                                    "condition": "",
                                    "sub_args": {}
                                },
                            }
                        },
                        "username": g.user.username,
                        "changed_on": datetime.datetime.now(),
                        "created_on": datetime.datetime.now(),
                        "label": "http请求",
                        "describe": "http请求",
                        "help_url": "",
                        "pass_through": {
                            # 无论什么内容  通过task的字段透传回来
                        }

                    }
                ],

            },
            "status": 0
        }
        index = 1
        for group in all_template['templte_list']:
            for template in all_template['templte_list'][group]:
                template['template_id'] = index
                template['changed_on'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                template['created_on'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                template['username'] = g.user.username,
                index += 1

        return jsonify(all_template)


class Service_Pipeline_ModelView(Service_Pipeline_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Service_Pipeline)
    # base_order = ("changed_on", "desc")
    # order_columns = ['changed_on']


appbuilder.add_view(Service_Pipeline_ModelView,"推理pipeline",href="/service_pipeline_modelview/list/",icon = 'fa-sitemap',category = '服务化')

# 添加api
class Service_Pipeline_ModelView_Api(Service_Pipeline_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Service_Pipeline)
    route_base = '/service_pipeline_modelview/api'
    show_columns = ['project','name','describe','namespace','node_selector','image_pull_policy','env','dag_json','run_id','created_by','changed_by','created_on','changed_on','expand']
    list_columns = show_columns
    add_columns = ['project','name','describe','namespace','node_selector','image_pull_policy','dag_json','env','expand']
    edit_columns = add_columns

appbuilder.add_api(Service_Pipeline_ModelView_Api)



