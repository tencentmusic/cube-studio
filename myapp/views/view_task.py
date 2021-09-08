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
from myapp.models.model_job import Repository,Images,Job_Template,Task,Pipeline,Workflow,Tfjob,Xgbjob,RunHistory,Pytorchjob
from myapp.models.model_team import Project,Project_User
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

from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
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
)
from myapp.views.base import CompactCRUDMixin
from flask_appbuilder import expose
import pysnooper,datetime,time,json
conf = app.config
logging = app.logger


class Task_ModelView_Base():
    label_title='任务'
    datamodel = SQLAInterface(Task)
    check_redirect_list_url = '/pipeline_modelview/edit/'
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    list_columns = ['name','label','job_template_url','volume_mount','debug','run','clear','log']
    show_columns = ['name', 'label','pipeline', 'job_template','volume_mount','command','overwrite_entrypoint','working_dir', 'args_html','resource_memory','resource_cpu','resource_gpu','timeout','retry','created_by','changed_by','created_on','changed_on','monitoring_html']
    add_columns = ['job_template', 'name', 'label', 'pipeline', 'volume_mount','command','working_dir']
    edit_columns = add_columns
    base_order = ('id','desc')
    order_columns = ['id']
    conv = GeneralModelConverter(datamodel)

    add_form_extra_fields = {
        "args": StringField(
            _(datamodel.obj.lab('args')),
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "pipeline": QuerySelectField(
            datamodel.obj.lab('pipeline'),
            query_factory=lambda: db.session.query(Pipeline),
            allow_blank=True,
            widget=Select2Widget(extra_classes="readonly"),
        ),
        "job_template": QuerySelectField(
            datamodel.obj.lab('job_template'),
            query_factory=lambda: db.session.query(Job_Template),
            allow_blank=True,
            widget=Select2Widget(),
        ),

        "name":StringField(
            label=_(datamodel.obj.lab('name')),
            description='英文名(字母、数字、- 组成)，最长50个字符',
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54),DataRequired()]
        ),
        "label":StringField(
            label=_(datamodel.obj.lab('label')),
            description='中文名',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "volume_mount":StringField(
            label = _(datamodel.obj.lab('volume_mount')),
            description='外部挂载，格式:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,注意pvc会自动挂载对应目录下的个人rtx子目录',
            widget=BS3TextFieldWidget(),
            default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives'
        ),
        "working_dir":  StringField(
            label = _(datamodel.obj.lab('working_dir')),
            description='工作目录，容器启动的初始所在目录，不填默认使用Dockerfile内定义的工作目录',
            widget=BS3TextFieldWidget()
        ),
        "command":StringField(
            label = _(datamodel.obj.lab('command')),
            description='启动命令',
            widget=MyBS3TextAreaFieldWidget(rows=3)
        ),
        "overwrite_entrypoint":BooleanField(
            label = _(datamodel.obj.lab('overwrite_entrypoint')),
            description='启动命令是否覆盖Dockerfile中ENTRYPOINT，不覆盖则叠加。'
        ),
        "node_selector": StringField(
            label = _(datamodel.obj.lab('node_selector')),
            description='运行当前task所在的机器', widget=BS3TextFieldWidget(),
            default=Task.node_selector.default.arg,
            validators=[DataRequired()]
        ),
        'resource_memory': StringField(
            label = _(datamodel.obj.lab('resource_memory')),
            default=Task.resource_memory.default.arg,
            description='内存的资源使用限制，示例1G，10G， 最大10G，如需更多联系管理员',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        'resource_cpu': StringField(
            label = _(datamodel.obj.lab('resource_cpu')),
            default=Task.resource_cpu.default.arg,
            description='cpu的资源使用限制(单位核)，示例 0.4，10，最大10核，如需更多联系管理员',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        'timeout': IntegerField(
            label = _(datamodel.obj.lab('timeout')),
            default=Task.timeout.default.arg,
            description='task运行时长限制，为0表示不限制(单位s)',
            widget=BS3TextFieldWidget()
        ),
        'retry': IntegerField(
            label = _(datamodel.obj.lab('retry')),
            default=Task.retry.default.arg, description='task重试次数',
            widget=BS3TextFieldWidget()
        ),
        'outputs': StringField(
            label = _(datamodel.obj.lab('outputs')),
            default=Task.outputs.default.arg,
            description='task输出文件，支持容器目录文件和minio存储路径',
            widget=MyBS3TextAreaFieldWidget(rows=3)
        ),

    }

    gpu_type = conf.get('GPU_TYPE')
    if gpu_type == 'TENCENT':
        add_form_extra_fields['resource_gpu'] = StringField(_(datamodel.obj.lab('resource_gpu')),
                                                                  default='0,0',
                                                                  description='gpu的资源使用限制(core,memory)，示例:10,2（10%的单卡核数和2*256M的显存），其中core为小于100的整数或100的整数倍，表示占用的单卡的百分比例，memory为整数，表示n*256M的显存',
                                                                  widget=BS3TextFieldWidget())
    if gpu_type == 'NVIDIA':
        add_form_extra_fields['resource_gpu'] = StringField(_(datamodel.obj.lab('resource_gpu')), default=0,
                                                                  description='gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡',
                                                                  widget=BS3TextFieldWidget())

    edit_form_extra_fields = add_form_extra_fields

    # 处理form请求
    # @pysnooper.snoop(watch_explode=('form'))
    def process_form(self, form, is_created):
        # from flask_appbuilder.forms import DynamicForm
        if 'job_describe' in form._fields:
            del form._fields['job_describe']  # 不处理这个字段


    # 验证args参数
    # @pysnooper.snoop(watch_explode=('item'))
    def task_args_check(self,item):
        core.validate_str(item.name, 'name')
        core.validate_json(item.args)
        task_args = json.loads(item.args)
        job_args = json.loads(item.job_template.args)
        item.args = json.dumps(core.validate_task_args(task_args, job_args), indent=4, ensure_ascii=False)

        if item.volume_mount and ":" not in item.volume_mount:
            raise MyappException('volume_mount is not valid, must contain : or null')

    # @pysnooper.snoop(watch_explode=('item'))
    def merge_args(self,item,action):

        logging.info(item)
        # 将字段合并为字典
        # @pysnooper.snoop()
        def nest_once(inp_dict):
            out = {}
            if isinstance(inp_dict, dict):
                for key, val in inp_dict.items():
                    if '.' in key:
                        keys = key.split('.')
                        sub_dict = out
                        for sub_key_index in range(len(keys)):
                            sub_key = keys[sub_key_index]
                            # 下面还有字典的情况
                            if sub_key_index!=len(keys)-1:
                                if sub_key not in sub_dict:
                                    sub_dict[sub_key]={}
                            else:
                                sub_dict[sub_key]=val
                            sub_dict=sub_dict[sub_key]

                    else:
                        out[key] = val
            return out

        args_json_column={}
        # 根据参数生成args字典。一层嵌套的形式
        for arg in item.__dict__:
            if arg[:5] == 'args.':
                task_attr_value = getattr(item,arg)
                # 如果是add
                # 用户没做任何修改，比如文件未做修改或者输入为空，那么后端采用不修改的方案
                if task_attr_value==None and action=='update':  # 必须不是None
                    # logging.info(item.args)
                    src_attr = arg[5:].split('.')  # 多级子属性
                    sub_src_attr=json.loads(item.args)
                    for sub_key in src_attr:
                        sub_src_attr=sub_src_attr[sub_key] if sub_key in sub_src_attr else ''
                    args_json_column[arg]=sub_src_attr
                elif task_attr_value==None and action=='add':  # 必须不是None
                    args_json_column[arg] = ''
                else:
                    args_json_column[arg] = task_attr_value

        # 如果是合并成的args
        if args_json_column:
            # 将一层嵌套的参数形式，改为多层嵌套的json形似
            des_merge_args = nest_once(args_json_column)
            item.args=json.dumps(des_merge_args.get('args',{}))
        # 如果是原始完成的args
        elif not item.args:
            item.args='{}'



    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self, item):

        item.name=item.name.replace('_','-')[0:54].lower()
        if item.job_template is None:
            raise MyappException("Job Template 为必选")

        if not item.volume_mount:
            item.volume_mount='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives'

        if item.job_template.volume_mount and item.job_template.volume_mount not in item.volume_mount:
            if item.volume_mount:
                item.volume_mount += ","+item.job_template.volume_mount
            else:
                item.volume_mount = item.job_template.volume_mount
        item.resource_memory = core.check_resource_memory(item.resource_memory)
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu)
        self.merge_args(item,'add')
        self.task_args_check(item)
        item.create_datetime=datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()

        if core.get_gpu(item.resource_gpu)[0]:
            item.node_selector = item.node_selector.replace('cpu=true','gpu=true')
        else:
            item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')


    # @pysnooper.snoop(watch_explode=('item'))
    def pre_update(self, item):
        item.name = item.name.replace('_', '-')[0:54].lower()
        if item.job_template is None:
            raise MyappException("Job Template 为必选")
        # if item.job_template.volume_mount and item.job_template.volume_mount not in item.volume_mount:
        #     if item.volume_mount:
        #         item.volume_mount += ","+item.job_template.volume_mount
        #     else:
        #         item.volume_mount = item.job_template.volume_mount
        if item.outputs:
            core.validate_json(item.outputs)
            item.outputs = json.dumps(json.loads(item.outputs),indent=4,ensure_ascii=False)
        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)


        item.resource_memory = core.check_resource_memory(item.resource_memory, self.src_item_json.get('resource_memory',None))
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu, self.src_item_json.get('resource_cpu',None))
        # item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_resource_memory)
        # item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_resource_cpu)

        self.merge_args(item,'update')
        self.task_args_check(item)
        item.change_datetime = datetime.datetime.now()


        if core.get_gpu(item.resource_gpu)[0]:
            item.node_selector = item.node_selector.replace('cpu=true','gpu=true')
        else:
            item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')


    # 添加和删除task以后要更新pipeline的信息,会报错is not bound to a Session
    # # @pysnooper.snoop()
    # def post_add(self, item):
    #     item.pipeline.pipeline_file = dag_to_pipeline(item.pipeline, db.session)
    #     pipeline_argo_id, version_id = upload_pipeline(item.pipeline)
    #     if pipeline_argo_id:
    #         item.pipeline.pipeline_argo_id = pipeline_argo_id
    #     if version_id:
    #         item.pipeline.version_id = version_id
    #     db.session.commit()


    # # @pysnooper.snoop(watch_explode=('item'))
    # def post_update(self, item):
    #     # if type(item)==UnmarshalResult:
    #     #     pass
    #     item.pipeline.pipeline_file = dag_to_pipeline(item.pipeline, db.session)
    #     pipeline_argo_id, version_id = upload_pipeline(item.pipeline)
    #     if pipeline_argo_id:
    #         item.pipeline.pipeline_argo_id = pipeline_argo_id
    #     if version_id:
    #         item.pipeline.version_id = version_id
    #     # db.session.update(item)
    #     db.session.commit()


    # 因为删除就找不到pipeline了
    def pre_delete(self, item):
        self.pipeline = item.pipeline


    widget_config = {
        "int": MyBS3TextFieldWidget,
        "float": MyBS3TextFieldWidget,
        "bool": None,
        "str": MyBS3TextFieldWidget,
        "text": MyBS3TextAreaFieldWidget,
        "json": MyBS3TextAreaFieldWidget,
        "date": DatePickerWidget,
        "datetime": DateTimePickerWidget,
        "password": BS3PasswordFieldWidget,
        "enum": Select2Widget,
        "multiple": Select2ManyWidget,
        "file": None,
        "dict": None,
        "list": None
    }


    field_config = {
        "int": IntegerField,
        "float": FloatField,
        "bool": BooleanField,
        "str": StringField,
        "text": StringField,
        "json": MyJSONField,  # MyJSONField   如果使用文本字段，传到后端的是又编过一次码的字符串
        "date": DateField,
        "datetime": DateTimeField,
        "password": StringField,
        "enum": SelectField,
        "multiple": SelectMultipleField,
        "file": FileField,
        "dict": None,
        "list": MyLineSeparatedListField
    }

    @event_logger.log_this
    @expose("/delete/<pk>")
    @has_access
    def delete(self, pk):
        pk = self._deserialize_pk_if_composite(pk)
        self.src_item_object = self.datamodel.get(pk, self._base_filters)
        if self.check_redirect_list_url:
            self.check_redirect_list_url = '/pipeline_modelview/edit/' + str(self.src_item_object.pipeline.id)
            try:
                self.check_edit_permission(self.src_item_object)
            except Exception as e:
                print(e)
                flash(str(e), 'warning')
                return redirect(self.check_redirect_list_url)

        self._delete(pk)
        return self.post_delete_redirect()


    def run_pod(self,task,k8s_client,run_id,namespace,pod_name,image,working_dir,command,args):

        # 模板中环境变量
        task_env = task.job_template.env + "\n"

        # 系统环境变量
        task_env += 'KFJ_TASK_ID=' + str(task.id) + "\n"
        task_env += 'KFJ_TASK_NAME=' + str(task.name) + "\n"
        task_env += 'KFJ_TASK_NODE_SELECTOR=' + str(task.get_node_selector()) + "\n"
        task_env += 'KFJ_TASK_VOLUME_MOUNT=' + str(task.volume_mount) + "\n"
        task_env += 'KFJ_TASK_IMAGES=' + str(task.job_template.images) + "\n"
        task_env += 'KFJ_TASK_RESOURCE_CPU=' + str(task.resource_cpu) + "\n"
        task_env += 'KFJ_TASK_RESOURCE_MEMORY=' + str(task.resource_memory) + "\n"
        task_env += 'KFJ_TASK_RESOURCE_GPU=' + str(task.resource_gpu.replace('+', '')) + "\n"
        task_env += 'KFJ_PIPELINE_ID=' + str(task.pipeline_id) + "\n"
        task_env += 'KFJ_RUN_ID=' + run_id + "\n"
        task_env += 'KFJ_CREATOR=' + str(task.pipeline.created_by.username) + "\n"
        task_env += 'KFJ_RUNNER=' + str(g.user.username) + "\n"
        task_env += 'KFJ_PIPELINE_NAME=' + str(task.pipeline.name) + "\n"
        task_env += 'KFJ_NAMESPACE=pipeline' + "\n"
        task_env += 'GPU_TYPE=%s' % os.environ.get("GPU_TYPE", "NVIDIA") + "\n"

        def template_str(src_str):
            rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
            des_str = rtemplate.render(creator=task.pipeline.created_by.username,
                                       datetime=datetime,
                                       runner=g.user.username if g and g.user and g.user.username else task.pipeline.created_by.username,
                                       uuid=uuid,
                                       pipeline_id=task.pipeline.id,
                                       pipeline_name=task.pipeline.name,
                                       cluster_name=task.pipeline.project.cluster['NAME']
                                       )
            return des_str
        # 全局环境变量
        pipeline_global_env = template_str(task.pipeline.global_env.strip()) if task.pipeline.global_env else ''  # 优先渲染，不然里面如果有date就可能存在不一致的问题
        pipeline_global_env = [env.strip() for env in pipeline_global_env.split('\n') if '=' in env.strip()]
        for env in pipeline_global_env:
            key, value = env[:env.index('=')], env[env.index('=') + 1:]
            if key not in task_env:
                task_env += key + '=' + value + "\n"

        platform_global_envs = json.loads(
            template_str(json.dumps(conf.get('GLOBAL_ENV', {}), indent=4, ensure_ascii=False)))
        for global_env_key in platform_global_envs:
            if global_env_key not in task_env:
                task_env += global_env_key + '=' + platform_global_envs[global_env_key] + "\n"

        volume_mount = task.volume_mount

        resource_cpu = task.job_template.get_env('TASK_RESOURCE_CPU') if task.job_template.get_env('TASK_RESOURCE_CPU') else task.resource_cpu
        resource_gpu = task.job_template.get_env('TASK_RESOURCE_GPU') if task.job_template.get_env('TASK_RESOURCE_GPU') else task.resource_gpu
        resource_memory = task.job_template.get_env('TASK_RESOURCE_MEMORY') if task.job_template.get_env('TASK_RESOURCE_MEMORY') else task.resource_memory
        hostAliases=conf.get('HOSTALIASES')
        if task.job_template.hostAliases:
            hostAliases+="\n"+task.job_template.hostAliases
        k8s_client.create_debug_pod(namespace,
                             name=pod_name,
                             labels={"pipeline": task.pipeline.name, 'task': task.name, 'run-rtx': g.user.username,'run-id': run_id},
                             command=command,
                             args=args,
                             volume_mount=volume_mount,
                             working_dir=working_dir,
                             node_selector=task.get_node_selector(), resource_memory=resource_memory,
                             resource_cpu=resource_cpu, resource_gpu=resource_gpu,
                             image_pull_policy=task.pipeline.image_pull_policy,
                             image_pull_secrets=[task.job_template.images.repository.hubsecret],
                             image=image,
                             hostAliases=hostAliases,
                             env=task_env, privileged=task.job_template.privileged,
                             accounts=task.job_template.accounts, username=task.pipeline.created_by.username)


    # @event_logger.log_this
    @expose("/debug/<task_id>", methods=["GET", "POST"])
    def debug(self,task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()
        if not g.user.is_admin() and task.job_template.created_by.username!=g.user.username:
            flash('仅管理员或当前任务模板创建者，可启动debug模式', 'warning')
            return redirect('/pipeline_modelview/web/%s' % str(task.pipeline.id))


        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster['KUBECONFIG'])
        namespace = conf.get('PIPELINE_NAMESPACE')
        pod_name="debug-"+task.pipeline.name.replace('_','-')+"-"+task.name.replace('_','-')
        pod_name=pod_name[:60]
        pod = k8s_client.get_pods(namespace=namespace,pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史非运行态，直接删除
        # if pod and (pod['status']!='Running' and pod['status']!='Pending'):
        if pod and pod['status'] == 'Succeeded':
            k8s_client.delete_pods(namespace=namespace,pod_name=pod_name)
            time.sleep(2)
            pod=None
        # 没有历史或者没有运行态，直接创建
        if not pod or pod['status']!='Running':
            run_id = "debug-" + str(uuid.uuid4().hex)
            command=['sh','-c','sleep 7200 && hour=`date +%H` && while [ $hour -ge 06 ];do sleep 3600;hour=`date +%H`;done']
            self.run_pod(
                task=task,
                k8s_client=k8s_client,
                run_id=run_id,
                namespace=namespace,
                pod_name=pod_name,
                image=json.loads(task.args)['images'] if task.job_template.name == conf.get('CUSTOMIZE_JOB') else task.job_template.images.name,
                working_dir='/mnt',
                command=command,
                args=None
            )

        try_num=5
        while(try_num>0):
            pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
            # print(pod)
            if pod:
                pod = pod[0]
            # 有历史非运行态，直接删除
            if pod and pod['status'] == 'Running':
                break
            try_num=try_num-1
            time.sleep(2)
        if try_num==0:
            flash('启动时间过长，一分钟后重试','warning')
            return redirect('/pipeline_modelview/web/%s'%str(task.pipeline.id))


        return redirect("/task_modelview/web/debug/%s/%s/%s"%(task.pipeline.project.cluster['NAME'],namespace,pod_name))


    @expose("/web/debug/<cluster_name>/<namespace>/<pod_name>", methods=["GET", "POST"])
    # @pysnooper.snoop()
    def web_debug(self,cluster_name,namespace,pod_name):
        cluster=conf.get('CLUSTERS',{})
        if cluster_name in cluster:
            pod_url = cluster[cluster_name].get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name,pod_name, namespace)
        else:
            pod_url = conf.get('K8S_DASHBOARD_CLUSTER') + '#/shell/%s/%s/%s?namespace=%s' % (namespace, pod_name, pod_name, namespace)
        print(pod_url)
        data = {
            "url": pod_url,
            "target":'div.kd-scroll-container', #  'div.kd-scroll-container.ng-star-inserted',
            "delay": 2000,
            "loading": True
        }
        # 返回模板
        if cluster_name==conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    @expose("/run/<task_id>", methods=["GET", "POST"])
    # @pysnooper.snoop(watch_explode=('ops_args',))
    def run_task(self,task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster['KUBECONFIG'])
        namespace = conf.get('PIPELINE_NAMESPACE')
        pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name[:60]
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            run_id = pod['labels'].get("run-id", '')
            if run_id:
                k8s_client.delete_workflow(all_crd_info=conf.get('CRD_INFO', {}), namespace=namespace, run_id=run_id)

            k8s_client.delete_pods(namespace=namespace, pod_name=pod_name)
            delete_time = datetime.datetime.now()
            while pod:
                time.sleep(2)
                pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
                check_date = datetime.datetime.now()
                if (check_date-delete_time).seconds>60:
                    flash("超时，请稍后重试",category='warning')
                    return redirect('/pipeline_modelview/web/%s' % str(task.pipeline.id))



        # 没有历史或者没有运行态，直接创建
        if not pod:
            command = None
            if task.job_template.entrypoint:
                command = task.job_template.entrypoint
            if task.command:
                command=task.command
            if command:
                command = command.split(" ")
                command = [com for com in command if com ]
            ops_args=[]

            task_args = json.loads(task.args) if task.args else {}

            for task_attr_name in task_args:
                # 添加参数名
                if type(task_args[task_attr_name]) == bool:
                    if task_args[task_attr_name]:
                        ops_args.append('%s' % str(task_attr_name))
                # 添加参数值
                elif type(task_args[task_attr_name]) == dict or type(task_args[task_attr_name]) == list:
                    ops_args.append('%s' % str(task_attr_name))
                    ops_args.append('%s' % json.dumps(task_args[task_attr_name], ensure_ascii=False))
                elif not task_args[task_attr_name]:  # 如果参数值为空，则都不添加
                    pass
                else:
                    ops_args.append('%s' % str(task_attr_name))
                    ops_args.append('%s' % str(task_args[task_attr_name]))  # 这里应该对不同类型的参数名称做不同的参数处理，比如bool型，只有参数，没有值



            # print(ops_args)
            run_id = "run-"+str(task.pipeline.id)+"-"+str(task.id)

            self.run_pod(
                task=task,
                k8s_client=k8s_client,
                run_id=run_id,
                namespace=namespace,
                pod_name=pod_name,
                image=json.loads(task.args)['images'] if task.job_template.name == conf.get('CUSTOMIZE_JOB') else task.job_template.images.name,
                working_dir=json.loads(task.args)['workdir'] if task.job_template.name == conf.get('CUSTOMIZE_JOB') else task.job_template.workdir,
                command=['bash','-c',json.loads(task.args)['command']] if task.job_template.name == conf.get('CUSTOMIZE_JOB') else command,
                args=None if task.job_template.name == conf.get('CUSTOMIZE_JOB') else ops_args)




        try_num = 5
        while (try_num > 0):
            pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
            # print(pod)
            if pod:
                break
            try_num = try_num - 1
            time.sleep(2)
        if try_num == 0:
            flash('启动时间过长，一分钟后重试', 'warning')
            return redirect('/pipeline_modelview/web/%s' % str(task.pipeline.id))

        return redirect("/myapp/web/log/%s/%s/%s" % (task.pipeline.project.cluster['NAME'],namespace, pod_name))




    @expose("/clear/<task_id>", methods=["GET", "POST"])
    def clear_task(self,task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster['KUBECONFIG'])
        namespace = conf.get('PIPELINE_NAMESPACE')

        # 删除运行时容器
        pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name[:60]
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s_client.delete_pods(namespace=namespace,pod_name=pod['name'])
            run_id = pod['labels'].get('run-id', '')
            if run_id:
                k8s_client.delete_workflow(all_crd_info = conf.get("CRD_INFO", {}), namespace=namespace,run_id=run_id)
                k8s_client.delete_pods(namespace=namespace, labels={"run-id": run_id})
                time.sleep(2)


        # 删除debug容器
        pod_name = "debug-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name[:60]
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s_client.delete_pods(namespace=namespace, pod_name=pod['name'])
            run_id = pod['labels'].get('run-id','')
            if run_id:
                k8s_client.delete_workflow(all_crd_info = conf.get("CRD_INFO", {}), namespace=namespace,run_id=run_id)
                k8s_client.delete_pods(namespace=namespace, labels={"run-id":run_id})
                time.sleep(2)
        flash("删除完成",category='success')
        # self.update_redirect()
        return redirect('/pipeline_modelview/web/%s' % str(task.pipeline.id))


    @expose("/log/<task_id>", methods=["GET", "POST"])
    def log_task(self,task_id):
        task = db.session.query(Task).filter_by(id=task_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s = K8s(task.pipeline.project.cluster['KUBECONFIG'])
        namespace = conf.get('PIPELINE_NAMESPACE')
        running_pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = running_pod_name[:60]
        pod = k8s.get_pods(namespace=namespace, pod_name=pod_name)
        if pod:
            pod = pod[0]
            return redirect("/myapp/web/log/%s/%s/%s" % (task.pipeline.project.cluster['NAME'],namespace, pod_name))

        flash("未检测到当前task正在运行的容器",category='success')
        return redirect('/pipeline_modelview/web/%s' % str(task.pipeline.id))



class Task_ModelView(Task_ModelView_Base,CompactCRUDMixin,MyappModelView):
    datamodel = SQLAInterface(Task)

# appbuilder.add_view(Task_ModelView,"Task",icon = 'fa-address-book-o',category = 'job',category_icon = 'fa-envelope')
appbuilder.add_view_no_menu(Task_ModelView)


# # 添加api
class Task_ModelView_Api(Task_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Task)
    route_base = '/task_modelview/api'
    # list_columns = ['name','label','job_template_url','volume_mount','debug']
    list_columns =['name', 'label','pipeline', 'job_template','volume_mount','node_selector','command','overwrite_entrypoint','working_dir', 'args','resource_memory','resource_cpu','resource_gpu','timeout','retry','created_by','changed_by','created_on','changed_on','monitoring','expand']
    add_columns = ['name','label','job_template','pipeline','working_dir','command','args','volume_mount','node_selector','resource_memory','resource_cpu','resource_gpu','timeout','retry','expand']
    edit_columns = add_columns
    show_columns = ['name', 'label','pipeline', 'job_template','volume_mount','node_selector','command','overwrite_entrypoint','working_dir', 'args','resource_memory','resource_cpu','resource_gpu','timeout','retry','created_by','changed_by','created_on','changed_on','monitoring','expand']


appbuilder.add_api(Task_ModelView_Api)





