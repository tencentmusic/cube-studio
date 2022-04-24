from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask import Blueprint, current_app, jsonify, make_response, request
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_serving import InferenceService
from myapp.models.model_team import Project,Project_User
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import action
from myapp import app, appbuilder,db,event_logger
import logging
from flask_babel import lazy_gettext,gettext
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


class InferenceService_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        return query.filter(self.model.project_id.in_(join_projects_id))


class InferenceService_ModelView(MyappModelView):
    datamodel = SQLAInterface(InferenceService)
    check_redirect_list_url = '/inferenceservice_modelview/list/'
    help_url = conf.get('HELP_URL', {}).get('inferenceservice','')
    # 外层的add_column和edit_columns 还有show_columns 一定要全，不然在gunicorn形式下get的不一定能被翻译
    # add_columns = ['service_type','project','name', 'label','images','resource_memory','resource_cpu','resource_gpu','min_replicas','max_replicas','ports','host','hpa','metrics','health']
    add_columns = ['service_type', 'project', 'label', 'model_name', 'model_version', 'images', 'model_path', 'model_input', 'model_output', 'resource_memory', 'resource_cpu', 'resource_gpu', 'min_replicas', 'max_replicas', 'hpa', 'canary', 'shadow', 'host', 'command', 'working_dir', 'env', 'ports', 'metrics', 'health', 'expand']
    show_columns = ['service_type','project', 'name', 'label','model_name', 'model_version', 'images', 'model_path', 'input_html', 'output_html', 'images', 'volume_mount','working_dir', 'command', 'env', 'resource_memory',
                    'resource_cpu', 'resource_gpu', 'min_replicas', 'max_replicas', 'ports', 'inference_host_url','hpa', 'canary', 'shadow', 'health','model_status', 'expand_html','metrics_html','deploy_history' ]

    list_columns = ['project','service_type','model_name_url','model_version','inference_host_url','model_status','creator','modified','operate_html']
    edit_columns = add_columns
    label_title = '推理服务'
    base_order = ('id','desc')
    order_columns = ['id']

    base_filters = [["id",InferenceService_Filter, lambda: []]]  # 设置权限过滤器
    custom_service = 'serving'
    # service_type_choices= ['',custom_service,'tfserving','torch-server','onnxruntime','triton-server','kfserving-tf','kfserving-torch','kfserving-onnx','kfserving-sklearn','kfserving-xgboost','kfserving-lightgbm','kfserving-paddle']
    service_type_choices= ['',custom_service,'tfserving','torch-server','onnxruntime','triton-server']
    # label_columns = {
    #     "host": _("域名：测试环境test.xx，调试环境 debug.xx"),
    # }
    service_type_choices = [x.replace('_','-') for x in service_type_choices]
    add_form_extra_fields={
        "project": QuerySelectField(
            _(datamodel.obj.lab('project')),
            query_factory=filter_join_org_project,
            allow_blank=True,
            widget=Select2Widget(),
            validators=[DataRequired()]
        ),
        "resource_memory":StringField(_(datamodel.obj.lab('resource_memory')),default='5G',description='内存的资源使用限制，示例1G，10G， 最大100G，如需更多联系管路员',widget=BS3TextFieldWidget(),validators=[DataRequired()]),
        "resource_cpu":StringField(_(datamodel.obj.lab('resource_cpu')), default='5',description='cpu的资源使用限制(单位核)，示例 0.4，10，最大50核，如需更多联系管路员',widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "min_replicas": StringField(_(datamodel.obj.lab('min_replicas')), default=InferenceService.min_replicas.default.arg,description='最小副本数，用来配置高可用，流量变动自动伸缩',widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "max_replicas": StringField(_(datamodel.obj.lab('max_replicas')), default=InferenceService.max_replicas.default.arg,
                                    description='最大副本数，用来配置高可用，流量变动自动伸缩', widget=BS3TextFieldWidget(),
                                    validators=[DataRequired()]),
        "host": StringField(_(datamodel.obj.lab('host')), default=InferenceService.host.default.arg,description='访问域名，xx.serving.%s'%conf.get('ISTIO_INGRESS_DOMAIN',''),widget=BS3TextFieldWidget()),
        "transformer":StringField(_(datamodel.obj.lab('transformer')), default=InferenceService.transformer.default.arg,description='前后置处理逻辑，用于原生开源框架的请求预处理和响应预处理，目前仅支持kfserving下框架',widget=BS3TextFieldWidget()),
        'resource_gpu':StringField(_(datamodel.obj.lab('resource_gpu')), default=0,
                                                        description='gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡。申请具体的卡型号，可以类似 1(V100),目前支持T4/V100/A100/VGPU',
                                                        widget=BS3TextFieldWidget()),

        'sidecar': MySelectMultipleField(
            _(datamodel.obj.lab('sidecar')), default='',
            description='容器的agent代理',
            widget=Select2ManyWidget(),
            choices=[['L5', 'L5'], ['DC', 'DC']]
        )
    }

    edit_form_extra_fields = add_form_extra_fields
    # edit_form_extra_fields['name']=StringField(_(datamodel.obj.lab('name')), description='英文名(字母、数字、- 组成)，最长50个字符',widget=MyBS3TextFieldWidget(readonly=True), validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54)]),


    # @pysnooper.snoop()
    def set_column(self, service=None):
        # 对编辑进行处理
        request_data = request.args.to_dict()
        service_type = request_data.get('service_type', '')
        if service:
            service_type = service.service_type


        if service:
            self.add_form_extra_fields['service_type'] = SelectField(
                _(self.datamodel.obj.lab('service_type')),
                description="推理框架类型",
                choices=[[x,x] for x in self.service_type_choices],
                widget=MySelect2Widget(extra_classes="readonly",value=service_type),
                validators=[DataRequired()]
            )
        else:
            self.add_form_extra_fields['service_type'] = SelectField(
                _(self.datamodel.obj.lab('service_type')),
                description="推理框架类型",
                widget=MySelect2Widget(new_web=True,value=service_type),
                choices=[[x,x] for x in self.service_type_choices],
                validators=[DataRequired()]
            )

        self.add_form_extra_fields['model_name'] = StringField(
            _('模型名称'),
            default=service.model_name if service else '',
            description='英文名(字母、数字、- 组成)，最长50个字符',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]
        )
        self.add_form_extra_fields['model_version'] = StringField(
            _('模型版本号'),
            default=service.model_version if service else datetime.datetime.now().strftime('v%Y.%m.%d.1'),
            description='版本号，时间格式',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^v[0-9.]*$"), Length(1, 54)]
        )
        self.add_form_extra_fields['model_path'] = StringField(
            _('模型地址'),
            default=service.model_path if service else '',
            description='英文名(字母、数字、- 组成)，最长50个字符',
            widget=BS3TextFieldWidget()
            # validators=[DataRequired()]
        )


        # 下面是公共配置，特定化值
        images = conf.get('INFERNENCE_IMAGES',{}).get(service_type,[])
        command = conf.get('INFERNENCE_COMMAND',{}).get(service_type,'')
        env = conf.get('INFERNENCE_ENV',{}).get(service_type,[])
        ports = conf.get('INFERNENCE_PORTS', {}).get(service_type, '80')
        metrics = conf.get('INFERNENCE_METRICS', {}).get(service_type, '')
        health = conf.get('INFERNENCE_HEALTH', {}).get(service_type, '')
        if service_type==self.custom_service:
            self.add_form_extra_fields['images'] = StringField(
                _(self.datamodel.obj.lab('images')),
                default=service.images if service else '',
                description="推理服务镜像",
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )
        else:
            self.add_form_extra_fields['images'] = SelectField(
                _(self.datamodel.obj.lab('images')),
                default=service.images if service else '',
                description="推理服务镜像",
                widget=Select2Widget(),
                choices=[[x,x] for x in images]
            )
        self.add_form_extra_fields['command'] = StringField(
            _(self.datamodel.obj.lab('command')),
            default=service.command if service else command,
            description='启动命令，支持多行命令，留空时将被自动重置',
            widget=MyBS3TextAreaFieldWidget(rows=3)
        )
        self.add_form_extra_fields['env'] = StringField(
            _(self.datamodel.obj.lab('env')),
            default=service.env if service else '\n'.join(env),
            description='使用模板的task自动添加的环境变量，支持模板变量。书写格式:每行一个环境变量env_key=env_value',
            widget=MyBS3TextAreaFieldWidget()
        )

        self.add_form_extra_fields['ports'] = StringField(
            _(self.datamodel.obj.lab('ports')),
            default=service.ports if service else ports,
            description='监听端口号，逗号分隔',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.add_form_extra_fields['metrics'] = StringField(
            _(self.datamodel.obj.lab('metrics')),
            default=service.metrics if service else metrics,
            description='请求指标采集，配置端口+url，示例：8080:/metrics',
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['health'] = StringField(
            _(self.datamodel.obj.lab('health')),
            default=service.health if service else health,
            description='健康检查接口，使用http接口或者shell命令，示例：8080:/health或者 shell:python health.py',
            widget=BS3TextFieldWidget()
        )

        # self.add_form_extra_fields['name'] = StringField(
        #     _(self.datamodel.obj.lab('name')),
        #     default=g.user.username+"-"+service_type+'-xx-v1',
        #     description='英文名(字母、数字、- 组成)，最长50个字符',
        #     widget=BS3TextFieldWidget(),
        #     validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]
        # )
        self.add_form_extra_fields['label'] = StringField(
            _(self.datamodel.obj.lab('label')),
            default="xx模型，%s框架，xx版"%service_type,
            description='中文描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.add_form_extra_fields["hpa"]=StringField(
            _(self.datamodel.obj.lab('hpa')),
            default=service.hpa if service else 'cpu:50%,gpu:50%',
            description='弹性伸缩容的触发条件：可以使用cpu/mem/gpu/qps等信息，可以使用其中一个指标或者多个指标，示例：cpu:50%,mem:%50,gpu:50%',
            widget=BS3TextFieldWidget()
        )

        self.add_form_extra_fields['expand'] = StringField(
            _(self.datamodel.obj.lab('expand')),
            default=service.expand if service else '{}',
            description='扩展字段',
            widget=MyBS3TextAreaFieldWidget(rows=12)
        )

        self.add_form_extra_fields['canary'] = StringField(
            _('流量分流'),
            default=json.loads(service.expand).get('canary','') if service else '',
            description='流量分流，将该服务的所有请求，按比例分流到目标服务上。格式 service1:20%,service2:30%，表示分流20%流量到service1，30%到service2',
            widget=BS3TextFieldWidget()
        )

        self.add_form_extra_fields['shadow'] = StringField(
            _('流量镜像'),
            default=json.loads(service.expand).get('shadow','') if service else '',
            description='流量镜像，将该服务的所有请求，按比例复制到目标服务上，格式 service1:20%,service2:30%，表示复制20%流量到service1，30%到service2',
            widget=BS3TextFieldWidget()
        )


        model_columns = ['service_type', 'project', 'label', 'model_name', 'model_version', 'images', 'model_path']
        service_columns = ['resource_memory', 'resource_cpu','resource_gpu', 'min_replicas', 'max_replicas', 'hpa','canary','shadow','host','sidecar']
        admin_columns = ['command','working_dir','env','ports','metrics','health','expand']


        if service_type=='tfserving' or service_type=='kfserving-tf':
            self.add_form_extra_fields['model_path'] = StringField(
                _('模型地址'),
                default=service.model_path if service else '/mnt/.../saved_model',
                description='仅支持tf save_model的模型存储方式',
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )


        if service_type=='torch-server' or service_type=='kfserving-torch':
            self.add_form_extra_fields['model_path'] = StringField(
                _('模型地址'),
                default=service.model_path if service else '/mnt/.../$model_name.mar',
                description='需保存完整模型信息，包括模型结构和模型参数，或者使用torch-model-archiver编译后的mar模型文件',
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )
            self.add_form_extra_fields['model_type'] = SelectField(
                _('模型类型'),
                default=service.model_type if service else 'image_classifier',
                description='模型的功能类型',
                widget=Select2Widget(),
                choices=[[x, x] for x in ["image_classifier","image_segmenter","object_detector","text_classifier"]],
                validators=[DataRequired()]
            )
            model_columns.append('model_type')


        if service_type=='onnxruntime' or service_type=='kfserving-onnx':
            self.add_form_extra_fields['model_path'] = StringField(
                _('模型地址'),
                default=service.model_path if service else '/mnt/.../$model_name.onnx',
                description='onnx模型文件的地址',
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )

        if service_type=='triton-server':
            self.add_form_extra_fields['model_path'] = StringField(
                _('模型地址'),
                default=service.model_path if service else 'onnx:/mnt/.../model.onnx(model.plan,model.bin,model.savedmodel/,model.pt,model.dali)',
                description='框架:地址。onnx:模型文件地址model.onnx，pytorch:torchscript模型文件地址model.pt，tf:模型目录地址saved_model，tensorrt:模型文件地址model.plan',
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )

            input_demo='''
[
    {
        name: "input_name"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [ 3, 224, 224 ]
        reshape: {
            shape: [ 1, 3, 224, 224 ]
        }
    }
]
            '''

            output_demo ='''
[
    {
        name: "output_name"
        data_type: TYPE_FP32
        dims: [ 1000 ]
        reshape: {
            shape: [ 1, 1000 ]
        }
    }
]
            '''

            self.add_form_extra_fields['model_input'] = StringField(
                _('模型输入'),
                default=service.model_input if service else input_demo,
                description='目前仅支持onnx/tensorrt/torch模型的triton gpu推理加速',
                widget=MyBS3TextAreaFieldWidget(rows=5),
                validators=[DataRequired()]
            )
            self.add_form_extra_fields['model_output'] = StringField(
                _('模型输出'),
                default=service.model_output if service else output_demo,
                description='目前仅支持onnx/tensorrt/torch模型的triton gpu推理加速',
                widget=MyBS3TextAreaFieldWidget(rows=5),
                validators=[DataRequired()]
            )

            model_columns.append('model_input')
            model_columns.append('model_output')

        # if 'kfserving' in service_type:
        # model_columns.append('transformer')


        add_fieldsets = [
            (
                lazy_gettext('模型配置'),
                {"fields": model_columns, "expanded": True},
            ),
            (
                lazy_gettext('推理配置'),
                {"fields": service_columns, "expanded": True},
            ),
            (
                lazy_gettext('管理员配置'),
                {"fields": admin_columns, "expanded": service_type==self.custom_service},
            )
        ]
        add_columns=model_columns+service_columns+admin_columns

        self.add_columns=add_columns
        self.edit_columns=self.add_columns
        self.add_fieldsets=add_fieldsets
        self.edit_fieldsets=self.add_fieldsets
        self.edit_form_extra_fields=self.add_form_extra_fields
        # self.show_columns=list(set(self.show_columns+add_columns+self.edit_columns+self.list_columns))
        # print('----------')
        # print(self.add_columns)
        # print(self.show_columns)
        # print('----------')



    pre_add_get=set_column
    pre_update_get=set_column


    # @pysnooper.snoop()
    def tfserving_model_config(self,model_name,model_version,model_path):
        config_str='''
model_config_list {
  config {
    name: "%s"
    base_path: "/%s/"
    model_platform: "tensorflow"
    model_version_policy {
        specific {
           versions: %s
        }
    }
  }
}
        '''%(model_name,model_path.strip('/'),model_version)
        return config_str


    def tfserving_monitoring_config(self):
        config_str='''
prometheus_config {
  enable: true
  path: "/metrics"
}
        '''
        return config_str

    def tfserving_platform_config(self):
        config_str = '''
platform_configs {
  key: "tensorflow"
  value {
    source_adapter_config {
      [type.googleapis.com/tensorflow.serving.SavedModelBundleSourceAdapterConfig] {
        legacy_config {
          session_config {
            gpu_options {
              allow_growth: true
            }
          }
        }
      }
    }
  }
}
        '''
        return config_str

# 这些配置可在环境变量中  TS_<PROPERTY_NAME>中实现
    def torch_config(self):
        config_str='''
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
cors_allowed_origin=*
cors_allowed_methods=GET, POST, PUT, OPTIONS
cors_allowed_headers=X-Custom-Header
number_of_netty_threads=32
enable_metrics_api=true
job_queue_size=1000
enable_envvars_config=true
async_logging=true
default_response_timeout=120
max_request_size=6553500
vmargs=-Dlog4j.configurationFile=file:///config/log4j2.xml
        '''
        return config_str

    def torch_log(self):
        config_str='''
<RollingFile
    name="access_log"
    fileName="${env:LOG_LOCATION:-logs}/access_log.log"
    filePattern="${env:LOG_LOCATION:-logs}/access_log.%d{dd-MMM}.log.gz">
  <PatternLayout pattern="%d{ISO8601} - %m%n"/>
  <Policies>
    <SizeBasedTriggeringPolicy size="100 MB"/>
    <TimeBasedTriggeringPolicy/>
  </Policies>
  <DefaultRolloverStrategy max="5"/>
</RollingFile>

<RollingFile
    name="ts_log"
    fileName="${env:LOG_LOCATION:-logs}/ts_log.log"
    filePattern="${env:LOG_LOCATION:-logs}/ts_log.%d{dd-MMM}.log.gz">
  <PatternLayout pattern="%d{ISO8601} [%-5p] %t %c - %m%n"/>
  <Policies>
    <SizeBasedTriggeringPolicy size="100 MB"/>
    <TimeBasedTriggeringPolicy/>
  </Policies>
  <DefaultRolloverStrategy max="5"/>
</RollingFile>
        '''
        return config_str

    def triton_config(self,model_name,model_input,model_output,model_type):
        plat_form={
            "onnx":"onnxruntime_onnx",
            "tensorrt":"tensorrt_plan",
            "torch":"pytorch_libtorch",
            "pytorch":"pytorch_libtorch",
            "tf":"tensorflow_savedmodel"
        }
        parameters=''
        if model_type == 'tf':
            parameters = '''
optimization { execution_accelerators { 
    gpu_execution_accelerator : [ { 
        name : "tensorrt"
        parameters { key: "precision_mode" value: "FP16" }}] 
}}
        '''
        if model_type=='onnx':
            parameters = '''
parameters { key: "intra_op_thread_count" value: { string_value: "0" } }
parameters { key: "execution_mode" value: { string_value: "1" } }
parameters { key: "inter_op_thread_count" value: { string_value: "0" } }
        '''
        if model_type=='pytorch' or model_type=='torch':
            parameters = '''
parameters: { key: "DISABLE_OPTIMIZED_EXECUTION" value: { string_value:"true" } }
parameters: { key: "INFERENCE_MODE" value: { string_value: "false" } }

            '''

        config_str = '''
name: "%s"
platform: "%s"
max_batch_size: 0
input %s
output %s
%s
instance_group [
    {
      count: 1
      kind: KIND_GPU
    }
]
        '''%(model_name,plat_form[model_type],model_input,model_output,parameters)
        return config_str

    # @pysnooper.snoop(watch_explode=('item'))
    def use_expand(self, item):

        # 先存储特定参数到expand
        expand = json.loads(item.expand) if item.expand else {}
        print(self.src_item_json)
        expand['service_token'] = json.loads(self.src_item_json['expand']).get('service_token','') if item.id else ''
        expand['alias_token'] = json.loads(self.src_item_json['expand']).get('alias_token', '') if item.id else ''
        expand['alias_l5'] = json.loads(self.src_item_json['expand']).get('alias_l5', '') if item.id else ''
        model_version = item.model_version.replace('v','').replace('.','').replace(':','')

        if item.service_type=='tfserving':
            model_path=item.model_path.strip('/')
            des_model_path = "/models/%s/" % (item.model_name,)
            des_version_path = "/models/%s/%s/"%(item.model_name,model_version)
            if not item.id or not item.command:
                item.command='mkdir -p %s && cp -r /%s/* %s  &&  /usr/bin/tf_serving_entrypoint.sh --model_config_file=/config/models.config --monitoring_config_file=/config/monitoring.config --platform_config_file=/config/platform.config'%(des_version_path,model_path,des_version_path)
            item.health='8501:/v1/models/%s/versions/%s/metadata'%(item.model_name,model_version)

            expand['models.config']=expand['models.config'] if expand.get('models.config','') else self.tfserving_model_config(item.model_name,model_version,des_model_path)
            expand['monitoring.config']=expand['monitoring.config'] if expand.get('monitoring.config','') else self.tfserving_monitoring_config()
            expand['platform.config'] = expand['platform.config'] if expand.get('platform.config','') else self.tfserving_platform_config()

        if item.service_type=='torch-server':
            if '.mar' not in item.model_path:
                tar_command = 'torch-model-archiver --model-name %s --version %s --handler %s --serialized-file %s --export-path /models -f'%(item.model_name,model_version,item.transformer or item.model_type,item.model_path)
            else:
                tar_command='cp %s /models/'%item.model_path
            if not item.id or not item.command:
                item.command='mkdir -p /models && cp /config/* /models/ && '+tar_command+' && torchserve --start --model-store /models --models %s=%s.mar --foreground --ts-config=/config/config.properties'%(item.model_name,item.model_name)
            if not item.working_dir:
                item.working_dir='/models'
            expand['config.properties'] = expand['config.properties'] if expand.get('config.properties','') else self.torch_config()
            expand['log4j2.xml'] = expand['log4j2.xml'] if expand.get('log4j2.xml','') else self.torch_log()

        if item.service_type=='triton-server':
            # 识别模型类型
            model_type = item.model_path.split(":")[0]
            model_path = item.model_path.split(":")[1]

            if not item.id or not item.command:
                if model_type=='tf':
                    item.command='mkdir -p /models/{model_name}/{model_version}/model.savedmodel && cp /config/* /models/{model_name}/ && cp -r /{model_path}/* /models/{model_name}/{model_version}/model.savedmodel && tritonserver --model-repository=/models --strict-model-config=true  --log-verbose=1'.format(model_path=model_path.strip('/'),model_name=item.model_name,model_version=model_version)
                else:
                    model_file_ext = item.model_path.split(".")[-1]
                    item.command='mkdir -p /models/{model_name}/{model_version}/ && cp /config/* /models/{model_name}/ && cp -r {model_path} /models/{model_name}/{model_version}/model.{model_file_ext} && tritonserver --model-repository=/models --strict-model-config=true  --log-verbose=1'.format(model_path=model_path,model_name=item.model_name,model_version=model_version,model_file_ext=model_file_ext)

            config_str = self.triton_config(item.model_name,item.model_input,item.model_output,model_type)
            old_config_str = json.loads(self.src_item_json['expand']).get('config.pbtxt','') if item.id else ''
            new_config_str = expand.get('config.pbtxt','')
            if not item.id:
                expand['config.pbtxt']=config_str
            elif new_config_str==old_config_str and new_config_str!=config_str:
                expand['config.pbtxt']=config_str
            elif not new_config_str:
                expand['config.pbtxt'] = config_str

        if item.service_type=='onnxruntime':
            if not item.id or not item.command:
                item.command='./onnxruntime_server --log_level info --model_path  %s'%item.model_path

        item.name=item.service_type+"-"+item.model_name+"-"+model_version
        item.expand = json.dumps(expand,indent=4,ensure_ascii=False)

    # @pysnooper.snoop()
    def pre_add(self, item):
        if not item.volume_mount:
            item.volume_mount=item.project.volume_mount
        self.use_expand(item)

    def delete_old_service(self,service_name,cluster):

        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(cluster['KUBECONFIG'])
        service_namespace = conf.get('SERVICE_NAMESPACE')
        kfserving_namespace = conf.get('KFSERVING_NAMESPACE')
        for namespace in [service_namespace,kfserving_namespace]:
            for name in [service_name,'debug-'+service_name,'test-'+service_name]:
                service_external_name = (name + "-external").lower()[:60].strip('-')
                k8s_client.delete_deployment(namespace=namespace, name=name)
                k8s_client.delete_service(namespace=namespace, name=name)
                k8s_client.delete_service(namespace=namespace, name=service_external_name)
                k8s_client.delete_istio_ingress(namespace=namespace, name=name)
                k8s_client.delete_hpa(namespace=namespace, name=name)
                k8s_client.delete_configmap(namespace=namespace, name=name)
                isvc_crd=conf.get('CRD_INFO')['inferenceservice']
                k8s_client.delete_crd(isvc_crd['group'],isvc_crd['version'],isvc_crd['plural'],namespace=namespace,name=name)


    # @pysnooper.snoop(watch_explode=('item',))
    def pre_update(self, item):
        # 修改了名称的话，要把之前的删掉
        self.use_expand(item)

    def pre_delete(self, item):
        self.delete_old_service(item.name,item.project.cluster)
        flash('服务已清理完成', category='warning')

    @expose('/clear/<service_id>', methods=['POST', "GET"])
    # @pysnooper.snoop()
    def clear(self, service_id):
        service = db.session.query(InferenceService).filter_by(id=service_id).first()
        self.delete_old_service(service.name,service.project.cluster)
        service.model_status='offline'
        if not service.deploy_history:
            service.deploy_history=''
        service.deploy_history = service.deploy_history + "\n" + "clear：%s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db.session.commit()
        flash('服务清理完成', category='warning')
        return redirect('/inferenceservice_modelview/list/')

    # @pysnooper.snoop()
    def create_polaris(self,service):
        try:
            # l5的值创建以后是不应该变的，一个服务对应一个固定的l5，不然客户端需要修改代码
            from myapp.utils.py.py_polaris import Polaris
            polaris = Polaris()

            alias_token = json.loads(service.expand).get('alias_token','') if service.expand else ''
            alias_l5 = json.loads(service.expand).get('alias_l5','') if service.expand else ''   # 只创建一次
            service_name = '%s.service' % (service.name)
            username = service.created_by.username + "," + conf.get('ADMIN_USER')

            if not alias_l5:

                alias = polaris.get_alias(service_name)
                if len(alias)>0 and alias[0]['alias']!=alias_l5:
                    flash('创建失败，存在系统无法识别的北极星别名，请先联系管理员手动处理','warning')
                    return

                polaris.delete_instances(service_name)
                polaris.delete_alias(service_name,alias_token)
                polaris.delete_service(service_name)
                polaris_service = polaris.register_service(username, service_name)
                print(polaris_service)
                service_token = polaris_service['token'] if polaris_service else ''

                alias = polaris.register_alias(username, service_name)

                expand = json.loads(service.expand) if service.expand else {}
                expand.update(
                    {
                        "service_token": service_token,
                        "alias_token": alias['service_token'],
                        "alias_l5": alias['alias'],
                    }
                )
                service.expand = json.dumps(expand, indent=4, ensure_ascii=False)
                db.session.commit()


            polaris.delete_instances(service_name)
            instances = polaris.register_instances(service_name,conf.get('SERVICE_EXTERNAL_IP'),30000+10*service.id)
            print(instances)


        except Exception as e:
            print(e)
            flash('部署北极星失败:%s'%str(e),'warning')

    #
    # # 针对kfserving框架，单独的部署方式
    # @pysnooper.snoop()
    # def deploy_kfserving(self,service):
    #     from myapp.utils.py.py_k8s import K8s
    #     k8s_client = K8s(service.project.cluster['KUBECONFIG'])
    #     namespace = conf.get('KFSERVING_NAMESPACE')
    #
    #     crd_info=conf.get('CRD_INFO')['inferenceservice']
    #
    #
    #     crd_list = k8s_client.get_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace)
    #     for vs_obj in crd_list:
    #         if vs_obj['name'] == service.name:
    #             k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=vs_obj['name'])
    #             time.sleep(1)
    #
    #     crd_json = {
    #         "apiVersion": "%s/%s"%(crd_info['group'],crd_info['version']),
    #         "kind": crd_info['kind'],
    #         "metadata": {
    #             "name": service.name,
    #             "namespace": namespace,
    #             "labels": {
    #                 "app": service.name,
    #                 "rtx-user": service.created_by.username
    #             }
    #         },
    #         "spec": {
    #             "predictor":
    #                 {
    #                     "min_replicas":service.min_replicas,
    #                     "max_replicas":service.max_replicas,
    #                     "pytorch": {
    #                         "storageUri": "gs://kfserving-examples/models/torchserve/image_classifier"
    #                     }
    #                 }
    #         }
    #     }
    #
    #     print(crd_json)
    #     crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=crd_json)
    #
    #
    #     flash('服务部署完成', category='success')
    #     return redirect('/inferenceservice_modelview/list/')


    @expose('/debug/<service_id>',methods=['POST',"GET"])
    # @pysnooper.snoop()
    def deploy_debug(self,service_id):
        return self.deploy(service_id,env='debug')

    @expose('/deploy/test/<service_id>',methods=['POST',"GET"])
    # @pysnooper.snoop()
    def deploy_test(self,service_id):
        return self.deploy(service_id,env='test')

    @expose('/deploy/prod/<service_id>', methods=['POST', "GET"])
    # @pysnooper.snoop()
    def deploy_prod(self, service_id):
        return self.deploy(service_id,env='prod')


    @pysnooper.snoop()
    def deploy(self,service_id,env='prod'):
        service = db.session.query(InferenceService).filter_by(id=service_id).first()
        namespace = conf.get('SERVICE_NAMESPACE','service')
        pre_namespace = conf.get('PRE_SERVICE_NAMESPACE','pre-service')
        name =  service.name
        command = service.command
        deployment_replicas = service.min_replicas
        if env=='debug':
            name = env+'-'+service.name
            command = 'sleep 43200'
            deployment_replicas = 1
            # namespace=pre_namespace

        if env =='test':
            name = env+'-'+service.name
            # namespace=pre_namespace

        # if 'kfserving' in service.service_type:
        #     return self.deploy_kfserving(service)

        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])


        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service.project.cluster['KUBECONFIG'])

        expand=json.loads(service.expand)
        config_data={}
        if service.service_type=='tfserving':
            config_data={
                "models.config":expand.get('models.config').replace('\r\n','\n'),
                "monitoring.config":expand.get('monitoring.config').replace('\r\n','\n'),
                "platform.config": expand.get('platform.config').replace('\r\n','\n')
            }

        if service.service_type=='torch-server':
            config_data={
                "config.properties":expand.get('config.properties').replace('\r\n','\n'),
                "log4j2.xml":expand.get('log4j2.xml').replace('\r\n','\n'),
            }

        if service.service_type=='triton-server':
            config_data={
                "config.pbtxt":expand.get('config.pbtxt').replace('\r\n','\n')
            }

        k8s_client.create_configmap(namespace=namespace,name=name,data=config_data,labels={'app':name})
        volume_mount = service.volume_mount+",%s(configmap):/config/"%name
        ports = [int(port) for port in service.ports.split(',')]


        pod_env = service.env
        pod_env+="\nKUBEFLOW_ENV="+env
        pod_env+='\nKUBEFLOW_MODEL_PATH='+service.model_path if service.model_path else ''
        pod_env+='\nKUBEFLOW_MODEL_VERSION='+service.model_version
        pod_env+='\nKUBEFLOW_MODEL_IMAGES='+service.images
        pod_env+='\nKUBEFLOW_MODEL_NAME='+service.model_name
        pod_env=pod_env.strip(',')


        if env=='test' or env =='debug':
            try:
                k8s_client.delete_deployment(namespace=namespace,name=name)
            except Exception as e:
                print(e)
        # 因为所有的服务流量通过ingress实现，所以没有isito的envoy代理
        try:
            k8s_client.create_deployment(
                namespace=namespace,
                name=name,
                replicas=deployment_replicas,
                labels={"app":name,"username":service.created_by.username},
                command=['sh','-c',command] if command else None,
                args=None,
                volume_mount=volume_mount,
                working_dir=service.working_dir,
                node_selector=service.get_node_selector(),
                resource_memory=service.resource_memory,
                resource_cpu=service.resource_cpu,
                resource_gpu=service.resource_gpu if service.resource_gpu else '',
                image_pull_policy='Always',
                image_pull_secrets=image_secrets,
                image=service.images,
                hostAliases=conf.get('HOSTALIASES',''),
                env=pod_env,
                privileged=False,
                accounts=None,
                username=service.created_by.username,
                ports=ports,
                health=service.health if ':' in service.health else None
            )
        except Exception as e:
            flash('deploymnet:'+str(e),'warning')


        # 监控
        if service.metrics:
            annotations = {
                "prometheus.io/scrape": "true",
                "prometheus.io/port": service.metrics.split(":")[0],
                "prometheus.io/path": service.metrics.split(":")[1]
            }
        else:
            annotations={}

        k8s_client.create_service(
            namespace=namespace,
            name=name,
            username=service.created_by.username,
            ports=ports,
            annotations=annotations
        )
        # 如果域名配置的gateway，就用这个
        if 'kfserving' in service.service_type:
            host = service.name + "." + service.project.cluster.get('KFSERVING_DOMAIN', conf.get('KFSERVING_DOMAIN'))
        else:
            host = service.name+"."+ service.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN'))

        if service.host:
            host=service.host.replace('http://','').replace('https://','').strip()
            if "/" in host:
                host = host[:host.index("/")]

        # 前缀来区分不同的环境服务
        if env=='debug' or env=='test':
            host=env+'.'+host

        k8s_client.create_istio_ingress(
            namespace=namespace,
            name=name,
            host = host,
            ports=service.ports.split(','),
            canary=service.canary,
            shadow=service.shadow
        )


        # # 以ip形式访问的话，使用的代理ip。不然不好处理机器服务化机器扩容和缩容时ip变化
        # SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP',None)
        # if SERVICE_EXTERNAL_IP:
        #     service_ports = [[20000+10*service.id+index,port] for index,port in enumerate(ports)]
        #     service_external_name = (service.name + "-external").lower()[:60].strip('-')
        #     k8s_client.create_service(
        #         namespace=namespace,
        #         name=service_external_name,
        #         username=service.created_by.username,
        #         ports=service_ports,
        #         selector={"app": service.name, 'user': service.created_by.username},
        #         externalIPs=conf.get('SERVICE_EXTERNAL_IP',None)
        #     )
        #     self.create_polaris(service)

        if env!='debug':
            hpas = re.split(',|;', service.hpa)
            if not int(service.resource_gpu):
                for hpa in copy.deepcopy(hpas):
                    if 'gpu' in hpa:
                        hpas.remove(hpa)

            # 伸缩容
            if int(service.max_replicas)>int(service.min_replicas) and service.hpa:
                try:
                    k8s_client.create_hpa(
                        namespace=namespace,
                        name=name,
                        min_replicas=int(service.min_replicas),
                        max_replicas=int(service.max_replicas),
                        hpa=','.join(hpas)
                    )
                except Exception as e:
                    flash('hpa:'+str(e),'warning')

        # # 使用激活器
        # if int(service.min_replicas)==0:
        #     flash('检测到最小副本为0，已加入激活器装置')
        #     pass

        # 不记录部署测试的情况
        if env =='debug' and service.model_status=='offline':
            service.model_status = 'debug'
        if env=='test' and service.model_status=='offline':
            service.model_status = 'test'

        if env=='prod':
            service.model_status = 'online'
        service.deploy_history=service.deploy_history+"\n"+"deploy %s：%s"%(env,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        db.session.commit()
        if env=="debug":
            time.sleep(2)
            pods = k8s_client.get_pods(namespace=namespace,labels={"app":name})
            if pods:
                pod = pods[0]
                return redirect("/myapp/web/debug/%s/%s/%s/%s" % (service.project.cluster['NAME'], namespace, pod['name'],name))
        flash('服务部署完成',category='success')
        return redirect('/inferenceservice_modelview/list/')





appbuilder.add_view(InferenceService_ModelView,"推理服务",icon = 'fa-space-shuttle',category = '服务化')


