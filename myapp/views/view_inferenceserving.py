import requests
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask import jsonify

from myapp.models.model_serving import InferenceService
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import action
from myapp import app, appbuilder,db
from flask_babel import lazy_gettext
import re
import pysnooper
import copy
from sqlalchemy.exc import InvalidRequestError
from myapp.models.model_job import Repository
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from myapp import security_manager
from wtforms.validators import DataRequired, Length, Regexp
from wtforms import SelectField, StringField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget, MyBS3TextFieldWidget,MySelectMultipleField
from myapp.views.view_team import Project_Join_Filter,filter_join_org_project
from flask import (
    flash,
    g,
    Markup,
    redirect,
    request
)
from .base import (
    MyappFilter,
    MyappModelView,

)
from .baseApi import (
    MyappModelRestApi
)

from flask_appbuilder import expose
import datetime,time,json
conf = app.config


class InferenceService_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        return query.filter(self.model.project_id.in_(join_projects_id))


class InferenceService_ModelView_base():
    datamodel = SQLAInterface(InferenceService)
    check_redirect_list_url = conf.get('MODEL_URLS',{}).get('inferenceservice','')


    # add_columns = ['service_type','project','name', 'label','images','resource_memory','resource_cpu','resource_gpu','min_replicas','max_replicas','ports','host','hpa','metrics','health']
    add_columns = ['service_type', 'project', 'label', 'model_name', 'model_version', 'images', 'model_path', 'resource_memory', 'resource_cpu', 'resource_gpu', 'min_replicas', 'max_replicas', 'hpa','priority', 'canary', 'shadow', 'host','inference_config',  'working_dir', 'command','volume_mount', 'env', 'ports', 'metrics', 'health','expand','sidecar']
    show_columns = ['service_type','project', 'name', 'label','model_name', 'model_version', 'images', 'model_path', 'images', 'volume_mount','sidecar','working_dir', 'command', 'env', 'resource_memory',
                    'resource_cpu', 'resource_gpu', 'min_replicas', 'max_replicas', 'ports', 'inference_host_url','hpa','priority', 'canary', 'shadow', 'health','model_status','expand','metrics','deploy_history','host','inference_config']

    edit_columns = add_columns

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields

    list_columns = ['project','service_type','label','model_name_url','model_version','inference_host_url','ip','model_status','resource','replicas_html','creator','modified','operate_html']
    cols_width={
        "project":{"type": "ellip2", "width": 150},
        "label": {"type": "ellip2", "width": 300},
        "service_type": {"type": "ellip2", "width": 100},
        "model_name_url":{"type": "ellip2", "width": 300},
        "model_version": {"type": "ellip2", "width": 200},
        "inference_host_url": {"type": "ellip2", "width": 500},
        "ip": {"type": "ellip2", "width": 250},
        "model_status": {"type": "ellip2", "width": 100},
        "modified": {"type": "ellip2", "width": 150},
        "operate_html": {"type": "ellip2", "width": 350},
        "resource": {"type": "ellip2", "width": 300},
    }
    search_columns = ['name','created_by','project','service_type','label','model_name','model_version','model_path','host','model_status','resource_gpu']

    label_title = '推理服务'
    base_order = ('id','desc')
    order_columns = ['id']

    base_filters = [["id",InferenceService_Filter, lambda: []]]
    images = []
    INFERNENCE_IMAGES = list(conf.get('INFERNENCE_IMAGES', {}).values())
    for item in INFERNENCE_IMAGES:
        images += item
    service_type_choices= ['serving','tfserving','torch-server','onnxruntime','triton-server']
    spec_label_columns = {
        # "host": _("域名：测试环境test.xx，调试环境 debug.xx"),
        "resource":"资源",
        "replicas_html":"副本数"
    }
    service_type_choices = [x.replace('_','-') for x in service_type_choices]
    host_rule=",".join([cluster+"集群:*."+conf.get('CLUSTERS')[cluster].get("SERVICE_DOMAIN",conf.get('SERVICE_DOMAIN','')) for cluster in conf.get('CLUSTERS') if conf.get('CLUSTERS')[cluster].get("SERVICE_DOMAIN",conf.get('SERVICE_DOMAIN',''))])
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
        "host": StringField(_(datamodel.obj.lab('host')), default=InferenceService.host.default.arg,description='访问域名，'+host_rule,widget=BS3TextFieldWidget()),
        "transformer":StringField(_(datamodel.obj.lab('transformer')), default=InferenceService.transformer.default.arg,description='前后置处理逻辑，用于原生开源框架的请求预处理和响应预处理，目前仅支持kfserving下框架',widget=BS3TextFieldWidget()),
        'resource_gpu':StringField(_(datamodel.obj.lab('resource_gpu')), default='0',
                                                        description='gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡。申请具体的卡型号，可以类似 1(V100),目前支持T4/V100/A100,或虚拟gpu，例如0.2(T4)',
                                                        widget=BS3TextFieldWidget(),validators=[DataRequired()]),

        'sidecar': MySelectMultipleField(
            _(datamodel.obj.lab('sidecar')),
            default='',
            description='容器的agent代理,istio用于服务网格',
            widget=Select2ManyWidget(),
            validators=[],
            choices=[['istio','istio']]
        ),
        "priority": SelectField(
            _('服务优先级'),
            widget=MySelect2Widget(),
            default=1,
            description='优先满足高优先级的资源需求，同时保证每个服务的最低pod副本数',
            choices=[[1, '高优先级'],[0, '低优先级']],
            validators=[DataRequired()]
        ),
        'model_name': StringField(
            _('模型名称'),
            default='',
            description='英文名(小写字母、数字、- 组成)，最长50个字符',
            widget=MyBS3TextFieldWidget(),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]
        ),
        'model_version': StringField(
            _('模型版本号'),
            default= datetime.datetime.now().strftime('v%Y.%m.%d.1'),
            description='版本号，时间格式',
            widget=MyBS3TextFieldWidget(),
            validators=[DataRequired(), Length(1, 54)]
        ),

        'service_type': SelectField(
            _(datamodel.obj.lab('service_type')),
            default='serving',
            description="推理框架类型",
            widget=MySelect2Widget(retry_info=True),
            choices=[[x, x] for x in service_type_choices],
            validators=[DataRequired()]
        ),
        'label': StringField(
            _(datamodel.obj.lab('label')),
            default="xx模型，%s框架，xx版",
            description='中文描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "hpa": StringField(
            _(datamodel.obj.lab('hpa')),
            default='cpu:50%,gpu:50%',
            description='弹性伸缩容的触发条件：可以使用cpu/mem/gpu/qps等信息，可以使用其中一个指标或者多个指标，示例：cpu:50%,mem:50%,gpu:50%',
            widget=BS3TextFieldWidget()
        ),

        'expand': StringField(
            _(datamodel.obj.lab('expand')),
            default=json.dumps({
                "help_url":"https://github.com/tencentmusic/cube-studio/tree/master/images/serving"
            },indent=4,ensure_ascii=False),
            description='扩展字段',
            widget=MyBS3TextAreaFieldWidget(rows=3)
        ),

        'canary': StringField(
            _('流量分流'),
            default='',
            description='流量分流，将该服务的所有请求，按比例分流到目标服务上。格式 service1:20%,service2:30%，表示分流20%流量到service1，30%到service2',
            widget=BS3TextFieldWidget()
        ),

        'shadow': StringField(
            _('流量复制'),
            default='',
            description='流量复制，将该服务的所有请求，按比例复制到目标服务上，格式 service1:20%,service2:30%，表示复制20%流量到service1，30%到service2',
            widget=BS3TextFieldWidget()
        ),
        'volume_mount':StringField(
            _(datamodel.obj.lab('volume_mount')),
            default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives',
            description='外部挂载，格式:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,4G(memory):/dev/shm,注意pvc会自动挂载对应目录下的个人rtx子目录',
            widget=BS3TextFieldWidget()
        ),
        'model_path':StringField(
            _('模型地址'),
            default='',
            description=Markup('tfserving：仅支持添加了服务签名的saved_model目录地址，例如 /xx/saved_model<br>'
                               'torch-server：torch-model-archiver编译后的mar模型文件，需保存模型结构和模型参数<br>'
                               'onnxruntime：onnx模型文件的地址<br>'
                               'triton-server：框架:地址。onnx:模型文件地址model.onnx，pytorch:torchscript模型文件地址model.pt，tf:模型目录地址saved_model，tensorrt:模型文件地址model.plan'),
            widget=BS3TextFieldWidget(),
            validators=[]
        ),
        'images': SelectField(
            _(datamodel.obj.lab('images')),
            default='',
            description="推理服务镜像",
            widget=MySelect2Widget(can_input=True),
            choices=[[x, x] for x in images]
        ),
        'command': StringField(
            _(datamodel.obj.lab('command')),
            default='',
            description='启动命令，<font color="#FF0000">留空时将被自动重置</font>',
            widget=MyBS3TextAreaFieldWidget(rows=3)
        ),
        'env':StringField(
            _(datamodel.obj.lab('env')),
            default='',
            description='使用模板的task自动添加的环境变量，支持模板变量。书写格式:每行一个环境变量env_key=env_value',
            widget=MyBS3TextAreaFieldWidget()
        ),
        'ports': StringField(
            _(datamodel.obj.lab('ports')),
            default='',
            description='监听端口号，逗号分隔',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        'metrics': StringField(
            _(datamodel.obj.lab('metrics')),
            default='',
            description='请求指标采集，配置端口+url，示例：8080:/metrics',
            widget=BS3TextFieldWidget()
        ),
        'health': StringField(
            _(datamodel.obj.lab('health')),
            default='',
            description='健康检查接口，使用http接口或者shell命令，示例：8080:/health或者 shell:python health.py',
            widget=BS3TextFieldWidget()
        ),

        'inference_config': StringField(
            _('推理配置文件'),
            default='',
            description='会配置文件的形式挂载到容器/config/目录下。<font color="#FF0000">留空时将被自动重置</font>，格式：<br>---文件名<br>多行文件内容<br>---文件名<br>多行文件内容',
            widget=MyBS3TextAreaFieldWidget(rows=5),
            validators=[]
        )

    }

    input_demo = '''
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

    output_demo = '''
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


    edit_form_extra_fields = add_form_extra_fields
    # edit_form_extra_fields['name']=StringField(_(datamodel.obj.lab('name')), description='英文名(小写字母、数字、- 组成)，最长50个字符',widget=MyBS3TextFieldWidget(readonly=True), validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54)]),

    model_columns = ['service_type', 'project', 'label', 'model_name', 'model_version', 'images', 'model_path']
    service_columns = ['resource_memory', 'resource_cpu', 'resource_gpu', 'min_replicas', 'max_replicas', 'hpa',
                       'priority', 'canary', 'shadow', 'host', 'volume_mount', 'sidecar']
    admin_columns = ['inference_config', 'working_dir', 'command', 'env', 'ports', 'metrics', 'health', 'expand']

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
            {"fields": admin_columns, "expanded": True},
        )
    ]
    add_columns = model_columns + service_columns + admin_columns

    edit_columns = add_columns

    edit_fieldsets = add_fieldsets

    def pre_add_web(self):
        self.default_filter = {
            "created_by": g.user.id
        }

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
<RollingFile name="access_log" fileName="${env:LOG_LOCATION:-logs}/access_log.log" filePattern="${env:LOG_LOCATION:-logs}/access_log.%d{dd-MMM}.log.gz"> 
  <PatternLayout pattern="%d{ISO8601} - %m%n"/>  
  <Policies> 
    <SizeBasedTriggeringPolicy size="100 MB"/>  
    <TimeBasedTriggeringPolicy/> 
  </Policies>  
  <DefaultRolloverStrategy max="5"/> 
</RollingFile>

        '''
        return config_str

    def triton_config(self,item,model_type):
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
        '''%(item.model_name,plat_form[model_type],self.input_demo,self.output_demo,parameters)
        return config_str

    # @pysnooper.snoop(watch_explode=('item'))
    def use_expand(self, item):
        #
        # item.ports = conf.get('INFERNENCE_PORTS',{}).get(item.service_type,item.ports)
        # item.env = '\n'.join(conf.get('INFERNENCE_ENV', {}).get(item.service_type, item.env.split('\n') if item.env else []))
        # item.metrics = conf.get('INFERNENCE_METRICS', {}).get(item.service_type, item.metrics)
        # item.health = conf.get('INFERNENCE_HEALTH', {}).get(item.service_type, item.health)

        # 先存储特定参数到expand
        expand = json.loads(item.expand) if item.expand else {}
        print(self.src_item_json)
        model_version = item.model_version.replace('v','').replace('.','').replace(':','')
        model_path = "/"+item.model_path.strip('/') if item.model_path else ''
        # 对网络地址先同一在命令中下载
        download_command=''
        if 'http:' in item.model_path or 'https:' in item.model_path:
            model_file = item.model_path[item.model_path.rindex('/')+1:]
            model_path = model_file
            download_command = 'wget %s && '%item.model_path
            if '.zip' in item.model_path:
                download_command+='unzip -O %s && '%model_file
                model_path = model_file.replace('.zip', '').replace('.tar.gz', '')  # 这就要求压缩文件和目录同名，并且下面直接就是目录。其他格式的文件不能压缩
            if '.tar.gz' in item.model_path:
                download_command += 'tar -zxvf %s && '%model_file
                model_path = model_file.replace('.zip','').replace('.tar.gz','')  # 这就要求压缩文件和目录同名，并且下面直接就是目录。其他格式的文件不能压缩

        if item.service_type=='tfserving':
            des_model_path = "/models/%s/" % (item.model_name,)
            des_version_path = "/models/%s/%s/"%(item.model_name,model_version)
            if not item.id or not item.command:
                item.command=download_command+'''mkdir -p %s && cp -r %s/* %s  &&  /usr/bin/tf_serving_entrypoint.sh --model_config_file=/config/models.config --monitoring_config_file=/config/monitoring.config --platform_config_file=/config/platform.config'''%(des_version_path,model_path,des_version_path)

            item.health='8501:/v1/models/%s/versions/%s/metadata'%(item.model_name,model_version)

            expand['models.config']=expand['models.config'] if expand.get('models.config','') else self.tfserving_model_config(item.model_name,model_version,des_model_path)
            expand['monitoring.config']=expand['monitoring.config'] if expand.get('monitoring.config','') else self.tfserving_monitoring_config()
            expand['platform.config'] = expand['platform.config'] if expand.get('platform.config','') else self.tfserving_platform_config()
            if not item.inference_config:
                item.inference_config='''
---models.config
%s
---monitoring.config
%s
---platform.config
%s
                '''%(
                    self.tfserving_model_config(item.model_name,model_version,des_model_path),
                    self.tfserving_monitoring_config(),
                    self.tfserving_platform_config()
                    )

        if item.service_type=='torch-server':
            if not item.working_dir:
                item.working_dir='/models'
            model_file = model_path[model_path.rindex('/') + 1:] if '/' in model_path else model_path
            tar_command='ls'
            if '.mar' not in model_path:
                tar_command = 'torch-model-archiver --model-name %s --version %s --handler %s --serialized-file %s --export-path /models -f'%(item.model_name,model_version,item.transformer or item.model_type,model_path)
            else:
                if ('http:' in item.model_path or 'https://' in item.model_path) and item.working_dir=='/models':
                    print('has download to des_version_path')
                else:
                    tar_command='cp -rf %s /models/'%(model_path)
            if not item.id or not item.command:
                item.command=download_command+'cp /config/* /models/ && '+tar_command+' && torchserve --start --model-store /models --models %s=%s.mar --foreground --ts-config=/config/config.properties'%(item.model_name,item.model_name)

            expand['config.properties'] = expand['config.properties'] if expand.get('config.properties','') else self.torch_config()
            expand['log4j2.xml'] = expand['log4j2.xml'] if expand.get('log4j2.xml','') else self.torch_log()

            if not item.inference_config:
                item.inference_config = '''
---config.properties
%s
---log4j2.xml
%s
                ''' % (
                    self.torch_config(),
                    self.torch_log()
                )


        if item.service_type=='triton-server':
            # 识别模型类型
            model_type = 'tf'
            if '.onnx' in model_path:
                model_type='onnx'
            if '.plan' in model_path:
                model_type = 'tensorrt'
            if '.pt' in model_path or '.pth' in model_path:
                model_type = 'pytorch'

            if not item.id or not item.command:
                if model_type=='tf':
                    item.command=download_command+'mkdir -p /models/{model_name}/{model_version}/model.savedmodel && cp /config/* /models/{model_name}/ && cp -r /{model_path}/* /models/{model_name}/{model_version}/model.savedmodel && tritonserver --model-repository=/models --strict-model-config=true  --log-verbose=1'.format(model_path=model_path.strip('/'),model_name=item.model_name,model_version=model_version)
                else:
                    model_file_ext = model_path.split(".")[-1]
                    item.command=download_command+'mkdir -p /models/{model_name}/{model_version}/ && cp /config/* /models/{model_name}/ && cp -r {model_path} /models/{model_name}/{model_version}/model.{model_file_ext} && tritonserver --model-repository=/models --strict-model-config=true  --log-verbose=1'.format(model_path=model_path,model_name=item.model_name,model_version=model_version,model_file_ext=model_file_ext)

            config_str = self.triton_config(item,model_type)
            old_config_str = json.loads(self.src_item_json['expand']).get('config.pbtxt','') if item.id else ''
            new_config_str = expand.get('config.pbtxt','')
            if not item.id:
                expand['config.pbtxt']=config_str
            elif new_config_str==old_config_str and new_config_str!=config_str:
                expand['config.pbtxt']=config_str
            elif not new_config_str:
                expand['config.pbtxt'] = config_str

            if not item.inference_config:
                item.inference_config = '''
---config.pbtxt
%s
                    ''' % (
                    config_str,
                )


        if item.service_type=='onnxruntime':
            if not item.id or not item.command:
                item.command=download_command+'./onnxruntime_server --log_level info --model_path  %s'%model_path

        item.name=item.service_type+"-"+item.model_name+"-"+model_version
        # item.expand = json.dumps(expand,indent=4,ensure_ascii=False)



    # @pysnooper.snoop()
    def pre_add(self, item):
        if not item.model_path:
            item.model_path=''
        if not item.volume_mount:
            item.volume_mount=item.project.volume_mount
        self.use_expand(item)

        if ('http:' in item.model_path or 'https:' in item.model_path) and ('.zip' in item.model_path or '.tar.gz' in item.model_path):
            try:
                flash('检测到模型地址为网络压缩文件，需压缩文件名和解压后文件夹名相同','warning')
            except Exception as e:
                pass
                print(e)

    def delete_old_service(self,service_name,cluster):
        try:
            from myapp.utils.py.py_k8s import K8s
            k8s_client = K8s(cluster.get('KUBECONFIG',''))
            service_namespace = conf.get('SERVICE_NAMESPACE')
            for namespace in [service_namespace,]:
                for name in [service_name,'debug-'+service_name,'test-'+service_name]:
                    service_external_name = (name + "-external").lower()[:60].strip('-')
                    k8s_client.delete_deployment(namespace=namespace, name=name)
                    k8s_client.delete_service(namespace=namespace, name=name)
                    k8s_client.delete_service(namespace=namespace, name=service_external_name)
                    k8s_client.delete_istio_ingress(namespace=namespace, name=name)
                    k8s_client.delete_hpa(namespace=namespace, name=name)
                    k8s_client.delete_configmap(namespace=namespace, name=name)
        except Exception as e:
            print(e)

    # @pysnooper.snoop(watch_explode=('item',))
    def pre_update(self, item):
        if not item.volume_mount:
            item.volume_mount=item.project.volume_mount
        item.name = item.name.replace("_","-")
        if ('http:' in item.model_path or 'https:' in item.model_path) and ('.zip' in item.model_path or '.tar.gz' in item.model_path):
            flash('检测到模型地址为网络压缩文件，需压缩文件名和解压后文件夹名相同','warning')

        # if ('http://' in item.model_path or 'https://' in item.model_path) and item.model_path!=self.src_item_json.get('model_path',''):
        #     # self.download_model(item)
        #     if '.zip' not in item.model_path and '.tar.gz' not in item.model_path:
        #         flash('未识别的模型网络地址','warning')

        # 修改了名称的话，要把之前的删掉
        self.use_expand(item)

        # 如果模型版本和模型名称变了，需要把之前的服务删除掉
        if self.src_item_json.get('name','') and item.name!=self.src_item_json.get('name',''):
            self.delete_old_service(self.src_item_json.get('name',''), item.project.cluster)
            flash('发现模型服务变更，启动清理服务%s:%s'%(self.src_item_json.get('model_name',''),self.src_item_json.get('model_version','')),'success')

    # 事后无法读取到project属性
    def pre_delete(self, item):
        self.delete_old_service(item.name,item.project.cluster)
        flash('服务已清理完成', category='success')

    @expose('/clear/<service_id>', methods=['POST', "GET"])
    def clear(self, service_id):
        service = db.session.query(InferenceService).filter_by(id=service_id).first()
        if service:
            self.delete_old_service(service.name, service.project.cluster)
            service.model_status='offline'
            if not service.deploy_history:
                service.deploy_history=''
            service.deploy_history = service.deploy_history + "\n" + "clear: %s %s" % (g.user.username,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            db.session.commit()
            flash('服务清理完成', category='success')
        return redirect(conf.get('MODEL_URLS',{}).get('inferenceservice',''))


    @expose('/deploy/debug/<service_id>',methods=['POST',"GET"])
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


    @expose('/deploy/update/', methods=['POST','GET'])
    # @pysnooper.snoop(watch_explode=('deploy'))
    def update_service(self):
        args = request.json if request.json else {}
        namespace = conf.get('SERVICE_NAMESPACE', 'service')
        args.update(request.args)
        service_id = int(args.get('service_id',0))
        service_name = args.get('service_name', '')
        model_name = args.get('model_name', '')
        model_version = args.get('model_version', '')
        service=None

        if service_id:
            service = db.session.query(InferenceService).filter_by(id=service_id).first()
        elif service_name:
            service = db.session.query(InferenceService).filter_by(name=service_name).first()
        elif model_name:
            if model_version:
                service = db.session.query(InferenceService)\
                    .filter(InferenceService.model_name == model_name)\
                    .filter(InferenceService.model_version == model_version)\
                    .filter(InferenceService.model_status == 'online')\
                    .order_by(InferenceService.id.desc()).first()
            else:
                service = db.session.query(InferenceService)\
                    .filter(InferenceService.model_name==model_name)\
                    .filter(InferenceService.model_status=='online')\
                    .order_by(InferenceService.id.desc()).first()

        if service:
            status=0
            message='success'
            if request.method=='POST':
                min_replicas = int(args.get('min_replicas',0))
                if min_replicas:
                    service.min_replicas = min_replicas
                    if service.max_replicas < min_replicas:
                        service.max_replicas=min_replicas
                    db.session.commit()
                    try:
                        self.deploy(service.id)
                    except Exception as e:
                        print(e)
                        status=-1
                        message=str(e)
                time.sleep(3)


            from myapp.utils.py.py_k8s import K8s
            k8s_client = K8s(service.project.cluster.get('KUBECONFIG',''))
            deploy=None
            try:
                deploy = k8s_client.AppsV1Api.read_namespaced_deployment(name=service.name,namespace=namespace)
            except Exception as e:
                print(e)
                status=-1,
                message=str(e)

            back={
                "result": {
                    "service":service.to_json(),
                    "deploy":deploy.to_dict() if deploy else {}
                },
                "status": status,
                "message": message
            }

            return jsonify(back)

        else:
            return jsonify({
                "result":"",
                "status":-1,
                "message":"service not exist or service not online"
            })

    # @pysnooper.snoop()
    def deploy(self,service_id,env='prod'):
        service = db.session.query(InferenceService).filter_by(id=service_id).first()
        namespace = conf.get('SERVICE_NAMESPACE','service')
        name = service.name
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

        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])


        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service.project.cluster.get('KUBECONFIG',''))

        config_datas = service.inference_config.strip().split("\n---") if service.inference_config else []
        config_datas = [x.strip() for x in config_datas if x.strip()]
        volume_mount = service.volume_mount
        print('文件个数：',len(config_datas))
        config_data={}
        for data in config_datas:
            file_name = re.sub('^-*', '',data.split('\n')[0]).strip()
            file_content = '\n'.join(data.split('\n')[1:])
            if file_name and file_content:
                config_data[file_name] = file_content
        if config_data:
            print('create configmap')
            k8s_client.create_configmap(namespace=namespace,name=name,data=config_data,labels={'app':name})
            volume_mount += ",%s(configmap):/config/"%name
        ports = [int(port) for port in service.ports.split(',')]


        pod_env = service.env
        pod_env += "\nKUBEFLOW_ENV=" + env
        pod_env += '\nKUBEFLOW_MODEL_PATH=' + service.model_path if service.model_path else ''
        pod_env += '\nKUBEFLOW_MODEL_VERSION=' + service.model_version
        pod_env += '\nKUBEFLOW_MODEL_IMAGES=' + service.images
        pod_env += '\nKUBEFLOW_MODEL_NAME=' + service.model_name
        pod_env += '\nKUBEFLOW_AREA=' + json.loads(service.project.expand).get('area', 'guangzhou')
        pod_env += "\nRESOURCE_CPU=" + service.resource_cpu
        pod_env += "\nRESOURCE_MEMORY=" + service.resource_memory
        pod_env = pod_env.strip(',')

        if env=='test' or env =='debug':
            try:
                print('delete deployment')
                k8s_client.delete_deployment(namespace=namespace,name=name)
            except Exception as e:
                print(e)
        # 因为所有的服务流量通过ingress实现，所以没有isito的envoy代理
        labels = {"app":name,"user":service.created_by.username,'pod-type':"inference"}

        try:
            pod_ports = copy.deepcopy(ports)
            try:
                if service.metrics.strip():
                    metrics_port = int(service.metrics[:service.metrics.index(":")])
                    pod_ports.append(metrics_port)
            except Exception as e:
                print(e)

            try:
                if service.health.strip():
                    health_port = int(service.health[:service.health.index(":")])
                    pod_ports.append(health_port)
            except Exception as e:
                print(e)

            pod_ports = list(set(pod_ports))
            print('create deployment')
            annotations={}
            # https://istio.io/latest/docs/reference/config/annotations/
            if service.sidecar and 'istio' in service.sidecar and service.service_type=='serving':
                labels['sidecar.istio.io/inject']='true'

            k8s_client.create_deployment(
                namespace=namespace,
                name=name,
                replicas=deployment_replicas,
                labels=labels,
                annotations=annotations,
                command=['sh','-c',command] if command else None,
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
                env=pod_env,
                privileged=False,
                accounts=None,
                username=service.created_by.username,
                ports=pod_ports,
                health=service.health if ':' in service.health and env!='debug' else None
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
        print('deploy service')
        k8s_client.create_service(
            namespace=namespace,
            name=name,
            username=service.created_by.username,
            ports=ports,
            annotations=annotations,
            selector=labels
        )
        # 如果域名配置的gateway，就用这个
        host = service.name+"."+ service.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN',''))

        if service.host:
            host=service.host.replace('http://','').replace('https://','').strip()
            if "/" in host:
                host = host[:host.index("/")]

        # 前缀来区分不同的环境服务
        if env=='debug' or env=='test':
            host=env+'.'+host
        try:
            print('deploy istio ingressgateway')
            k8s_client.create_istio_ingress(
                namespace=namespace,
                name=name,
                host = host,
                ports=service.ports.split(','),
                canary=service.canary,
                shadow=service.shadow
            )
        except Exception as e:
            print(e)

        # 以ip形式访问的话，使用的代理ip。不然不好处理机器服务化机器扩容和缩容时ip变化

        SERVICE_EXTERNAL_IP=[]
        # 使用项目组ip
        if service.project.expand:
            ip = json.loads(service.project.expand).get('SERVICE_EXTERNAL_IP', '')
            if ip and type(ip) == str:
                SERVICE_EXTERNAL_IP = [ip]
            if ip and type(ip) == list:
                SERVICE_EXTERNAL_IP = ip

        # 使用全局ip
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', None)

        # 使用当前ip
        if not SERVICE_EXTERNAL_IP:
            ip = request.host[:request.host.rindex(':')] if ':' in request.host else request.host # 如果捕获到端口号，要去掉
            if core.checkip(ip):
                SERVICE_EXTERNAL_IP=[ip]


        if SERVICE_EXTERNAL_IP:
            # 对于多网卡模式，或者单域名模式，代理需要配置内网ip，界面访问需要公网ip或域名
            SERVICE_EXTERNAL_IP = [ip.split('|')[0].strip() for ip in SERVICE_EXTERNAL_IP]

            service_ports = [[20000+10*service.id+index,port] for index,port in enumerate(ports)]
            service_external_name = (service.name + "-external").lower()[:60].strip('-')
            print('deploy proxy ip')
            k8s_client.create_service(
                namespace=namespace,
                name=service_external_name,
                username=service.created_by.username,
                ports=service_ports,
                selector=labels,
                external_ip=SERVICE_EXTERNAL_IP
            )



        if env=='prod':
            hpas = re.split(',|;', service.hpa)
            regex = re.compile(r"\(.*\)")
            if float(regex.sub('', service.resource_gpu))<1:
                for hpa in copy.deepcopy(hpas):
                    if 'gpu' in hpa:
                        hpas.remove(hpa)

            # 伸缩容
            if int(service.max_replicas)>int(service.min_replicas) and service.hpa:
                try:
                    # 创建+绑定deployment
                    print('create hpa')
                    k8s_client.create_hpa(
                        namespace=namespace,
                        name=name,
                        min_replicas=int(service.min_replicas),
                        max_replicas=int(service.max_replicas),
                        hpa=','.join(hpas)
                    )
                except Exception as e:
                    flash('hpa:'+str(e),'warning')
            else:
                k8s_client.delete_hpa(namespace=namespace,name=name)

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
        service.deploy_history=service.deploy_history+"\n"+"deploy %s: %s %s"%(env,g.user.username,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        service.deploy_history = '\n'.join(service.deploy_history.split("\n")[-10:])
        db.session.commit()
        if env=="debug":
            time.sleep(2)
            pods = k8s_client.get_pods(namespace=namespace,labels={"app":name})
            if pods:
                pod = pods[0]
                print('deploy debug success')
                return redirect("/k8s/web/debug/%s/%s/%s/%s" % (service.project.cluster['NAME'], namespace, pod['name'],name))

        # 生产环境才有域名代理灰度的问题
        if env=='prod':
            from myapp.tasks.async_task import upgrade_service
            kwargs = {
                "service_id": service.id,
                "name":service.name,
                "namespace":namespace
            }
            upgrade_service.apply_async(kwargs=kwargs)

        flash('服务部署完成，正在进行同域名服务版本切换', category='success')
        print('deploy prod success')
        return redirect(conf.get('MODEL_URLS',{}).get('inferenceservice',''))


    @action(
        "copy", __("Copy service"), confirmation=__('Copy Service'), icon="fa-copy",multiple=True, single=False
    )
    def copy(self, services):
        if not isinstance(services, list):
            services = [services]
        try:
            for service in services:
                new_services = service.clone()
                index=1
                model_version = datetime.datetime.now().strftime('v%Y.%m.%d.1')
                while True:
                    model_version = datetime.datetime.now().strftime('v%Y.%m.%d.'+str(index))
                    exits_service = db.session.query(InferenceService).filter_by(model_version=model_version).filter_by(model_name=new_services.model_name).first()
                    if exits_service:
                        index+=1
                    else:
                        break

                new_services.model_version=model_version
                new_services.name = new_services.service_type+"-"+new_services.model_name+"-"+new_services.model_version.replace('v','').replace('.','')
                new_services.created_on = datetime.datetime.now()
                new_services.changed_on = datetime.datetime.now()
                db.session.add(new_services)
                db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            raise e
        return redirect(request.referrer)



class InferenceService_ModelView(InferenceService_ModelView_base,MyappModelView):
    datamodel = SQLAInterface(InferenceService)


appbuilder.add_view_no_menu(InferenceService_ModelView)

# 添加api
class InferenceService_ModelView_Api(InferenceService_ModelView_base,MyappModelRestApi):
    datamodel = SQLAInterface(InferenceService)
    route_base = '/inferenceservice_modelview/api'


    def set_columns_related(self,exist_add_args,response_add_columns):
        exist_service_type = exist_add_args.get('service_type','')
        service_model_path = {
            "tfserving": "/mnt/.../saved_model",
            "torch-server": "/mnt/.../$model_name.mar",
            "onnxruntime": "/mnt/.../$model_name.onnx",
            "triton-server": "onnx:/mnt/.../model.onnx(model.plan,model.bin,model.savedmodel/,model.pt,model.dali)"
        }
        response_add_columns['images']['values'] = [{"id":x,"value":x} for x in conf.get('INFERNENCE_IMAGES',{}).get(exist_service_type,[])]
        response_add_columns['model_path']['default']=service_model_path.get(exist_service_type,'')
        response_add_columns['command']['default'] = conf.get('INFERNENCE_COMMAND',{}).get(exist_service_type,'')
        response_add_columns['env']['default'] = '\n'.join(conf.get('INFERNENCE_ENV',{}).get(exist_service_type,[]))
        response_add_columns['ports']['default'] = conf.get('INFERNENCE_PORTS',{}).get(exist_service_type,'80')
        response_add_columns['metrics']['default'] = conf.get('INFERNENCE_METRICS',{}).get(exist_service_type,'')
        response_add_columns['health']['default'] = conf.get('INFERNENCE_HEALTH',{}).get(exist_service_type,'')

        # if exist_service_type!='triton-server' and "inference_config" in response_add_columns:
        #     del response_add_columns['inference_config']


appbuilder.add_api(InferenceService_ModelView_Api)





