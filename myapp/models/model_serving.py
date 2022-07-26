from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, ForeignKey,Float
from sqlalchemy.orm import relationship
import datetime,time,json
from sqlalchemy import (
    Boolean,
    Column,
    create_engine,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Enum,
)
import pysnooper
from myapp.utils import core
import re
from myapp.utils.py.py_k8s import K8s
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from flask import escape, g, Markup, request
from .model_team import Project
from .model_job import Pipeline
from myapp import app,db
from myapp.models.base import MyappModelBase
from myapp.models.helpers import ImportMixin
# 添加自定义model
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
import datetime
metadata = Model.metadata
conf = app.config



class service_common():
    @property
    def monitoring_url(self):
        # return Markup(f'<a href="/service_modelview/clear/{self.id}">清理</a>')
        url=self.project.cluster.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_SERVICE_PATH')+self.name
        return Markup(f'<a href="{url}">监控</a>')
        # https://www.angularjswiki.com/fontawesome/fa-flask/    <i class="fa-solid fa-monitor-waveform"></i>



    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'service')



class Service(Model,AuditMixinNullable,MyappModelBase,service_common):
    __tablename__ = 'service'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))  # 定义外键
    project = relationship(
        Project, foreign_keys=[project_id]
    )

    name = Column(String(100), nullable=False,unique=True)   # 用于产生pod  service和location
    label = Column(String(100), nullable=False)   # 别名
    images = Column(String(200), nullable=False)   # 别名
    working_dir = Column(String(100),default='')
    command = Column(String(1000),default='')
    args = Column(Text,default='')
    env = Column(Text,default='')  # 默认自带的环境变量
    volume_mount = Column(String(200),default='')   # 挂载
    node_selector = Column(String(100),default='cpu=true,serving=true')   # 挂载
    replicas = Column(Integer,default=1)
    ports = Column(String(100),default='80')   # 挂载
    resource_memory = Column(String(100),default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu= Column(String(100), default='0')
    deploy_time = Column(String(100), nullable=False,default=datetime.datetime.now)
    host = Column(String(200), default='')  # 挂载
    expand = Column(Text(65536), default='')


    @property
    def deploy(self):
        url = self.project.cluster.get('GRAFANA_HOST', '').strip('/') + conf.get('GRAFANA_SERVICE_PATH') + self.name
        return Markup(f'<a href="/service_modelview/deploy/{self.id}">部署</a> | <a href="{url}">监控</a> | <a href="/service_modelview/clear/{self.id}">清理</a>')

    @property
    def clear(self):
        return Markup(f'<a href="/service_modelview/clear/{self.id}">清理</a>')


    @property
    def name_url(self):
        # user_roles = [role.name.lower() for role in list(g.user.roles)]
        # if "admin" in user_roles:
        url = self.project.cluster['K8S_DASHBOARD_CLUSTER'] + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))
        # else:
        #     url = conf.get('K8S_DASHBOARD_PIPELINE', '') + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))

        return Markup(f'<a target=_blank href="{url}">{self.label}</a>')

    def __repr__(self):
        return self.name

    @property
    def ip(self):
        port = 30000+10*self.id
        # 优先使用项目组配置的代理ip
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None
        if not SERVICE_EXTERNAL_IP:
            # 再使用全局配置代理ip
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP',[])
            if SERVICE_EXTERNAL_IP:
                SERVICE_EXTERNAL_IP=SERVICE_EXTERNAL_IP[0]

        if not SERVICE_EXTERNAL_IP:
            ip = request.host[:request.host.rindex(':')] if ':' in request.host else request.host # 如果捕获到端口号，要去掉
            if core.checkip(ip):
                SERVICE_EXTERNAL_IP = ip

        if SERVICE_EXTERNAL_IP:
            host = SERVICE_EXTERNAL_IP + ":" + str(port)
            return Markup(f'<a target=_blank href="http://{host}/">{host}</a>')
        else:
            return "未开通"

    @property
    def host_url(self):
        url = "http://" + self.name + "." + conf.get('SERVICE_DOMAIN')
        if self.host:
            if 'http://' in self.host or 'https://' in self.host:
                url = self.host
            else:
                url = "http://"+self.host

        return Markup(f'<a target=_blank href="{url}">{url}</a>')


class InferenceService(Model,AuditMixinNullable,MyappModelBase,service_common):
    __tablename__ = 'inferenceservice'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))  # 定义外键
    project = relationship(
        Project, foreign_keys=[project_id]
    )

    name = Column(String(100), nullable=True,unique=True)   # 用于产生pod  service和location
    label = Column(String(100), nullable=False)   # 别名

    service_type= Column(String(100),nullable=True,default='serving')
    model_name = Column(String(200),default='')
    model_version = Column(String(200),default='')
    model_path = Column(String(200),default='')
    model_type = Column(String(200),default='')
    model_input = Column(Text(65536), default='')
    model_output = Column(Text(65536), default='')
    model_status = Column(String(200),default='offline')
    # model_status = Column(Enum('offline','test','online','delete'),nullable=True,default='offline')

    transformer=Column(String(200),default='')  # 前后置处理逻辑的文件

    images = Column(String(200), nullable=False)   # 别名
    working_dir = Column(String(100),default='')
    command = Column(String(1000),default='')
    args = Column(Text,default='')
    env = Column(Text,default='')  # 默认自带的环境变量
    volume_mount = Column(String(2000),default='')   # 挂载
    node_selector = Column(String(100),default='cpu=true,serving=true')   # 挂载
    min_replicas = Column(Integer,default=1)
    max_replicas = Column(Integer, default=1)
    hpa = Column(String(400), default='')  # 弹性伸缩
    metrics = Column(Text(65536), default='')  # 指标监控
    health = Column(String(400), default='')  # 健康检查
    sidecar = Column(String(400), default='')  # sidecar的名称
    ports = Column(String(100),default='80')   # 挂载
    resource_memory = Column(String(100),default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu= Column(String(100), default='0')
    deploy_time = Column(String(100), nullable=True,default=datetime.datetime.now)
    host = Column(String(200), default='')  # 挂载
    expand = Column(Text(65536), default='{}')
    canary = Column(String(400), default='')   # 分流
    shadow = Column(String(400), default='')  # 镜像

    run_id = Column(String(100),nullable=True)   # 可能同一个pipeline产生多个模型
    run_time = Column(String(100))
    deploy_history = Column(Text(65536), default='')  # 部署记录



    @property
    def model_name_url(self):
        url = self.project.cluster['K8S_DASHBOARD_CLUSTER'] + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))
        return Markup(f'<a target=_blank href="{url}">{self.model_name}</a>')

    @property
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')

    @property
    def input_html(self):
        return Markup('<pre><code>' + self.model_input + '</code></pre>')

    @property
    def operate_html(self):
        url=self.project.cluster.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_SERVICE_PATH')+self.name
        # if self.created_by.username==g.user.username or g.user.is_admin():
        dom = f'''
                <a target=_blank href="/inferenceservice_modelview/deploy/debug/{self.id}">调试</a> | 
                <a href="/inferenceservice_modelview/deploy/test/{self.id}">部署测试</a> | 
                <a href="/inferenceservice_modelview/deploy/prod/{self.id}">部署生产</a> |
                <a target=_blank href="{url}">监控</a> |
                <a href="/inferenceservice_modelview/clear/{self.id}">清理</a>
                '''
        # else:
        #     dom = f'''调试 | 部署测试 | 部署生产 | <a target=_blank href="{url}">监控</a> | 清理'''
        #     # dom = f'''调试 | 部署测试</a> | 部署生产 | 监控 | 清理 '''
        return Markup(dom)

    @property
    def output_html(self):
        return Markup('<pre><code>' + self.model_output + '</code></pre>')

    @property
    def metrics_html(self):
        return Markup('<pre><code>' + self.model_output + '</code></pre>')

    @property
    def debug(self):
        return Markup(f'<a target=_blank href="/inferenceservice_modelview/debug/{self.id}">调试</a>')

    @property
    def test_deploy(self):
        return Markup(f'<a href="/inferenceservice_modelview/deploy/test/{self.id}">部署测试</a>')

    @property
    def deploy(self):
        return Markup(f'<a href="/inferenceservice_modelview/deploy/prod/{self.id}">部署生产</a>')

    @property
    def clear(self):
        return Markup(f'<a href="/inferenceservice_modelview/clear/{self.id}">清理</a>')



    @property
    def ip(self):

        port = 20000+10*self.id
        # 优先使用项目组配置的代理ip
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None
        if not SERVICE_EXTERNAL_IP:
            # 再使用全局配置代理ip
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', [])
            if SERVICE_EXTERNAL_IP:
                SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP[0]

        if not SERVICE_EXTERNAL_IP:
            ip = request.host[:request.host.rindex(':')] if ':' in request.host else request.host  # 如果捕获到端口号，要去掉
            if core.checkip(ip):
                SERVICE_EXTERNAL_IP = ip

        if SERVICE_EXTERNAL_IP:
            host = SERVICE_EXTERNAL_IP + ":" + str(port)
            return Markup(f'<a target=_blank href="http://{host}/">{host}</a>')
        else:
            return "未开通"


    def __repr__(self):
        return self.name



    @property
    def inference_host_url(self):
        url = "http://" + self.name + "." + self.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN'))
        if self.host:
            if 'http://' in self.host or 'https://' in self.host:
                url = self.host
            else:
                url = "http://"+self.host

        link = url
        if self.service_type=='tfserving':
            link+="/v1/models/"+self.model_name
        if self.service_type=='torch-server':
            link+=":8080/models"
        hosts=f'''
        <a target=_blank href="{link}">{url}</a>
        <br><a target=_blank href="{link.replace('http://','http://debug.').replace('https://','https://debug.')}">{url.replace('http://','http://debug.').replace('https://','https://debug.')}</a>
        <br><a target=_blank href="{link.replace('http://','http://test.').replace('https://','https://test.')}">{url.replace('http://','http://test.').replace('https://','https://test.')}</a>
        '''

        hosts=f'<a target=_blank href="{link}">{url}</a>'
        return Markup(hosts)



    def clone(self):
        return InferenceService(
            project_id=self.project_id,
            name = self.name+"-copy",
            label = self.label,
            service_type = self.service_type,
            model_name = self.model_name,
            model_version = self.model_version,
            model_path = self.model_path,
            model_type = self.model_type,
            model_input = self.model_input,
            model_output = self.model_output,
            model_status = 'offline',

            transformer = self.transformer,

            images = self.images,
            working_dir = self.working_dir,
            command = self.command,
            args = self.args,
            env = self.env,
            volume_mount =self.volume_mount,
            node_selector = self.node_selector,
            min_replicas = self.min_replicas,
            max_replicas = self.max_replicas,
            hpa = self.hpa,
            metrics = self.metrics,
            health = self.health,
            sidecar = self.sidecar,
            ports = self.ports,
            resource_memory = self.resource_memory,
            resource_cpu = self.resource_cpu,
            resource_gpu = self.resource_gpu,
            deploy_time = '',
            host = self.host,
            expand = '{}',
            canary = '',
            shadow = '',

            run_id = '',
            run_time = '',
            deploy_history = ''

        )
