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
        return Markup(f'<a href="/service_modelview/deploy/{self.id}">部署</a>')

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

    @property
    def link(self):
        namespace = conf.get('SERVICE_NAMESPACE')
        # hosts = '%s.%s:%s' % (self.name, namespace, self.ports)

        hosts='/service_modelview/link/%s'%self.id
        return Markup(f'<a href="{hosts}">{hosts}</a>')

    def __repr__(self):
        return self.name

    @property
    def ip(self):

        # 优先使用项目组配置的代理ip
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None
        if SERVICE_EXTERNAL_IP:
            host = SERVICE_EXTERNAL_IP+":"+str(30000+10*self.id)
            return Markup(f'<a target=_blank href="http://{host}/">{host}</a>')

        # 再使用全局配置代理ip
        SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP',[])

        if SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP[0]
            host = SERVICE_EXTERNAL_IP + ":" + str(30000 + 10 * self.id)
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
    volume_mount = Column(String(200),default='')   # 挂载
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
    expand = Column(Text(65536), default='')
    canary = Column(String(400), default='')   # 分流
    shadow = Column(String(400), default='')  # 镜像

    run_id = Column(String(100),nullable=True)   # 可能同一个pipeline产生多个模型
    run_time = Column(String(100))
    deploy_history = Column(Text(65536), default='')  # 部署记录



    @property
    def model_name_url(self):
        if 'kfserving' not in self.service_type:
            url = self.project.cluster['K8S_DASHBOARD_CLUSTER'] + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))
        else:
            url = self.project.cluster['K8S_DASHBOARD_CLUSTER'] + '#/search?namespace=%s&q=%s' % (conf.get('KFSERVING_NAMESPACE'), self.name.replace('_', '-'))
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
        dom=f'''
        <a target=_blank href="/inferenceservice_modelview/debug/{self.id}">调试</a> | 
        <a href="/inferenceservice_modelview/deploy/test/{self.id}">部署测试</a> | 
        <a href="/inferenceservice_modelview/deploy/prod/{self.id}">部署生产</a> |
        <a target=_blank href="{url}">监控</a> |
        <a href="/inferenceservice_modelview/clear/{self.id}">清理</a>
        '''
        return Markup(dom)

    @property
    def output_html(self):
        return Markup('<pre><code>' + self.model_output + '</code></pre>')

    @property
    def metric_html(self):
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

    def __repr__(self):
        return self.name



    @property
    def inference_host_url(self):
        if 'kfserving' in self.service_type:
            url = "http://" + self.name + "." + self.project.cluster.get('KFSERVING_DOMAIN',conf.get('KFSERVING_DOMAIN'))
        else:
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


# 蓝绿部署（全新机器，流量切分）、金丝雀发布（灰度发布）（逐步替换旧机器）、A/B测试的准确定义（多个成熟的服务，看效果）
class KfService(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'kfservice'

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))  # 定义外键
    project = relationship(
        Project, foreign_keys=[project_id]
    )
    name = Column(String(100), nullable=False)   # 用于产生pod  service和location
    label = Column(String(100), nullable=False)   # 别名
    service_type= Column(Enum('predictor','transformer','explainer'),nullable=False,default='predictor')
    default_service_id = Column(Integer, ForeignKey('service.id'))  # 定义外键
    default_service = relationship(
        Service, foreign_keys=[default_service_id]
    )
    canary_service_id = Column(Integer, ForeignKey('service.id'))  # 定义外键
    canary_service = relationship(
        Service, foreign_keys=[canary_service_id]
    )
    canary_traffic_percent = Column(Integer,default=0)

    label_columns_spec = {
        "default_service": "默认服务",
        "canary_service": "灰度服务",
        "canary_traffic_percent": "灰度流量",
    }
    label_columns = MyappModelBase.label_columns.copy()
    label_columns.update(label_columns_spec)

    @property
    def host(self):
        domain = conf.get('KFSERVING_DOMAIN')
        hosts = '%s.%s'%(self.name,domain)
        if self.default_service:
            hosts += '<br>'
            hosts += '%s-%s-default.%s (%s%%)'%(self.name,self.service_type,domain,100-self.canary_traffic_percent)
        if self.canary_service:
            hosts += '<br>'
            hosts += '%s-%s-canary.%s  (%s%%)' % (self.name, self.service_type, domain,self.canary_traffic_percent)
        return Markup(f'<a href="">{hosts}</a>')

    @property
    def service(self):
        url = ''
        if self.default_service:
            url += '<a href="/service_modelview/edit/%s">default(%s)</a>'%(self.default_service.id,self.default_service.name)
        if self.canary_service:
            if url:
                url += '<br>'
            url += '<a href="/service_modelview/edit/%s">canary(%s)</a>'%(self.canary_service.id,self.canary_service.name)
        return Markup(f'%s'%url)


    @property
    def deploy(self):
        return Markup(f'<a href="/kfservice_modelview/deploy/{self.id}">部署</a>')

    @property
    def roll(self):
        return Markup(f'<a href="/kfservice_modelview/roll/{self.id}">灰度</a>')

    @property
    def label_url(self):
        # user_roles = [role.name.lower() for role in list(g.user.roles)]
        # if "admin" in user_roles:
        url = self.project.cluster['K8S_DASHBOARD_CLUSTER'] + '#/search?namespace=%s&q=%s' % (conf.get('KFSERVING_NAMESPACE'), self.name.replace('_', '-'))
        # else:
        #     url = conf.get('K8S_DASHBOARD_PIPELINE', '') + '#/search?namespace=%s&q=%s' % (conf.get('KFSERVING_NAMESPACE'), self.name.replace('_', '-'))

        return Markup(f'<a target=_blank href="{url}">{self.label}</a>')

    @property
    def k8s_yaml(self):
        k8s = K8s(self.project.cluster['KUBECONFIG'])
        namespace = conf.get('KFSERVING_NAMESPACE')
        crd_info = conf.get('CRD_INFO')['inferenceservice']
        crd_yaml = k8s.get_one_crd_yaml(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,name=self.name)
        return Markup('<pre><code>' + crd_yaml + '</code></pre>')

    @property
    def status(self):
        try:
            k8s = K8s(self.project.cluster['KUBECONFIG'])
            namespace = conf.get('KFSERVING_NAMESPACE')
            crd_info = conf.get('CRD_INFO')['inferenceservice']
            crd = k8s.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=self.name)
            if crd:
                status_more = json.loads(crd['status_more'])
            # print(status_more)
                url = crd['status']+":"+str(status_more.get('traffic',0))+"%"
                return Markup(f'%s'%url)

        except Exception as e:
            print(e)
        return 'unknown'


    def __repr__(self):
        return self.name