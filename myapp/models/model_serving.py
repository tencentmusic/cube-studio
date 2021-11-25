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



class Service(Model,AuditMixinNullable,MyappModelBase):
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
    min_replicas = Column(Integer,default=1)
    max_replicas = Column(Integer, default=1)
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
    def monitoring_url(self):
        # return Markup(f'<a href="/service_modelview/clear/{self.id}">清理</a>')
        url=self.project.cluster.get('GRAFANA_SERVICE','')+self.name
        return Markup(f'<a href="{url}">监控</a>')
        # https://www.angularjswiki.com/fontawesome/fa-flask/    <i class="fa-solid fa-monitor-waveform"></i>


    @property
    def name_url(self):
        # user_roles = [role.name.lower() for role in list(g.user.roles)]
        # if "admin" in user_roles:
        url = self.project.cluster['K8S_DASHBOARD_CLUSTER'] + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))
        # else:
        #     url = conf.get('K8S_DASHBOARD_PIPELINE', '') + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))

        return Markup(f'<a target=_blank href="{url}">{self.label}</a>')

    @property
    def host_url(self):
        url = "http://" + self.name + "." + conf.get('SERVICE_DOMAIN')
        if self.host:
            if 'http://' in self.host or 'https://' in self.host:
                url = self.host
            else:
                url = "http://"+self.host

        return Markup(f'<a target=_blank href="{url}">{url}</a>')

    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'service')



    @property
    def link(self):
        namespace = conf.get('SERVICE_NAMESPACE')
        # hosts = '%s.%s:%s' % (self.name, namespace, self.ports)

        hosts='/service_modelview/link/%s'%self.id
        return Markup(f'<a href="{hosts}">{hosts}</a>')

    def __repr__(self):
        return self.name

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