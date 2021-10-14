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
from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from flask import escape, g, Markup, request
from myapp import app,db
from myapp.models.helpers import ImportMixin
# 添加自定义model
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
import datetime
metadata = Model.metadata
conf = app.config
from myapp.utils.py import py_k8s


# 定义model
class Notebook(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'notebook'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False)  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    name = Column(String(200), unique = True, nullable=True)
    describe = Column(Text)
    namespace = Column(String(200), nullable=True,default='jupyter')
    images=Column(String(200), nullable=True,default='')
    ide_type = Column(String(200), default='jupyter')
    working_dir = Column(String(200), default='')  # 挂载
    volume_mount = Column(String(400), default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives')  # 挂载
    node_selector = Column(String(200), default='cpu=true,notebook=true')  # 挂载
    image_pull_policy = Column(Enum('Always', 'IfNotPresent'), nullable=True, default='Always')
    resource_memory = Column(String(100), default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu = Column(String(100), default='0')

    def __repr__(self):
        return self.name

    @property
    def name_url(self):
        if self.ide_type=='theia':
            url = "/notebook/"+self.namespace + "/" + self.name+"/"
        else:
            url = "/notebook/"+self.namespace + "/" + self.name+"/lab?"

        url=url + "#"+self.mount
        # if "(pvc)" not in self.volume_mount:
        #     url = url + "#/mnt"
        # else:
        #     if self.created_by:
        #         url = url + "#/mnt/%s"%self.created_by.username
        host = "http://"+self.cluster['JUPYTER_DOMAIN']
        return Markup(f'<a target=_blank href="{host}{url}">{self.name}</a>')

    @property
    def mount(self):
        if "(hostpath)" in self.volume_mount:
            mount = self.volume_mount[self.volume_mount.index('(hostpath)'):]
            mount=mount.replace('(hostpath):','')
            if ',' in mount:
                mount = mount[:mount.index(',')]
                return mount
            else:
                return mount
        else:
            if self.created_by:
                return "/mnt/%s"% self.created_by.username
            else:
                return "/mnt/"


    @property
    def resource(self):
        return self.resource_cpu+"(cpu)"+self.resource_memory+"(memory)"+self.resource_gpu+"(gpu)"

    @property
    def status(self):
        try:
            k8s_client = py_k8s.K8s(self.cluster['KUBECONFIG'])
            namespace = conf.get('NOTEBOOK_NAMESPACE')
            pods = k8s_client.get_pods(namespace=namespace,pod_name=self.name)
            status = pods[0]['status']
            if g.user.is_admin():
                k8s_dash_url = self.cluster.get('K8S_DASHBOARD_CLUSTER') + "#/search?namespace=jupyter&q=" + self.name
                url = Markup(f'<a target=_blank href="{k8s_dash_url}">{status}</a>')
                return url
            return status

        except Exception as e:
            # print(e)
            return "unknown"

    @property
    def cluster(self):
        if self.project:
            return self.project.cluster
        else:
            return conf.get('CLUSTERS')[conf.get('ENVIRONMENT')]

    # 清空激活
    @property
    def renew(self):
        object_info = conf.get("CRD_INFO", {}).get('notebook', {})
        timeout = int(object_info.get('timeout', 60 * 60 * 24 * 3))

        end = self.changed_on+datetime.timedelta(seconds=timeout)
        end = end.strftime('%Y-%m-%d')
        return Markup(f'<a href="/notebook_modelview/renew/{self.id}">截至 {end}</a>')


    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'notebook')


    # 清空激活
    @property
    def reset(self):
        return Markup(f'<a href="/notebook_modelview/reset/{self.id}">reset</a>')


