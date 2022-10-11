from flask_appbuilder import Model
from sqlalchemy.orm import relationship
import json
from sqlalchemy import (
    Text,
    Enum,
)
from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable
from flask import g, request
from myapp import app

from sqlalchemy import Column, Integer, String, ForeignKey
from flask import Markup
import datetime
metadata = Model.metadata
conf = app.config
from myapp.utils.py import py_k8s



class Notebook(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'notebook'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False)  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    name = Column(String(200), unique = True, nullable=True)
    describe = Column(String(200), nullable=True)
    namespace = Column(String(200), nullable=True,default='jupyter')
    images=Column(String(200), nullable=True,default='')
    ide_type = Column(String(100), default='jupyter')
    working_dir = Column(String(200), default='')  # 挂载
    volume_mount = Column(String(400), default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives')  # 挂载
    node_selector = Column(String(200), default='cpu=true,notebook=true')  # 挂载
    image_pull_policy = Column(Enum('Always', 'IfNotPresent'), nullable=True, default='Always')
    resource_memory = Column(String(100), default='10G')
    resource_cpu = Column(String(100), default='10')
    resource_gpu = Column(String(100), default='0')
    expand = Column(Text(65536), default='')

    def __repr__(self):
        return self.name

    @property
    def name_url(self):
        if self.ide_type=='theia':
            url = "/notebook/"+self.namespace + "/" + self.name+"/" + "#"+self.mount
        else:
            url = "/notebook/"+self.namespace + "/" + self.name+"/lab?#"+self.mount

        # url= url + "#"+self.mount
        JUPYTER_DOMAIN = self.project.cluster.get('JUPYTER_DOMAIN',request.host)
        if JUPYTER_DOMAIN:
            host = "http://"+JUPYTER_DOMAIN
        else:
            host = request.host_url.strip('/') # 使用当前域名打开

        # 对于有边缘节点，直接使用边缘集群的代理ip
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None
        if SERVICE_EXTERNAL_IP:
            service_ports = 10000 + 10 * self.id
            host = "http://%s:%s"%(SERVICE_EXTERNAL_IP,str(service_ports))
            if self.ide_type=='theia':
                url = "/" + "#/mnt/" + self.created_by.username
            else:
                url = '/notebook/jupyter/%s/lab/tree/mnt/%s'%(self.name,self.created_by.username)
        return Markup(f'<a target=_blank href="{host}{url}">{self.name}</a>')

    @property
    def mount(self):
        # if "(hostpath)" in self.volume_mount:
        #     mount = self.volume_mount[self.volume_mount.index('(hostpath)'):]
        #     mount=mount.replace('(hostpath):','')
        #     if ',' in mount:
        #         mount = mount[:mount.index(',')]
        #         return mount
        #     else:
        #         return mount
        # else:
        if self.created_by:
            return "/mnt/%s"% self.created_by.username
        else:
            return "/mnt/%s"%g.user.username


    @property
    def resource(self):
        return self.resource_cpu+"(cpu)"+self.resource_memory+"(memory)"+self.resource_gpu+"(gpu)"

    @property
    def status(self):
        try:
            k8s_client = py_k8s.K8s(self.cluster.get('KUBECONFIG',''))
            namespace = conf.get('NOTEBOOK_NAMESPACE')
            pods = k8s_client.get_pods(namespace=namespace,pod_name=self.name)
            status = pods[0]['status']
            if g.user.is_admin():
                k8s_dash_url = self.cluster.get('K8S_DASHBOARD_CLUSTER') + "#/search?namespace=jupyter&q=" + self.name
                url = Markup(f'<a target=_blank href="{k8s_dash_url}">{status}</a>')
                return url
            return status

        except Exception:
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


