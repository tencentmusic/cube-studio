from flask_appbuilder import Model
from sqlalchemy.orm import relationship
import json
from sqlalchemy import (
    Text,
    Enum,
)
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.base import MyappModelBase
from myapp.models.model_team import Project
from myapp.models.helpers import AuditMixinNullable
from flask import g, request
from myapp import app

from sqlalchemy import Column, Integer, String, ForeignKey
from flask import Markup
import datetime
import pysnooper
metadata = Model.metadata
conf = app.config
from myapp.utils.py import py_k8s



class Notebook(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'notebook'
    id = Column(Integer, primary_key=True,comment='id主键')
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False,comment='项目组id')  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    name = Column(String(200), unique = True, nullable=True,comment='英文名')
    describe = Column(String(200), nullable=True,comment='描述')
    namespace = Column(String(200), nullable=True,default='jupyter',comment='命名空间')
    images=Column(String(200), nullable=True,default='',comment='镜像')
    ide_type = Column(String(100), default='jupyter',comment='ide类型')
    working_dir = Column(String(200), default='',comment='工作目录')
    env = Column(String(400),default='',comment='环境变量') #
    volume_mount = Column(String(2000), default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives',comment='挂载')  #
    node_selector = Column(String(200), default='cpu=true,notebook=true',comment='机器选择器')  #
    image_pull_policy = Column(Enum('Always', 'IfNotPresent',name='image_pull_policy'), nullable=True, default='Always',comment='镜像拉取策略')
    resource_memory = Column(String(100), default='10G',comment='申请内存')
    resource_cpu = Column(String(100), default='10',comment='申请cpu')
    resource_gpu = Column(String(100), default='0',comment='申请gpu')
    expand = Column(Text(65536), default='{}',comment='扩展参数')

    def __repr__(self):
        return self.name

    @property
    def name_url(self):
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None

        host = "//" + self.project.cluster.get('HOST', request.host)

        expand = json.loads(self.expand) if self.expand else {}
        root = expand.get('root','')

        if self.ide_type=='theia':
            url = "/notebook/"+self.namespace + "/" + self.name+"/" + "#"+self.mount
        elif self.ide_type=='matlab':
            url = "/notebook/"+self.namespace + "/" + self.name+"/index.html"
        elif self.ide_type=='rstudio' and not SERVICE_EXTERNAL_IP:
            url1 = host+"/notebook/" + self.namespace + "/" + self.name + "/auth-sign-in?appUri=%2F"
            url2 = host+"/notebook/" + self.namespace + "/" + self.name+"/"
            a_html='''<a onclick="(function (){window.open('%s','_blank');window.open('%s','_blank')})()">%s</a>'''%(url1,url2,self.name)
            return Markup(a_html)

            # url = "/notebook/" + self.namespace + "/" + self.name+"/"
        else:
            if root:
                url = '/notebook/jupyter/%s/lab/tree/%s' % (self.name,root.lstrip('/'))
            else:
                url = "/notebook/"+self.namespace + "/" + self.name+"/lab?#"+self.mount

        # url= url + "#"+self.mount

        # 对于有边缘节点，直接使用边缘集群的代理ip
        if SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP.split('|')[-1].strip()
            from myapp.utils import core
            meet_ports = core.get_not_black_port(10000 + 10 * self.id)
            host = "//%s:%s"%(SERVICE_EXTERNAL_IP,str(meet_ports[0]))
            if self.ide_type=='theia':
                url = "/" + "#/mnt/" + self.created_by.username
            elif self.ide_type == 'matlab':
                url = "/notebook/" + self.namespace + "/" + self.name + "/index.html"
            elif self.ide_type == 'rstudio':
                url = '/'
            else:
                url = "/notebook/" + self.namespace + "/" + self.name + "/lab?#" + self.mount
                # url = '/notebook/jupyter/%s/lab/tree/mnt/%s'%(self.name,self.created_by.username)
        return Markup(f'<a target=_blank href="{host}{url}">{self.name}</a>')

    @property
    def ide_type_html(self):
        images = dict([[x[0],x[1]] for x in conf.get('NOTEBOOK_IMAGES', [])])
        if self.images in images:
            return images[self.images]
        return self.ide_type

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
            if pods and len(pods)>0:
                status = pods[0]['status']
                if g.user.is_admin():
                    k8s_dash_url = f'/k8s/web/search/{self.cluster["NAME"]}/jupyter/{self.name}'
                    url = Markup(f'<a target=_blank style="color:#008000;" href="{k8s_dash_url}">{status}</a>')
                    return url
                return status

        except Exception as e:
            print(e)

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
        return Markup(f'<a href="/notebook_modelview/api/renew/{self.id}">{__("截至")} {end}</a>')


    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'notebook')


    # 清空激活
    @property
    def reset(self):
        return Markup(f'<a href="/notebook_modelview/api/reset/{self.id}">reset</a>')


    # 镜像保存
    @property
    def save(self):
        return Markup(f'<span style="color:red;">环境保存(企业版)</span>')

