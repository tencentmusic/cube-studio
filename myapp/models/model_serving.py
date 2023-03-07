from flask_appbuilder import Model
from sqlalchemy.orm import relationship
import json
from sqlalchemy import Text
from myapp.utils import core

from myapp.models.helpers import AuditMixinNullable
from flask import request
from .model_team import Project

from myapp import app
from myapp.models.base import MyappModelBase
from sqlalchemy import Column, Integer, String, ForeignKey

from flask import Markup
import datetime
metadata = Model.metadata
conf = app.config



class service_common():
    @property
    def monitoring_url(self):
        # return Markup(f'<a href="/service_modelview/clear/{self.id}">清理</a>')
        url="http://"+self.project.cluster.get('HOST',request.host)+conf.get('GRAFANA_SERVICE_PATH')+self.name
        return Markup(f'<a href="{url}">监控</a>')
        # https://www.angularjswiki.com/fontawesome/fa-flask/    <i class="fa-solid fa-monitor-waveform"></i>



    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'service')



class Service(Model,AuditMixinNullable,MyappModelBase,service_common):
    __tablename__ = 'service'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(
        Project, foreign_keys=[project_id]
    )

    name = Column(String(100), nullable=False,unique=True)   # Used to generate pod service and vs
    label = Column(String(100), nullable=False)
    images = Column(String(200), nullable=False)
    working_dir = Column(String(100),default='')
    command = Column(String(1000),default='')
    args = Column(Text,default='')
    env = Column(Text,default='')
    volume_mount = Column(String(200),default='')
    node_selector = Column(String(100),default='cpu=true,serving=true')
    replicas = Column(Integer,default=1)
    ports = Column(String(100),default='80')
    resource_memory = Column(String(100),default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu= Column(String(100), default='0')
    deploy_time = Column(String(100), nullable=False,default=datetime.datetime.now)
    host = Column(String(200), default='')
    expand = Column(Text(65536), default='{}')

    @property
    def deploy(self):
        monitoring_url = "http://"+self.project.cluster.get('HOST', request.host) + conf.get('GRAFANA_SERVICE_PATH') + self.name
        help_url=''
        try:
            help_url = json.loads(self.expand).get('help_url','') if self.expand else ''
        except Exception as e:
            print(e)

        if help_url:
            return Markup(f'<a target=_blank href="{help_url}">帮助</a> | <a href="/service_modelview/deploy/{self.id}">部署</a> | <a href="{monitoring_url}">监控</a> | <a href="/service_modelview/clear/{self.id}">清理</a>')
        else:
            return Markup(f'帮助 | <a href="/service_modelview/deploy/{self.id}">部署</a> | <a href="{monitoring_url}">监控</a> | <a href="/service_modelview/clear/{self.id}">清理</a>')


    @property
    def clear(self):
        return Markup(f'<a href="/service_modelview/clear/{self.id}">清理</a>')


    @property
    def name_url(self):
        # user_roles = [role.name.lower() for role in list(g.user.roles)]
        # if "admin" in user_roles:
        url = "http://"+self.project.cluster.get('HOST',request.host)+conf.get('K8S_DASHBOARD_CLUSTER') + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))

        return Markup(f'<a target=_blank href="{url}">{self.label}</a>')

    def __repr__(self):
        return self.name

    @property
    def ip(self):
        port = 30000+10*self.id
        # first, Use the proxy ip configured by the project group
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None
        if not SERVICE_EXTERNAL_IP:
            # second, Use the global configuration proxy ip
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP',[])
            if SERVICE_EXTERNAL_IP:
                SERVICE_EXTERNAL_IP=SERVICE_EXTERNAL_IP[0]

        if not SERVICE_EXTERNAL_IP:
            ip = request.host[:request.host.rindex(':')] if ':' in request.host else request.host # remove port in host
            if core.checkip(ip):
                SERVICE_EXTERNAL_IP = ip

        if SERVICE_EXTERNAL_IP:
            # 对于多网卡或者单域名模式，这里需要使用公网ip或者域名打开
            if '|' in SERVICE_EXTERNAL_IP:
                SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP.split('|')[1].strip()

            host = SERVICE_EXTERNAL_IP + ":" + str(port)
            return Markup(f'<a target=_blank href="http://{host}/">{host}</a>')
        else:
            return "未开通"

    @property
    def host_url(self):
        url = "http://" + self.name + "." + self.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN',''))
        if self.host:
            if 'http://' in self.host or 'https://' in self.host:
                url = self.host
            else:
                url = "http://"+self.host

        return Markup(f'<a target=_blank href="{url}">{url}</a>')


class InferenceService(Model,AuditMixinNullable,MyappModelBase,service_common):
    __tablename__ = 'inferenceservice'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(
        Project, foreign_keys=[project_id]
    )

    name = Column(String(100), nullable=True,unique=True)
    label = Column(String(100), nullable=False)

    service_type= Column(String(100),nullable=True,default='serving')
    model_name = Column(String(200),default='')
    model_version = Column(String(200),default='')
    model_path = Column(String(200),default='')
    model_type = Column(String(200),default='')
    model_input = Column(Text(65536), default='')
    model_output = Column(Text(65536), default='')
    inference_config = Column(Text(65536), default='')   # make configmap
    model_status = Column(String(200),default='offline')
    # model_status = Column(Enum('offline','test','online','delete'),nullable=True,default='offline')

    transformer=Column(String(200),default='')  # pre process and post process

    images = Column(String(200), nullable=False)
    working_dir = Column(String(100),default='')
    command = Column(String(1000),default='')
    args = Column(Text,default='')
    env = Column(Text,default='')
    volume_mount = Column(String(2000),default='')
    node_selector = Column(String(100),default='cpu=true,serving=true')
    min_replicas = Column(Integer,default=1)
    max_replicas = Column(Integer, default=1)
    hpa = Column(String(400), default='')
    metrics = Column(Text(65536), default='')
    health = Column(String(400), default='')
    sidecar = Column(String(400), default='')
    ports = Column(String(100),default='80')
    resource_memory = Column(String(100),default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu= Column(String(100), default='0')
    deploy_time = Column(String(100), nullable=True,default=datetime.datetime.now)
    host = Column(String(200), default='')
    expand = Column(Text(65536), default='{}')
    canary = Column(String(400), default='')
    shadow = Column(String(400), default='')

    run_id = Column(String(100),nullable=True)
    run_time = Column(String(100))
    deploy_history = Column(Text(65536), default='')

    priority = Column(Integer,default=1)   # giving priority to meeting high-priority resource needs

    @property
    def model_name_url(self):
        url = "http://"+self.project.cluster.get('HOST',request.host)+conf.get('K8S_DASHBOARD_CLUSTER') + '#/search?namespace=%s&q=%s' % (conf.get('SERVICE_NAMESPACE'), self.name.replace('_', '-'))
        return Markup(f'<a target=_blank href="{url}">{self.model_name}</a>')

    @property
    def replicas_html(self):
        return "%s~%s"%(self.min_replicas,self.max_replicas)


    @property
    def resource(self):
        return 'cpu:%s,memory:%s,gpu:%s'%(self.resource_cpu,self.resource_memory,self.resource_gpu)

    @property
    def operate_html(self):
        help_url=''
        try:
            help_url = json.loads(self.expand).get('help_url','') if self.expand else ''
        except Exception as e:
            print(e)

        monitoring_url="http://"+self.project.cluster.get('HOST', request.host)+conf.get('GRAFANA_SERVICE_PATH')+self.name
        # if self.created_by.username==g.user.username or g.user.is_admin():
        dom = f'''
                <a target=_blank href="/inferenceservice_modelview/deploy/debug/{self.id}">调试</a> | 
                <a href="/inferenceservice_modelview/deploy/test/{self.id}">部署测试</a> | 
                <a href="/inferenceservice_modelview/deploy/prod/{self.id}">部署生产</a> |
                <a target=_blank href="{monitoring_url}">监控</a> |
                <a href="/inferenceservice_modelview/clear/{self.id}">清理</a>
                '''
        if help_url:
            dom=f'<a target=_blank href="{help_url}">帮助</a> | '+dom
        else:
            dom = '帮助 | ' + dom
        return Markup(dom)

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

        TKE_EXISTED_LBID = json.loads(self.project.expand).get('TKE_EXISTED_LBID', self.project.cluster.get("TKE_EXISTED_LBID",conf.get('TKE_EXISTED_LBID','')))

        # first, Use the proxy ip configured by the project group
        SERVICE_EXTERNAL_IP = json.loads(self.project.expand).get('SERVICE_EXTERNAL_IP',None) if self.project.expand else None
        if not SERVICE_EXTERNAL_IP:
            # second, Use the global configuration proxy ip
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', [])
            if SERVICE_EXTERNAL_IP:
                SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP[0]

        if not SERVICE_EXTERNAL_IP:
            ip = request.host[:request.host.rindex(':')] if ':' in request.host else request.host  #  remove port in host
            if core.checkip(ip):
                SERVICE_EXTERNAL_IP = ip

        if SERVICE_EXTERNAL_IP:
            # 对于多网卡或者单域名模式，这里需要使用公网ip或者域名打开
            if '|' in SERVICE_EXTERNAL_IP:
                SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP.split('|')[1].strip()

            host = SERVICE_EXTERNAL_IP + ":" + str(port)
            return Markup(f'<a target=_blank href="http://{host}/">{host}</a>')

        elif TKE_EXISTED_LBID:
            TKE_EXISTED_LBID = TKE_EXISTED_LBID.split('|')
            if len(TKE_EXISTED_LBID)>1:
                host = TKE_EXISTED_LBID[1] + ":" + str(port)
                return Markup(f'<a target=_blank href="http://{host}/">{host}</a>')

        return "未开通"


    def __repr__(self):
        return self.name



    @property
    def inference_host_url(self):
        url = "http://" + self.name + "." + self.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN',''))
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
