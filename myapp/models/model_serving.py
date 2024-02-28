from flask_appbuilder import Model
from sqlalchemy.orm import relationship
import json
from sqlalchemy import Text
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.helpers import AuditMixinNullable
from flask import request,g
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
        url="//"+self.project.cluster.get('HOST',request.host)+conf.get('GRAFANA_SERVICE_PATH')+self.name
        return Markup(f'<a href="{url}">{__("监控")}</a>')
        # https://www.angularjswiki.com/fontawesome/fa-flask/    <i class="fa-solid fa-monitor-waveform"></i>



    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'service')



class Service(Model,AuditMixinNullable,MyappModelBase,service_common):
    __tablename__ = 'service'
    id = Column(Integer, primary_key=True,comment='id主键')
    project_id = Column(Integer, ForeignKey('project.id'),comment='项目组id')
    project = relationship(
        Project, foreign_keys=[project_id]
    )

    name = Column(String(100), nullable=False,unique=True,comment='英文名')   # Used to generate pod service and vs
    label = Column(String(100), nullable=False,comment='中文名')
    images = Column(String(200), nullable=False,comment='镜像')
    working_dir = Column(String(100),default='',comment='启动目录')
    command = Column(String(1000),default='',comment='启动命令')
    args = Column(Text,default='',comment='启动参数')
    env = Column(Text,default='',comment='环境变量')
    volume_mount = Column(String(2000),default='',comment='挂载')
    node_selector = Column(String(100),default='cpu=true,serving=true',comment='机器选择器')
    replicas = Column(Integer,default=1,comment='副本')
    ports = Column(String(100),default='80',comment='端口')
    resource_memory = Column(String(100),default='2G',comment='申请内存')
    resource_cpu = Column(String(100), default='2',comment='申请cpu')
    resource_gpu= Column(String(100), default='0',comment='申请gpu')
    deploy_time = Column(String(100), nullable=False,default=datetime.datetime.now,comment='部署时间')
    host = Column(String(200), default='',comment='域名')
    expand = Column(Text(65536), default='{}',comment='扩展参数')

    @property
    def deploy(self):
        monitoring_url = "//"+self.project.cluster.get('HOST', request.host) + conf.get('GRAFANA_SERVICE_PATH') + self.name
        help_url=''
        try:
            help_url = json.loads(self.expand).get('help_url','') if self.expand else ''
        except Exception as e:
            print(e)

        if help_url:
            return Markup(f'<a href="/service_modelview/deploy/{self.id}">{__("部署")}</a> | <a href="{monitoring_url}">{__("监控")}</a> | <a href="/service_modelview/clear/{self.id}">{__("清理")}</a>')
        else:
            return Markup(f'<a href="/service_modelview/deploy/{self.id}">{__("部署")}</a> | <a href="{monitoring_url}">{__("监控")}</a> | <a href="/service_modelview/clear/{self.id}">{__("清理")}</a>')


    @property
    def clear(self):
        return Markup(f'<a href="/service_modelview/clear/{self.id}">{__("清理")}</a>')


    @property
    def name_url(self):
        # user_roles = [role.name.lower() for role in list(g.user.roles)]
        # if "admin" in user_roles:
        url = f'/k8s/web/search/{self.project.cluster["NAME"]}/{conf.get("SERVICE_NAMESPACE")}/{self.name.replace("_", "-")}'

        return Markup(f'<a target=_blank href="{url}">{self.label}</a>')

    def __repr__(self):
        return self.name

    @property
    def ip(self):
        from myapp.utils import core
        meet_ports = core.get_not_black_port(30000 + 10 * self.id)
        port = meet_ports[0]
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
            return Markup(f'<a target=_blank href="//{host}">{host}</a>')
        else:
            return "未开通"

    @property
    def host_url(self):
        # 泛域名先统一http
        url = "http://" + self.name + "." + self.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN',''))
        if self.host:
            if 'http://' in self.host or 'https://' in self.host:
                url = self.host
            else:
                url = "http://"+self.host

        return Markup(f'<a target=_blank href="{url}">{url}</a>')


class InferenceService(Model,AuditMixinNullable,MyappModelBase,service_common):
    __tablename__ = 'inferenceservice'
    id = Column(Integer, primary_key=True,comment='id主键')
    project_id = Column(Integer, ForeignKey('project.id'),comment='项目组id')
    project = relationship(
        Project, foreign_keys=[project_id]
    )

    name = Column(String(100), nullable=True,unique=True,comment='英文名')
    label = Column(String(100), nullable=False,comment='中文名')

    service_type= Column(String(100),nullable=True,default='serving',comment='服务类型')
    model_name = Column(String(200),default='',comment='模型名')
    model_version = Column(String(200),default='',comment='模型版本')
    model_path = Column(String(200),default='',comment='模型地址')
    model_type = Column(String(200),default='',comment='模型类型')
    model_input = Column(Text(65536), default='',comment='模型输入')
    model_output = Column(Text(65536), default='',comment='模型输出')
    inference_config = Column(Text(65536), default='',comment='配置文件')   # make configmap
    model_status = Column(String(200),default='offline',comment='服务状态')
    # model_status = Column(Enum('offline','test','online','delete',name='model_status'),nullable=True,default='offline')

    transformer=Column(String(200),default='',comment='预处理')  # pre process and post process

    images = Column(String(200), nullable=False,comment='镜像')
    working_dir = Column(String(100),default='',comment='启动目录')
    command = Column(String(1000),default='',comment='启动命令')
    args = Column(Text,default='',comment='启动参数')
    env = Column(Text,default='',comment='环境变量')
    volume_mount = Column(String(2000),default='',comment='挂载')
    node_selector = Column(String(100),default='cpu=true,serving=true',comment='机器选择器')
    min_replicas = Column(Integer,default=1,comment='最小副本数')
    max_replicas = Column(Integer, default=1,comment='最大副本数')
    hpa = Column(String(400), default='',comment='弹性伸缩')
    metrics = Column(Text(65536), default='',comment='监控接口')
    health = Column(String(400), default='',comment='健康检查')
    sidecar = Column(String(400), default='',comment='伴随容器')
    ports = Column(String(100),default='80',comment='端口')
    resource_memory = Column(String(100),default='2G',comment='申请内存')
    resource_cpu = Column(String(100), default='2',comment='申请cpu')
    resource_gpu= Column(String(100), default='0',comment='申请gpu')
    deploy_time = Column(String(100), nullable=True,default=datetime.datetime.now,comment='部署时间')
    host = Column(String(200), default='',comment='域名')
    expand = Column(Text(65536), default='{}',comment='扩展参数')
    canary = Column(String(400), default='',comment='灰度发布')
    shadow = Column(String(400), default='',comment='灰度发布')

    run_id = Column(String(100),nullable=True,comment='run id')
    run_time = Column(String(100),comment='运行时间')
    deploy_history = Column(Text(65536), default='',comment='部署历史')

    priority = Column(Integer,default=1,comment='优先级')   # giving priority to meeting high-priority resource needs

    @property
    def model_name_url(self):
        url = f'/k8s/web/search/{self.project.cluster["NAME"]}/{conf.get("SERVICE_NAMESPACE")}/{self.name.replace("_", "-")}'

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

        monitoring_url="//"+self.project.cluster.get('HOST', request.host)+conf.get('GRAFANA_SERVICE_PATH')+self.name
        # if self.created_by.username==g.user.username or g.user.is_admin():
        if self.created_by.id == g.user.id or self.project.user_role(g.user.id)=='creator':
            dom = f'''
                    <a target=_blank href="/inferenceservice_modelview/deploy/debug/{self.id}">{__("调试")}</a> | 
                    <a href="/inferenceservice_modelview/deploy/test/{self.id}">{__("部署测试")}</a> | 
                    <a href="/inferenceservice_modelview/deploy/prod/{self.id}">{__("部署生产")}</a> |
                    <a target=_blank href="{monitoring_url}">{__("监控")}</a> |
                    <a href="/inferenceservice_modelview/clear/{self.id}">{__("清理")}</a>
                    '''
        else:
            dom = f''' {__("调试")} | {__("部署测试")} | {__("部署生产")}</a> | <a target=_blank href="{monitoring_url}">{__("监控")}</a> | {__("清理")} '''

        # if help_url:
        #     dom=f'<a target=_blank href="{help_url}">{__("帮助")}</a> | '+dom
        # else:
        #     dom = f'{__("帮助")} | ' + dom
        return Markup(dom)

    @property
    def debug(self):
        return Markup(f'<a target=_blank href="/inferenceservice_modelview/debug/{self.id}">{__("调试")}</a>')

    @property
    def test_deploy(self):
        return Markup(f'<a href="/inferenceservice_modelview/deploy/test/{self.id}">{__("部署测试")}</a>')

    @property
    def deploy(self):
        return Markup(f'<a href="/inferenceservice_modelview/deploy/prod/{self.id}">{__("部署生产")}</a>')

    @property
    def clear(self):
        return Markup(f'<a href="/inferenceservice_modelview/clear/{self.id}">{__("清理")}</a>')

    @property
    def ip(self):
        from myapp.utils import core
        meet_ports = core.get_not_black_port(20000 + 10 * self.id)
        port = meet_ports[0]

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
            url = host

            # 如果是tfserving
            if self.service_type == 'tfserving':
                url = host + f"/v1/models/{self.model_name}/metadata"
            # 如果是torch-server
            if self.service_type == 'torch-server':
                url = SERVICE_EXTERNAL_IP + f":{port + 1}/models"

            # 处理业务自己配置的host的特殊配置
            if self.host:
                from myapp.utils.core import split_url
                host_temp, port_temp, path_temp = split_url(self.host)

                if port_temp and port_temp in self.ports:
                    # 查看是第几个端口
                    if self.ports.find(port_temp) > self.ports.find(','):
                        port = port + 1
                url = SERVICE_EXTERNAL_IP+":"+str(port)+path_temp
            return Markup(f'<a target=_blank href="http://{url}">{host}</a>')

        elif TKE_EXISTED_LBID:
            TKE_EXISTED_LBID = TKE_EXISTED_LBID.split('|')
            if len(TKE_EXISTED_LBID)>1:
                host = TKE_EXISTED_LBID[1] + ":" + str(port)
                return Markup(f'<a target=_blank href="http://{host}">{host}</a>')

        return __("未开通")


    def __repr__(self):
        return self.name


    @property
    def inference_host_url(self):
        # 泛域名先使用http
        host = self.name + "." + self.project.cluster.get('SERVICE_DOMAIN',conf.get('SERVICE_DOMAIN',''))
        link = host

        if not self.host and self.service_type=='tfserving':
            link=host+f"/v1/models/{self.model_name}/metadata"
        if not self.host and self.service_type=='torch-server':
            link=host+":8080/models"

        if self.host:
            from myapp.utils.core import split_url
            host_temp, port_temp, path_temp = split_url(self.host)

            if host_temp:
                host=host_temp
            if port_temp and port_temp in self.ports:
                # 查看是第几个端口
                if self.ports.find(port_temp) > self.ports.find(','):
                    link = host+":8080"+path_temp
            else:
                if port_temp:
                    link = host+":"+port_temp+path_temp
                else:
                    link = host+path_temp

        hosts=f'<a target=_blank href="http://{link}">{host}</a>'
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
            inference_config=self.inference_config,
            expand = '{}',
            canary = '',
            shadow = '',

            run_id = '',
            run_time = '',
            deploy_history = ''

        )
