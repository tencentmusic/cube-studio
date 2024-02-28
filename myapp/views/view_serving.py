import copy

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from myapp.models.model_serving import Service
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder, db
from myapp.models.model_job import Repository
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from myapp import security_manager
from wtforms.validators import DataRequired, Length, Regexp
from wtforms import StringField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MyBS3TextFieldWidget
from flask import (
    flash,
    g,
    redirect,
    request
)
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,

)
from .baseApi import (
    MyappModelRestApi
)
from myapp.views.view_team import Project_Join_Filter, filter_join_org_project
from flask_appbuilder import expose
import json

conf = app.config


class Service_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # public_project_id =
        # logging.info(join_projects_id)
        return query.filter(self.model.project_id.in_(join_projects_id))


class Service_ModelView_base():
    datamodel = SQLAInterface(Service)

    show_columns = ['project', 'name', 'label', 'images', 'volume_mount', 'working_dir', 'command', 'env',
                    'resource_memory', 'resource_cpu', 'resource_gpu', 'replicas', 'ports', 'host']
    add_columns = ['project', 'name', 'label', 'images', 'working_dir', 'command', 'env', 'resource_memory',
                   'resource_cpu', 'resource_gpu', 'replicas', 'ports', 'host']
    list_columns = ['project', 'name_url', 'host_url', 'ip', 'deploy', 'creator', 'modified']
    cols_width = {
        "name_url": {"type": "ellip2", "width": 200},
        "host_url": {"type": "ellip2", "width": 400},
        "ip": {"type": "ellip2", "width": 250},
        "deploy": {"type": "ellip2", "width": 230},
        "modified": {"type": "ellip2", "width": 150}
    }
    search_columns = ['created_by', 'project', 'name', 'label', 'images', 'resource_memory', 'resource_cpu', 'resource_gpu', 'volume_mount', 'host']

    edit_columns = ['project', 'name', 'label', 'images', 'working_dir', 'command', 'env', 'resource_memory', 'resource_cpu', 'resource_gpu', 'replicas', 'ports', 'volume_mount', 'host', ]
    base_order = ('id', 'desc')
    order_columns = ['id']
    label_title = _('云原生服务')
    base_filters = [["id", Service_Filter, lambda: []]]
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields
    host_rule = ", ".join([cluster + "cluster:*." + conf.get('CLUSTERS')[cluster].get("SERVICE_DOMAIN", conf.get('SERVICE_DOMAIN','')) for cluster in conf.get('CLUSTERS') if conf.get('CLUSTERS')[cluster].get("SERVICE_DOMAIN", conf.get('SERVICE_DOMAIN',''))])
    add_form_extra_fields={
        "project": QuerySelectField(_('项目组'),query_factory=filter_join_org_project,allow_blank=True,widget=Select2Widget()),
        "name":StringField(_('名称'), description= _('英文名(小写字母、数字、- 组成)，最长50个字符'),widget=BS3TextFieldWidget(), validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54)]),
        "label":StringField(_('标签'), description= _('中文名'), widget=BS3TextFieldWidget(),validators=[DataRequired()]),
        "images": StringField(_('镜像'), description= _('镜像全称'), widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "volume_mount":StringField(_('挂载'),description= _('外部挂载，格式:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,4G(memory):/dev/shm,注意pvc会自动挂载对应目录下的个人rtx子目录'),widget=BS3TextFieldWidget(),default=''),
        "working_dir": StringField(_('工作目录'),description= _('工作目录，容器启动的初始所在目录，不填默认使用Dockerfile内定义的工作目录'),widget=BS3TextFieldWidget()),
        "command":StringField(_('启动命令'), description= _('启动命令，支持多行命令'),widget=MyBS3TextAreaFieldWidget(rows=3)),
        "node_selector":StringField(_('机器选择'), description= _('运行当前服务所在的机器'),widget=BS3TextFieldWidget(),default='cpu=true,serving=true'),
        "resource_memory":StringField(_('memory'),default=Service.resource_memory.default.arg,description= _('内存的资源使用限制，示例1G，10G， 最大100G，如需更多联系管路员'),widget=BS3TextFieldWidget(),validators=[DataRequired()]),
        "resource_cpu":StringField(_('cpu'), default=Service.resource_cpu.default.arg,description= _('cpu的资源使用限制(单位核)，示例 0.4，10，最大50核，如需更多联系管路员'),widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "resource_gpu": StringField(_('gpu'), default='0',description= _('gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡'), widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "replicas": StringField(_('副本数'), default=Service.replicas.default.arg,description= _('pod副本数，用来配置高可用'),widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "ports": StringField(_('端口'), default=Service.ports.default.arg,description= _('进程端口号，逗号分隔'),widget=BS3TextFieldWidget(), validators=[DataRequired()]),
        "env": StringField(_('环境变量'), default=Service.env.default.arg, description= _('使用模板的task自动添加的环境变量，支持模板变量。书写格式:每行一个环境变量env_key=env_value'),widget=MyBS3TextAreaFieldWidget()),
        "host": StringField(_('域名'), default=Service.host.default.arg,description= _('访问域名，') + host_rule, widget=BS3TextFieldWidget()),
    }

    edit_form_extra_fields = copy.deepcopy(add_form_extra_fields)


    def set_column(self, service=None):
        host_field = StringField(_('域名'), default=Service.host.default.arg,description= _('访问域名，') + self.host_rule, widget=BS3TextFieldWidget())
        self.edit_form_extra_fields['host'] = host_field
        self.add_form_extra_fields['host'] = host_field

    pre_update_web = set_column
    pre_add_web = set_column

    def pre_add(self, item):
        if not item.volume_mount:
            item.volume_mount = item.project.volume_mount

    def delete_old_service(self, service_name, cluster):
        service_external_name = (service_name + "-external").lower()[:60].strip('-')
        from myapp.utils.py.py_k8s import K8s
        k8s = K8s(cluster.get('KUBECONFIG', ''))
        namespace = conf.get('SERVICE_NAMESPACE')
        k8s.delete_deployment(namespace=namespace, name=service_name)
        k8s.delete_service(namespace=namespace, name=service_name)
        k8s.delete_service(namespace=namespace, name=service_external_name)
        k8s.delete_istio_ingress(namespace=namespace, name=service_name)

    def pre_update(self, item):
        # 修改了名称的话，要把之前的删掉
        if self.src_item_json.get('name', '') != item.name:
            self.delete_old_service(self.src_item_json.get('name', ''), item.project.cluster)
            flash(__('检测到修改名称，旧服务已清理完成'), category='warning')

    def pre_delete(self, item):
        self.delete_old_service(item.name, item.project.cluster)
        flash(__('服务清理完成'), category='success')

    @expose('/clear/<service_id>', methods=['POST', "GET"])
    def clear(self, service_id):
        service = db.session.query(Service).filter_by(id=service_id).first()
        self.delete_old_service(service.name, service.project.cluster)
        flash(__('服务清理完成'), category='success')
        return redirect(conf.get('MODEL_URLS', {}).get('service', ''))

    @expose('/deploy/<service_id>', methods=['POST', "GET"])
    def deploy(self, service_id):

        image_pull_secrets = conf.get('HUBSECRET', [])
        user_repositorys = db.session.query(Repository).filter(Repository.created_by_fk == g.user.id).all()
        image_pull_secrets = list(set(image_pull_secrets + [rep.hubsecret for rep in user_repositorys]))

        service = db.session.query(Service).filter_by(id=service_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(service.project.cluster.get('KUBECONFIG', ''))
        namespace = conf.get('SERVICE_NAMESPACE')

        volume_mount = service.volume_mount
        labels = {"app": service.name, "user": service.created_by.username, "pod-type": "service"}
        env = service.env
        env += "\nRESOURCE_CPU=" + service.resource_cpu
        env += "\nRESOURCE_MEMORY=" + service.resource_memory
        env += "\nUSERNAME=" + service.created_by.username

        annotations = {
            'project': service.project.name
        }
        k8s_client.create_deployment(namespace=namespace,
                                     name=service.name,
                                     replicas=service.replicas,
                                     labels=labels,
                                     annotations=annotations,
                                     command=['bash', '-c', service.command] if service.command else None,
                                     args=None,
                                     volume_mount=volume_mount,
                                     working_dir=service.working_dir,
                                     node_selector=service.get_node_selector(),
                                     resource_memory= service.resource_memory if conf.get('SERVICE_EXCLUSIVE',False) else ("0~" + service.resource_memory),
                                     resource_cpu=service.resource_cpu if conf.get('SERVICE_EXCLUSIVE',False) else ("0~" + service.resource_cpu),
                                     resource_gpu=service.resource_gpu if service.resource_gpu else '0',
                                     image_pull_policy=conf.get('IMAGE_PULL_POLICY', 'Always'),
                                     image_pull_secrets=image_pull_secrets,
                                     image=service.images,
                                     hostAliases=conf.get('HOSTALIASES', ''),
                                     env=env,
                                     privileged=False,
                                     accounts=None,
                                     username=service.created_by.username,
                                     ports=[int(port) for port in service.ports.split(',')]
                                     )

        ports = [int(port) for port in service.ports.split(',')]

        k8s_client.create_service(
            namespace=namespace,
            name=service.name,
            username=service.created_by.username,
            ports=ports,
            selector=labels
        )
        # 如果域名配置的gateway，就用这个
        host = service.name + "." + service.project.cluster.get('SERVICE_DOMAIN', conf.get('SERVICE_DOMAIN', ''))
        if service.host:
            host = service.host.replace('http://', '').replace('https://', '').strip()
            if "/" in host:
                host = host[:host.index("/")]
            if ":" in host:
                host = host[:host.index(":")]
        k8s_client.create_istio_ingress(namespace=namespace,
                                        name=service.name,
                                        host=host,
                                        ports=service.ports.split(',')
                                        )

        # 以ip形式访问的话，使用的代理ip。不然不好处理机器服务化机器扩容和缩容时ip变化
        # 创建EXTERNAL_IP的服务

        SERVICE_EXTERNAL_IP = []
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
            ip = request.host[:request.host.rindex(':')] if ':' in request.host else request.host  # 如果捕获到端口号，要去掉
            if ip == '127.0.0.1':
                host = service.project.cluster.get('HOST', '')
                if host:
                    host = host[:host.rindex(':')] if ':' in host else host
                    SERVICE_EXTERNAL_IP = [host]
            elif core.checkip(ip):
                SERVICE_EXTERNAL_IP = [ip]

        if SERVICE_EXTERNAL_IP:
            # 对于多网卡模式，或者单域名模式，代理需要配置内网ip，界面访问需要公网ip或域名
            SERVICE_EXTERNAL_IP = [ip.split('|')[0].strip() for ip in SERVICE_EXTERNAL_IP]

            meet_ports = core.get_not_black_port(30000 + 10 * service.id)
            service_ports = [[meet_ports[index], port] for index, port in enumerate(ports)]
            service_external_name = (service.name + "-external").lower()[:60].strip('-')
            k8s_client.create_service(
                namespace=namespace,
                name=service_external_name,
                username=service.created_by.username,
                ports=service_ports,
                selector=labels,
                service_type='ClusterIP' if conf.get('K8S_NETWORK_MODE', 'iptables') != 'ipvs' else 'NodePort',
                external_ip=SERVICE_EXTERNAL_IP if conf.get('K8S_NETWORK_MODE', 'iptables') != 'ipvs' else None
            )

        flash(__('服务部署完成'), category='success')
        return redirect(conf.get("MODEL_URLS", {}).get("service", '/'))


class Service_ModelView(Service_ModelView_base, MyappModelView, DeleteMixin):
    datamodel = SQLAInterface(Service)


appbuilder.add_view_no_menu(Service_ModelView)


class Service_ModelView_Api(Service_ModelView_base, MyappModelRestApi):
    datamodel = SQLAInterface(Service)
    route_base = '/service_modelview/api'


appbuilder.add_api(Service_ModelView_Api)
