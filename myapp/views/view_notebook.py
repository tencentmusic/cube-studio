import os

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import pysnooper
import uuid
from myapp.models.model_notebook import Notebook
from myapp.models.model_job import Repository
from flask_appbuilder.actions import action
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder, db, event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.validators import DataRequired, Length, Regexp
from wtforms import SelectField, StringField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MySelect2Widget, MyBS3TextFieldWidget
from flask import Markup
from myapp.utils.py.py_k8s import K8s
from flask import (
    abort,
    flash,
    g,
    redirect,
    request, make_response,
)
from .baseApi import (
    MyappModelRestApi
)
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import expose
import datetime, time, json
from myapp.views.view_team import Project_Join_Filter, filter_join_org_project
from myapp.models.model_team import Project

conf = app.config


class Notebook_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        return query.filter(self.model.created_by_fk == g.user.id)


class Notebook_ModelView_Base():
    datamodel = SQLAInterface(Notebook)
    label_title = _('notebook')
    check_redirect_list_url = conf.get('MODEL_URLS', {}).get('notebook', '')
    crd_name = 'notebook'
    conv = GeneralModelConverter(datamodel)
    base_permissions = ['can_add', 'can_delete', 'can_edit', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    base_filters = [["id", Notebook_Filter, lambda: []]]
    order_columns = ['id']
    search_columns = ['created_by', 'name']
    add_columns = ['project', 'name', 'describe', 'images', 'working_dir', 'volume_mount', 'resource_memory','resource_cpu', 'resource_gpu']
    list_columns = ['project', 'ide_type_html', 'name_url', 'status', 'describe','reset', 'resource', 'renew', 'save']
    cols_width = {
        "project": {"type": "ellip2", "width": 150},
        "ide_type_html": {"type": "ellip2", "width": 200},
        "name_url": {"type": "ellip2", "width": 250},
        "describe": {"type": "ellip2", "width": 300},
        "resource": {"type": "ellip2", "width": 300},
        "status": {"type": "ellip2", "width": 100},
        "renew": {"type": "ellip2", "width": 200},
        "save": {"type": "ellip2", "width": 200}
    }
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields

    # @pysnooper.snoop()
    def set_column(self, notebook=None):
        # 对编辑进行处理
        self.add_form_extra_fields['name'] = StringField(
            _('名称'),
            default="%s-" % g.user.username + uuid.uuid4().hex[:4],
            description= _('英文名(小写字母、数字、-组成)，最长50个字符'),
            widget=MyBS3TextFieldWidget(readonly=True if notebook else False),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54)]  # 注意不能以-开头和结尾
        )
        self.add_form_extra_fields['describe'] = StringField(
            _('描述'),
            default='%s-notebook' % g.user.username,
            description= _('中文描述'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        # "project": QuerySelectField(
        #     _('项目组'),
        #     query_factory=filter_join_org_project,
        #     allow_blank=True,
        #     widget=Select2Widget()
        # ),

        self.add_form_extra_fields['project'] = QuerySelectField(
            _('项目组'),
            default='',
            description= _('部署项目组'),
            query_factory=filter_join_org_project,
            widget=MySelect2Widget(extra_classes="readonly" if notebook else None, new_web=False),
        )
        self.add_form_extra_fields['images'] = SelectField(
            _('镜像'),
            description= _('notebook基础环境镜像，如果显示不准确，请删除新建notebook'),
            widget=MySelect2Widget(extra_classes="readonly" if notebook else None, new_web=False, can_input=True),
            choices=[[x[0], x[1]] for x in conf.get('NOTEBOOK_IMAGES', [])],
            validators=[DataRequired()]
        )
        self.add_form_extra_fields['node_selector'] = StringField(
            _('机器选择'),
            default='cpu=true,notebook=true',
            description= _("部署task所在的机器"),
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['image_pull_policy'] = SelectField(
            _('拉取策略'),
            description= _("镜像拉取策略(Always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)"),
            widget=Select2Widget(),
            choices=[['Always', 'Always'], ['IfNotPresent', 'IfNotPresent']]
        )
        self.add_form_extra_fields['volume_mount'] = StringField(
            _('挂载'),
            default=notebook.project.volume_mount if notebook else '',
            description= _('外部挂载，格式:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,4G(memory):/dev/shm,注意pvc会自动挂载对应目录下的个人rtx子目录'),
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['working_dir'] = StringField(
            _('工作目录'),
            default='/mnt',
            description= _("工作目录，如果为空，则使用Dockerfile中定义的workingdir"),
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['resource_memory'] = StringField(
            _('内存'),
            default=Notebook.resource_memory.default.arg,
            description= _('内存的资源使用限制，示例：1G，20G'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.add_form_extra_fields['resource_cpu'] = StringField(
            _('cpu'),
            default=Notebook.resource_cpu.default.arg,
            description= _('cpu的资源使用限制(单位：核)，示例：2'), widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        self.add_form_extra_fields['resource_gpu'] = StringField(
            _('gpu'),
            default='0',
            description= _('gpu的资源使用限gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡。申请具体的卡型号，可以类似 1(V100)，<span style="color:red;">虚拟化占用和共享模式占用仅企业版支持</span>'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        columns = ['name', 'describe', 'images', 'resource_memory', 'resource_cpu', 'resource_gpu']

        self.add_columns = ['project'] + columns  # 添加的时候没有挂载配置，使用项目中的挂载配置

        # 修改的时候管理员可以在上面添加一些特殊的挂载配置，适应一些特殊情况
        if g.user.is_admin():
            columns.append('volume_mount')
        self.edit_columns = ['project'] + columns
        self.edit_form_extra_fields = self.add_form_extra_fields
        self.default_filter = {
            "created_by": g.user.id
        }

    # @pysnooper.snoop()
    def pre_add(self, item):
        item.name = item.name.replace("_", "-")[0:54].lower()

        # 不需要用户自己填写node selector
        # if core.get_gpu(item.resource_gpu)[0]:
        #     item.node_selector = item.node_selector.replace('cpu=true','gpu=true')
        # else:
        #     item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')

        item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_item_json.get('resource_memory',None))
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_item_json.get('resource_cpu',None))
        item.namespace = json.loads(item.project.expand).get('NOTEBOOK_NAMESPACE', conf.get('NOTEBOOK_NAMESPACE'))

        if 'theia' in item.images or 'vscode' in item.images:
            item.ide_type = 'theia'
        elif 'matlab' in item.images:
            item.ide_type = 'matlab'
        elif 'rstudio' in item.images.lower() or 'rserver' in item.images.lower():
            item.ide_type = 'rstudio'
        else:
            item.ide_type = 'jupyter'

        if not item.id:
            item.volume_mount = item.project.volume_mount

        all_images={x[1]:x[0] for x in conf.get('NOTEBOOK_IMAGES', []) }
        if item.images in all_images:
            item.images = all_images[item.images]


    # @pysnooper.snoop(watch_explode=('item'))
    def pre_update(self, item):

        # if item.changed_by_fk:
        #     item.changed_by=db.session.query(MyUser).filter_by(id=item.changed_by_fk).first()
        # if item.created_by_fk:
        #     item.created_by=db.session.query(MyUser).filter_by(id=item.created_by_fk).first()

        self.pre_add(item)

    def post_add(self, item):
        flash(__('自动reset 一分钟后生效'), 'warning')
        try:
            self.reset_notebook(item)
        except Exception as e:
            print(e)
            flash(__('reset后查看运行运行状态'), 'warning')

    # @pysnooper.snoop(watch_explode=('item'))
    def post_update(self, item):
        flash(__('reset以后配置方可生效'), 'warning')

        # item.changed_on = datetime.datetime.now()
        # db.session.commit()
        # self.reset_notebook(item)

        # flash('自动reset 一分钟后生效', 'warning')
        if self.src_item_json:
            item.changed_by_fk = int(self.src_item_json.get('changed_by_fk'))
        if self.src_item_json:
            item.created_by_fk = int(self.src_item_json.get('created_by_fk'))

        db.session.commit()

    def post_list(self,items):
        flash(__('注意：个人重要文件本地git保存，notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。<br>个人持久化目录在/mnt/')+g.user.username,category='info')
        return items

    # @event_logger.log_this
    # @expose("/add", methods=["GET", "POST"])
    # @has_access
    # def add(self):
    #     self.set_column()
    #     self.add_form = self.conv.create_form(
    #             self.label_columns,
    #             self.add_columns,
    #             self.description_columns,
    #             self.validators_columns,
    #             self.add_form_extra_fields,
    #             self.add_form_query_rel_fields,
    #         )
    #     widget = self._add()
    #     if not widget:
    #         return redirect('/notebook_modelview/list/') # self.post_add_redirect()
    #     else:
    #         return self.render_template(
    #             self.add_template, title=self.add_title, widgets=widget
    #         )

    pre_update_web = set_column
    pre_add_web = set_column


    # @pysnooper.snoop(watch_explode=('notebook'))
    def reset_notebook(self, notebook):
        notebook.changed_on = datetime.datetime.now()
        db.session.commit()
        self.reset_theia(notebook)

    # 部署pod，service，VirtualService
    # @pysnooper.snoop(watch_explode=('notebook',))
    def reset_theia(self, notebook):
        try:
            # 先清理notebook
            self.pre_delete(notebook)
        except:
            pass
        k8s_client = K8s(notebook.cluster.get('KUBECONFIG', ''))
        namespace = notebook.namespace
        SERVICE_EXTERNAL_IP = []

        # 先使用项目组的
        if notebook.project.expand:
            SERVICE_EXTERNAL_IP = json.loads(notebook.project.expand).get('SERVICE_EXTERNAL_IP', '')
            if SERVICE_EXTERNAL_IP and type(SERVICE_EXTERNAL_IP) == str:
                SERVICE_EXTERNAL_IP = SERVICE_EXTERNAL_IP.split('|')[0].strip()
                SERVICE_EXTERNAL_IP = [SERVICE_EXTERNAL_IP]

        # 使用全局ip
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', None)

        # 使用当前
        if not SERVICE_EXTERNAL_IP:
            if core.checkip(request.host):
                ip = request.host
                ip = ip[:ip.index(':')] if ":" in ip else ip
                SERVICE_EXTERNAL_IP = [ip]

        port = 3000

        command = None
        workingDir = None
        volume_mount = notebook.volume_mount
        # 端口+0是jupyterlab  +1是sshd   +2 +3 是预留的用户自己启动应用占用的端口

        meet_ports = core.get_not_black_port(10000 + 10 * notebook.id)
        env = {
            "NO_AUTH": "true",
            "USERNAME": notebook.created_by.username,
            "NODE_OPTIONS": "--max-old-space-size=%s" % str(int(notebook.resource_memory.replace("G", '')) * 1024),
            "SSH_PORT": str(meet_ports[1]),
            "PORT1": str(meet_ports[2]),
            "PORT2": str(meet_ports[3]),
            "NOTEBOOK_NAME":notebook.name
        }

        rewrite_url = '/'
        pre_command = '(nohup sh /init.sh > /notebook_init.log 2>&1 &) ; (nohup sh /mnt/%s/init.sh > /init.log 2>&1 &) ; ' % notebook.created_by.username
        if notebook.ide_type == 'jupyter' or notebook.ide_type == 'bigdata' or notebook.ide_type == 'machinelearning' or notebook.ide_type == 'deeplearning':
            rewrite_url = '/notebook/jupyter/%s/' % notebook.name
            workingDir = '/mnt/%s' % notebook.created_by.username
            command = ["sh", "-c", "%s jupyter lab --notebook-dir=%s --ip=0.0.0.0 "
                                   "--no-browser --allow-root --port=%s "
                                   "--NotebookApp.token='' --NotebookApp.password='' --ServerApp.disable_check_xsrf=True "
                                   "--NotebookApp.allow_origin='*' "
                                   "--NotebookApp.base_url=%s" % (pre_command, notebook.mount, port, rewrite_url)]

            # command = ["sh", "-c", "%s jupyter lab --notebook-dir=/ --ip=0.0.0.0 "
            #                         "--no-browser --allow-root --port=%s "
            #                         "--NotebookApp.token='' --NotebookApp.password='' "
            #                         "--NotebookApp.allow_origin='*' "
            #                         "--NotebookApp.base_url=%s" % (pre_command,port,rewrite_url)]


        elif notebook.ide_type=='theia':
            command = ["bash",'-c','%s node /home/theia/src-gen/backend/main.js /home/project --hostname=0.0.0.0 --port=%s'%(pre_command,port)]
            workingDir = '/home/theia'

        print(command)
        print(workingDir)

        image_pull_secrets = conf.get('HUBSECRET', [])
        user_repositorys = db.session.query(Repository).filter(Repository.created_by_fk == g.user.id).all()
        image_pull_secrets = list(set(image_pull_secrets + [rep.hubsecret for rep in user_repositorys]))

        labels = {"app": notebook.name, 'user': notebook.created_by.username, 'pod-type': "notebook"}

        notebook_env = []
        if notebook.env:
            notebook_env = [x.strip() for x in notebook.env.split('\n') if x.strip()]
            notebook_env = [env.split("=") for env in notebook_env if '=' in env]
            notebook_env = dict(zip([env[0] for env in notebook_env], [env[1] for env in notebook_env]))
        if notebook_env:
            env.update(notebook_env)
        if SERVICE_EXTERNAL_IP:
            env["SERVICE_EXTERNAL_IP"] = SERVICE_EXTERNAL_IP[0]


        annotations={
            'project': notebook.project.name
        }
        k8s_client.create_debug_pod(
            namespace=namespace,
            name=notebook.name,
            labels=labels,
            annotations=annotations,
            command=command,
            args=None,
            volume_mount=volume_mount,
            working_dir=workingDir,
            node_selector=notebook.get_node_selector(),
            resource_memory=notebook.resource_memory if conf.get('NOTEBOOK_EXCLUSIVE',False) else ("0G~" + notebook.resource_memory),
            resource_cpu=notebook.resource_cpu if conf.get('NOTEBOOK_EXCLUSIVE',False) else ("0G~" + notebook.resource_cpu),
            resource_gpu=notebook.resource_gpu,
            image_pull_policy=conf.get('IMAGE_PULL_POLICY', 'Always'),
            image_pull_secrets=image_pull_secrets,
            image=notebook.images,
            hostAliases=conf.get('HOSTALIASES', ''),
            env=env,
            privileged=None,
            accounts=conf.get('JUPYTER_ACCOUNTS'),
            username=notebook.created_by.username
        )
        k8s_client.create_service(
            namespace=namespace,
            name=notebook.name,
            username=notebook.created_by.username,
            ports=[port, ],
            selector=labels
        )

        crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})
        crd_name = "notebook-jupyter-%s"%notebook.name.replace('_', '-') #  notebook.name.replace('_', '-')
        vs_obj = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
        if vs_obj:
            k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
            time.sleep(1)
        host=None
        # 优先使用项目组配置的
        if SERVICE_EXTERNAL_IP:
            host = SERVICE_EXTERNAL_IP[0]
        # 再使用项目配置
        if not host:
            host = notebook.project.cluster.get('HOST', request.host)
        # 最后使用当前域名
        if not host:
            host = request.host
        if ':' in host:
            host = host[:host.rindex(':')]  # 如果捕获到端口号，要去掉
        crd_json = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": crd_name,
                "namespace": namespace
            },
            "spec": {
                "gateways": [
                    "kubeflow/kubeflow-gateway"
                ],
                "hosts": [
                    "*" if core.checkip(host) else host
                ],
                "http": [
                    {
                        "match": [
                            {
                                "uri": {
                                    "prefix": "/notebook/%s/%s/" % (namespace, notebook.name)
                                }
                            }
                        ],
                        "rewrite": {
                            "uri": rewrite_url
                        },
                        "route": [
                            {
                                "destination": {
                                    "host": "%s.%s.svc.cluster.local" % (notebook.name, namespace),
                                    "port": {
                                        "number": port
                                    }
                                }
                            }
                        ],
                        "timeout": "300s"
                    }
                ]
            }
        }

        # print(crd_json)
        crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=crd_json)

        # 边缘模式时，需要根据项目组中的配置设置代理ip

        if SERVICE_EXTERNAL_IP and SERVICE_EXTERNAL_IP[0]!='127.0.0.1':
            SERVICE_EXTERNAL_IP = [ip.split('|')[0].strip() for ip in SERVICE_EXTERNAL_IP]
            ports = [port]
            # if notebook.ide_type=='bigdata':
            for index in range(1, 4):
                ports.append(meet_ports[index])

            # ports = list(set(ports))  # 这里会乱序
            service_ports = [[meet_ports[index], port] for index, port in enumerate(ports)]
            service_external_name = (notebook.name + "-external").lower()[:60].strip('-')
            k8s_client.create_service(
                namespace=namespace,
                name=service_external_name,
                username=notebook.created_by.username,
                ports=service_ports,
                selector=labels,
                service_type='ClusterIP' if conf.get('K8S_NETWORK_MODE','iptables')!='ipvs' else 'NodePort',
                external_ip=SERVICE_EXTERNAL_IP if conf.get('K8S_NETWORK_MODE','iptables')!='ipvs' else None
            )

        return crd

    # @event_logger.log_this
    @expose('/reset/<notebook_id>', methods=['GET', 'POST'])
    def reset(self, notebook_id):

        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        try:
            self.reset_notebook(notebook)
            flash(__('已重置，Running状态后可进入。注意：notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。'), 'info')
        except Exception as e:
            message = __('重置失败，稍后重试。') + str(e)
            flash(message, 'warning')
            return self.response(400, **{"message": message, "status": 1, "result": {}})

        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))

    # @event_logger.log_this
    @expose('/renew/<notebook_id>', methods=['GET', 'POST'])
    def renew(self, notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        notebook.changed_on = datetime.datetime.now()
        db.session.commit()
        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))

    # 基础批量删除
    # @pysnooper.snoop()
    def base_muldelete(self, items):
        if not items:
            abort(404)
        for item in items:
            try:
                k8s_client = K8s(item.cluster.get('KUBECONFIG',''))
                k8s_client.delete_pods(namespace=item.namespace,pod_name=item.name)
                k8s_client.delete_service(namespace=item.namespace,name=item.name)
                k8s_client.delete_service(namespace=item.namespace, name=(item.name + "-external").lower()[:60].strip('-'))
                crd_info = conf.get("CRD_INFO", {}).get('virtualservice', {})
                if crd_info:
                    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],
                                          plural=crd_info['plural'], namespace=item.namespace, name="notebook-jupyter-%s" % item.name.replace('_', '-'))

            except Exception as e:
                flash(str(e), "warning")

    def pre_delete(self, item):
        self.base_muldelete([item])

    @expose("/list/")
    @has_access
    def list(self):
        args = request.args.to_dict()
        if '_flt_0_created_by' in args and args['_flt_0_created_by'] == '':
            print(request.url)
            print(request.path)
            return redirect(request.url.replace('_flt_0_created_by=', '_flt_0_created_by=%s' % g.user.id))

        widgets = self._list()
        res = self.render_template(
            self.list_template, title=self.list_title, widgets=widgets
        )
        return res

    # @event_logger.log_this
    # @expose("/delete/<pk>")
    # @has_access
    # def delete(self, pk):
    #     pk = self._deserialize_pk_if_composite(pk)
    #     self.base_delete(pk)
    #     url = url_for(f"{self.endpoint}.list")
    #     return redirect(url)

    @action("stop_all", "停止", "停止所有选中的notebook?", "fa-trash", single=False)
    def stop_all(self, items):
        self.base_muldelete(items)
        self.update_redirect()
        return redirect(self.get_redirect())


class Notebook_ModelView(Notebook_ModelView_Base, MyappModelView, DeleteMixin):
    datamodel = SQLAInterface(Notebook)


# 添加视图和菜单
appbuilder.add_view_no_menu(Notebook_ModelView)


# 添加api
class Notebook_ModelView_Api(Notebook_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Notebook)
    route_base = '/notebook_modelview/api'


appbuilder.add_api(Notebook_ModelView_Api)


# 添加api
class Notebook_ModelView_SDK_Api(Notebook_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Notebook)
    route_base = '/notebook_modelview/sdk'
    add_columns = ['project', 'name', 'describe', 'images', 'working_dir', 'volume_mount', 'resource_memory','resource_cpu', 'resource_gpu','volume_mount','image_pull_policy','expand']
    edit_columns = add_columns
    list_columns = ['project', 'ide_type_html', 'name_url', 'status', 'describe', 'reset', 'resource', 'renew']
    show_columns = ['project', 'name', 'namespace', 'describe', 'images', 'working_dir', 'env', 'volume_mount','resource_memory', 'resource_cpu', 'resource_gpu', 'status', 'ide_type', 'image_pull_policy','node_selector', 'expand']
appbuilder.add_api(Notebook_ModelView_SDK_Api)
