import os
import re

from flask_appbuilder.baseviews import expose_api

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
        if g.user.is_admin():
            return query

        return query.filter(self.model.created_by_fk == g.user.id)


class Notebook_ModelView_Base():
    datamodel = SQLAInterface(Notebook)
    label_title = _('notebook')
    crd_name = 'notebook'
    conv = GeneralModelConverter(datamodel)
    base_permissions = ['can_add', 'can_delete', 'can_edit', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    base_filters = [["id", Notebook_Filter, lambda: []]]
    order_columns = ['id']
    search_columns = ['created_by', 'name']

    add_columns = ['project', 'name', 'describe', 'images', 'working_dir', 'volume_mount', 'resource_memory','resource_cpu', 'resource_gpu']
    list_columns = ['project', 'ide_type_html', 'name_url', 'status', 'describe','reset', 'resource', 'renew', 'save']
    show_columns = ['project', 'name', 'namespace', 'describe', 'images', 'working_dir', 'env', 'volume_mount','resource_memory', 'resource_cpu', 'resource_gpu', 'expand']
    cols_width = {
        "project": {"type": "ellip2", "width": 120},
        "ide_type_html": {"type": "ellip2", "width": 200},
        "name_url": {"type": "ellip2", "width": 150},
        "describe": {"type": "ellip2", "width": 180},
        "resource": {"type": "ellip2", "width": 270},
        "status": {"type": "ellip2", "width": 140},
        "renew": {"type": "ellip2", "width": 150},
        "save": {"type": "ellip2", "width": 200},
        "ops_html": {"type": "ellip2", "width": 130}
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
            description= _('部署项目组，在切换项目组前注意先停止当前notebook'),
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
            description= _('外部挂载，格式:<br>$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2<br>注意pvc会自动挂载对应目录下的个人username子目录') if g.user.is_admin() else _('外部挂载，格式: $ip/$path(nfs):/nfs，逗号分隔多个挂载'),
            widget=BS3TextFieldWidget(),
            validators=[Regexp('^[\x00-\x7F]*$')]
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
            description= _('内存的资源使用配置，示例：1G，20G'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Regexp("^.*G$")]
        )
        self.add_form_extra_fields['resource_cpu'] = StringField(
            _('cpu'),
            default=Notebook.resource_cpu.default.arg,
            description= _('cpu的资源使用配置(单位：核)，示例：2'), widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp("^[0-9]*$")]
        )

        self.add_form_extra_fields['resource_gpu'] = StringField(
            _('gpu'),
            default='0',
            description= _('申请的gpu卡数目，示例:2，每个容器独占整卡。-1为共享占用方式，小数(0.1)为vgpu方式，申请具体的卡型号，可以类似 1(V100)'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp('^[\-\.0-9,a-zA-Z\(\)]*$')]
        )

        columns = ['name', 'describe', 'images', 'resource_memory', 'resource_cpu', 'resource_gpu']

        self.add_columns = ['project'] + columns  # 添加的时候没有挂载配置，使用项目中的挂载配置

        # 修改的时候管理员可以在上面添加一些特殊的挂载配置，适应一些特殊情况
        if not conf.get('ENABLE_USER_VOLUME',False) and not g.user.is_admin():
            self.add_columns = ['project'] + columns
            self.edit_columns = ['project'] + columns
        else:
            self.add_columns = ['project'] + columns
            self.edit_columns = ['project'] + columns + ['volume_mount']

        self.edit_form_extra_fields = self.add_form_extra_fields
        self.default_filter = {
            "created_by": g.user.id
        }

    def pre_add(self, item):
        item.name = item.name.replace("_", "-")[0:54].lower()
        item.resource_gpu = item.resource_gpu.upper() if item.resource_gpu else '0'

        # 不需要用户自己填写node selector
        # if core.get_gpu(item.resource_gpu)[0]:
        #     item.node_selector = item.node_selector.replace('cpu=true','gpu=true')
        # else:
        #     item.node_selector = item.node_selector.replace('gpu=true', 'cpu=true')

        item.resource_memory=core.check_resource_memory(item.resource_memory,self.src_item_json.get('resource_memory',None))
        item.resource_cpu = core.check_resource_cpu(item.resource_cpu,self.src_item_json.get('resource_cpu',None))
        item.namespace = json.loads(item.project.expand).get('NOTEBOOK_NAMESPACE', conf.get('NOTEBOOK_NAMESPACE'))
        if not item.namespace:
            item.namespace = item.project.notebook_namespace

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
        else:
            if conf.get('ENABLE_USER_VOLUME',False) and not g.user.is_admin():
                volume_mounts_temp = re.split(',|;', item.volume_mount)
                volume_mount_arr=[]
                for volume_mount in volume_mounts_temp:
                    match = re.search(r'\((.*?)\)', volume_mount)
                    if match:
                        volume_type = match.group(1)
                        re_str = conf.get('ENABLE_USER_VOLUME_CONFIG', {}).get(volume_type, '')
                        if re_str:
                            if re.match(re_str, volume_mount):
                                volume_mount_arr.append(volume_mount)

                item.volume_mount = ','.join(volume_mount_arr).strip(',')
            # 合并项目组的挂载
            item.volume_mount = core.merge_volume_mount(item.project.volume_mount,item.volume_mount)



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




        # 如果修改了基础镜像，就把debug中的任务删除掉
        if self.src_item_json:
            # k8s集群更换了，要删除原来的
            if str(self.src_item_json.get('project_id', '1')) != str(item.project.id):
                src_project = db.session.query(Project).filter_by(id=int(self.src_item_json.get('project_id', '1'))).first()
                if src_project and src_project.cluster['NAME'] != item.project.cluster['NAME']:
                    self.base_muldelete([item],src_project.cluster['NAME'])
                    flash(__('发现集群更换，已帮你删除之前启动的notebook'), 'success')

    def post_add(self, item):

        try:
            self.reset_notebook(item)
        except Exception as e:
            print(e)
            flash(__('start fail, please reset notebook: ')+str(e), 'warning')
            return

        flash(__('自动reset 一分钟后生效'), 'info')

    # @pysnooper.snoop(watch_explode=('item'))
    def post_update(self, item):
        flash(__('reset以后配置方可生效'), 'info')

        # item.changed_on = datetime.datetime.now()
        # db.session.commit()
        # self.reset_notebook(item)

        # flash('自动reset 一分钟后生效', 'warning')
        if self.src_item_json:
            changed_by_fk = self.src_item_json.get('changed_by_fk','')
            if changed_by_fk:
                item.changed_by_fk = int(self.src_item_json.get('changed_by_fk'))
        if self.src_item_json:
            created_by_fk = self.src_item_json.get('created_by_fk','')
            if created_by_fk:
                item.created_by_fk = int(self.src_item_json.get('created_by_fk'))

        db.session.commit()

    def post_list(self,items):
        flash(__('注意：个人重要文件本地git保存，notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。<br>个人持久化目录在/mnt/')+g.user.username,category='info')
        return items

    pre_update_web = set_column
    pre_add_web = set_column

    @expose_api(description="创建打开jupyter",url='/entry/jupyter', methods=['GET', 'DELETE'])
    def entry_jupyter(self):
        data=request.args
        project_name=data.get('project_name','public')
        name = data.get('name',g.user.username+'-pipeline').replace("_",'')[:56]
        label = data.get('label','打开目录')
        resource_memory = data.get('resource_memory',"10G")
        resource_cpu = data.get('resource_cpu',"10")
        volume_mount = data.get('volume_mount','kubeflow-user-workspace(pvc):/mnt')
        file_path = data.get('file_path', '')   # 一定要是文件在 notebook的容器目录
        if 'http://' in file_path or 'https://' in file_path:
            file_path = '/mnt/{{creator}}'

        def template_str(src_str):
            from jinja2 import Environment, BaseLoader, DebugUndefined
            rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
            des_str = rtemplate.render(creator=g.user.username,
                                       datetime=datetime,
                                       runner=g.user.username,
                                       uuid=uuid
                                       )
            return des_str

        file_path = template_str(file_path)
        if f'/mnt/{g.user.username}' in file_path:
            file_path = file_path.replace(f'/mnt/{g.user.username}','')
        # 如果是打不开的文本文件，就自动变为目录
        file_name = file_path.split('/')[-1]
        text_file_extensions = {'ipynb', 'py', 'R', 'txt', 'csv', 'json', 'xml', 'py', 'html', 'css', 'js', 'md', 'yaml', 'yml', 'html', 'jpg', 'jpeg', 'png', 'sh'}
        if '.' in file_name:
            if file_name.split('.')[-1] not in text_file_extensions:
                file_path = file_path.replace(file_name,'')

        images = data.get('images',f'{conf.get("REPOSITORY_ORG","ccr.ccs.tencentyun.com/cube-studio/")}notebook:jupyter-ubuntu22.04')
        project = db.session.query(Project).filter(Project.name==project_name).filter(Project.type=='org').first()
        notebook = db.session.query(Notebook).filter(Notebook.name==name).first()
        if not project:
            res = make_response("项目组%s不存在"%project_name)
            res.status_code = 405
            return res

        if project:
            if not notebook:
                notebook = Notebook()
            notebook.project = project
            notebook.project_id = project.id
            notebook.name = name
            notebook.describe = label
            notebook.images = images
            notebook.ide_type = 'jupyter'
            notebook.working_dir = ''
            notebook.volume_mount = volume_mount
            notebook.resource_memory = resource_memory
            notebook.created_by=g.user
            notebook.changed_by=g.user
            notebook.resource_cpu = resource_cpu
            if file_path.strip('/'):
                notebook.expand = json.dumps({
                    "root":file_path
                })
            if not notebook.id:
                notebook.created_on=datetime.datetime.now()
                db.session.add(notebook)

            db.session.commit()

        notebook_id = notebook.id
        k8s_client = K8s(notebook.cluster.get('KUBECONFIG', ''))
        namespace = notebook.project.notebook_namespace
        del_namespace = notebook.namespace
        crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})

        # 删除
        if request.method=='DELETE':
            try:
                k8s_client.delete_pods(namespace=del_namespace,pod_name=name)
            except Exception as e:
                print(e)
            try:
                k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=del_namespace, name=name)
            except Exception as e:
                print(e)

            try:
                k8s_client.delete_service(namespace=del_namespace,name=name)
            except Exception as e:
                print(e)

            notebook.delete()
            db.session.commit()
            res = make_response(__("删除成功"))
            res.status_code = 200
            return res

        notebook = db.session.query(Notebook).filter(Notebook.id==int(notebook_id)).first()

        port = 3000
        name = notebook.name

        # 创建新的service
        labels = {"app": notebook.name, 'user': notebook.created_by.username, 'pod-type': "notebook"}
        try:
            exist_service = k8s_client.v1.read_namespaced_service(name=name, namespace=namespace)
        except:
            exist_service = None
        if not exist_service:
            k8s_client.create_service(
                namespace=namespace,
                name=name,
                username=notebook.created_by.username,
                ports=[port],
                selector=labels
            )
        try:
            exist_crd = k8s_client.CustomObjectsApi.get_namespaced_custom_object(
                group=crd_info['group'], version=crd_info['version'],
                plural=crd_info['plural'], namespace=namespace,
                name=name
            )
        except:
            exist_crd=None
        host = notebook.project.cluster.get('HOST', request.host).split('|')[0].strip().split(':')[0]

        if not exist_crd:
            # 创建vs
            crd_json = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "VirtualService",
                "metadata": {
                    "name": name,
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
                                        "prefix": f"/notebook/jupyter/{notebook.name}/"
                                    },
                                    "headers": {
                                        "cookie": {
                                            "regex": ".*myapp_username=.*"
                                        }
                                    }

                                }
                            ],
                            "rewrite": {
                                "uri": '/notebook/jupyter/%s/' % notebook.name
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
            crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                                        namespace=namespace, body=crd_json)

        exist_pod = k8s_client.get_pods(pod_name=name, namespace=namespace)
        if exist_pod:
            exist_pod = exist_pod[0]
            if exist_pod['status'].lower() != 'running':
                k8s_client.v1.delete_namespaced_pod(name, namespace, grace_period_seconds=0)
                exist_pod = None

        rewrite_url = '/notebook/jupyter/%s/' % notebook.name
        username=g.user.username
        if not exist_pod:

            pre_command = '(nohup sh /init.sh > /notebook_init.log 2>&1 &) ; (nohup sh /mnt/%s/init.sh > /init.log 2>&1 &) ; ' % username
            working_dir = '/mnt/%s' % username
            command = ["sh", "-c", "%s jupyter lab --notebook-dir=%s --ip=0.0.0.0 "
                                   "--no-browser --allow-root --port=%s "
                                   "--NotebookApp.token='' --NotebookApp.password='' --ServerApp.disable_check_xsrf=True "
                                   "--NotebookApp.allow_origin='*' "
                                   "--NotebookApp.base_url=%s" % (pre_command, "/mnt/"+username, port, rewrite_url)]
            env = {
                "NO_AUTH": "true",
            }
            pod, pod_spec = k8s_client.make_pod(
                namespace=namespace,
                name=name,
                labels=labels,
                annotations = {"project":project_name},
                command=command,
                args=None,
                volume_mount=notebook.volume_mount,
                working_dir=working_dir,
                node_selector=notebook.get_node_selector(),
                resource_memory="0G~" + notebook.resource_memory,
                resource_cpu="0~" + notebook.resource_cpu,
                resource_gpu=notebook.resource_gpu,
                image_pull_policy=conf.get('IMAGE_PULL_POLICY', 'Always'),
                image_pull_secrets=conf.get('HUBSECRET', []),
                image=notebook.images,
                hostAliases=conf.get('HOSTALIASES', ''),
                env=env,
                privileged=None,
                accounts=conf.get('JUPYTER_ACCOUNTS',''),
                username=username,
                restart_policy='Never'
            )
            # print(pod)
            try:
                pod = k8s_client.v1.create_namespaced_pod(namespace, pod)
                notebook.namespace=namespace
                db.session.commit()
                time.sleep(1)
            except Exception as e:
                print(e)
        # print(pod)

        left_retry = 10
        status=''
        while(left_retry):
            pod = k8s_client.get_pods(namespace=namespace, pod_name=name)
            if pod:
                pod=pod[0]
                status=json.dumps(pod['status_more'],ensure_ascii=False,indent=4, default=str).replace('\n',"<br>")
                if pod['status']=='Running':
                    if file_path:
                        if file_path.lstrip('/'):
                            file_path=f'/notebook/jupyter/{name}/lab/tree/'+file_path.strip('/')
                            time.sleep(2)
                            return redirect(file_path)
                        else:
                            file_path = f'/notebook/jupyter/{name}/lab?#/mnt/{g.user.username}'
                            time.sleep(2)
                            return redirect(file_path)

                    return redirect('%s%s'%(request.host_url.strip('/'),rewrite_url))
            left_retry=left_retry-1
            time.sleep(2)
        res = make_response(__("notebook未就绪，刷新此页面。<br> notebook状态：<br><br>")+Markup(status))
        res.status_code=200
        return res


    # @pysnooper.snoop(watch_explode=('message'))
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
        k8s_client = K8s(notebook.project.cluster.get('KUBECONFIG', ''))
        new_namespace = notebook.project.notebook_namespace
        old_namespace = notebook.namespace
        SERVICE_EXTERNAL_IP = []

        # 先使用项目组的
        if notebook.project.expand:
            SERVICE_EXTERNAL_IP = json.loads(notebook.project.expand).get('SERVICE_EXTERNAL_IP', '')
        # 使用集群的ip
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = notebook.project.cluster.get('HOST','').split('|')[0].strip().split(':')[0]
        # 使用全局ip
        if not SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = conf.get('SERVICE_EXTERNAL_IP', None)
        # 使用当前url的
        if not SERVICE_EXTERNAL_IP:
            if core.checkip(request.host.split(':')[0]):
                SERVICE_EXTERNAL_IP = request.host.split(':')[0]

        if SERVICE_EXTERNAL_IP and type(SERVICE_EXTERNAL_IP) == str:
            SERVICE_EXTERNAL_IP = [SERVICE_EXTERNAL_IP]
        if SERVICE_EXTERNAL_IP:
            SERVICE_EXTERNAL_IP = [x.split('|')[0].strip().split(':')[0] for x in SERVICE_EXTERNAL_IP]

        port = 3000

        command = None
        workingDir = None
        health=None
        volume_mount = notebook.volume_mount
        # 端口+0是jupyterlab  +1是sshd   +2 +3 是预留的用户自己启动应用占用的端口
        port_str = conf.get('NOTEBOOK_PORT','10000+10*ID').replace('ID', str(notebook.id))
        meet_ports = core.get_not_black_port(int(eval(port_str)))
        env = {
            "NO_AUTH": "true",
            "DISPLAY": ":10.0",   # 屏幕投屏时使用
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
            env["SERVICE_EXTERNAL_IP"] = SERVICE_EXTERNAL_IP[0].split('|')[-1].split(':')[0]


        annotations={
            'project': notebook.project.name
        }
        notebook.namespace = new_namespace
        db.session.commit()

        k8s_client.create_debug_pod(
            namespace=new_namespace,
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
            privileged=None,   # 这里设置privileged 才能看到所有的gpu卡，小心权限太高
            accounts=conf.get('JUPYTER_ACCOUNTS',''),
            username=notebook.created_by.username
        )
        k8s_client.create_service(
            namespace=new_namespace,
            name=notebook.name,
            username=notebook.created_by.username,
            ports=[port, ],
            selector=labels
        )

        crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})
        old_crd_name = f"notebook-{old_namespace}-{notebook.name.replace('_', '-')}" #  notebook.name.replace('_', '-')
        new_crd_name = f"notebook-{new_namespace}-{notebook.name.replace('_', '-')}"  # notebook.name.replace('_', '-')
        vs_obj = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=old_namespace, name=old_crd_name)
        if vs_obj:
            k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=old_namespace, name=old_crd_name)
            time.sleep(1)
        host=None
        # 优先使用项目组配置的
        if SERVICE_EXTERNAL_IP:
            host = SERVICE_EXTERNAL_IP[0]
        # 再使用项目配置
        if not host:
            host = notebook.project.cluster.get('HOST', request.host).split('|')[0].strip().split(':')[0]
        # 最后使用当前域名
        if not host:
            host = request.host.split(':')[0]
        crd_json = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": new_crd_name,
                "namespace": new_namespace
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
                                    "prefix": f"/notebook/jupyter/{notebook.name}/"
                                },
                                "headers": {
                                    "cookie":{
                                        "regex": ".*myapp_username=.*"
                                    }
                                }
                            }
                        ],
                        "rewrite": {
                            "uri": rewrite_url
                        },
                        "route": [
                            {
                                "destination": {
                                    "host": "%s.%s.svc.cluster.local" % (notebook.name, new_namespace),
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
        try:
            crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=new_namespace, body=crd_json)
        except:
            pass
        # 边缘模式时，需要根据项目组中的配置设置代理ip
        if meet_ports[0]>20000:
            flash(__('端口已耗尽，ssh连接notebook请通过跳板机跳转连接'), 'warning')

        if SERVICE_EXTERNAL_IP and SERVICE_EXTERNAL_IP[0]!='127.0.0.1' and meet_ports[0]<20000:
            SERVICE_EXTERNAL_IP = [ip.split('|')[0].strip().split(':')[0] for ip in SERVICE_EXTERNAL_IP]
            ports = [port]
            ports.append(meet_ports[1])   # 给每个notebook多开一个端口，ssh的端口

            # for index in range(1, 4):
            #     ports.append(meet_ports[index])

            # ports = list(set(ports))  # 这里会乱序
            service_ports = [[meet_ports[index], port] for index, port in enumerate(ports)]
            service_external_name = (notebook.name + "-external").lower()[:60].strip('-')
            k8s_client.create_service(
                namespace=new_namespace,
                name=service_external_name,
                username=notebook.created_by.username,
                ports=service_ports,
                selector=labels,
                service_type='ClusterIP' if conf.get('K8S_NETWORK_MODE','iptables')!='ipvs' else 'NodePort',
                external_ip=SERVICE_EXTERNAL_IP if conf.get('K8S_NETWORK_MODE','iptables')!='ipvs' else None
            )

    # @event_logger.log_this
    @expose_api(description="重置在线ide",url='/reset/<notebook_id>', methods=['GET', 'POST'])
    def reset(self, notebook_id):

        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        try:
            self.reset_notebook(notebook)
            flash(__('已重置，Running状态后可进入。注意：notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。'), 'info')
        except Exception as e:
            message = __('重置失败，稍后重试。') + str(e)
            flash(message, 'warning')
            return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))
            # return self.response(400, **{"message": message, "status": 1, "result": {}})

        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))

    # @event_logger.log_this
    @expose_api(description="在线ide续期",url='/renew/<notebook_id>', methods=['GET', 'POST'])
    def renew(self, notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        notebook.changed_on = datetime.datetime.now()
        db.session.commit()
        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))

    # 基础批量删除
    # @pysnooper.snoop()
    def base_muldelete(self, items, cluster=None):
        if not items:
            return
            # abort(404)

        for item in items:
            try:
                if not cluster:
                    cluster = item.project.cluster['NAME']
                k8s_client = K8s(conf.get('CLUSTERS').get(cluster).get('KUBECONFIG', ''))
                namespace = item.namespace
                k8s_client.delete_pods(namespace=namespace,pod_name=item.name)
                k8s_client.delete_service(namespace=namespace,name=item.name)
                k8s_client.delete_service(namespace=namespace, name=(item.name + "-external").lower()[:60].strip('-'))
                crd_info = conf.get("CRD_INFO", {}).get('virtualservice', {})
                if crd_info:
                    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=item.namespace, name="notebook-jupyter-%s" % item.name.replace('_', '-'))

            except Exception as e:
                flash(str(e), "warning")

    def pre_delete(self, item):
        self.base_muldelete([item])

    @expose_api(description="在线ide的列表查询",url="/list/")
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
    # @expose_api(description="",url="/delete/<pk>")
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

    @expose_api(description="停止在线ide",url='/stop/<notebook_id>', methods=['GET', 'POST'])
    def stop(self, notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        self.base_muldelete([notebook])
        flash(_('清理完成'),'info')
        return redirect(conf.get('MODEL_URLS', {}).get('notebook', ''))
    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        return self._muldelete(items)

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
    show_columns = ['project', 'name', 'namespace', 'describe', 'images', 'working_dir', 'env', 'volume_mount','resource_memory', 'resource_cpu', 'resource_gpu', 'status', 'ide_type', 'image_pull_policy', 'expand']
appbuilder.add_api(Notebook_ModelView_SDK_Api)
