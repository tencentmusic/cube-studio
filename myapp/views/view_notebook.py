from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import random
# 将model添加成视图，并控制在前端的显示
import uuid
from myapp.models.model_notebook import Notebook
from myapp.models.model_job import Repository
from flask_appbuilder.actions import action
from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith,FilterEqual,FilterNotEqual
from wtforms.validators import EqualTo,Length
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db,event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField, SelectField, StringField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MyCommaSeparatedListField,MySelectMultipleField
from myapp.security import MyUser
from myapp.utils.py import py_k8s
from flask_wtf.file import FileField
import shlex
import re,copy
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from .baseApi import (
    MyappModelRestApi
)
from myapp import security_manager
from werkzeug.datastructures import FileStorage
from kubernetes import client as k8s_client
from .base import (
    api,
    BaseMyappView,
    check_ownership,
    data_payload_response,
    DeleteMixin,
    generate_download_headers,
    get_error_msg,
    get_user_roles,
    handle_api_exception,
    json_error_response,
    json_success,
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
from myapp.views.view_team import Project_Filter,Project_Join_Filter,filter_join_org_project
from kubernetes.client import V1ObjectMeta

conf = app.config

class Notebook_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        return query.filter(self.model.created_by_fk==g.user.id)



# 定义数据库视图
class Notebook_ModelView_Base():
    datamodel = SQLAInterface(Notebook)
    label_title='notebook'
    check_redirect_list_url = '/notebook_modelview/list/'
    crd_name = 'notebook'
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    datamodel = SQLAInterface(Notebook)
    conv = GeneralModelConverter(datamodel)
    base_permissions = ['can_add', 'can_delete','can_edit', 'can_list', 'can_show']  # 默认为这些
    base_order = ('changed_on', 'desc')
    base_filters = [["id", Notebook_Filter, lambda: []]]  # 设置权限过滤器
    order_columns = ['id']
    add_columns = ['project','name','describe','images','working_dir','volume_mount','resource_memory','resource_cpu']
    list_columns = ['project','ide_type','name_url','resource','status','renew','reset']
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields
    # @pysnooper.snoop()
    def set_column(self, notebook=None):
        # 对编辑进行处理
        self.add_form_extra_fields['name'] = StringField(
            _(self.datamodel.obj.lab('name')),
            default="%s-"%g.user.username+uuid.uuid4().hex[:4],
            description='英文名(字母、数字、-组成)，最长50个字符',
            widget=MyBS3TextFieldWidget(readonly=True if notebook else False),
            validators=[DataRequired(),Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54)]   # 注意不能以-开头和结尾
        )
        self.add_form_extra_fields['describe'] = StringField(
            _(self.datamodel.obj.lab('describe')),
            default='%s的个人notebook'%g.user.username,
            description='中文描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )

        # "project": QuerySelectField(
        #     _(datamodel.obj.lab('project')),
        #     query_factory=filter_join_org_project,
        #     allow_blank=True,
        #     widget=Select2Widget()
        # ),

        self.add_form_extra_fields['project'] = QuerySelectField(
            _(self.datamodel.obj.lab('project')),
            description=_(r'部署项目组'),
            query_factory=filter_join_org_project,
            widget=MySelect2Widget(extra_classes="readonly" if notebook else None, new_web=False),
        )
        self.add_form_extra_fields['images'] = SelectField(
            _(self.datamodel.obj.lab('images')),
            description=_(r'notebook基础环境镜像，如果显示不准确，请删除新建notebook'),
            widget=MySelect2Widget(extra_classes="readonly" if notebook else None,new_web=False),
            choices=conf.get('NOTEBOOK_IMAGES',[]),
            # validators=[DataRequired()]
        )

        self.add_form_extra_fields['node_selector'] = StringField(
            _(self.datamodel.obj.lab('node_selector')),
            default='cpu=true,notebook=true',
            description="部署task所在的机器",
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['image_pull_policy'] = SelectField(
            _(self.datamodel.obj.lab('image_pull_policy')),
            description="镜像拉取策略(Always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)",
            widget=Select2Widget(),
            choices=[['Always', 'Always'], ['IfNotPresent', 'IfNotPresent']]
        )
        self.add_form_extra_fields['volume_mount'] = StringField(
            _(self.datamodel.obj.lab('volume_mount')),
            default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives',
            description='外部挂载，格式:$pvc_name1(pvc):/$container_path1,$pvc_name2(pvc):/$container_path2',
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['working_dir'] = StringField(
            _(self.datamodel.obj.lab('working_dir')),
            default='/mnt',
            description="工作目录，如果为空，则使用Dockerfile中定义的workingdir",
            widget=BS3TextFieldWidget()
        )
        self.add_form_extra_fields['resource_memory'] = StringField(
            _(self.datamodel.obj.lab('resource_memory')),
            default=Notebook.resource_memory.default.arg,
            description='内存的资源使用限制，示例：1G，20G',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )
        self.add_form_extra_fields['resource_cpu'] = StringField(
            _(self.datamodel.obj.lab('resource_cpu')),
            default=Notebook.resource_cpu.default.arg,
            description='cpu的资源使用限制(单位：核)，示例：2', widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        )



        gpu_type = conf.get("GPU_TYPE", "NVIDIA")  # TENCENT
        if gpu_type=='NVIDIA':
            self.add_form_extra_fields['resource_gpu'] = StringField(
                _(self.datamodel.obj.lab('resource_gpu')),
                default='0',
                description='gpu的资源使用限gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡',
                widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )
        if gpu_type=='TENCENT':
            self.add_form_extra_fields['resource_gpu'] = StringField(
                _(self.datamodel.obj.lab('resource_gpu')),
                default='0,0',
                description='gpu的资源使用限制(core,memory)，示例:10,2（10%的单卡核数和2G的显存,最大100,13），其中core为小于100的整数或100的整数倍，表示占用的单卡的百分比例，memory为整数，表示n(G)的显存', widget=BS3TextFieldWidget(),
                validators=[DataRequired()]
            )
        columns = ['name','describe','images','resource_memory','resource_cpu','resource_gpu']

        if g.user.is_admin():
            columns.append('volume_mount')
        self.add_columns = ['project']+columns
        self.edit_columns = ['project']+columns
        self.edit_form_extra_fields=self.add_form_extra_fields


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

        if 'theia' in item.images or 'vscode' in item.images:
            item.ide_type = 'theia'
        else:
            item.ide_type = 'jupyter'

    # @pysnooper.snoop(watch_explode=('item'))
    def pre_update(self, item):

        # if item.changed_by_fk:
        #     item.changed_by=db.session.query(MyUser).filter_by(id=item.changed_by_fk).first()
        # if item.created_by_fk:
        #     item.created_by=db.session.query(MyUser).filter_by(id=item.created_by_fk).first()

        self.pre_add(item)


    def post_add(self, item):
        flash('自动reset 一分钟后生效','warning')
        self.reset_notebook(item)

    @pysnooper.snoop(watch_explode=('item'))
    def post_update(self, item):
        flash('reset以后配置方可生效', 'warning')

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
        flash('注意：notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行',category='warning')
        # items.sort(key=lambda item:item.created_by.username==g.user.username,reverse=True)
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

    pre_update_get=set_column
    pre_add_get=set_column


    # @pysnooper.snoop(watch_explode=('notebook'))
    def reset_notebook(self, notebook):
        self.reset_theia(notebook)


    # 部署pod，service，VirtualService
    # @pysnooper.snoop(watch_explode=('notebook',))
    def reset_theia(self, notebook):
        from myapp.utils.py.py_k8s import K8s

        k8s = K8s(notebook.cluster['KUBECONFIG'])
        namespace = conf.get('NOTEBOOK_NAMESPACE')
        port=3000

        command=None
        workingDir=None
        volume_mount = notebook.volume_mount
        rewrite_url = '/'
        if notebook.ide_type=='jupyter':
            rewrite_url = '/notebook/jupyter/%s/' % notebook.name
            workingDir = '/mnt/%s' % notebook.created_by.username
            command = ["sh", "-c", "jupyter lab --notebook-dir=%s --ip=0.0.0.0 "
                                    "--no-browser --allow-root --port=%s "
                                    "--NotebookApp.token='' --NotebookApp.password='' "
                                    "--NotebookApp.allow_origin='*' "
                                    "--NotebookApp.base_url=%s" % ('/mnt/%s' % notebook.created_by.username if "(pvc)" in notebook.volume_mount else "/mnt/",port,rewrite_url)]
            volume_mount +=',2G(memory):/dev/shm'

        elif notebook.ide_type=='theia':
            command = ["node","/home/theia/src-gen/backend/main.js",  "/home/project","--hostname=0.0.0.0","--port=%s"%port]
            workingDir = '/home/theia'
        print(command)
        print(workingDir)


        image_secrets = conf.get('HUBSECRET', [])
        user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == notebook.created_by.id).all()
        if user_hubsecrets:
            for hubsecret in user_hubsecrets:
                if hubsecret[0] not in image_secrets:
                    image_secrets.append(hubsecret[0])


        k8s.create_debug_pod(
            namespace=namespace,
            name=notebook.name,
            labels={"app":notebook.name,'user':notebook.created_by.username},
            command=command,
            args=None,
            volume_mount=volume_mount,
            working_dir=workingDir,
            node_selector=notebook.get_node_selector(),
            resource_memory=notebook.resource_memory,
            resource_cpu=notebook.resource_cpu,
            resource_gpu=notebook.resource_gpu,
            image_pull_policy=notebook.image_pull_policy,
            image_pull_secrets=image_secrets,
            image=notebook.images,
            hostAliases=conf.get('HOSTALIASES',''),
            env={
             "NO_AUTH": "true"
            },
            privileged=None,
            accounts=conf.get('JUPYTER_ACCOUNTS'),
            username=notebook.created_by.username
        )
        k8s.create_service(
            namespace=namespace,
            name=notebook.name,
            username=notebook.created_by.username,
            ports=[port,])

        crd_info = conf.get('CRD_INFO', {}).get('virtualservice', {})
        crd_name = "notebook-jupyter-%s"%notebook.name.replace('_', '-') #  notebook.name.replace('_', '-')
        vs_obj = k8s.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
        if vs_obj:
            k8s.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
            time.sleep(1)

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
                   "*" if core.checkip(notebook.cluster.get('JUPYTER_DOMAIN')) else notebook.cluster.get('JUPYTER_DOMAIN')
                ],
                "http": [
                    {
                        "match": [
                            {
                                "uri": {
                                    "prefix": "/notebook/%s/%s/"%(namespace,notebook.name)
                                }
                            }
                        ],
                        "rewrite": {
                            "uri": rewrite_url
                        },
                        "route": [
                            {
                                "destination": {
                                    "host": "%s.%s.svc.cluster.local"%(notebook.name,namespace),
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
        crd = k8s.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=crd_json)
        return crd

    # @event_logger.log_this
    @expose('/reset/<notebook_id>',methods=['GET','POST'])
    def reset(self,notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        try:
            notebook_crd = self.reset_notebook(notebook)
            flash('已重置，Running状态后可进入。注意：notebook会定时清理，如要运行长期任务请在pipeline中创建任务流进行。','warning')
        except Exception as e:
            flash('重置失败，稍后重试。%s'%str(e), 'warning')
        return redirect(self.check_redirect_list_url)

    # @event_logger.log_this
    @expose('/renew/<notebook_id>',methods=['GET','POST'])
    def renew(self,notebook_id):
        notebook = db.session.query(Notebook).filter_by(id=notebook_id).first()
        notebook.changed_on=datetime.datetime.now()
        db.session.commit()
        return redirect(self.check_redirect_list_url)

    # 基础批量删除
    # @pysnooper.snoop()
    def base_muldelete(self,items):
        if not items:
            abort(404)
        for item in items:
            try:
                k8s_client = py_k8s.K8s(item.cluster['KUBECONFIG'])
                k8s_client.delete_pods(namespace=item.namespace,pod_name=item.name)
                k8s_client.delete_service(namespace=item.namespace,name=item.name)
                crd_info = conf.get("CRD_INFO", {}).get('virtualservice', {})
                if crd_info:
                    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'],
                                      plural=crd_info['plural'], namespace=item.namespace, name="notebook-jupyter-%s"%item.name.replace('_', '-'))

            except Exception as e:
                flash(str(e), "warning")

    def pre_delete(self,item):
        self.base_muldelete([item])



    @event_logger.log_this
    @expose("/list/")
    @has_access
    def list(self):
        args = request.args.to_dict()
        if '_flt_0_created_by' in args and args['_flt_0_created_by']=='':
            print(request.url)
            print(request.path)
            return redirect(request.url.replace('_flt_0_created_by=','_flt_0_created_by=%s'%g.user.id))

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

    @action(
        "stop_all", __("Stop"), __("Stop all Really?"), "fa-trash", single=False
    )
    def stop_all(self, items):
        self.base_muldelete(items)
        self.update_redirect()
        return redirect(self.get_redirect())



class Notebook_ModelView(Notebook_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Notebook)
# 添加视图和菜单
appbuilder.add_view(Notebook_ModelView,"notebook",href="/notebook_modelview/list/?_flt_0_created_by=",icon = 'fa-shopping-basket',category = '在线开发',category_icon = 'fa-glass')


# 添加api
class Notebook_ModelView_Api(Notebook_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Notebook)
    route_base = '/notebook_modelview/api'

appbuilder.add_api(Notebook_ModelView_Api)


