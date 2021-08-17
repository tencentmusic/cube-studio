from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.forms import GeneralModelConverter
import uuid
import re
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images
from myapp.views.view_team import Project_Filter
from myapp import app, appbuilder,db,event_logger

from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField

from .baseApi import (
    MyappModelRestApi
)
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
from myapp import security_manager
import kfp    # 使用自定义的就要把pip安装的删除了
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
conf = app.config
logging = app.logger



# 定义数据库视图
class Repository_ModelView_Base():
    datamodel = SQLAInterface(Repository)

    label_title='仓库'
    check_redirect_list_url = '/repository_modelview/list/'
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']  # 默认为这些
    base_order = ('id', 'desc')
    order_columns = ['id']
    list_columns = ['name','hubsecret','creator','modified']
    show_exclude_columns = ['password']
    add_columns = ['name','server','user','password','hubsecret']
    edit_columns = add_columns

    add_form_extra_fields = {
        "password": StringField(
            _(datamodel.obj.lab('password')),
            widget=BS3PasswordFieldWidget()  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )
    }
    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def set_column(self):
        self.add_form_extra_fields['name'] = StringField(
            _(self.datamodel.obj.lab('name')),
            default=g.user.username+"-",
            widget=BS3TextFieldWidget()  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )

        self.add_form_extra_fields['hubsecret'] = StringField(
            _(self.datamodel.obj.lab('hubsecret')),
            default=g.user.username + "-",
            widget=BS3TextFieldWidget()  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )

    pre_add_get = set_column

    # 直接创建hubsecret
    def apply_hubsecret(self,hubsecret):
        from myapp.utils.py.py_k8s import K8s
        all_cluster=conf.get('CLUSTERS',{})
        all_kubeconfig = [all_cluster[cluster]['KUBECONFIG'] for cluster in all_cluster]+['']
        all_kubeconfig = list(set(all_kubeconfig))
        for kubeconfig in all_kubeconfig:
            k8s = K8s(kubeconfig)
            namespaces = conf.get('HUBSECRET_NAMESPACE')
            for namespace in namespaces:
                k8s.apply_hubsecret(namespace=namespace,
                                    name=hubsecret.hubsecret,
                                    user=hubsecret.user,
                                    password=hubsecret.password,
                                    server=hubsecret.server
                                    )

    def post_add(self, item):
        self.apply_hubsecret(item)

    def post_update(self, item):
        self.apply_hubsecret(item)

class Repository_ModelView(Repository_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Repository)

# 添加视图和菜单
appbuilder.add_view(Repository_ModelView,"仓库",icon = 'fa-shopping-basket',category = '训练',category_icon = 'fa-tasks')

# 添加api
class Repository_ModelView_Api(Repository_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Repository)

    route_base = '/repository_modelview/api'

appbuilder.add_api(Repository_ModelView_Api)


# 只能查看到自己归属的项目组的镜像
class Images_Filter(MyappFilter):
    # @pysnooper.snoop(watch_explode=('result'))
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query
        from flask_sqlalchemy import BaseQuery
        # join_projects_id = security_manager.get_join_projects_id(db.session)
        # logging.info(join_projects_id)
        # result = query.filter(self.model.project_id.in_(join_projects_id)).order_by(self.model.changed_on.desc())

        result = query.order_by(self.model.id.desc())
        return result



# 定义数据库视图
class Images_ModelView_Base():
    label_title='镜像'
    datamodel = SQLAInterface(Images)
    check_redirect_list_url = '/images_modelview/list/?_flt_2_name='
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    list_columns = ['project','images_url','creator','modified']

    base_order = ('id', 'desc')
    order_columns = ['id']
    add_columns = ['project', 'repository', 'name', 'describe', 'dockerfile', 'gitpath']
    edit_columns = add_columns
    add_form_query_rel_fields = {
        "project": [["name", Project_Filter, 'job-template']]
    }

    edit_form_query_rel_fields = add_form_query_rel_fields
    add_form_extra_fields = {
        "dockerfile": StringField(
            _(datamodel.obj.lab('dockerfile')),
            description='镜像的构建Dockerfile全部内容',
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "name": StringField(
            _(datamodel.obj.lab('name')),
            description='镜像名称全称，例如ubuntu:20.04',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "entrypoint": StringField(
            _(datamodel.obj.lab('entrypoint')),
            description='镜像的入口命令，直接写成单行字符串，例如python xx.py，无需添加[]',
            widget=BS3TextFieldWidget(),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        )
    }
    # show_columns = ['project','repository','created_by','changed_by','created_on','changed_on','name','describe','entrypoint','dockerfile','gitpath']

    edit_form_extra_fields = add_form_extra_fields
    base_filters = [["id", Images_Filter, lambda: []]]  # 设置权限过滤器



class Images_ModelView(Images_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Images)

appbuilder.add_view(Images_ModelView,"镜像",href="/images_modelview/list/?_flt_2_name=",icon = 'fa-file-image-o',category = '训练',category_icon = 'fa-envelope')


# 添加api
class Images_ModelView_Api(Images_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Images)
    route_base = '/images_modelview/api'
    list_columns = ['project','images_url', 'repository', 'name', 'describe', 'dockerfile', 'gitpath','modified','creator']

appbuilder.add_api(Images_ModelView_Api)


appbuilder.add_separator("训练")   # 在指定菜单栏下面的每个子菜单中间添加一个分割线的显示。

