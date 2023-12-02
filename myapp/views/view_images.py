import os

from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import lazy_gettext as _
from myapp.models.model_job import Repository,Images
from myapp import app, appbuilder, db
from wtforms.validators import DataRequired, Length, Regexp
from wtforms import StringField, SelectField
import pysnooper
import json
from flask import redirect, flash
from myapp.utils.py.py_k8s import K8s
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget
from flask_appbuilder import expose
from .baseApi import MyappModelRestApi
from flask import g
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
from .view_team import Project_Join_Filter

conf = app.config
logging = app.logger


class Repository_ModelView_Base():
    datamodel = SQLAInterface(Repository)

    label_title = '仓库'
    check_redirect_list_url = conf.get('MODEL_URLS', {}).get('repository', '')
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    order_columns = ['id']
    search_columns = ['name', 'server', 'hubsecret', 'user']
    list_columns = ['name', 'hubsecret', 'creator', 'modified']
    cols_width = {
        "name": {"type": "ellip2", "width": 250},
        "hubsecret": {"type": "ellip2", "width": 250},
    }
    show_exclude_columns = ['password']
    add_columns = ['name', 'server', 'user', 'password', 'hubsecret']
    edit_columns = add_columns

    add_form_extra_fields = {
        "server": SelectField(
            _(datamodel.obj.lab('server')),
            widget=MySelect2Widget(can_input=True),
            default='harbor.oa.com',
            choices=[['harbor.oa.com','harbor.oa.com'],['ccr.ccs.tencentyun.com','ccr.ccs.tencentyun.com'],['registry.docker-cn.com','registry.docker-cn.com']],
            description="镜像仓库服务地址"
        ),
        "user": StringField(
            _(datamodel.obj.lab('user')),
            default='',
            widget=BS3TextFieldWidget(),
            description="镜像仓库的用户名"
        ),
        "password": StringField(
            _(datamodel.obj.lab('password')),
            default='',
            widget=BS3TextFieldWidget(),
            description="镜像仓库的链接密码"
        )
    }

    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def set_column(self):
        self.add_form_extra_fields['name'] = StringField(
            _(self.datamodel.obj.lab('name')),
            default=g.user.username + "-",
            widget=BS3TextFieldWidget(),
            description="仓库名称"
        )

        self.add_form_extra_fields['hubsecret'] = StringField(
            _(self.datamodel.obj.lab('hubsecret')),
            default=g.user.username + "-hubsecret",
            widget=BS3TextFieldWidget(),
            description="在k8s中创建的hub secret",
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54), DataRequired()]
        )

    pre_add_web = set_column

    # create hubsecret
    # @pysnooper.snoop()
    def apply_hubsecret(self, hubsecret):
        from myapp.utils.py.py_k8s import K8s
        all_cluster = conf.get('CLUSTERS', {})
        all_kubeconfig = [all_cluster[cluster].get('KUBECONFIG', '') for cluster in all_cluster] + ['']
        all_kubeconfig = list(set(all_kubeconfig))
        for kubeconfig in all_kubeconfig:
            try:
                k8s = K8s(kubeconfig)
                namespaces = conf.get('HUBSECRET_NAMESPACE')
                for namespace in namespaces:
                    try:
                        k8s.apply_hubsecret(namespace=namespace,
                                            name=hubsecret.hubsecret,
                                            user=hubsecret.user,
                                            password=hubsecret.password,
                                            server=hubsecret.server
                                            )
                    except Exception as e1:
                        print(e1)
            except Exception as e:
                print(e)

    def post_add(self, item):
        self.apply_hubsecret(item)

    def post_update(self, item):
        self.apply_hubsecret(item)


# class Repository_ModelView(Repository_ModelView_Base,MyappModelView,DeleteMixin):
#     datamodel = SQLAInterface(Repository)
#
# appbuilder.add_view(Repository_ModelView,"仓库",icon = 'fa-shopping-basket',category = '训练',category_icon = 'fa-sitemap')


class Repository_ModelView_Api(Repository_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Repository)

    route_base = '/repository_modelview/api'


appbuilder.add_api(Repository_ModelView_Api)


# 只能查看到自己归属的项目组的镜像
class Images_Filter(MyappFilter):
    # @pysnooper.snoop(watch_explode=('result'))
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query.order_by(self.model.id.desc())

        result = query.order_by(self.model.id.desc())
        return result


class Images_ModelView_Base():
    label_title = '镜像'
    datamodel = SQLAInterface(Images)
    check_redirect_list_url = conf.get('MODEL_URLS', {}).get('images', '')

    list_columns = ['project','images_url', 'creator', 'modified']
    cols_width = {
        "images_url": {"type": "ellip2", "width": 500},
    }
    search_columns = ['created_by', 'project', 'repository', 'name', 'describe']
    base_order = ('id', 'desc')
    order_columns = ['id']
    add_columns = ['project','repository', 'name', 'describe', 'dockerfile', 'gitpath']
    edit_columns = add_columns
    spec_label_columns={
        "build":"构建"
    }
    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'job-template']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields


    add_form_extra_fields = {
        "dockerfile": StringField(
            _(datamodel.obj.lab('dockerfile')),
            description='镜像的构建Dockerfile全部内容，/mnt/$username/是个人存储目录，可以从此目录下复制文件到镜像中',
            default='',
            widget=MyBS3TextAreaFieldWidget(rows=10),
        ),
        "name": StringField(
            _(datamodel.obj.lab('name')),
            description='镜像名称全称，例如ubuntu:20.04',
            default='',
            widget=BS3TextFieldWidget(),
        ),
        "entrypoint": StringField(
            _(datamodel.obj.lab('entrypoint')),
            description='镜像的入口命令，直接写成单行字符串，例如python xx.py，无需添加[]',
            default='',
            widget=BS3TextFieldWidget(),
        )
    }

    edit_form_extra_fields = add_form_extra_fields
    base_filters = [["id", Images_Filter, lambda: []]]





class Images_ModelView_Api(Images_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Images)
    route_base = '/images_modelview/api'


appbuilder.add_api(Images_ModelView_Api)
