import os

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.model_job import Repository,Images
from myapp.views.view_team import Creator_Filter, Project_Join_Filter, Project_Filter
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

conf = app.config




class Repository_ModelView_Base():
    datamodel = SQLAInterface(Repository)

    label_title = _('仓库')
    base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
    base_order = ('id', 'desc')
    # base_filters = [["id", Creator_Filter, lambda: []]]
    order_columns = ['id']
    search_columns = ['name', 'server', 'hubsecret', 'user']
    list_columns = ['name', 'server', 'hubsecret', 'creator', 'modified']
    cols_width = {
        "name": {"type": "ellip2", "width": 250},
        "hubsecret": {"type": "ellip2", "width": 250},
    }
    show_exclude_columns = ['password']
    add_columns = ['name', 'server', 'user', 'password', 'hubsecret']
    edit_columns = add_columns

    spec_label_columns={
        "server": _('仓库'),
        "user": _("用户名"),
        "hubsecret": 'k8s hubsecret',
    }

    add_form_extra_fields = {
        "server": SelectField(
            _('服务地址'),
            widget=MySelect2Widget(can_input=True),
            default='harbor.oa.com/xx/',
            choices=[['harbor.oa.com/xx/','harbor.oa.com/xx/'],['ccr.ccs.tencentyun.com/xx/','ccr.ccs.tencentyun.com/xx/'],['registry.docker-cn.com','registry.docker-cn.com']],
            description= _("镜像仓库地址，示例：")+conf.get('REPOSITORY_ORG','')
        ),
        "user": StringField(
            _('用户名'),
            default='',
            widget=BS3TextFieldWidget(),
            description= _("镜像仓库的用户名")
        ),
        "password": StringField(
            _('密码'),
            default='',
            widget=BS3TextFieldWidget(),
            description= _("镜像仓库的链接密码")
        )
    }

    edit_form_extra_fields = add_form_extra_fields

    # @pysnooper.snoop()
    def set_column(self):
        self.add_form_extra_fields['name'] = StringField(
            _('名称'),
            default=g.user.username + "-",
            widget=BS3TextFieldWidget(),
            description= _("仓库名称")
        )

        self.add_form_extra_fields['hubsecret'] = StringField(
            "hubsecret",
            default=g.user.username + "-hubsecret",
            widget=BS3TextFieldWidget(),
            description= _("在k8s中创建的hub secret"),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"), Length(1, 54), DataRequired()]
        )

    pre_add_web = set_column

    def check_edit_permission(self, item):
        if not g.user.is_admin() and g.user.username != item.created_by.username:
            return False
        return True
    check_delete_permission = check_edit_permission

    # create hubsecret
    # @pysnooper.snoop()
    def apply_hubsecret(self, repo):
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
                        server = repo.server[:repo.server.index('/')] if '/' in repo.server else repo.server
                        # print(server)
                        k8s.apply_hubsecret(namespace=namespace,
                                            name=repo.hubsecret,
                                            user=repo.user,
                                            password=repo.password,
                                            server=server
                                            )
                    except Exception as e1:
                        print(e1)
            except Exception as e:
                print(e)

    def post_add(self, item):
        self.apply_hubsecret(item)

    def post_update(self, item):
        self.apply_hubsecret(item)


class Repository_ModelView_Api(Repository_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Repository)

    route_base = '/repository_modelview/api'

appbuilder.add_api(Repository_ModelView_Api)



class Images_ModelView_Base():
    label_title = _('镜像')
    datamodel = SQLAInterface(Images)

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
        "project": _("功能分类")
    }
    add_form_query_rel_fields = {
        "project": [["name", Project_Filter, 'job-template']]
    }
    edit_form_query_rel_fields = add_form_query_rel_fields


    add_form_extra_fields = {
        "dockerfile": StringField(
            'dockerfile',
            description= _('镜像的构建Dockerfile全部内容，/mnt/$username/是个人存储目录，可以从此目录下复制文件到镜像中'),
            default='',
            widget=MyBS3TextAreaFieldWidget(rows=10),
        ),
        "name": StringField(
            _('名称'),
            description= _('镜像名称全称，例如ubuntu:20.04'),
            default='',
            widget=BS3TextFieldWidget(),
        ),
        "entrypoint": StringField(
            _('启动命令'),
            description= _('镜像的入口命令，直接写成单行字符串，例如python xx.py，无需添加[]'),
            default='',
            widget=BS3TextFieldWidget(),
        )
    }

    edit_form_extra_fields = add_form_extra_fields
    # base_filters = [["id", Creator_Filter, lambda: []]]

    def check_edit_permission(self, item):
        if not g.user.is_admin() and g.user.username != item.created_by.username:
            return False
        return True
    check_delete_permission = check_edit_permission




class Images_ModelView_Api(Images_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(Images)
    route_base = '/images_modelview/api'


appbuilder.add_api(Images_ModelView_Api)
