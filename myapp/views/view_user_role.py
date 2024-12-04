import re

from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from wtforms import SelectField, StringField
from myapp import appbuilder, conf
from flask_appbuilder.fieldwidgets import Select2Widget, BS3TextFieldWidget, Select2ManyWidget

from wtforms.validators import DataRequired, Regexp
from flask import (
    flash,
    g
)
from .baseApi import (
    MyappModelRestApi
)
import json
from flask_appbuilder import CompactCRUDMixin, expose
from myapp.security import MyUser, MyRole, MyUserRemoteUserModelView_Base


class User_ModelView_Base(MyUserRemoteUserModelView_Base):

    list_columns = ["username", "active", "roles_html"]

    spec_label_columns = {
        "get_full_name": _("全名称"),
        "first_name": _("姓"),
        "last_name": _("名"),
        "username": _("用户名"),
        "password": _("密码"),
        "active": _("激活"),
        "email": _("邮件"),
        "roles": _("角色"),
        "roles_html": _("角色"),
        "last_login": _("最近一次登录"),
        "login_count": _("登录次数"),
        "fail_login_count": _("登录失败次数"),
        "created_on": _("创建时间"),
        "created_by": _("创建者"),
        "changed_on": _("修改时间"),
        "changed_by": _("修改者"),
        "secret": _("秘钥"),
        "quota": _('额度'),
        "org": _("组织架构")
    }

# 添加api
class User_ModelView_Api(User_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(MyUser)
    route_base = '/users/api'

appbuilder.add_api(User_ModelView_Api)


class Role_ModelView_Base():
    label_title = _('角色')
    datamodel = SQLAInterface(MyRole)

    base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']
    edit_columns = ["name", 'permissions']
    add_columns = edit_columns
    show_columns = ["name", "permissions"]
    list_columns = ["name", "permissions_html"]
    spec_label_columns = {
        "name":_("名称"),
        "permissions":_("权限"),
        "permissions_html": _("权限"),
        "user":_("用户"),
        "user_html": _("用户"),
    }
    cols_width = {
        "name": {"type": "ellip2", "width": 100},
        "permissions_html": {"type": "ellip2", "width": 700}
    }

    order_columns=['id']
    search_columns = ["name"]
    base_order = ('id', 'desc')

# 添加api
class Role_ModelView_Api(Role_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(MyRole)
    route_base = '/roles/api'

appbuilder.add_api(Role_ModelView_Api)
