import re

from flask_login import current_user
import logging
import jwt
from flask_jwt_extended import current_user as current_user_jwt
import pysnooper
from flask import current_app
from flask_appbuilder.security.sqla import models as ab_models
from flask_appbuilder.security.sqla.manager import SecurityManager
from flask_babel import lazy_gettext as _
from flask_appbuilder.security.views import (
    PermissionModelView,
    PermissionViewModelView,
    RoleModelView,
    UserModelView
)
from markupsafe import Markup
from werkzeug.security import generate_password_hash,check_password_hash
from flask_appbuilder.security.sqla.models import assoc_user_role

from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget
from sqlalchemy import or_

from flask_appbuilder.security.views import expose

from flask import g, flash, request
from sqlalchemy import (
    Boolean,
    Text
)
from flask_appbuilder.security.sqla.models import assoc_permissionview_role
from sqlalchemy import select
from flask_appbuilder.const import (
    AUTH_DB,
    AUTH_LDAP,
    AUTH_OAUTH,
    AUTH_OID,
    AUTH_REMOTE_USER,
    LOGMSG_WAR_SEC_LOGIN_FAILED
)
from flask_appbuilder.security.views import SimpleFormView
from flask_appbuilder._compat import as_unicode
from wtforms import StringField
from wtforms.validators import DataRequired,Regexp,Length

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from flask_appbuilder.forms import DynamicForm


# expand user
from flask_appbuilder.security.sqla.models import User,Role
from sqlalchemy import Column, String,Integer,ForeignKey
from sqlalchemy.orm import backref, relationship
from sqlalchemy.ext.declarative import declared_attr

from myapp.models.base import MyappModelBase
class MyUser(User,MyappModelBase):
    __tablename__ = 'ab_user'
    nickname = Column(String(200))   # 昵称
    org = Column(String(200))   # Organization
    quota = Column(String(2000))  # 资源配额
    active = Column(Boolean,default=True)
    balance = Column(Text(),default="{}")  # 余额，冻结余额，余额不足是否关机，真实还是虚拟余额
    wechat = Column(String(200))  # 微信
    phone = Column(String(200))  # 电话号码
    coupon = Column(Text(),default="{}")  # 优惠券：名称，使用条件，生效时间，失效时间，余额，原始面值，状态
    voucher = Column(Text(),default="{}")  # 代金券：名称，类型，规则，最高抵扣，面额，折扣，生效时间，失效时间，状态，适用产品，适用范围
    bill = Column(Text(),default="{}")  # 发票信息:发票抬头,发票类型,纳税人识别号,开户银行名称,基本开户账号,注册场所地址,注册固定电话,收件人姓名,手机号,地址,
    real_name_authentication = Column(Text(),default="{}")  # 实名认证：真实姓名，身份证；企业证件类型，企业证件附件，企业名称，企业证件号码，法人/被授权人身份(法定代表人，被授予人)，身份证件附件正面，身份证件背面，姓名，证件号码，状态
    subaccount = Column(Text(),default="{}")  # 子用户，用于表征子账户的信息：父账户，累计消息，共享主帐号余额，子账户权限

    def get_full_name(self):
        return self.username

    def __repr__(self):
        return self.username

    def is_admin(self):
        user_roles = [role.name.lower() for role in list(self.roles)]
        if "admin" in user_roles:
            return True
        return False

    @property
    def secret(self):
        if self.changed_on:
            pass
            # help(self.changed_on)
            # timestamp = int(func.date_format(self.changed_on))
            timestamp = int(self.changed_on.timestamp())
            payload = {
                "iss": "cube-studio",
                "sub":self.username
                # "iat": timestamp,  # Issue period
                # "nbf": timestamp,  # Effective Date
                # "exp": timestamp + 60 * 60 * 24 * 30 * 12,  # Valid for 12 months
            }

            from myapp import conf
            global_password = conf.get('JWT_PASSWORD','cube-studio')
            encoded_jwt = jwt.encode(payload, global_password, algorithm='HS256')
            return encoded_jwt
        return ''

    @property
    def roles_html(self):
        return "["+', '.join([role.name for role in self.roles]) +"]"

import pysnooper
class MyRole(Role,MyappModelBase):
    __tablename__ = 'ab_role'

    @property
    def permissions_html(self):
        return "["+'],\n['.join([permission.permission.name + " on "+permission.view_menu.name for permission in self.permissions]) +"]"

# customize role view
class MyRoleModelView(RoleModelView):

    datamodel = SQLAInterface(MyRole)
    order_columns = ["id"]
    base_order = ('id', 'desc')
    route_base = "/roles"
    list_columns = ["name", "permissions"]
    base_permissions = ['can_list', 'can_edit', 'can_add', 'can_show']


class MyUserRemoteUserModelView_Base():
    label_title = _('用户')
    datamodel = SQLAInterface(MyUser)

    base_permissions = ['can_list', 'can_edit', 'can_add', 'can_show']

    list_columns = ["username",'nickname', "active", "roles"]

    edit_columns = ["username",'nickname','password', "active", "email", "roles", 'org']
    add_columns = ["username",'nickname','password', "email", "roles", 'org']
    show_columns = ["username", 'nickname', "active",'email','org', "roles",'secret','password']
    describe_columns={
        "roles": "Admin角色拥有管理员权限，Gamma为普通用户角色"
    }

    label_columns = {
        "get_full_name": _("全名称"),
        "first_name": _("姓"),
        "last_name": _("名"),
        "username": _("用户名"),
        'nickname':_("昵称"),
        "password": _("密码"),
        "active": _("激活"),
        "email": _("邮箱"),
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
    spec_label_columns = label_columns

    order_columns=['id']
    search_columns = ["username",'nickname', 'org']
    base_order = ('id', 'desc')
    # 个人查看详情额展示的信息
    user_show_fieldsets = [
        (
            _("用户信息"),
            {"fields": ["username",'nickname', "active", "roles", "email",'secret','org']},
        )
    ]
    show_fieldsets = user_show_fieldsets

    add_form_extra_fields = {
        "username" : StringField(
            _("用户名"),
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$")],
            widget=BS3TextFieldWidget(),
            description=_("用户名只能由小写字母、数字、-组成"),
        ),
        "nickname": StringField(
            _("昵称"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget(),
            description=_("中文名"),
        ),
        "password": StringField(
            _("密码"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget()
        ),
        "email": StringField(
            _("邮箱"),
            validators=[DataRequired(), Regexp(".*@.*\..*")],
            widget=BS3TextFieldWidget()
        ),
        "org": StringField(
            _("组织架构"),
            widget=BS3TextFieldWidget(),
            description=_("组织架构，自行填写"),
        )
    }
    edit_form_extra_fields = add_form_extra_fields


    # 添加默认gamma角色
    # @pysnooper.snoop()
    def post_add(self,user):
        from myapp import security_manager,db
        gamma_role = security_manager.find_role('Gamma')
        if gamma_role not in user.roles and not user.roles:
            user.roles.append(gamma_role)
            user.active=True
            db.session.commit()

        # 添加到public项目组
        try:
            from myapp.models.model_team import Project_User, Project
            public_project = db.session.query(Project).filter(Project.name == "public").filter(Project.type == "org").first()
            if public_project:
                project_user = Project_User()
                project_user.project = public_project
                project_user.role = 'dev'
                project_user.user_id = user.id
                db.session.add(project_user)
                db.session.commit()
        except Exception:
            db.session.rollback()

    def post_update(self,user):
        # 如果修改了账户，要更改labelstudio中的账户
        self.post_add(user)

    def pre_add(self,user):
        user.first_name = user.username
        user.last_name = ''
        user.active=True
        user.password = generate_password_hash(user.password)

    # @pysnooper.snoop()
    def pre_update(self,user):
        user.first_name = user.username
        user.last_name = ''
        if 'pbkdf2:sha256:' not in user.password:
            user.password = generate_password_hash(user.password)

class MyUserRemoteUserModelView(MyUserRemoteUserModelView_Base,UserModelView):
    datamodel = SQLAInterface(MyUser)


from myapp.project import MyCustomRemoteUserView
from myapp.project import Myauthdbview
# myapp自带的角色和角色权限，自定义了各种权限
# 基础类fab-Security-Manager中 def load_user(self, pk):  是用来认证用户的
# before_request是user赋值给g.user
# @pysnooper.snoop()
class MyappSecurityManager(SecurityManager):

    user_model = MyUser
    rolemodelview = MyRoleModelView  #

    # Remote Authentication
    userremoteusermodelview = MyUserRemoteUserModelView
    authremoteuserview = MyCustomRemoteUserView

    # Account password authentication
    userdbmodelview = MyUserRemoteUserModelView
    authdbview = Myauthdbview

    # 构建启动前工作，认证
    @staticmethod
    def before_request():
        g.user = current_user

    def __init__(self, appbuilder):
        super(MyappSecurityManager, self).__init__(appbuilder)
        # 添加从header中进行认证的方式
        self.lm.header_loader(self.load_user_from_header)

    # 使用header 认证，通过username名获取用户
    # @pysnooper.snoop()
    def load_user_from_header(self, authorization_value):
        # token=None
        # if 'token' in request.headers:
        #     token = request.headers['token']
        if authorization_value:
            from myapp import conf
            # username 免认证，设计到多平台调用时打开
            if len(authorization_value) < 40: # 任务模板请求后端api调用  and conf.get('AUTH_PLATFORM_ACCESS',False):
                username = authorization_value
                if username:
                    user = self.find_user(username)
                    g.user = user
                    return user
            else:  # token 认证
                encoded_jwt = authorization_value.encode('utf-8')
                payload = jwt.decode(encoded_jwt, conf.get('JWT_PASSWORD','cube-studio'), algorithms=['HS256'])
                # if payload['iat'] > time.time():
                #     return
                # elif payload['exp'] < time.time():
                #     return
                # else:
                user = self.find_user(payload['sub'])
                g.user = user
                return user

    # 自定义登录用户
    def load_user(self, pk):
        user = self.get_user_by_id(int(pk))
        # set cookie
        return user


    # 注册security菜单栏下的子菜单和链接
    # @pysnooper.snoop()
    def register_views(self):

        # Security APIs
        self.appbuilder.add_api(self.security_api)

        if self.auth_type == AUTH_DB:
            self.user_view = self.userdbmodelview
            self.auth_view = self.authdbview()

        elif self.auth_type == AUTH_LDAP:
            self.user_view = self.userdbmodelview
            self.auth_view = self.authdbview()
        elif self.auth_type == AUTH_OAUTH:
            self.user_view = self.userremoteusermodelview
            self.auth_view = self.authremoteuserview()
        elif self.auth_type == AUTH_REMOTE_USER:
            self.user_view = self.userremoteusermodelview
            self.auth_view = self.authremoteuserview()
        else:
            self.user_view = self.useroidmodelview
            self.auth_view = self.authoidview()
            if self.auth_user_registration:
                pass
                self.registeruser_view = self.registeruseroidview()
                self.appbuilder.add_view_no_menu(self.registeruser_view)

        self.appbuilder.add_view_no_menu(self.auth_view)

    # @pysnooper.snoop()
    def add_org_user(self,username,first_name,last_name,org,email,roles,password="",hashed_password=""):
        """
            Generic function to create user
        """
        try:
            user = self.user_model()
            user.first_name = first_name
            user.org = org
            user.last_name = last_name
            user.username = username
            user.email = email
            user.active = True
            user.roles+=roles   # 添加默认注册角色
            if password:
                user.password=password
            if hashed_password:
                user.password = generate_password_hash(hashed_password)
            self.get_session.add(user)
            self.get_session.commit()
            # 在项目组中添加用户
            try:
                from myapp.models.model_team import Project_User, Project
                public_project = self.get_session.query(Project).filter(Project.name == "public").filter(Project.type == "org").first()
                if public_project:
                    project_user = Project_User()
                    project_user.project = public_project
                    project_user.role = 'dev'
                    project_user.user_id = user.id
                    self.get_session.add(project_user)
                    self.get_session.commit()
            except Exception:
                self.get_session.rollback()

            return user
        except Exception:
            self.get_session.rollback()
            return False


    # 添加注册远程用户
    # @pysnooper.snoop()
    def auth_user_remote_org_user(self, username,org_name='',password='',hashed_password='',email='',first_name='',last_name=''):
        if not username:
            return None
        # 查找用户
        from myapp import conf
        user = self.find_user(username=username)
        # 添加以组织同名的角色，同时添加上级角色

        # 如果用户不存在就注册用户
        if user is None:
            user = self.add_org_user(
                username=username,
                first_name=first_name if first_name else username,
                last_name=last_name if last_name else username,
                password=password,
                hashed_password=hashed_password,
                org=org_name,               # 添加组织架构
                email=username + f"@{conf.get('APP_NAME','cube-studio').replace(' ','').lower()}.com" if not email else email,
                roles=[self.find_role(self.auth_user_registration_role)] if self.find_role(self.auth_user_registration_role) else []  #  org_role   添加gamma默认角色,    组织架构角色先不自动添加
            )
        elif not user.is_active:  # 如果用户未激活不允许接入
            print(LOGMSG_WAR_SEC_LOGIN_FAILED.format(username))
            return None
        if user:
            user.org = org_name if org_name else user.org
            user.email = email if email else user.email
            user.first_name = first_name if first_name else user.first_name
            user.last_name = last_name if last_name else user.last_name

            gamma_role = self.find_role(self.auth_user_registration_role)
            if gamma_role and gamma_role not in user.roles:
                user.roles.append(gamma_role)

            # 更新用户信息
            if org_name:
                user.org = org_name    # 更新组织架构字段
                # org_role = self.add_role(org_name)
                # if org_role not in user.roles:
                #     user.roles.append(org_role)

            self.update_user_auth_stat(user)

        return user

    # 只有admin才能看到的视图
    ADMIN_ONLY_VIEW_MENUS = {
        "Node_ModelView_Api",
        "Storage_ModelView_Api",
        "User_ModelView_Api",
        "Role_ModelView_Api",
        "LOG_ModelView_Api"
    }
    # 只有admin才有的权限
    ADMIN_ONLY_PERMISSIONS = {
        "can_override_role_permissions",
        "can_override_role_permissions",
        "can_approve",   # db owner需要授权approve 权限后才能授权
        "can_update_role",
    }
    # 只读者的权限，也就是所有人都有的权限
    PUBLIC_ONLY_PERMISSION = {"can_show", "can_list"}

    #
    # @pysnooper.snoop()
    def has_access(self, permission_name: str, view_name: str) -> bool:
        if current_user.is_authenticated:
            if current_user.is_admin():
                return True
            return self._has_view_access(g.user, permission_name, view_name)
        elif current_user_jwt:
            return self._has_view_access(current_user_jwt, permission_name, view_name)
        else:
            return self.is_item_public(permission_name, view_name)


    # 获取用户是否有在指定视图上的指定权限名
    # @pysnooper.snoop()
    def can_access(self, permission_name, view_name):
        """Protecting from has_access failing from missing perms/view"""
        user = g.user
        if user.is_anonymous:
            return self.is_item_public(permission_name, view_name)
        return self._has_view_access(user, permission_name, view_name)

    # 获取用户具有指定权限的视图
    def user_view_menu_names(self, permission_name: str):
        from myapp import db
        base_query = (
            db.session.query(self.viewmenu_model.name)
                .join(self.permissionview_model)
                .join(self.permission_model)
                .join(assoc_permissionview_role)
                .join(self.role_model)
        )

        # 非匿名用户
        if not g.user.is_anonymous:
            # filter by user id
            view_menu_names = (
                base_query.join(assoc_user_role)
                    .join(self.user_model)
                    .filter(self.user_model.id == g.user.id)
                    .filter(self.permission_model.name == permission_name)
            ).all()
            return list(set([s.name for s in view_menu_names]))

        # Properly treat anonymous user 匿名用户
        public_role = self.get_public_role()
        if public_role:
            # filter by public role
            view_menu_names = (
                base_query.filter(self.role_model.id == public_role.id).filter(
                    self.permission_model.name == permission_name
                )
            ).all()
            return list(set([s.name for s in view_menu_names]))
        return []


    # 在视图上添加权限
    def merge_perm(self, permission_name, view_menu_name):
        logging.warning(
            "This method 'merge_perm' is deprecated use add_permission_view_menu"
        )
        self.add_permission_view_menu(permission_name, view_menu_name)

    # 初始化自定义角色，将对应的权限加到对应的角色上
    # @pysnooper.snoop()
    def sync_role_definitions(self):
        """Inits the Myapp application with security roles and such"""

        logging.info("Syncing role definition")

        # Creating default roles
        self.set_role("Admin", self.is_admin_pvm)
        self.set_role("Gamma", self.is_gamma_pvm)
        self.set_role("Public", self.is_public_pvm)
        # commit role and view menu updates
        self.get_session.commit()
        self.clean_perms()


    # 清理权限
    def clean_perms(self):
        """FAB leaves faulty permissions that need to be cleaned up"""
        logging.info("Cleaning faulty perms")
        sesh = self.get_session
        pvms = sesh.query(ab_models.PermissionView).filter(
            or_(
                ab_models.PermissionView.permission == None,  # NOQA
                ab_models.PermissionView.view_menu == None,  # NOQA
            )
        )
        deleted_count = pvms.delete()
        sesh.commit()
        if deleted_count:
            logging.info("Deleted {} faulty permissions".format(deleted_count))

    # 为角色添加权限，pvm_check为自定义权限校验函数。这样变量权限，就能通过pvm_check函数知道时候应该把权限加到角色上
    def set_role(self, role_name, pvm_check):
        logging.info("Syncing {} perms".format(role_name))
        sesh = self.get_session
        # 获取所有的pv记录
        pvms = sesh.query(ab_models.PermissionView).all()
        # 获取权限和视图都有值的pv
        pvms = [p for p in pvms if p.permission and p.view_menu]
        # 添加或者获取role
        role = self.add_role(role_name)
        # 检查pv是否归属于该role
        role_pvms = [p for p in pvms if pvm_check(p)]
        role.permissions = role_pvms
        # 添加pv-role记录
        sesh.merge(role)
        sesh.commit()


    # 看一个权限是否是只有admin角色该有的权限
    def is_admin_only(self, pvm):

        return (
            pvm.view_menu.name in self.ADMIN_ONLY_VIEW_MENUS
            or pvm.permission.name in self.ADMIN_ONLY_PERMISSIONS
        )

    # 校验权限是否是默认所有人可接受的
    def is_accessible_to_all(self, pvm):
        return pvm.permission.name in self.PUBLIC_ONLY_PERMISSION

    # 看一个权限是否是admin角色该有的权限
    def is_admin_pvm(self, pvm):
        return True

    # 看一个权限是否是gamma角色该有的权限，只要不是admin特有的，就都给gamma
    def is_gamma_pvm(self, pvm):
        return not self.is_admin_only(pvm)

    # 看一个权限是否是public角色该有的权限，不是admin的视图并且是public的权限，才可以
    def is_public_pvm(self, pvm):
        if (
            pvm.view_menu.name not in self.ADMIN_ONLY_VIEW_MENUS
            and pvm.permission.name in self.PUBLIC_ONLY_PERMISSION
        ):
            return True

        return False



    # 创建视图，创建权限，创建视图-权限绑定记录。
    # @pysnooper.snoop()
    def set_perm(self, view_menu_name,permission_name):
        connection = self.get_session
        permission = self.find_permission(permission_name)
        view_menu = self.find_view_menu(view_menu_name)
        pv = None
        # 如果权限不存存在就创建
        if not permission:
            permission_table = (
                self.permission_model.__table__  # pylint: disable=no-member
            )
            connection.execute(permission_table.insert().values(name=permission_name))
            connection.commit()
            permission = self.find_permission(permission_name)

        # 如果视图不存在就创建
        if not view_menu:
            view_menu_table = self.viewmenu_model.__table__  # pylint: disable=no-member
            connection.execute(view_menu_table.insert().values(name=view_menu_name))
            connection.commit()
            view_menu = self.find_view_menu(view_menu_name)

        # 获取是否存在 视图-权限绑定  记录
        if permission and view_menu:
            pv = (
                self.get_session.query(self.permissionview_model)
                .filter_by(permission=permission, view_menu=view_menu)
                .first()
            )

        # 如果没有视图-权限绑定 记录，就创建
        if not pv and permission and view_menu:
            permission_view_table = (
                self.permissionview_model.__table__  # pylint: disable=no-member
            )
            connection.execute(
                permission_view_table.insert().values(
                    permission_id=permission.id, view_menu_id=view_menu.id
                )
            )
            connection.commit()
            # 重新获取权限视图绑定记录
            pv = (
                self.get_session.query(self.permissionview_model)
                    .filter_by(permission=permission, view_menu=view_menu)
                    .first()
            )
        return pv


    # 根据权限，视图，添加到相关pv-role
    # @pysnooper.snoop()
    def add_pv_role(self,permission_name,view_menu_name,role_name):
        pv = self.set_perm(view_menu_name=view_menu_name,permission_name=permission_name)
        if pv:
            try:
                session = self.get_session
                role = session.query(self.role_model).filter_by(name=role_name).first()
                if role:
                    # 为pvm-role表中添加记录
                    pv_role = session.query(assoc_permissionview_role.c.id).filter(assoc_permissionview_role.c.permission_view_id==pv.id).filter(assoc_permissionview_role.c.role_id==role.id).first()
                    if not pv_role:
                        session.execute(assoc_permissionview_role.insert().values(permission_view_id=pv.id, role_id=role.id))
                        session.commit()
            except Exception as e:
                logging.error(e)



    @classmethod
    def get_join_projects_id(self,session):
        from myapp.models.model_team import Project_User
        if g.user:
            project_users = session.query(Project_User).filter(Project_User.user_id == User.get_user_id()).all()

            projects_id = [project_user.project_id for project_user in project_users if project_user.project.type=='org']
            return projects_id
        else:
            return []

    @classmethod
    def get_join_projects(self,session):
        from myapp.models.model_team import Project_User
        if g.user:
            project_users = session.query(Project_User).filter(Project_User.user_id == g.user.id).all()
            return [project_user.project for project_user in project_users if project_user.project.type=='org']
        else:
            return []

    @classmethod
    def get_create_pipeline_ids(self,session):
        from myapp.models.model_job import Pipeline
        if g.user:
            pipeline_ids = session.query(Pipeline.id).filter(Pipeline.created_by_fk == User.get_user_id()).all()
            pipeline_ids = [pipeline_id[0] for pipeline_id in pipeline_ids]
            return pipeline_ids
        else:
            return []

    @classmethod
    def get_all_namespace(self,session):
        from myapp.models.model_team import Project
        all_namespaces = []
        projects = session.query(Project).all()
        for project in projects:
            all_namespaces.append(project.notebook_namespace)
            all_namespaces.append(project.pipeline_namespace)
            all_namespaces.append(project.service_namespace)
            all_namespaces.append(project.automl_namespace)

        all_namespaces = list(set(all_namespaces))
        return all_namespaces

    @classmethod
    def get_all_notebook_namespace(self,session):
        from myapp.models.model_team import Project
        all_namespaces = []
        projects = session.query(Project).all()
        for project in projects:
            all_namespaces.append(project.notebook_namespace)
        all_namespaces = list(set(all_namespaces))
        return all_namespaces

    @classmethod
    def get_all_pipeline_namespace(self,session):
        from myapp.models.model_team import Project
        all_namespaces = []
        projects = session.query(Project).all()
        for project in projects:
            all_namespaces.append(project.pipeline_namespace)
        all_namespaces = list(set(all_namespaces))
        return all_namespaces

    @classmethod
    def get_all_service_namespace(self,session):
        from myapp.models.model_team import Project
        all_namespaces = []
        projects = session.query(Project).all()
        for project in projects:
            all_namespaces.append(project.service_namespace)
        all_namespaces = list(set(all_namespaces))
        return all_namespaces

    @classmethod
    def get_all_automl_namespace(self,session):
        from myapp.models.model_team import Project
        all_namespaces = []
        projects = session.query(Project).all()
        for project in projects:
            all_namespaces.append(project.automl_namespace)
        all_namespaces = list(set(all_namespaces))
        return all_namespaces