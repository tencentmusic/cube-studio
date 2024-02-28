from flask_login import current_user
import logging
import jwt

from flask_babel import lazy_gettext
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
from werkzeug.security import generate_password_hash,check_password_hash
from flask_appbuilder.security.sqla.models import assoc_user_role

from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget
from sqlalchemy import or_

from flask_appbuilder.security.views import expose

from flask import g, flash, request

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




# user list page template
class MyappSecurityListWidget(ListWidget):
    """
        Redeclaring to avoid circular imports
    """
    template = "myapp/fab_overrides/list.html"


# role list page template
class MyappRoleListWidget(ListWidget):
    """
        Role model view from FAB already uses a custom list widget override
        So we override the override
    """
    template = "myapp/fab_overrides/list_role.html"
    def __init__(self, **kwargs):
        kwargs["appbuilder"] = current_app.appbuilder
        super().__init__(**kwargs)



# customize list,add,edit page
UserModelView.list_columns= ["username", "active", "roles"]
UserModelView.edit_columns= ["first_name", "last_name", "username", "active", "email"]
UserModelView.add_columns= ["first_name", "last_name", "username", "email", "active", "roles"]



UserModelView.list_widget = MyappSecurityListWidget
RoleModelView.list_widget = MyappRoleListWidget
PermissionViewModelView.list_widget = MyappSecurityListWidget
PermissionModelView.list_widget = MyappSecurityListWidget


# expand user
from flask_appbuilder.security.sqla.models import User,Role
from sqlalchemy import Column, String



class MyUser(User):
    __tablename__ = 'ab_user'
    org = Column(String(200))   # Organization
    quota = Column(String(2000))  # 资源配额


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
                "iss": self.username
                # "iat": timestamp,  # Issue period
                # "nbf": timestamp,  # Effective Date
                # "exp": timestamp + 60 * 60 * 24 * 30 * 12,  # Valid for 12 months
            }

            global_password = 'myapp'
            encoded_jwt = jwt.encode(payload, global_password, algorithm='HS256')
            return encoded_jwt
        return ''


# customize role view
class MyRoleModelView(RoleModelView):

    datamodel = SQLAInterface(Role)
    order_columns = ["id"]
    route_base = "/roles"
    list_columns = ["name", "permissions"]


class MyUserRemoteUserModelView_Base():
    datamodel = SQLAInterface(MyUser)
    list_columns = ["username", "active", "roles", ]
    edit_columns = ["first_name", "last_name", "username",'password', "active", "email", "roles", 'org', 'quota' ]
    add_columns = ["first_name", "last_name", "username",'password', "active", "email", "roles", 'org', 'quota']
    show_columns = ["username", "active", "roles", "login_count"]
    describe_columns={
        "org":"组织架构，自行填写",
        "quota": '资源限额，额度填写方式 $集群名,$资源组名,$命名空间,$资源类型,$限制类型,$限制值，其中$命名空间包含all,jupyter,pipeline,service,automl,aihub,$资源类型包含cpu,memory,gpu,$限制类型包含single,concurrent,total'
    }
    list_widget = MyappSecurityListWidget
    label_columns = {
        "get_full_name": lazy_gettext("Full Name"),
        "first_name": lazy_gettext("First Name"),
        "last_name": lazy_gettext("Last Name"),
        "username": lazy_gettext("User Name"),
        "password": lazy_gettext("Password"),
        "active": lazy_gettext("Is Active?"),
        "email": lazy_gettext("Email"),
        "roles": lazy_gettext("Role"),
        "last_login": lazy_gettext("Last login"),
        "login_count": lazy_gettext("Login count"),
        "fail_login_count": lazy_gettext("Failed login count"),
        "created_on": lazy_gettext("Created on"),
        "created_by": lazy_gettext("Created by"),
        "changed_on": lazy_gettext("Changed on"),
        "changed_by": lazy_gettext("Changed by"),
        "secret": lazy_gettext("Authorization"),
        "quota": lazy_gettext("quota"),
    }
    # 用户show页面
    show_fieldsets = [
        (
            lazy_gettext("User info"),
            {"fields": ["username", "active", "roles", "login_count",'secret']},
        ),
        (
            lazy_gettext("Personal Info"),
            {"fields": ["first_name", "last_name", "email",'org','quota'], "expanded": True},
        ),
        (
            lazy_gettext("Audit Info"),
            {
                "fields": [
                    "last_login",
                    "fail_login_count",
                    "created_on",
                    "created_by",
                    "changed_on",
                    "changed_by",
                ],
                "expanded": False,
            },
        ),
    ]
    # 个人查看详情额展示的信息
    user_show_fieldsets = [
        (
            lazy_gettext("User info"),
            {"fields": ["username", "active", "roles", "login_count",'secret']},
        ),
        (
            lazy_gettext("Personal Info"),
            {"fields": ["first_name", "last_name", "email",'org','quota'], "expanded": True},
        ),
    ]


    @expose("/userinfo/")
    @has_access
    def userinfo(self):
        item = self.datamodel.get(g.user.id, self._base_filters)
        widgets = self._get_show_widget(
            g.user.id, item, show_fieldsets=self.user_show_fieldsets
        )
        self.update_redirect()
        return self.render_template(
            self.show_template,
            title=self.user_info_title,
            widgets=widgets,
            appbuilder=self.appbuilder,
        )

    # 添加默认gamma角色
    # @pysnooper.snoop()
    def post_add(self,user):
        from myapp import security_manager,db
        gamma_role = security_manager.find_role('Gamma')
        if gamma_role not in user.roles:
            user.roles.append(gamma_role)
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



class MyUserRemoteUserModelView(MyUserRemoteUserModelView_Base,UserModelView):
    datamodel = SQLAInterface(MyUser)

from flask_appbuilder.security.views import SimpleFormView
from flask_appbuilder._compat import as_unicode
from flask_babel import lazy_gettext
from wtforms import StringField
from wtforms.validators import DataRequired

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from flask_appbuilder.forms import DynamicForm


class UserInfoEditView(SimpleFormView):

    class UserInfoEdit(DynamicForm):
        first_name = StringField(
            lazy_gettext("First Name"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget(),
            description=lazy_gettext("Write the user first name or names"),
        )
        last_name = StringField(
            lazy_gettext("Last Name"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget(),
            description=lazy_gettext("Write the user last name"),
        )
        username = StringField(
            lazy_gettext("User Name"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget(),
            description=lazy_gettext("Write the Username"),
        )
        password = StringField(
            lazy_gettext("Password"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget(),
            description=lazy_gettext("Password"),
        )
        email = StringField(
            lazy_gettext("Email"),
            validators=[DataRequired()],
            widget=BS3TextFieldWidget(),
            description=lazy_gettext("Write the Email"),
        )
        org = StringField(
            lazy_gettext("Org"),
            widget=BS3TextFieldWidget(),
            description=lazy_gettext("organization name"),
        )

    form = UserInfoEdit
    form_title = lazy_gettext("Edit User Information")
    redirect_url = "/"
    message = lazy_gettext("User information changed")

    def form_get(self, form):
        item = self.appbuilder.sm.get_user_by_id(g.user.id)
        # fills the form generic solution
        for key, value in form.data.items():
            if key == "csrf_token":
                continue
            form_field = getattr(form, key)
            form_field.data = getattr(item, key)

    def form_post(self, form):
        form = self.form.refresh(request.form)
        item = self.appbuilder.sm.get_user_by_id(g.user.id)
        form.populate_obj(item)
        self.appbuilder.sm.update_user(item)
        flash(as_unicode(self.message), "info")



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

    # userinfo edit view
    userinfoeditview = UserInfoEditView

    # 构建启动前工作，认证
    @staticmethod
    def before_request():
        g.user = current_user

    def __init__(self, appbuilder):
        super(MyappSecurityManager, self).__init__(appbuilder)
        # 添加从header中进行认证的方式
        self.lm.header_loader(self.load_user_from_header)

    # 使用header 认证，通过rtx名获取用户
    # @pysnooper.snoop()
    def load_user_from_header(self, authorization_value):
        # token=None
        # if 'token' in request.headers:
        #     token = request.headers['token']
        if authorization_value:
            # rtx 免认证
            if len(authorization_value) < 20:
                username = authorization_value
                if username:
                    user = self.find_user(username)
                    g.user = user
                    return user
            else:  # token 认证
                encoded_jwt = authorization_value.encode('utf-8')
                payload = jwt.decode(encoded_jwt, 'myapp', algorithms=['HS256'])
                # if payload['iat'] > time.time():
                #     return
                # elif payload['exp'] < time.time():
                #     return
                # else:
                user = self.find_user(payload['iss'])
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
        if not self.appbuilder.app.config.get('FAB_ADD_SECURITY_VIEWS', True):
            return
        # Security APIs
        self.appbuilder.add_api(self.security_api)

        if self.auth_user_registration:
            if self.auth_type == AUTH_DB:
                self.registeruser_view = self.registeruserdbview()
            elif self.auth_type == AUTH_OID:
                self.registeruser_view = self.registeruseroidview()
            elif self.auth_type == AUTH_OAUTH:
                self.registeruser_view = self.registeruseroauthview()
            if self.registeruser_view:
                self.appbuilder.add_view_no_menu(self.registeruser_view)

        self.appbuilder.add_view_no_menu(self.resetpasswordview())
        self.appbuilder.add_view_no_menu(self.resetmypasswordview())
        self.appbuilder.add_view_no_menu(self.userinfoeditview())



        if self.auth_type == AUTH_DB:
            self.user_view = self.userdbmodelview
            self.auth_view = self.authdbview()

        elif self.auth_type == AUTH_LDAP:
            self.user_view = self.userldapmodelview
            self.auth_view = self.authldapview()
        elif self.auth_type == AUTH_OAUTH:
            self.user_view = self.useroauthmodelview
            self.auth_view = self.authoauthview()
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

        self.user_view = self.appbuilder.add_view(
            self.user_view,
            "List Users",
            icon="fa-user",
            href="/users/list/?_flt_2_username=",
            label=_("List Users"),
            category="Security",
            category_icon="fa-cogs",
            category_label=_("Security"),
        )
        role_view = self.appbuilder.add_view(
            self.rolemodelview,
            "List Roles",
            icon="fa-group",
            href="/roles/list/?_flt_2_name=",
            label=_("List Roles"),
            category="Security",
            category_icon="fa-cogs",
        )
        role_view.related_views = [self.user_view.__class__]

        if self.userstatschartview:
            self.appbuilder.add_view(
                self.userstatschartview,
                "User's Statistics",
                icon="fa-bar-chart-o",
                label=_("User's Statistics"),
                category="Security",
            )
        if self.auth_user_registration:
            self.appbuilder.add_view(
                self.registerusermodelview,
                "User's Statistics",
                icon="fa-user-plus",
                label=_("User Registrations"),
                category="Security",
            )
        self.appbuilder.menu.add_separator("Security")
        self.appbuilder.add_view(
            self.permissionmodelview,
            "Base Permissions",
            icon="fa-lock",
            label=_("Base Permissions"),
            category="Security",
        )
        self.appbuilder.add_view(
            self.viewmenumodelview,
            "Views/Menus",
            icon="fa-list-alt",
            label=_("Views/Menus"),
            category="Security",
        )
        self.appbuilder.add_view(
            self.permissionviewmodelview,
            "Permission on Views/Menus",
            icon="fa-link",
            label=_("Permission on Views/Menus"),
            category="Security",
        )


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

        # 添加public项目组



    # 添加注册远程用户
    # @pysnooper.snoop()
    def auth_user_remote_org_user(self, username,org_name='',password='',email='',first_name='',last_name=''):
        if not username:
            return None
        # 查找用户
        from myapp import conf
        user = self.find_user(username=username)
        # 添加以组织同名的角色，同时添加上级角色
        # # 注册rtx同名角色
        # rtx_role = self.add_role(username)
        # 如果用户不存在就注册用户
        if user is None:
            user = self.add_org_user(
                username=username,
                first_name=first_name if first_name else username,
                last_name=last_name if last_name else username,
                password=password,
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
            # if rtx_role and rtx_role not in user.roles:
            #     user.roles.append(rtx_role)
            # 更新用户信息
            if org_name:
                user.org = org_name    # 更新组织架构字段
                # org_role = self.add_role(org_name)
                # if org_role not in user.roles:
                #     user.roles.append(org_role)

            self.update_user_auth_stat(user)

        return user



    READ_ONLY_MODEL_VIEWS = {
        'link','Minio','Kubernetes Dashboard','Granfana','Wiki'
    }


    USER_MODEL_VIEWS = {
        "UserDBModelView",
        "UserLDAPModelView",
        "UserOAuthModelView",
        "UserOIDModelView",
        "UserRemoteUserModelView",
    }


    # 只有admin才能看到的menu
    ADMIN_ONLY_VIEW_MENUS = {
        "ResetPasswordView",
        "RoleModelView",
        "List Users",
        "List Roles",
        "UserStatsChartView",
        "Base Permissions",
        "Permission on Views/Menus",
        "Action Log",
        "Views/Menus",
        "ViewMenuModelView",
        "User's Statistics",
        "Security",
    } | USER_MODEL_VIEWS

    ALPHA_ONLY_VIEW_MENUS = {}
    # 只有admin才有的权限
    ADMIN_ONLY_PERMISSIONS = {
        "can_override_role_permissions",
        "can_override_role_permissions",
        # "can_approve",   # db owner需要授权approve 权限后才能授权
        "can_update_role",
    }

    READ_ONLY_PERMISSION = {"can_show", "can_list",'can_add'}

    ALPHA_ONLY_PERMISSIONS = {
        "muldelete"
    }

    # 用户创建menu才有的权限
    OBJECT_SPEC_PERMISSIONS = {
        "can_only_access_owned_queries",
    }

    # 所有人都可以有的基本权限
    ACCESSIBLE_PERMS = {"can_userinfo","can_request_access","can_approve"}

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
            return set([s.name for s in view_menu_names])

        # Properly treat anonymous user 匿名用户
        public_role = self.get_public_role()
        if public_role:
            # filter by public role
            view_menu_names = (
                base_query.filter(self.role_model.id == public_role.id).filter(
                    self.permission_model.name == permission_name
                )
            ).all()
            return set([s.name for s in view_menu_names])
        return set()


    # 在视图上添加权限
    def merge_perm(self, permission_name, view_menu_name):
        logging.warning(
            "This method 'merge_perm' is deprecated use add_permission_view_menu"
        )
        self.add_permission_view_menu(permission_name, view_menu_name)

    # 判断权限是否是user自定义权限
    def is_user_defined_permission(self, perm):
        return perm.permission.name in self.OBJECT_SPEC_PERMISSIONS




    # 初始化自定义角色，将对应的权限加到对应的角色上
    # @pysnooper.snoop()
    def sync_role_definitions(self):
        """Inits the Myapp application with security roles and such"""

        logging.info("Syncing role definition")

        # Creating default roles
        self.set_role("Admin", self.is_admin_pvm)
        self.set_role("Gamma", self.is_gamma_pvm)


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
        # not readonly operations on read only model views allowed only for admins
        if (
            pvm.view_menu.name in self.READ_ONLY_MODEL_VIEWS
            and pvm.permission.name not in self.READ_ONLY_PERMISSION
        ):
            return True
        return (
            pvm.view_menu.name in self.ADMIN_ONLY_VIEW_MENUS
            or pvm.permission.name in self.ADMIN_ONLY_PERMISSIONS
        )


    # 校验权限是否是默认所有人可接受的
    def is_accessible_to_all(self, pvm):
        return pvm.permission.name in self.ACCESSIBLE_PERMS

    # 看一个权限是否是admin角色该有的权限
    def is_admin_pvm(self, pvm):
        return not self.is_user_defined_permission(pvm)

    # 看一个权限是否是gamma角色该有的权限
    def is_gamma_pvm(self, pvm):
        return not (
            self.is_user_defined_permission(pvm)
            or self.is_admin_only(pvm)
        ) or self.is_accessible_to_all(pvm)


    # 创建视图，创建权限，创建视图-权限绑定记录。
    def set_perm(self, mapper, connection, target,permission_name):  # noqa
        #
        # connection is sql
        # target is tables/db  model

        if target.perm != target.get_perm():
            link_table = target.__table__
            connection.execute(
                link_table.update()
                .where(link_table.c.id == target.id)
                .values(perm=target.get_perm())
            )

        # add to view menu if not already exists
        permission_name = permission_name
        view_menu_name = target.get_perm()
        permission = self.find_permission(permission_name)
        view_menu = self.find_view_menu(view_menu_name)
        pv = None
        # 如果权限不存存在就创建
        if not permission:
            permission_table = (
                self.permission_model.__table__  # pylint: disable=no-member
            )
            connection.execute(permission_table.insert().values(name=permission_name))
            permission = self.find_permission(permission_name)

        # 如果视图不存在就创建
        if not view_menu:
            view_menu_table = self.viewmenu_model.__table__  # pylint: disable=no-member
            connection.execute(view_menu_table.insert().values(name=view_menu_name))
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
            # 重新获取权限视图绑定记录
            pv = (
                self.get_session.query(self.permissionview_model)
                    .filter_by(permission=permission, view_menu=view_menu)
                    .first()
            )
        return pv



    # 根据权限，视图，添加到相关pv-role
    @classmethod
    def add_pv_role(self,permission_name,view_menu_name,session):
        permission = session.query(self.permission_model).filter_by(name=permission_name).first()
        view_menu = session.query(self.viewmenu_model).filter_by(name=view_menu_name).first()
        # 获取是否存在 视图-权限绑定  记录
        if permission and view_menu:
            pv = (
                session.query(self.permissionview_model)
                    .filter_by(permission=permission, view_menu=view_menu)
                    .first()
            )
            try:
                # 为用户所属组织架构都添加该pv
                if pv and g.user and g.user.org:
                    roles = session.query(self.role_model).all()  # 获取所有角色，自动在相应角色下面添加pv
                    if roles:
                        for role in roles:
                            if role.name in g.user.org:
                                # 为pvm-role表中添加记录
                                pv_role = session.execute(select([assoc_permissionview_role.c.id]).where(assoc_permissionview_role.c.permission_view_id==pv.id)
                                                          .where(assoc_permissionview_role.c.role_id==role.id)
                                                          .limit(1)
                                                          ).fetchall()
                                if not pv_role:
                                    session.execute(assoc_permissionview_role.insert().values(
                                        permission_view_id=pv.id, role_id=role.id
                                        )
                                    )
            except Exception as e:
                logging.error(e)



    @classmethod
    def get_join_projects_id(self,session):
        from myapp.models.model_team import Project_User
        if g.user:
            projects_id = session.query(Project_User.project_id).filter(Project_User.user_id == User.get_user_id()).all()
            projects_id = [project_id[0] for project_id in projects_id]
            return projects_id
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








