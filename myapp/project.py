
from werkzeug.security import check_password_hash
from flask_appbuilder.security.sqla.models import (
    assoc_permissionview_role,
    assoc_user_role,
)

from flask import g

from flask_appbuilder.security.views import AuthDBView
from flask_appbuilder.security.views import expose
from flask_appbuilder.const import (
    AUTH_DB,
    AUTH_LDAP,
    AUTH_OAUTH,
    AUTH_OID,
    AUTH_REMOTE_USER,
    LOGMSG_ERR_SEC_AUTH_LDAP,
    LOGMSG_ERR_SEC_AUTH_LDAP_TLS,
    LOGMSG_WAR_SEC_LOGIN_FAILED,
    LOGMSG_WAR_SEC_NO_USER,
    LOGMSG_WAR_SEC_NOLDAP_OBJ,
    PERMISSION_PREFIX
)
import pysnooper
import json







# 首页显示内容

# 数据格式说明 dict:
# 'id': 唯一标识,
# 'type': 类型 一级支持 title | boxlist | list 二级支持 text | markdown,
# 'open': 是否当前页打开 1 从当前窗口打开 默认打开新的标签页,
# 'url': 点击时打开的链接 仅支持 http(s)://开头,
# 'cover': 封面图链接 支持base64图片编码,
# 'content': 内容 支持markdown(需设置类型为markdown),
# 'data': 嵌套下级内容数组

HOME_CONFIG = [
    {
        'id': 2,
        'type': 'title',
        'content': '平台主功能',
        'data': [],
    }, {
        'id': 102,
        'type': 'boxlist',
        'content': 'Pipelines',
        'data': [{
            'id': 1021,
            'type': 'base',
            'open': 1,
            'url': '/pipeline_modelview/list/',
            'cover': '/static/assets/images/home/dag.jpg',
            'content': '机器学习pipeline流水线',
        }, {
            'id': 1022,
            'type': 'base',
            'open': 1,
            'url': '/notebook_modelview/add',
            'cover': '/static/assets/images/home/vscode.png',
            'content': '在线ide编辑器',
        },
            {
                'id': 1023,
                'type': 'base',
                'open': 1,
                'url': '/nni_modelview/add',
                'cover': '/static/assets/images/home/private.png',
                'content': '超参搜索',
            }, {
                'id': 1024,
                'type': 'base',
                'open': 1,
                'url': '/service_modelview/list/',
                'cover': '/static/assets/images/home/service.png',
                'content': '模型服务化',
            }],
    }
]



# 推送给管理员消息的函数
def push_admin(message):
    pass

# 推送消息给用户的函数
def push_message(receivers,message,link=None):
    pass



import logging as log
import datetime
import logging
import re

from flask import abort, current_app, flash, g, redirect, request, session, url_for
from flask_babel import lazy_gettext
from flask_login import login_user, logout_user
import jwt
from werkzeug.security import generate_password_hash
from flask_appbuilder.security.forms import LoginForm_db, LoginForm_oid, ResetPasswordForm, UserInfoEdit
from flask_appbuilder._compat import as_unicode
import pysnooper


class MyCustomRemoteUserView():
    pass

class Myauthdbview(AuthDBView):
    login_template = "appbuilder/general/security/login_db.html"

    @expose("/login/", methods=["GET", "POST"])
    @pysnooper.snoop(watch_explode=('form',))
    def login(self):

        if 'rtx' in request.args:
            if request.args.get('rtx'):
                username = request.args.get('rtx')
                user = self.appbuilder.sm.find_user(username)
                if user:
                    login_user(user, remember=True)
                    return redirect(self.appbuilder.get_url_for_index)


        if g.user is not None and g.user.is_authenticated:
            return redirect(self.appbuilder.get_url_for_index)

        form = LoginForm_db()
        method = request.method
        # 如果提交请求。就是认证
        if form.validate_on_submit():
            username = form.username.data
            password = form.password.data

            user = self.appbuilder.sm.find_user(username=username)
            if user is None:
                user = self.appbuilder.sm.find_user(email=username)
            if user is None or (not user.is_active):
                log.info(LOGMSG_WAR_SEC_LOGIN_FAILED.format(username))
                user = None
            elif check_password_hash(user.password, password):
                self.appbuilder.sm.update_user_auth_stat(user, True)
            elif user.password==password:
                self.appbuilder.sm.update_user_auth_stat(user, True)
            else:
                self.appbuilder.sm.update_user_auth_stat(user, False)
                log.info(LOGMSG_WAR_SEC_LOGIN_FAILED.format(username))
                user = None

            if not user:
                user = self.appbuilder.sm.find_user(form.username.data)
                if user:
                    # 有用户，但是密码不对
                    flash('发现用户%s已存在，但输入密码不对'%form.username.data, "warning")
                    return redirect(self.appbuilder.get_url_for_login)
                else:
                    # 没有用户的时候自动注册用户
                    user = self.appbuilder.sm.auth_user_remote_org_user(username=form.username.data, org_name='',
                                                                        password=form.password.data)
                    flash('发现用户%s不存在，已自动注册' % form.username.data, "warning")

            login_user(user, remember=True)
            return redirect(self.appbuilder.get_url_for_index)
        return self.render_template(
            self.login_template, title=self.title, form=form, appbuilder=self.appbuilder
        )







