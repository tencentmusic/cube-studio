from werkzeug.security import check_password_hash
from flask_appbuilder.security.views import AuthDBView, AuthRemoteUserView
from flask_appbuilder.security.views import expose
from flask_appbuilder.const import LOGMSG_WAR_SEC_LOGIN_FAILED
from flask import send_file, jsonify
import os

# 推送资源申请消息
def push_resource_apply(notebook_id=None,pipeline_id=None,task_id=None,service_id=None,**kwargs):
    from myapp.models.model_job import Task,Pipeline
    from myapp.models.model_notebook import Notebook
    from myapp.models.model_serving import InferenceService

    pass

# 推送资源审批消息
def push_resource_approve(notebook_id=None,pipeline_id=None,task_id=None,service_id=None,**kwargs):
    from myapp.models.model_job import Task,Pipeline
    from myapp.models.model_notebook import Notebook
    from myapp.models.model_serving import InferenceService

    pass

# 推送给管理员消息的函数
def push_admin(message):
    pass


# 推送消息给用户的函数
def push_message(receivers, message, link=None):
    pass


import logging
from flask import flash, g, redirect, request, session
from flask_login import login_user, logout_user
from flask_appbuilder.security.forms import LoginForm_db
import pysnooper

portal_url = 'http://127.0.0.1/xxxx/login?staffCode=admin'

# 自定义远程用户视图
# @pysnooper.snoop()
class MyCustomRemoteUserView(AuthRemoteUserView):

    @expose('/xx/xx/explorer/login')
    @pysnooper.snoop(watch_explode=('request_data',))
    def login(self):

        request_data = request.args.to_dict()

        username = request_data.get('staffCode','').lower().replace('_','-').replace('.','')
        if not username:
            print('no find user')
            return redirect(portal_url)
        # 处理特殊符号
        email = ''
        if '@' in username:
            email = username
            username = username[:username.index('@')]

        # 先查询用户是否存在
        if email:
            user = self.appbuilder.sm.find_user(email=email)
        else:
            user = self.appbuilder.sm.find_user(username=username)

        if user and (not user.is_active):
            logging.info(LOGMSG_WAR_SEC_LOGIN_FAILED.format(username))
            print('用户未激活，联系管理员激活')
            flash('user not active',category='warning')
            return redirect(portal_url)

        if not user:
            # 没有用户的时候自动注册用户
            user = self.appbuilder.sm.auth_user_remote_org_user(
                username=username,
                org_name='',
                password='123456',
                email=email,
                first_name=username.split('.')[0] if '.' in username else username,
                last_name=username.split('.')[1] if '.' in username else username
            )
            flash('发现用户%s不存在，已自动注册' % username, "success")
        if not user:
            return redirect(portal_url)
        login_user(user, remember=True)
        # 添加到public项目组
        from myapp.security import MyUserRemoteUserModelView_Base
        user_view = MyUserRemoteUserModelView_Base()
        user_view.post_add(user)
        res = redirect('/frontend/')
        res.set_cookie('myapp_username', username)  # 让前端认为也成功了
        return res
        # return redirect(self.appbuilder.get_url_for_index)

    @expose('/login/')
    def _login(self):
        if 'username' in request.args:
            if request.args.get('username'):
                username = request.args.get('username')
                user = self.appbuilder.sm.find_user(username)
                if user:
                    login_user(user, remember=True)
                    return redirect(self.appbuilder.get_url_for_index)

        return redirect(portal_url)

    @expose('/logout')
    # @pysnooper.snoop()
    def logout(self):
        session.pop('user', None)
        logout_user()
        return redirect(portal_url)


# 账号密码登录方式的登录界面

class Myauthdbview(AuthDBView):
    login_template = "appbuilder/general/security/login_db.html"

    @expose("/login/api/", methods=["GET", "POST"])
    # @pysnooper.snoop(watch_explode=('form',))
    def login_api(self):
        request_data = request.args.to_dict()
        if request.get_json(silent=True):
            request_data.update(request.get_json(silent=True))
        token = request_data.get('token', '')
        uuid = request_data.get('uuid', '')
        if token:
            user = self.appbuilder.sm.find_user(token)
            if user:
                login_user(user, remember=True)
                if uuid:
                    user.org = uuid
                    from myapp import db, app
                    db.session.commit()

                return jsonify({
                    "status": 0,
                    "message": '登录成功',
                    "result": {}
                })
            else:
                return jsonify({
                    "status": 1,
                    "message": '未发现用户',
                    "result": {}
                })

    @expose("/login/", methods=["GET", "POST"])
    def login(self):
        request_data = request.args.to_dict()
        comed_url = request_data.get('login_url', '')

        if 'username' in request_data:
            if request_data.get('username'):
                username = request_data.get('username')
                user = self.appbuilder.sm.find_user(username)
                if user:
                    login_user(user, remember=True)
                    if comed_url:
                        return redirect(comed_url)
                    return redirect(self.appbuilder.get_url_for_index)
        # 如果已经登录了，那么直接进去首页
        if g.user is not None and g.user.is_authenticated:
            return redirect(self.appbuilder.get_url_for_index)

        form = LoginForm_db()
        # 如果提交请求。就是认证
        if form.validate_on_submit():
            username = form.username.data
            import re
            if not re.match('^[a-z][a-z0-9\-]*[a-z0-9]$',username):
                flash('用户名只能由小写字母、数字、-组成',"warning")
                return redirect(self.appbuilder.get_url_for_login)

            # 密码加密
            password = form.password.data
            print("The message was: ", password)

            # 根据用户名登录
            user = self.appbuilder.sm.find_user(username=username)
            if user is None:
                # 根据用户邮箱登录
                user = self.appbuilder.sm.find_user(email=username)
            if user is None:
                logging.info(LOGMSG_WAR_SEC_LOGIN_FAILED.format(username))
            # 用户未激活报错
            elif not user.is_active:
                flash('发现用户%s已存在，但未激活，请联系管理员激活' % form.username.data, "warning")
                return redirect(self.appbuilder.get_url_for_login)
            # 校验是否和加密密码对应
            elif check_password_hash(user.password, password):
                self.appbuilder.sm.update_user_auth_stat(user, True)
            # 校验是否和明文密码对应
            elif user.password == password:
                self.appbuilder.sm.update_user_auth_stat(user, True)
            else:
                self.appbuilder.sm.update_user_auth_stat(user, False)
                logging.info(LOGMSG_WAR_SEC_LOGIN_FAILED.format(username))
                user = None

            if not user:
                import re
                # 查看用户不存在还是密码不对
                user = self.appbuilder.sm.find_user(username)
                if user:
                    # 有用户，但是密码不对
                    flash('发现用户%s已存在，但输入密码不对' % username, "warning")

                    return redirect(self.appbuilder.get_url_for_login)
                else:
                    # 不自动创建，联系管理员创建
                    # flash('未发现%s用户，联系管理员创建' % form.username.data, "warning")
                    # return redirect(self.appbuilder.get_url_for_login)
                    # 没有用户的时候自动注册用户
                    user = self.appbuilder.sm.auth_user_remote_org_user(username=username, org_name='', password=password)   # hashed_password=password可以密文存储
                    flash('发现用户%s不存在，已自动注册' % username, "warning")


            login_user(user, remember=True)
            # 添加到public项目组
            from myapp.security import MyUserRemoteUserModelView_Base
            user_view = MyUserRemoteUserModelView_Base()
            user_view.post_add(user)

            # 拷贝示例数据
            import shutil

            user_workspace = f'/data/k8s/kubeflow/pipeline/workspace/{username}'
            os.makedirs(user_workspace,exist_ok=True)
            if not os.path.exists(os.path.join(user_workspace,'pipeline')):
                os.makedirs(os.path.join(user_workspace,'pipeline'),exist_ok=True)
                shutil.copytree(f'myapp/example/pipeline',f'/data/k8s/kubeflow/pipeline/workspace/{username}/pipeline/example')

            return redirect(comed_url if comed_url else self.appbuilder.get_url_for_index)


        return self.render_template(
            self.login_template, title=self.title, form=form, appbuilder=self.appbuilder
        )

    @expose('/logout')
    def logout(self):
        login_url = request.host_url.strip('/') + '/login/'
        session.pop('user', None)
        g.user = None
        logout_user()
        return redirect(login_url)



