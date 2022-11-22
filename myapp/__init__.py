# 避免多进程同时启动对系统cpu负载过高
# time.sleep(random.randint(1,10))
from copy import deepcopy
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from flask import g, abort
import os
from flask import render_template
from flask import Flask, redirect
from flask_appbuilder import AppBuilder, IndexView, SQLA
from flask_appbuilder.baseviews import expose
from flask_compress import Compress
from flask_migrate import Migrate
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix
import wtforms_json

from myapp.security import MyappSecurityManager
from myapp.utils.core import pessimistic_connection_handling, setup_cache
from myapp.utils.log import DBEventLogger
wtforms_json.init()

# 在这个文件里面只创建app，不要做view层面的事情。

APP_DIR = os.path.dirname(__file__)

CONFIG_MODULE = os.environ.get("MYAPP_CONFIG", "myapp.config")



# app = Flask(__name__,static_url_path='/static',static_folder='static',template_folder='templates')
app = Flask(__name__)  # ,static_folder='/mnt',static_url_path='/mnt'

app.config.from_object(CONFIG_MODULE)
conf = app.config

if conf.get('DATA_DIR',''):
    if not os.path.exists(conf['DATA_DIR']):
        os.makedirs(conf['DATA_DIR'],exist_ok=True)

print(conf.get('SQLALCHEMY_DATABASE_URI',''))

#################################################################
# Handling manifest file logic at app start
#################################################################
# 依赖js和css的配置文件
# MANIFEST_FILE = APP_DIR + "/static/assets/dist/manifest.json"
MANIFEST_FILE = APP_DIR + "/assets/dist/manifest.json"
manifest = {}

# @pysnooper.snoop()
def parse_manifest_json():
    global manifest
    try:
        with open(MANIFEST_FILE, "r") as f:
            # the manifest inclues non-entry files
            # we only need entries in templates
            full_manifest = json.load(f)
            manifest = full_manifest.get("entrypoints", {})
    except Exception:
        pass

# 获取依赖的js文件地址
# @pysnooper.snoop()
def get_js_manifest_files(filename):
    if app.debug:
        parse_manifest_json()
    entry_files = manifest.get(filename, {})
    return entry_files.get("js", [])

# 获取依赖的css文件地址
# @pysnooper.snoop()
def get_css_manifest_files(filename):
    if app.debug:
        parse_manifest_json()
    entry_files = manifest.get(filename, {})
    return entry_files.get("css", [])


def get_unloaded_chunks(files, loaded_chunks):
    filtered_files = [f for f in files if f not in loaded_chunks]
    for f in filtered_files:
        loaded_chunks.add(f)
    return filtered_files


parse_manifest_json()


# 字典每一个key，例如css_manifest都可以在模板中使用
@app.context_processor
def get_manifest():
    return dict(
        loaded_chunks=set(),
        get_unloaded_chunks=get_unloaded_chunks,
        js_manifest=get_js_manifest_files,
        css_manifest=get_css_manifest_files,
    )


#######################################blueprints##########################

if conf.get("BLUEPRINTS"):
    for bp in conf.get("BLUEPRINTS"):
        try:
            print("Registering blueprint: '{}'".format(bp.name))
            app.register_blueprint(bp)
        except Exception as e:
            print("blueprint registration failed")
            logging.exception(e)

if conf.get("SILENCE_FAB"):
    logging.getLogger("flask_appbuilder").setLevel(logging.ERROR)

if app.debug:
    app.logger.setLevel(logging.DEBUG)  # pylint: disable=no-member
else:
    # In production mode, add log handler to sys.stderr.
    app.logger.addHandler(logging.StreamHandler())  # pylint: disable=no-member
    app.logger.setLevel(logging.INFO)  # pylint: disable=no-member

db = SQLA(app)

if conf.get("WTF_CSRF_ENABLED"):
    csrf = CSRFProtect(app)
    csrf_exempt_list = conf.get("WTF_CSRF_EXEMPT_LIST", [])
    for ex in csrf_exempt_list:
        csrf.exempt(ex)

pessimistic_connection_handling(db.engine)

cache = setup_cache(app, conf.get("CACHE_CONFIG"))

migrate = Migrate(app, db, directory=APP_DIR + "/migrations")

# Logging configuration
logging.basicConfig(format=app.config.get("LOG_FORMAT"))
logging.getLogger().setLevel( app.config.get("LOG_LEVEL") if app.config.get("LOG_LEVEL") else 1)

# 系统日志输出，myapp的输出。在gunicor是使用
if conf.get("ENABLE_TIME_ROTATE"):

    logging.getLogger().setLevel(conf.get("TIME_ROTATE_LOG_LEVEL"))
    handler = TimedRotatingFileHandler(
        conf.get("FILENAME"),
        when=conf.get("ROLLOVER"),
        interval=conf.get("INTERVAL"),
        backupCount=conf.get("BACKUP_COUNT"),
    )
    logging.getLogger().addHandler(handler)

if conf.get("ENABLE_CORS"):
    from flask_cors import CORS

    CORS(app, **conf.get("CORS_OPTIONS"))

if conf.get("ENABLE_PROXY_FIX"):
    app.wsgi_app = ProxyFix(app.wsgi_app)

if conf.get("ENABLE_CHUNK_ENCODING"):

    class ChunkedEncodingFix(object):
        def __init__(self, app):
            self.app = app

        def __call__(self, environ, start_response):
            # Setting wsgi.input_terminated tells werkzeug.wsgi to ignore
            # content-length and read the stream till the end.
            if environ.get("HTTP_TRANSFER_ENCODING", "").lower() == u"chunked":
                environ["wsgi.input_terminated"] = True
            return self.app(environ, start_response)

    app.wsgi_app = ChunkedEncodingFix(app.wsgi_app)

if conf.get("UPLOAD_FOLDER"):
    try:
        os.makedirs(conf.get("UPLOAD_FOLDER"))
    except OSError:
        pass

if conf.get("ADDITIONAL_MIDDLEWARE"):
    for middleware in conf.get("ADDITIONAL_MIDDLEWARE"):
        app.wsgi_app = middleware(app.wsgi_app)


class MyIndexView(IndexView):

    @expose("/")
    def index(self):
        if not g.user or not g.user.get_id():
            return redirect(appbuilder.get_url_for_login)
        # return redirect("/myapp/home")
        return redirect("/frontend/")


custom_sm = conf.get("CUSTOM_SECURITY_MANAGER") or MyappSecurityManager
if not issubclass(custom_sm, MyappSecurityManager):
    raise Exception(
        """Your CUSTOM_SECURITY_MANAGER must now extend MyappSecurityManager,
         not FAB's security manager.
         See [4565] in UPDATING.md"""
    )


# 创建appbuilder
with app.app_context():
    # 创建所有表
    # db.create_all()
    # 创建fab
    appbuilder = AppBuilder(
        app,
        db.session,
        base_template="myapp/base.html",
        indexview=MyIndexView,   # 首页
        security_manager_class=custom_sm,   # 自定义认证方式
        # Run `myapp init` to update FAB's perms,设置为true就可以自动更新了，这样才能自动添加新建权限
        update_perms=True,
    )

security_manager = appbuilder.sm

results_backend = conf.get("RESULTS_BACKEND")

# Merge user defined feature flags with default feature flags
_feature_flags = conf.get("DEFAULT_FEATURE_FLAGS") or {}
_feature_flags.update(conf.get("FEATURE_FLAGS") or {})
# Event Logger
event_logger = conf.get("EVENT_LOGGER", DBEventLogger)()

def get_feature_flags():
    GET_FEATURE_FLAGS_FUNC = conf.get("GET_FEATURE_FLAGS_FUNC")
    if GET_FEATURE_FLAGS_FUNC:
        return GET_FEATURE_FLAGS_FUNC(deepcopy(_feature_flags))
    return _feature_flags


def is_feature_enabled(feature):
    """Utility function for checking whether a feature is turned on"""
    return get_feature_flags().get(feature)


# Flask-Compress
if conf.get("ENABLE_FLASK_COMPRESS"):
    Compress(app)

if conf.get("TALISMAN_ENABLED"):
    talisman_config = conf.get("TALISMAN_CONFIG")
    Talisman(app, **talisman_config)

# Hook that provides administrators a handle on the Flask APP
# after initialization
flask_app_mutator = conf.get("FLASK_APP_MUTATOR")
if flask_app_mutator:
    flask_app_mutator(app)



# from flask import Flask, json, make_response
# 添加每次请求后的操作函数，必须要返回res
from flask import request

import pysnooper

@app.before_request
# @pysnooper.snoop(watch_explode='aa')
def check_login():
    if '/static' in request.path or '/logout' in request.path or '/login' in request.path or '/health' in request.path or '/wechat' in request.path:
        return

    if not g.user or not g.user.get_id():

        # 支持跨域名cookie登录
        myapp_username = request.cookies.get('myapp_username', '')
        if myapp_username:
            try:
                user = security_manager.find_user(myapp_username)
                g.user = user
                return
            except Exception as e:
                print(e)

        abort(401)



@app.after_request
def myapp_after_request(resp):
    try:
        if g.user and g.user.username:

            resp.set_cookie('myapp_username', g.user.username,domain=conf.get('COOKIE_DOMAIN',None) if conf.get('COOKIE_DOMAIN',None) else None)  # 设置用户信息传递
            # resp.set_cookie('myapp_username', g.user.username)  # 设置用户信息传递

            if hasattr(g, 'id'):
                resp.set_cookie('id', str(g.id),max_age=3)   # 设置有效期
        if g.user and g.user.first_name:
            if 'vip' in g.user.first_name:
                resp.set_cookie('version', str('vip'))  # 设置有效期



    except Exception as e:
        print(e)
        resp.set_cookie('myapp_username', 'myapp')
        # resp.delete_cookie('id')
    return resp

# 配置影响后操作
@app.after_request
def apply_http_headers(response):
    """Applies the configuration's http headers to all responses"""
    for k, v in conf.get("HTTP_HEADERS").items():
        response.headers[k] = v
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@appbuilder.app.errorhandler(404)
def page_not_found(e):
    return (
        render_template(
            "404.html", base_template=appbuilder.base_template, appbuilder=appbuilder
        ),
        404,
    )

# 配置werkzeug的日志级别为error，这样就不会频繁的打印访问路径了。
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

if __name__ != '__main__':
    # 如果不是直接运行，则将日志输出到 gunicorn 中
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# 引入视图
from myapp import views


# def can_access(menuitem):
#     print(menuitem.name,menuitem.label)
#     return security_manager.can_access("menu access",menuitem.label)
#
#
# for item1 in appbuilder.menu.get_list():
#     item1.can_access = can_access


