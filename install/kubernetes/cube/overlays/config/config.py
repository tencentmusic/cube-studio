

import imp
import json
import os
import sys

from dateutil import tz

from flask_appbuilder.security.manager import AUTH_REMOTE_USER, AUTH_DB
from myapp.stats_logger import DummyStatsLogger


# Realtime stats logger, a StatsD implementation exists
STATS_LOGGER = DummyStatsLogger()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if "MYAPP_HOME" in os.environ:
    DATA_DIR = os.environ["MYAPP_HOME"]
else:
    DATA_DIR = os.path.join(os.path.expanduser("~"), ".myapp")

APP_THEME = "readable.css"

FAB_UPDATE_PERMS=True
FAB_STATIC_FOLDER = BASE_DIR + "/static/appbuilder/"

MYAPP_WORKERS = 2  # deprecated
MYAPP_CELERY_WORKERS = 32  # deprecated

MYAPP_WEBSERVER_ADDRESS = "0.0.0.0"
MYAPP_WEBSERVER_PORT = 80

# 前面页面静态文件的缓存超时时间，单位s
MYAPP_WEBSERVER_TIMEOUT = 300


# Your App secret key
# 设置才能正常使用session
SECRET_KEY = "\2\1thisismyscretkey\1\2\e\y\y\h"  # noqa

# csv导出文件编码
CSV_EXPORT = {"encoding": "utf_8_sig"}
# Flask-WTF flag for CSRF
# 跨域配置
WTF_CSRF_ENABLED = False

# 跨域访问允许通过的站点
WTF_CSRF_EXEMPT_LIST = ["myapp.views.core.log"]

# 是否debug模式运行
DEBUG = os.environ.get("FLASK_ENV") == "development"
FLASK_USE_RELOAD = True

# Myapp allows server-side python stacktraces to be surfaced to the
# user when this feature is on. This may has security implications
# and it's more secure to turn it off in production settings.
SHOW_STACKTRACE = True

# Extract and use X-Forwarded-For/X-Forwarded-Proto headers?
ENABLE_PROXY_FIX = False

# ------------------------------
# GLOBALS FOR APP Builder
# ------------------------------
# 应用名
APP_NAME = "Cube-Studio"

# 图标
APP_ICON = "/static/assets/images/myapp-logo.png"
APP_ICON_WIDTH = 126

# 配置logo点击后的跳转链接，例如'/welcome'  会跳转到'/myapp/welcome'
LOGO_TARGET_PATH = None


# ----------------------------------------------------
# 认证相关的配置
# ----------------------------------------------------
# 认证类型
# AUTH_OID : OpenID认证
# AUTH_DB : 数据库账号密码配置
# AUTH_LDAP : LDAP认证
# AUTH_REMOTE_USER : 远程用户认证
AUTH_TYPE = AUTH_DB

# AUTH_TYPE = AUTH_REMOTE_USER
# Uncomment to setup Full admin role name
# AUTH_ROLE_ADMIN = 'Admin'

# Uncomment to setup Public role name, no authentication needed
# AUTH_ROLE_PUBLIC = 'Public'

# 是否允许用户注册
AUTH_USER_REGISTRATION = False

# 用户的默认角色
AUTH_USER_REGISTRATION_ROLE = "Gamma"

# RECAPTCHA_PUBLIC_KEY = 'GOOGLE PUBLIC KEY FOR RECAPTCHA'
# RECAPTCHA_PRIVATE_KEY = 'GOOGLE PRIVATE KEY FOR RECAPTCHA'

OAUTH_PROVIDERS=[]

#LDAP认证时, ldap server
# AUTH_LDAP_SERVER = "ldap://ldapserver.new"

# OpenID认证的提供方
# OPENID_PROVIDERS = [
#    { 'name': 'Yahoo', 'url': 'https://open.login.yahoo.com/' },
#    { 'name': 'Flickr', 'url': 'https://www.flickr.com/<username>' },


# ---------------------------------------------------
# 语言翻译上的配置
# ---------------------------------------------------
# 默认使用的语言
BABEL_DEFAULT_LOCALE = os.getenv('LOCALE','zh')
# Your application default translation path
BABEL_DEFAULT_FOLDER = "myapp/translations"
# The allowed translation for you app
LANGUAGES = {
    "en": {"flag": "us", "name": "English"},
    "zh": {"flag": "cn", "name": "Chinese"},
}

# ---------------------------------------------------
# Feature flags
# ---------------------------------------------------
# Feature flags that are set by default go here. Their values can be
# For example, DEFAULT_FEATURE_FLAGS = { 'FOO': True, 'BAR': False } here
# and FEATURE_FLAGS = { 'BAR': True, 'BAZ': True } in myapp_config.py
# will result in combined feature flags of { 'FOO': True, 'BAR': True, 'BAZ': True }
DEFAULT_FEATURE_FLAGS = {
    # Experimental feature introducing a client (browser) cache
    "CLIENT_CACHE": False,
    "ENABLE_EXPLORE_JSON_CSRF_PROTECTION": False,
}

# A function that receives a dict of all feature flags
# (DEFAULT_FEATURE_FLAGS merged with FEATURE_FLAGS)
# can alter it, and returns a similar dict. Note the dict of feature
# flags passed to the function is a deepcopy of the dict in the config,
# and can therefore be mutated without side-effect
#
# GET_FEATURE_FLAGS_FUNC can be used to implement progressive rollouts,
# role-based features, or a full on A/B testing framework.
#
# from flask import g, request
# def GET_FEATURE_FLAGS_FUNC(feature_flags_dict):
#     feature_flags_dict['some_feature'] = g.user and g.user.id == 5
#     return feature_flags_dict
GET_FEATURE_FLAGS_FUNC = None

# ---------------------------------------------------
# 图片和文件相关的配置
# ---------------------------------------------------
# The file upload folder, when using models with files
UPLOAD_FOLDER = BASE_DIR + "/static/file/uploads/"
DOWNLOAD_FOLDER = BASE_DIR + "/static/file/download/"
DOWNLOAD_URL = "/static/file/download/"
# The image upload folder, when using models with images
IMG_UPLOAD_FOLDER = BASE_DIR + "/static/file/uploads/"

# The image upload url, when using models with images
IMG_UPLOAD_URL = "/static/file/uploads/"

# Setup image size default is (300, 200, True)
# IMG_SIZE = (300, 200, True)

# CORS Options
ENABLE_CORS = True
CORS_OPTIONS = {"supports_credentials":True}

# Chrome allows up to 6 open connections per domain at a time. When there are more
# than 6 slices in dashboard, a lot of time fetch requests are queued up and wait for
# next available socket. PR #5039 is trying to allow domain sharding for Myapp,
# and this feature will be enabled by configuration only (by default Myapp
# doesn't allow cross-domain request).
MYAPP_WEBSERVER_DOMAINS = None

# ---------------------------------------------------
# Time grain configurations
# ---------------------------------------------------
# List of time grains to disable in the application (see list of builtin
# time grains in myapp/db_engine_specs.builtin_time_grains).
# For example: to disable 1 second time grain:
# TIME_GRAIN_BLACKLIST = ['PT1S']
TIME_GRAIN_BLACKLIST = []

# Additional time grains to be supported using similar definitions as in
# myapp/db_engine_specs.builtin_time_grains.
# For example: To add a new 2 second time grain:
# TIME_GRAIN_ADDONS = {'PT2S': '2 second'}
TIME_GRAIN_ADDONS = {}

# Implementation of additional time grains per engine.
# For example: To implement 2 second time grain on clickhouse engine:
# TIME_GRAIN_ADDON_FUNCTIONS = {
#     'clickhouse': {
#         'PT2S': 'toDateTime(intDiv(toUInt32(toDateTime({col})), 2)*2)'
#     }
# }
TIME_GRAIN_ADDON_FUNCTIONS = {}

# Console Log Settings
ADDITIONAL_MODULE_DS_MAP = {}
ADDITIONAL_MIDDLEWARE = []

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
LOG_LEVEL = "INFO"

# ---------------------------------------------------
# Enable Time Rotate Log Handler
# ---------------------------------------------------
# LOG_LEVEL = DEBUG, INFO, WARNING, ERROR, CRITICAL
# 控制日志是否输出到文件
ENABLE_TIME_ROTATE = False
# 输出到文件的日志级别
TIME_ROTATE_LOG_LEVEL = "DEBUG"
# 日志地址
FILENAME = os.path.join(DATA_DIR, "myapp.log")
# 这意味着每天午夜时分，日志文件都会自动轮换
ROLLOVER = "midnight"
# 这意味着每天都会轮换一次日志文件
INTERVAL = 1
# 这意味着系统将保留最近 30 天的日志文件
BACKUP_COUNT = 30

# If defined, shows this text in an alert-warning box in the navbar
# one example use case may be "STAGING" to make it clear that this is
# not the production version of the site.
WARNING_MSG = None

# 自动添加到响应头的配置
HTTP_HEADERS = {
    "Access-Control-Allow-Origin":"*",
    "Access-Control-Allow-Methods":"*",
    "Access-Control-Allow-Headers":"*",
}


# A dictionary of items that gets merged into the Jinja context for
# SQL Lab. The existing context gets updated with this dictionary,
# meaning values for existing keys get overwritten by the content of this
# dictionary.
JINJA_CONTEXT_ADDONS = {}

# Roles that are controlled by the API / Myapp and should not be changes
# by humans.
ROBOT_PERMISSION_ROLES = ["Gamma", "Admin"]

CONFIG_PATH_ENV_VAR = "MYAPP_CONFIG_PATH"

# If a callable is specified, it will be called at app startup while passing
# a reference to the Flask app. This can be used to alter the Flask app
# in whatever way.
# example: FLASK_APP_MUTATOR = lambda x: x.before_request = f
FLASK_APP_MUTATOR = None

# Set this to false if you don't want users to be able to request/grant
ENABLE_ACCESS_REQUEST = True

# smtp server configuration
EMAIL_NOTIFICATIONS = False  # all the emails are sent using dryrun
SMTP_HOST = "localhost"
SMTP_STARTTLS = True
SMTP_SSL = False
SMTP_USER = ""
SMTP_PORT = 25
SMTP_PASSWORD = ""
SMTP_MAIL_FROM = ""


# Whether to bump the logging level to ERROR on the flask_appbuilder package
# Set to False if/when debugging FAB related issues like
# permission management
SILENCE_FAB = True

# The link to a page containing common errors and their resolutions
# It will be appended at the bottom of sql_lab errors.
TROUBLESHOOTING_LINK = ""

# CSRF token timeout, set to None for a token that never expires
WTF_CSRF_TIME_LIMIT = 60 * 60 * 24 * 7

# This link should lead to a page with instructions on how to gain access to a
PERMISSION_INSTRUCTIONS_LINK = ""

# Integrate external Blueprints to the app by passing them to your
# configuration. These blueprints will get integrated in the app
BLUEPRINTS = []

# When not using gunicorn, (nginx for instance), you may want to disable
# using flask-compress
ENABLE_FLASK_COMPRESS = True

# 任务的最小执行间隔min
PIPELINE_TASK_CRON_RESOLUTION = 10

# Send bcc of all reports to this address. Set to None to disable.
# This is useful for maintaining an audit trail of all email deliveries.
# 响应支持中文序列化
JSON_AS_ASCII = False

# User credentials to use for generating reports
# This user should have permissions to browse all the dashboards and
# slices.
# TODO: In the future, login as the owner of the item to generate reports
EMAIL_REPORTS_USER = "admin"
EMAIL_REPORTS_SUBJECT_PREFIX = "[Report] "

# The webdriver to use for generating reports. Use one of the following
# firefox
#   Requires: geckodriver and firefox installations
#   Limitations: can be buggy at times
# chrome:
#   Requires: headless chrome
#   Limitations: unable to generate screenshots of elements
EMAIL_REPORTS_WEBDRIVER = "chrome"

# Any config options to be passed as-is to the webdriver
WEBDRIVER_CONFIGURATION = {}

# The base URL to query for accessing the user interface

# Send user to a link where they can report bugs
BUG_REPORT_URL = None
# Send user to a link where they can read more about Myapp
DOCUMENTATION_URL = None

# Do you want Talisman enabled?
TALISMAN_ENABLED = False
# If you want Talisman, how do you want it configured??
TALISMAN_CONFIG = {
    "content_security_policy": None,
    "force_https": True,
    "force_https_permanent": False,
}
# 前端静态文件的默认缓存时间
SEND_FILE_MAX_AGE_DEFAULT=300

try:
    if CONFIG_PATH_ENV_VAR in os.environ:
        # Explicitly import config module that is not in pythonpath; useful
        # for case where app is being executed via pex.
        print(
            "Loaded your LOCAL configuration at [{}]".format(
                os.environ[CONFIG_PATH_ENV_VAR]
            )
        )
        module = sys.modules[__name__]
        override_conf = imp.load_source(
            "myapp_config", os.environ[CONFIG_PATH_ENV_VAR]
        )
        for key in dir(override_conf):
            if key.isupper():
                setattr(module, key, getattr(override_conf, key))

    else:
        from myapp_config import *  # noqa
        import myapp_config

        print(
            "Loaded your LOCAL configuration at [{}]".format(myapp_config.__file__)
        )
except ImportError:
    pass


def get_env_variable(var_name, default=None):
    """Get the environment variable or raise exception."""
    try:
        return os.environ[var_name]
    except KeyError:
        if default is not None:
            return default
        else:
            error_msg = 'The environment variable {} was missing, abort...'.format(var_name)
            raise EnvironmentError(error_msg)

# 当前控制器所在的集群
ENVIRONMENT=get_env_variable('ENVIRONMENT','DEV').lower()

SQLALCHEMY_POOL_SIZE = 300
SQLALCHEMY_POOL_RECYCLE = 300  # 超时重连， 必须小于数据库的超时终端时间
SQLALCHEMY_MAX_OVERFLOW = 800
SQLALCHEMY_TRACK_MODIFICATIONS=False


# redis的配置
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'admin')   # default must set None
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
SOCKETIO_MESSAGE_QUEUE = 'redis://:%s@%s:%s/2'%(REDIS_PASSWORD,REDIS_HOST,str(REDIS_PORT)) if REDIS_PASSWORD else 'redis://%s:%s/1'%(REDIS_HOST,str(REDIS_PORT))

# 数据库配置地址
SQLALCHEMY_DATABASE_URI = os.getenv('MYSQL_SERVICE','')

SQLALCHEMY_BINDS = {}
from celery.schedules import crontab

CACHE_DEFAULT_TIMEOUT = 60 * 60 * 24  # cache默认超时是24小时，一天才过期

CACHE_CONFIG = {
    'CACHE_TYPE': 'redis', # 使用 Redis
    'CACHE_REDIS_HOST': REDIS_HOST, # 配置域名
    'CACHE_REDIS_PORT': int(REDIS_PORT), # 配置端口号
    'CACHE_REDIS_URL':'redis://:%s@%s:%s/0'%(REDIS_PASSWORD,REDIS_HOST,str(REDIS_PORT)) if REDIS_PASSWORD else 'redis://%s:%s/1'%(REDIS_HOST,str(REDIS_PORT))   # 0，1为数据库编号（redis有0-16个数据库）
}

class CeleryConfig(object):
    # 任务队列
    broker_url = 'redis://:%s@%s:%s/0'%(REDIS_PASSWORD,REDIS_HOST,str(REDIS_PORT)) if REDIS_PASSWORD else 'redis://%s:%s/0'%(REDIS_HOST,str(REDIS_PORT))
    # celery_task的定义模块
    imports = (
        'myapp.tasks',
    )
    # 结果存储
    result_backend = 'redis://:%s@%s:%s/0'%(REDIS_PASSWORD,REDIS_HOST,str(REDIS_PORT)) if REDIS_PASSWORD else 'redis://%s:%s/0'%(REDIS_HOST,str(REDIS_PORT))
    worker_redirect_stdouts = True
    worker_redirect_stdouts_level = 'DEBUG'
    # celery worker每次去redis取任务的数量
    worker_prefetch_multiplier = 10
    # 每个worker执行了多少次任务后就会死掉，建议数量大一些
    worker_max_tasks_per_child = 12000
    # celery任务执行结果的超时时间
    result_expires = 3600
    # 单个任务的运行时间限制，否则会被杀死
    # task_time_limit = 600
    # 单个任务的运行时间限制，会报错，可以捕获。
    # task_soft_time_limit=3600
    # 任务完成前还是完成后进行确认
    task_acks_late = True
    task_send_sent_event = True
    # celery worker的并发数，默认是服务器的内核数目, 也是命令行 - c参数指定的数目
    # worker_concurrency = 4
    timezone = 'Asia/Shanghai'
    enable_utc = False
    # 任务失败或者超时也确认
    task_acks_on_failure_or_timeout = True
    broker_connection_retry_on_startup = True
    # worker 将不会存储任务状态并返回此任务的值
    task_ignore_result = True
    # celery是否拦截系统根日志
    # worker_hijack_root_logger = False
    # worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
    # worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s]%(task_name)s: %(message)s"


    # 任务的限制，key是celery_task的name，值是限制配置
    task_annotations = {
        # 删除历史workflow，以及相关任务
        'task.delete_workflow': {
            'rate_limit': '1/h',
            'soft_time_limit': 600,  # 运行时长限制soft_time_limit 可以内catch
            "expires": 3600,   # 上一次的直接跳过
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        # 检查运行定时pipeline
        'task.make_timerun_config': {
            'rate_limit': '1/m',
            'soft_time_limit': 300,
            "expires": 300,
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        # 异步任务，检查在线构建镜像的docker pod
        'task.check_docker_commit': {
            'rate_limit': '1/s',
            'soft_time_limit': 600,
            "expires": 600,
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        # 异步任务，检查notebook在线构建pod
        'task.check_notebook_commit': {
            'rate_limit': '1/s',
            'soft_time_limit': 600,
            "expires": 600,
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        # 异步升级服务
        'task.upgrade_service': {
            'rate_limit': '1/s',
            'soft_time_limit': 3600,
            "expires": 3600,
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        # 上传workflow信息
        'task.upload_workflow': {
            'rate_limit': '10/s',
            'soft_time_limit': 600,
            "expires": 600,
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        'task.check_pod_terminating':{
            'rate_limit': '1/s',
            'soft_time_limit': 600,
            "expires": 600,
            'max_retries': 0,
            "reject_on_worker_lost": False
        },
        'task.exec_command': {
            'rate_limit': '1/s',
            'soft_time_limit': 600,
            "expires": 600,
            'max_retries': 0,
            "reject_on_worker_lost": False
        }
    }

    # 定时任务的配置项，key为celery_task的name，值是调度配置
    beat_schedule = {
        'task_delete_workflow': {
            'task': 'task.delete_workflow',   # 定时删除旧的workflow
            'schedule': crontab(minute='1'),
        },
        'task_make_timerun_config': {
            'task': 'task.make_timerun_config',  # 定时产生定时任务的yaml信息
            'schedule': crontab(minute='*/5'),
        },
        'task_delete_old_data': {
            'task': 'task.delete_old_data',   # 定时删除旧数据
            'schedule': crontab(minute='1', hour='1'),
        },
        # 'task_delete_notebook': {
        #     'task': 'task.delete_notebook',  # 定时停止notebook
        #     # 'schedule': 10.0,
        #     'schedule': crontab(minute='1', hour='4'),
        # },
        # 'task_push_workspace_size': {
        #     'task': 'task.push_workspace_size',   # 定时推送用户文件大小
        #     # 'schedule': 10.0,
        #     'schedule': crontab(minute='10', hour='10'),
        # },
        'task_check_pipeline_run':{
            'task':"task.check_pipeline_run",   # 定时检查pipeline的运行时长
            'schedule': crontab(minute='10', hour='11'),
        },
        'task_delete_debug_docker': {
            'task': 'task.delete_debug_docker',   # 定时删除debug的pod
            # 'schedule': 10.0,
            'schedule': crontab(minute='30', hour='23'),
        },
        'task_watch_gpu': {
            'task': 'task.watch_gpu',   # 定时推送gpu的使用情况
            'schedule': crontab(minute='10',hour='8-23/2'),
        },
        # 'task_adjust_node_resource': {
        #     'task': 'task.adjust_node_resource',  # 定时在多项目组间进行资源均衡
        #     'schedule': crontab(minute='*/10'),
        # },
        'task_watch_pod_utilization': {
            'task': 'task.watch_pod_utilization',   # 定时推送低负载利用率的pod
            'schedule': crontab(minute='10',hour='11'),
        },
        "task_check_pod_terminating": {
            "task": "task.check_pod_terminating",
            'schedule': crontab(minute='*/10'),
        }
    }

 # 帮助文档地址，显示在web导航栏
DOCUMENTATION_URL='https://github.com/tencentmusic/cube-studio/wiki'
BUG_REPORT_URL = 'https://github.com/tencentmusic/cube-studio/issues/new'
GIT_URL = 'https://github.com/tencentmusic/cube-studio/tree/master'


ROBOT_PERMISSION_ROLES=[]   # 角色黑名单

FAB_API_MAX_PAGE_SIZE=2000    # 最大翻页数目，不设置的话就会是20
MAX_TASK_CPU=50  # 最大任务cpu申请值
MAX_TASK_MEM=100  # 最大任务内存申请值
ENABLE_TASK_APPROVE=False   # 是否启动任务申请授权，需要重写推送通知，由推送通知进行对接授权接口

CELERY_CONFIG = CeleryConfig

REMEMBER_COOKIE_NAME="remember_token"   # 使用cookie认证用户的方式
# api方式访问认证header头
AUTH_HEADER_NAME = 'Authorization'   # header方式认证的header 头

# k8s中用到的各种自动自定义资源
# timeout为自定义资源需要定期删除时，设置的过期时长。创建timeout秒的实例会被认为是太久远的实例，方便及时清理过期任务
CRD_INFO={
    "workflow":{
        "group":"argoproj.io",
        "version":"v1alpha1",
        "plural":"workflows",
        'kind':'Workflow',
        "timeout": 60*60*24*2
    },
    "mpijob": {
        "group": "kubeflow.org",
        "version": "v1",
        "plural": "mpijobs",
        'kind': 'MPIJob',
        "timeout": 60 * 60 * 24 * 2
    },
    "tfjob": {
        "group": "kubeflow.org",
        "version": "v1",
        "plural": "tfjobs",
        'kind':'TFJob',
        "timeout": 60*60*24*2
    },
    "xgbjob": {
        "group": "xgboostjob.kubeflow.org",
        "version": "v1alpha1",
        "plural": "xgboostjobs",
        "timeout": 60*60*24*2
    },
    "experiment":{
        "group": "kubeflow.org",
        "version": 'v1alpha3',  # "v1alpha3",
        "plural": "experiments",
        'kind':'Experiment',
        "timeout": 60 * 60 * 24 * 2
    },
    "pytorchjob": {
        "group": "kubeflow.org",
        "version": "v1",
        'kind':'PyTorchJob',
        "plural": "pytorchjobs",
        "timeout": 60 * 60 * 24 * 2
    },
    "virtualservice": {
        "group": "networking.istio.io",
        "version": "v1alpha3",
        "plural": "virtualservices",
        'kind': 'VirtualService',
        "timeout": 60 * 60 * 24 * 1
    },
    "vcjob": {
        "group": "batch.volcano.sh",
        "version": "v1alpha1",
        'kind': 'Job',
        "plural": "jobs",
        "timeout": 60 * 60 * 24 * 2
    },
    "sparkjob": {
        "group": "sparkoperator.k8s.io",
        "version": "v1beta2",
        'kind': 'SparkApplication',
        "plural": "sparkapplications",
        "timeout": 60 * 60 * 24 * 2
    },
    "paddlejob":{
        "group": "kubeflow.org",
        "version": "v1",
        'kind': 'PaddleJob',
        "plural": "paddlejobs",
        "timeout": 60 * 60 * 24 * 2
    },
    "mxjob":{
        "group": "kubeflow.org",
        "version": "v1",
        'kind': 'MXJob',
        "plural": "mxjobs",
        "timeout": 60 * 60 * 24 * 2
    }
}

# 每个task都会携带的任务环境变量，{{}}模板变量会在插入前进行渲染
GLOBAL_ENV={
    "KFJ_PIPELINE_ID":"{{pipeline_id}}",
    "KFJ_RUN_ID":"{{uuid.uuid4().hex}}",
    "KFJ_CREATOR":"{{creator}}",
    "KFJ_RUNNER":"{{runner}}",
    "KFJ_MODEL_REPO_API_URL":"http://kubeflow-dashboard.infra",
    "KFJ_ARCHIVE_BASE_PATH":"/archives",
    "KFJ_PIPELINE_NAME":"{{pipeline_name}}",
    "KFJ_NAMESPACE":"pipeline",
    "KFJ_GPU_MEM_MIN":"13G",
    "KFJ_GPU_MEM_MAX":"13G",
    "KFJ_ENVIRONMENT":"{{cluster_name}}",
}

GPU_RESOURCE={
    "gpu":"nvidia.com/gpu"
}
DEFAULT_GPU_RESOURCE_NAME='nvidia.com/gpu'

# 配置禁用gpu的方法，不然对复合共用型机器，gpu会被共享使用
GPU_NONE={
    "gpu":['NVIDIA_VISIBLE_DEVICES','none']
}

# vgpu的类型方式
VGPU_RESOURCE={
    "mgpu":"tencent.com/vcuda-core"
}
VGPU_DRIVE_TYPE = "mgpu"   # tke gpumanager的方式


RDMA_RESOURCE_NAME=''

DEFAULT_POD_RESOURCES={}

# 各类model list界面的帮助文档
HELP_URL={}

# 不使用模板中定义的镜像而直接使用用户镜像的模板名称
CUSTOMIZE_JOB='自定义镜像' if BABEL_DEFAULT_LOCALE=='zh' else 'customize'
LOGICAL_JOB = 'logical'
PYTHON_JOB = 'python'
USER_CUSTOMIZE_IMAGES=[CUSTOMIZE_JOB,'hyperparam-search-nni']  # 使用用户自定义的镜像而不使用模板的镜像，工作目录和启动命令，需要这些模板有images，command，workdir参数

# admin管理员用户
ADMIN_USER='admin'
# pipeline任务的运行空间，目前必填pipeline
PIPELINE_NAMESPACE = 'pipeline'
# 服务pipeline运行的空间，必填service
SERVICE_PIPELINE_NAMESPACE='service'
# 超参搜索命名空间，必填automl
AUTOML_NAMESPACE = 'automl'
# notebook必填空间，必填jupyter
NOTEBOOK_NAMESPACE = 'jupyter'
# 内部服务命名空间，必填service
SERVICE_NAMESPACE = 'service'
# aihub的命令空间
AIHUB_NAMESPACE = 'aihub'
# 服务链路追踪地址
SERVICE_PIPELINE_ZIPKIN='http://xx.xx.xx.xx:9401'
SERVICE_PIPELINE_JAEGER='tracing.service'
# 拉取私有仓库镜像默认携带的k8s hubsecret名称
HUBSECRET = ['hubsecret']
# 私有仓库的组织名，用户在线构建的镜像自动推送这个组织下面
REPOSITORY_ORG='ccr.ccs.tencentyun.com/cube-studio/'
# 用户常用默认镜像
USER_IMAGE = 'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9'
# notebook每个pod使用的用户账号
JUPYTER_ACCOUNTS=''
HUBSECRET_NAMESPACE=[PIPELINE_NAMESPACE,AUTOML_NAMESPACE,NOTEBOOK_NAMESPACE,SERVICE_NAMESPACE,AIHUB_NAMESPACE]

# notebook使用的镜像
NOTEBOOK_IMAGES=[
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-cpu-base', 'vscode（cpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-gpu-base', 'vscode（gpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu22.04', 'jupyter（cpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu22.04-cuda11.8.0-cudnn8','jupyter（gpu）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-bigdata', 'jupyter（bigdata）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-machinelearning', 'jupyter（machinelearning）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-deeplearning', 'jupyter（deeplearning）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:enterprise-jupyter-ubuntu-cpu-pro', 'jupyter-conda-pro（企业版）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:enterprise-matlab-ubuntu-deeplearning', 'matlab（企业版）'],
    ['ccr.ccs.tencentyun.com/cube-studio/notebook:enterprise-rstudio-ubuntu-bigdata', 'rstudio（企业版）'],
]

# 定时检查大小的目录列表。需要再celery中启动检查任务
CHECK_WORKSPACE_SIZE = [
    "/data/k8s/kubeflow/pipeline/workspace",
    "/data/k8s/kubeflow/pipeline/archives",
]
# 定时定时检查清理旧数据的目录
DELETE_OLD_DATA = [
    "/data/k8s/kubeflow/minio/mlpipeline",
    "/data/k8s/kubeflow/minio/mlpipeline/pipelines",
    "/data/k8s/kubeflow/minio/mlpipeline/artifacts"
]
# 用户工作目录，下面会自动创建每个用户的个人目录
WORKSPACE_HOST_PATH = '/data/k8s/kubeflow/pipeline/workspace'
# 每个用户的归档目录，可以用来存储训练模型
ARCHIVES_HOST_PATH = "/data/k8s/kubeflow/pipeline/archives"
# prometheus地址
PROMETHEUS = 'prometheus-k8s.monitoring:9090'
# nni默认镜像
NNI_IMAGES='ccr.ccs.tencentyun.com/cube-studio/nni:20211003'

# 数据集的存储地址
DATASET_SAVEPATH = '/dataset/'
STORE_TYPE=""   # 目前不支持备份到云上
STORE_CONFIG = {
    "appid": "xx",
    "secret_id": "xx",
    "secret_key": "xx",
    "region": "ap-nanjing",
    "bucket_name": "xx",
    "root": "/dataset",
    "download_host":"https://xx.cos.ap-nanjing.myqcloud.com/"
}

K8S_DASHBOARD_CLUSTER = '/k8s/dashboard/cluster/'  #
BLACK_PORT = [10250]   # 黑名单端口，cube-studio将不会占用这些端口，10250是kubelet的端口。

K8S_NETWORK_MODE = 'iptables'   # iptables ipvs
NOTEBOOK_EXCLUSIVE = False   # notebook 启动是否独占资源
SERVICE_EXCLUSIVE = False   # 内部服务 启动是否独占资源

MINIO_HOST = 'minio.kubeflow:9000'

# 多行分割内网特定host
HOSTALIASES='''
127.0.0.1 localhost
'''
# 默认服务代理的ip
SERVICE_EXTERNAL_IP=[]

# json响应是否按字母顺序排序
JSON_SORT_KEYS=False
# 链接菜单
ALL_LINKS=[
    {
        "label": "K8s Dashboard",
        "name": "kubernetes_dashboard",
        "url": K8S_DASHBOARD_CLUSTER+"#/pod?namespace=infra"
    },
    {
        "label":"Grafana",
        "name":"grafana",
        "url": '/grafana/d/pod-info/pod-info?orgId=1&refresh=5s&from=now-15m&to=now'  # 访问grafana的域名地址
    }
]

# 推理服务的各种配置
TFSERVING_IMAGES=['tensorflow/serving:2.14.1-gpu','tensorflow/serving:2.14.1','tensorflow/serving:2.13.1-gpu','tensorflow/serving:2.13.1','tensorflow/serving:2.12.2-gpu','tensorflow/serving:2.12.2','tensorflow/serving:2.11.1-gpu','tensorflow/serving:2.11.1','tensorflow/serving:2.10.1-gpu','tensorflow/serving:2.10.1','tensorflow/serving:2.9.3-gpu','tensorflow/serving:2.9.3','tensorflow/serving:2.8.4-gpu','tensorflow/serving:2.8.4','tensorflow/serving:2.7.4-gpu','tensorflow/serving:2.7.4','tensorflow/serving:2.6.5-gpu','tensorflow/serving:2.6.5','tensorflow/serving:2.5.4-gpu','tensorflow/serving:2.5.4']
TORCHSERVER_IMAGES=['pytorch/torchserve:0.9.0-gpu','pytorch/torchserve:0.9.0-cpu','pytorch/torchserve:0.8.2-gpu','pytorch/torchserve:0.8.2-cpu','pytorch/torchserve:0.7.1-gpu','pytorch/torchserve:0.7.1-cpu']
ONNXRUNTIME_IMAGES=['ccr.ccs.tencentyun.com/cube-studio/onnxruntime:latest','ccr.ccs.tencentyun.com/cube-studio/onnxruntime:latest-cuda']
TRITONSERVER_IMAGES=['ccr.ccs.tencentyun.com/cube-studio/tritonserver:24.01-py3','ccr.ccs.tencentyun.com/cube-studio/tritonserver:23.12-py3','ccr.ccs.tencentyun.com/cube-studio/tritonserver:22.12-py3','ccr.ccs.tencentyun.com/cube-studio/tritonserver:21.12-py3','ccr.ccs.tencentyun.com/cube-studio/tritonserver:20.12-py3']

INFERNENCE_IMAGES={
    "tfserving":TFSERVING_IMAGES,
    'torch-server':TORCHSERVER_IMAGES,
    'onnxruntime':ONNXRUNTIME_IMAGES,
    'triton-server':TRITONSERVER_IMAGES
}

INFERNENCE_COMMAND={
    "tfserving":"/usr/bin/tf_serving_entrypoint.sh --model_config_file=/config/models.config --monitoring_config_file=/config/monitoring.config --platform_config_file=/config/platform.config",
    "torch-server":"torchserve --start --model-store /models/$model_name/ --models $model_name=$model_name.mar --foreground --log-config /config/log4j2.xml",
    "onnxruntime":"onnxruntime_server --model_path /models/",
    "triton-server":'tritonserver --model-repository=/models/ --strict-model-config=true --log-verbose=1'
}
INFERNENCE_ENV={
    "tfserving":['TF_CPP_VMODULE=http_server=1','TZ=Asia/Shanghai'],
}
INFERNENCE_PORTS={
    "tfserving":'8501',
    "torch-server":"8080,8081",
    "onnxruntime":"8001",
    "triton-server":"8000,8002"
}
INFERNENCE_METRICS={
    "tfserving":'8501:/metrics',
    "torch-server":"8082:/metrics",
    "triton-server":"8002:/metrics"
}
INFERNENCE_HEALTH={
    "tfserving":'8501:/v1/models/$model_name/versions/$model_version/metadata',
    "torch-server":"8080:/ping",
    "triton-server":"8000:/v2/health/ready"
}

CONTAINER_CLI='docker'   # 或者 docker nerdctl

DOCKER_IMAGES='docker:23.0.4'
NERDCTL_IMAGES='ccr.ccs.tencentyun.com/cube-studio/nerdctl:1.7.2'

WAIT_POD_IMAGES='ccr.ccs.tencentyun.com/cube-studio/wait-pod:v1'
# notebook，pipeline镜像拉取策略
IMAGE_PULL_POLICY='Always'    # IfNotPresent   Always

# 任务资源使用情况地址
GRAFANA_TASK_PATH='/grafana/d/pod-info/pod-info?var-pod='
# 推理服务监控地址
GRAFANA_SERVICE_PATH="/grafana/d/istio-service/istio-service?var-namespace=service&var-service="
# 集群资源监控地址
GRAFANA_CLUSTER_PATH="/grafana/d/all-node/all-node?var-org="
# 节点资源监控地址
GRAFANA_NODE_PATH="/grafana/d/node/node?var-node="
# GPU资源监控地址
GRAFANA_GPU_PATH="/grafana/d/dcgm/gpu"

MODEL_URLS = {
    "sqllab":"/frontend/data/datasearch/data_search",
    "metadata_table":"/frontend/data/metadata/metadata_table",
    "data_blood":"/frontend/data/metadata/data_blood",
    "metadata_metric":"/frontend/data/metadata/metadata_metric",
    "dimension": "/frontend/data/metadata/metadata_dimension",
    "feast":"/frontend/data/feast/feast",
    "dataset":"/frontend/data/media_data/dataset",
    "label_platform":"/frontend/data/media_data/label_platform",

    "repository": "/frontend/dev/images/docker_repository",
    "docker": "/frontend/dev/images/docker",
    "template_images": "/frontend/dev/images/template_images",
    "notebook": "/frontend/dev/dev_online/notebook",
    "etl_pipeline":"/frontend/dev/data_pipeline/etl_pipeline",
    "etl_task":"/frontend/dev/data_pipeline/task_manager",
    "etl_task_instance":"/frontend/dev/data_pipeline/instance_manager",

    "job_template": "/frontend/train/train_template/job_template",
    "pipeline": "/frontend/train/train_task/pipeline",
    "runhistory": "/frontend/train/train_task/runhistory",
    "workflow": "/frontend/train/train_task/workflow",
    "nni": "/frontend/train/automl/hyperparameter_search",

    "total_resource": "/frontend/service/total_resource",
    "service": "/frontend/service/k8s_service",
    "train_model": "/frontend/service/inferenceservice/model_manager",
    "inferenceservice": "/frontend/service/inferenceservice/inferenceservice_manager",

    "model_market_visual": "/frontend/ai_hub/model_market/model_visual",
    "model_market_voice": "/frontend/ai_hub/model_market/model_voice",
    "model_market_language": "/frontend/ai_hub/model_market/model_language",
}
 # 可以跨域分享cookie的子域名，例如.svc.local.com
COOKIE_DOMAIN = ''
SERVICE_DOMAIN='service.svc.cluster.local'
CHATGPT_TOKEN = [""]
CHATGPT_CHAT_URL = ['']


# 所有训练集群的信息
CLUSTERS={
    # 和project expand里面的名称一致
    "dev":{
        "NAME":"dev",
        "KUBECONFIG":'/home/myapp/kubeconfig/dev-kubeconfig',
        # "SERVICE_DOMAIN": 'service.local.com',
    }
}

HOST = CLUSTERS[ENVIRONMENT].get('HOST',None)
