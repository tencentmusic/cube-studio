from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.forms import GeneralModelConverter
import uuid
from flask_appbuilder.actions import action
import re,os
import traceback
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp import app, appbuilder,db,event_logger
from myapp.utils import core
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from flask import jsonify
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
from werkzeug.datastructures import FileStorage
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
import requests
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
from myapp.security import MyUser
from myapp.utils.celery import session_scope
import logging
from myapp.models.model_sqllab_query import Sqllab_Query
conf = app.config

def db_commit_helper(dbsession):
    try:
        dbsession.commit()
    except Exception as e:
        dbsession.rollback()
        raise e

# 新引擎在此添加
from myapp.utils.sqllab.idex_impl import Idex_Impl
from myapp.utils.sqllab.kugou_impl import Kugou_Impl
engine_impls = {'tme': Idex_Impl(), 'kugou':Kugou_Impl() , 'hive': None, 'mysql':None}

def add_task(req_data):
    try:
        eng = req_data['biz']
        group_id= req_data['tdw_app_group']
        qsql= req_data['sql']
    except Exception as e:
        raise KeyError("添加任务参数缺失")
    
    try:
        engine_impl = engine_impls[eng]
    except Exception as e:
        raise KeyError("%s引擎未实现"%eng)

    with session_scope(nullpool=True) as dbsession:
        q = Sqllab_Query(submit_time = datetime.datetime.now(),
                        group_id= group_id,
                        qsql= qsql,
                        rtx = g.user.username,
                        biz = eng 
                        )
        dbsession.add(q)
        db_commit_helper(dbsession)
        qid = q.id
    return qid, engine_impl

def check_task_engine(qid):
    with session_scope(nullpool=True) as dbsession:
        q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(qid)).first()
        if not q:
            raise RuntimeError("任务数据库记录不存在，id:" + str(qid))
        eng = q.biz
    try:
        engine_impl = engine_impls[eng]
    except Exception as e:
        raise KeyError("%s引擎未实现"%eng)
    return qid, engine_impl 

def check_process_res(res_keys, res):
    try:
        res = {key:res[key] for key in res_keys}
    except Exception as e:
        raise KeyError("返回值异常，检查引擎实现，需包含：" + str(res_keys))
    return res


class Sqllab_Query_View(BaseMyappView):
    route_base='/idex'
    # 获取配置信息
    @expose('/config',methods=(["GET","POST"]))
    def sqllab_config(self):
        return jsonify({
            "status":0,
            "message":"",
            "result":{
                "clusters_group":   #  可能有多个关联组，一个组内前后关联
                {
                    "label": [
                        {
                            "name":"enginer_type",
                            "label":"引擎",
                            "ui-type":"select"
                        },
                        {
                            "name":"db_uri",
                            "label":"数据库",
                            "ui-type": "input-select"
                        }
                    ],
                    "value": {
                        "enginer_type":[
                            {
                                "id":"mysql",
                                "value":"mysql",
                                "db_uri":[
                                    {
                                        "id":item,
                                        "value":item
                                    } for item in ['mysql+pymysql://账号:密码@host:port/db?charset=utf8']
                                ]
                            },
                            {
                                "id": "postgresql",
                                "value": "postgresql",
                                "db_uri": [
                                    {
                                        "id":item,
                                        "value":item
                                    } for item in ['postgresql://账号:密码@host:port/db']
                                ]
                            },
                            {
                                "id": "clickhouse",
                                "value": "clickhouse",
                                "db_uri": [
                                    {
                                        "id": item,
                                        "value": item
                                    } for item in ['clickhouse+native://账号:密码@host:port/db']
                                ]
                            }
                        ]
                    }
                }

            }
        })
        pass
    # 提交异步查询（必须）
    @expose('/submit_task',methods=(["POST"]))
    def submit_task_impl(self, args_in=None):
        if args_in:
            req_data = args_in
        else:
            req_data = request.json
        qid, engine_impl = add_task(req_data)
        res = engine_impl.submit_task_impl(qid)
        res_keys = ["err_msg", "task_id"]
        return check_process_res(res_keys, res)

    # 获取结果（必须）
    @expose('/result/<task_id>',methods=(["GET"]))
    def get_result(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.get_result(qid)
        res_keys = ["err_msg", "result"]
        return check_process_res(res_keys, res)

    # 下载结果（必须）
    @expose('/download_url/<task_id>',methods=(["GET"]))
    def download_url(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.download_url(qid)
        res_keys = ["err_msg", "download_url"]
        return check_process_res(res_keys, res)

    # 终止任务（必须）
    @expose('/stop/<task_id>',methods=(["GET"]))
    def stop(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.stop(qid)
        res_keys = ["err_msg"]
        return check_process_res(res_keys, res)

    # 获取任务状态（必须）
    @expose('/look/<task_id>',methods=(["GET"]))
    def check_task_impl(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.check_task_impl(qid)
        res_keys = ['stage', 'state', 'err_msg', "spark_log_url", "spark_ui_url"]
        return check_process_res(res_keys, res)

    # 杀
    @expose('/kill/<task_id>',methods=(["GET"]))
    def kill(self, task_id):
        try:
            import requests
            data = '{"id":1037,"act":"query","task_value":{"jobid":"%s"}}'%task_id
            print(data)
            headers = {"content-type":"application/json", "token":"11f901ef6973c79d0dd275c7bf632cbf", "staffname":"uthermai", "projectname":"p-tdw-1"}
            res = requests.get(url="http://openapi.zhiyan.woa.com/operate/v1/exec_task", data=data ,headers=headers)
            if not res.status_code in (200,201):
                raise RuntimeError("kill失败，status_code: " + str(res.status_code) +"，desc:" + str(res.text))
        except:
            err_msg = traceback.format_exc()
            return {'err_msg':err_msg, 'succ': False}
        return {'err_msg':"", 'succ': True}

appbuilder.add_api(Sqllab_Query_View)

