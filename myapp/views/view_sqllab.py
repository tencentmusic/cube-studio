
import traceback
# 将model添加成视图，并控制在前端的显示
from myapp import app, appbuilder,db,event_logger
from flask import jsonify,g,request
from .base import BaseMyappView
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
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
from myapp.utils.sqllab.base_impl import Base_Impl
engine_impls = {
    'mysql': Base_Impl(),
    'presto':Base_Impl(),
    'clikchouse': Base_Impl(),
    'postgres':Base_Impl(),
    "impala":Base_Impl(),
    "oracle":Base_Impl(),
    "mssql":Base_Impl()
}
db_uri_demo = {
    'mysql': 'mysql+pymysql://username:password@host:port/database',
    'presto': 'presto://username:password@host:port/database',
    'clikchouse': 'clickhouse+native://username:password@host:port/database',
    'postgres': 'postgresql+psycopg2://username:password@host:port/database',
    "impala": 'impala://host:port/database',
    "oracle":'oracle://username:password@host:port/database',
    "mssql": 'mssql+pymssql://username:password@host:port/database'
}

def add_task(req_data):
    try:
        engine_arg1 = req_data['engine_arg1']
        engine_arg2= req_data['engine_arg2']
        qsql= req_data['sql']
    except Exception as e:
        raise KeyError("添加任务参数缺失")
    
    try:
        engine_impl = engine_impls[engine_arg1]
    except Exception as e:
        raise KeyError("%s引擎未实现"%engine_arg1)

    q = Sqllab_Query(
        submit_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        engine_arg1=engine_arg1,
        engine_arg2= engine_arg2,
        qsql= qsql,
        username = g.user.username
    )
    db.session.add(q)
    db.session.commit()
    qid = q.id
    return qid, engine_impl

def check_task_engine(qid):
    with session_scope(nullpool=True) as dbsession:
        q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(qid)).first()
        if not q:
            raise RuntimeError("任务数据库记录不存在，id:" + str(qid))
        eng = q.engine_arg1
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

# @pysnooper.snoop()
class Sqllab_Query_View(BaseMyappView):
    route_base='/idex'

    @expose('/config', methods=(["GET", "POST"]))
    def sqllab_config(self):
        config = {
            "status": 0,
            "message": "",
            "result": [
                {
                    "type": 'select',
                    "label": '引擎',
                    "id": 'engine_arg1',
                    "value": [
                        {
                            "label": enginer,
                            "value": enginer,
                            "relate": {
                                "relateId": 'engine_arg2',
                                "value": [
                                    {
                                        "label": db_uri_demo[enginer],
                                        "value": db_uri_demo[enginer],
                                    }
                                ]
                            }
                        } for enginer in db_uri_demo
                    ]
                },
                {
                    "type": 'input-select',
                    "label": '数据库',
                    "id": 'engine_arg2',
                    "value": [],
                }
            ]
        }
        print(config)
        return jsonify(config)


    # 提交异步查询（必须）
    @expose('/submit_task',methods=(["POST"]))
    def submit_task(self, args_in=None):
        if args_in:
            req_data = args_in
        else:
            req_data = request.json
        qid, engine_impl = add_task(req_data)
        res = engine_impl.submit_task(qid)
        res_keys = ["err_msg", "task_id"]
        return check_process_res(res_keys, res)

    # 获取任务状态（必须）
    @expose('/look/<task_id>',methods=(["GET"]))
    def check_task(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.check_task_status(qid)
        res_keys = ['stage', 'state', 'err_msg', "spark_log_url", "spark_ui_url"]
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

appbuilder.add_api(Sqllab_Query_View)

