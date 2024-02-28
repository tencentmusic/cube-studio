import copy
import traceback
# 将model添加成视图，并控制在前端的显示
from myapp import app, appbuilder, db, event_logger, cache
from flask import jsonify, g, request
from .base import BaseMyappView
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper, datetime, time, json
from myapp.utils.celery import session_scope
import logging
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
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
    'presto': Base_Impl(),
    'clikchouse': Base_Impl(),
    'postgres': Base_Impl(),
    "impala": Base_Impl(),
    "oracle": Base_Impl(),
    "mssql": Base_Impl()
}
db_uri_demo = {
    'mysql': ['mysql+pymysql://username:password@host:port/database'],
    'postgres': ['postgresql+psycopg2://username:password@host:port/database'],
    'presto': ['presto://username:password@host:port/database'],
    'clikchouse': ['clickhouse+native://username:password@host:port/database'],
    "impala": ['impala://host:port/database'],
    "oracle": ['oracle://username:password@host:port/database'],
    "mssql": ['mssql+pymssql://username:password@host:port/database']
}

# @pysnooper.snoop()
def add_task(req_data):
    try:
        engine_arg1 = req_data['engine_arg1']
        engine_arg2 = req_data['engine_arg2']
        qsql = req_data['sql']
    except Exception as e:
        raise KeyError(__("添加任务参数缺失"))

    try:
        engine_impl = engine_impls[engine_arg1]
    except Exception as e:
        raise KeyError(engine_arg1+__("引擎未实现"))

    q = Sqllab_Query(
        submit_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        engine_arg1=engine_arg1,
        engine_arg2=engine_arg2,
        qsql=qsql,
        username=g.user.username
    )
    db.session.add(q)
    db.session.commit()
    qid = q.id
    return qid, engine_impl


def check_task_engine(qid):
    with session_scope(nullpool=True) as dbsession:
        q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id == int(qid)).first()
        if not q:
            raise RuntimeError(__("任务数据库记录不存在，id:") + str(qid))
        eng = q.engine_arg1
    try:
        engine_impl = engine_impls[eng]
    except Exception as e:
        raise KeyError(eng+__("引擎未实现"))
    return qid, engine_impl


def check_process_res(res_keys, res):
    try:
        res = {key: res[key] for key in res_keys}
    except Exception as e:
        raise KeyError(__("返回值异常，检查引擎实现，需包含：") + str(res_keys))
    return res


# @pysnooper.snoop()
class Sqllab_Query_View(BaseMyappView):
    route_base = '/idex'

    @expose('/config', methods=(["GET", "POST"]))
    def sqllab_config(self):
        all_uri = copy.deepcopy(db_uri_demo)
        try:
            success_uris = db.session.query(Sqllab_Query.engine_arg1,Sqllab_Query.engine_arg2).filter(Sqllab_Query.username==g.user.username).filter(Sqllab_Query.status=='success').filter(Sqllab_Query.submit_time>(datetime.datetime.now()-datetime.timedelta(days=30)).strftime("%Y-%m-%d")).group_by(Sqllab_Query.engine_arg1,Sqllab_Query.engine_arg2).all()
            for success_uri in success_uris:
                all_uri[success_uri[0]].append(success_uri[1])
            # cache_uri = cache.get('sqllab_uri')
            # if cache_uri:
            #     cache_uri = json.loads()
            #     cache_uri = cache_uri.get(g.user.username)
            #     for enginer in db_uri_demo:
            #         all_uri[enginer] = all_uri[enginer]+cache_uri[enginer]
        except Exception as e:
            print(e)

        # print(all_uri)
        config = {
            "status": 0,
            "message": "",
            "result": [
                {
                    "type": 'select',
                    "label": __('引擎'),
                    "id": 'engine_arg1',
                    "value": [
                        {
                            "label": enginer,
                            "value": enginer,
                            "relate": {
                                "relateId": 'engine_arg2',
                                "value": [
                                    {
                                        "label": x,
                                        "value": x,
                                    } for x in all_uri[enginer]
                                ]
                            }
                        } for enginer in all_uri
                    ]
                },
                {
                    "type": 'input-select',
                    "label": __('数据库'),
                    "id": 'engine_arg2',
                    "value": [],
                }
            ]
        }
        if 'SQLLAB' in conf:
            config['result']=conf.get('SQLLAB')
        # print(config)
        return jsonify(config)

    # 提交异步查询（必须）
    @expose('/submit_task', methods=(["POST"]))
    def submit_task(self, args_in=None):
        if args_in:
            req_data = args_in
        else:
            req_data = request.get_json(silent=True)
        if conf.get('SQLLAB_ARGS',{}):
            req_data.update(conf.get('SQLLAB_ARGS',{}))

        qid, engine_impl = add_task(req_data)
        res = engine_impl.submit_task(qid)
        res_keys = ["err_msg", "task_id"]
        return check_process_res(res_keys, res)

    # 获取任务状态（必须）
    @expose('/look/<task_id>', methods=(["GET"]))
    def check_task(self, task_id):
        task_id = int(task_id)
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.check_task_status(qid)
        res_keys = ['stage', 'state', 'err_msg', "spark_log_url", "spark_ui_url"]
        return check_process_res(res_keys, res)

    # 获取结果（必须）
    @expose('/result/<task_id>', methods=(["GET"]))
    def get_result(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.get_result(qid)
        res_keys = ["err_msg", "result"]
        return check_process_res(res_keys, res)

    # 下载结果（必须）
    @expose('/download_url/<task_id>', methods=(["GET"]))
    def download_url(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.download_url(qid)
        res_keys = ["err_msg", "download_url"]
        return check_process_res(res_keys, res)

    # 终止任务（必须）
    @expose('/stop/<task_id>', methods=(["GET"]))
    def stop(self, task_id):
        qid, engine_impl = check_task_engine(task_id)
        res = engine_impl.stop(qid)
        res_keys = ["err_msg"]
        return check_process_res(res_keys, res)


appbuilder.add_api(Sqllab_Query_View)
