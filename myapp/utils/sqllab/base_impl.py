import os, sys
import logging
import pandas as pd
from io import StringIO
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _

import pysnooper, datetime, time, json
from myapp.utils.celery import session_scope
from myapp.tasks.celery_app import celery_app
from flask import g, request
import traceback, requests
from myapp.models.model_sqllab_query import Sqllab_Query
from myapp import app, db

BASE_LOGGING_CONF = '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s\n'


def db_commit_helper(dbsession):
    try:
        dbsession.commit()
    except Exception as e:
        dbsession.rollback()
        raise e


def convert_to_dataframe(res_text, sep=chr(0x01)):
    data_steam = StringIO(res_text)
    try:
        df = pd.read_csv(data_steam, sep=sep)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df = df.fillna("")
    return df


def convert_to_str(res_text):
    return res_text


@celery_app.task(name="task.idex.handle_base_task", bind=False)
def handle_task(qid, username=""):
    logging.info("============= begin run sqllab_base_task start, id:" + str(qid))
    with session_scope(nullpool=True) as dbsession:
        try:
            # 获取数据库记录
            q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id == int(qid)).first()
            if not q:
                raise RuntimeError(__("任务异常，数据库记录不存在"))

            if not username:
                raise RuntimeError(__("无法识别账号"))
            if not 'limit' in q.qsql:
                raise RuntimeError(__("查询sql必须包含limit"))

            q.start_time = str(datetime.datetime.now())

            # 校验参数
            q.status = 'running'
            # 提交任务
            q.stage = 'execute'
            dbsession.commit()

            # 发起远程sql查询
            from sqlalchemy import create_engine
            import pandas as pd

            engine = create_engine(q.engine_arg2)
            df = pd.read_sql_query(q.qsql, engine)
            save_path = f"/data/k8s/kubeflow/global/sqllab/result/{qid}.csv"
            print(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, encoding='utf-8-sig', index=None, header=True)
            print(df)

            q.stage = 'end'
            q.status = 'success'
            dbsession.commit()

        except Exception as e:
            print(e)
            # 记录异常信息
            err_msg = traceback.format_exc()
            q.err_msg = err_msg
            q.status = 'failure'
            dbsession.commit()
        finally:
            q.end_time = str(datetime.datetime.now())
            dbsession.commit()

        return q.stage, q.status, q.err_msg


class Base_Impl():
    # 提交任务
    # @pysnooper.snoop()
    def submit_task(self, qid, enable_async=True):
        err_msg = ""
        result = ""
        try:
            if enable_async:
                async_task = handle_task.delay(qid, username=g.user.username)
            else:
                stage, status, _err_msg = handle_task(qid, username=g.user.username)
                if _err_msg != "":
                    raise RuntimeError(_err_msg)

                res = self.get_result(qid)
                if res['err_msg'] != "":
                    raise RuntimeError(res['err_msg'])
                result = res['result']
        except Exception as e:
            err_msg = traceback.format_exc()

        if enable_async:
            return {"err_msg": err_msg, "task_id": qid}
        return {"err_msg": err_msg, "task_id": qid, "result": result}

    # 检查任务运行状态和结果
    def check_task_status(self, qid):
        stage = 'unknow'
        status = 'unknow'
        err_msg = ""
        result = []
        ui_url = ""
        log_url = ""
        try:
            with session_scope(nullpool=True) as dbsession:
                # 获取数据库记录
                q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id == int(qid)).first()
                if not q:
                    raise RuntimeError(__("任务异常，数据库记录不存在"))
                status = q.status
                stage = q.stage
                err_msg = q.err_msg
                ui_url = q.ui_url
                log_url = q.log_url

        except:
            status = 'failure'
            err_msg = traceback.format_exc()

        return {'stage': stage, 'state': status, 'err_msg': err_msg, "spark_log_url": ui_url, "spark_ui_url": log_url}

    # 同步或者异步的任务
    # @pysnooper.snoop()
    def get_result(self, task_id):
        err_msg = ""
        qid = task_id
        try:
            csv_path = f"/data/k8s/kubeflow/global/sqllab/result/{qid}.csv"
            df = pd.read_csv(csv_path, encoding='utf-8-sig', header=0)
            df = df.fillna('')

            res = [df.columns.values.tolist()] + df.values.tolist()
        except:
            err_msg = traceback.format_exc()
            raise RuntimeError(__("下载失败，desc: ") + err_msg)
        return {"err_msg": err_msg, "result": res}

    # 根据分隔符生成下载文件地址
    def download_url(self, task_id):
        err_msg = ""
        qid = task_id
        deli = '<@>'
        separator = request.args.get('separator')
        result_route = '/data/k8s/kubeflow/global/sqllab/result/'
        result_line_num = 0
        name_map = {
            ",": "comma",
            "|": "vertical",
            "TAB": "tab"
        }
        result_path = result_route + str(qid) + '.csv'
        back_path = result_route + f'{qid}.{name_map[separator]}.csv'
        os.makedirs(os.path.dirname(back_path), exist_ok=True)

        if not os.path.exists(back_path):
            with open(result_path, 'r') as f:
                data = f.read()
            result_line_num = data.count('\n') - 1
            with open(back_path, 'w') as f:
                f.write(data.replace(deli, separator))
                f.close()

        url = f'{request.host_url.rstrip("/")}/static/global/sqllab/result/{qid}.{name_map[separator]}.csv'

        # 获取数据库记录
        q = db.session.query(Sqllab_Query).filter(Sqllab_Query.id == int(qid)).first()
        if not q:
            raise RuntimeError(__("任务异常，数据库记录不存在"))
        q.result_line_num = str(result_line_num)
        db.session.commit()

        return {"err_msg": "", "download_url": url}

    def stop(self, task_id):
        err_msg = __("暂未实现远程数据库kill操作")
        return {"err_msg": err_msg}
