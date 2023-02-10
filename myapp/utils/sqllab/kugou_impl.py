import os,sys
import logging
import pandas as pd
from io import StringIO
import pysnooper,datetime,time,json
from myapp.security import MyUser
from myapp.utils.celery import session_scope
from myapp.tasks.celery_app import celery_app
from flask_appbuilder import CompactCRUDMixin, expose
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
import traceback,requests
from myapp.models.model_sqllab_query import Sqllab_Query

BASE_LOGGING_CONF = '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s\n'

def db_commit_helper(dbsession):
    try:
        dbsession.commit()
    except Exception as e:
        dbsession.rollback()
        raise e

def convert_to_dataframe(res_text, sep=chr(0x01)):
    data_steam=StringIO(res_text)
    try:
        df = pd.read_csv(data_steam, sep=sep)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df = df.fillna("")
    return df

def convert_to_str(res_text):
    return res_text

@celery_app.task(name="task.idex.handle_kugou_task", bind=False)
def handle_kugou_task(qid, rtx="", token="", bind_oa=""):
    logging.info("idex_kugou_task start, id:" + str(qid))
    with session_scope(nullpool=True) as dbsession:
        # 获取数据库记录
        q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(qid)).first()
        if not q:
            raise RuntimeError("任务异常，数据库记录不存在")

        try:
            q.start_time = str(datetime.datetime.now())
            db_commit_helper(dbsession)

            # 初始化重试次数
            from requests.adapters import HTTPAdapter
            rs = requests.Session()
            rs.mount('http://', HTTPAdapter(max_retries=3))
            rs.mount('https://', HTTPAdapter(max_retries=3))

            # 校验参数
            q.stage = 'start'
            q.status = 'running'
            db_commit_helper(dbsession)

            import hashlib
            hl = hashlib.md5()
            if not rtx:
                raise RuntimeError("无法识别账号")
            hl.update((rtx + 'PGDwmcqI7MdSrrNlG5KU4IGVpYMW0Zj6').encode(encoding='utf-8'))
            sign=hl.hexdigest()
            group_id = q.group_id
            proxies = { "http": "http://11.181.93.67:8080", "https": "http://11.181.93.67:8080"}
            header = {"Content-Type": "application/json", "userName":rtx, "Authorization": sign}
            data = {"sql": q.qsql, "engine":"hive", "queue": group_id}
            deli='<@>'

            if not 'limit' in q.qsql:
                raise RuntimeError("试验功能，只能 limit 100 条")

            # 提交任务
            q.stage = 'execute'
            db_commit_helper(dbsession)
            res = rs.post(url="http://49.7.106.165:7001/datacenter/query/tasks", headers=header, json=data, proxies=proxies)
#            print(res.status_code)
#            print(res.text)
            if (not res.status_code in [200, 201]) or json.loads(res.text).get('result_msg')!='SUCCESS':
                raise RuntimeError("任务提交kugou失败，desc: " + res.text)

            result = json.loads(res.text)['result']
            batchCode = result['batchCode']
            queryDetailId = result['queryDetailIds'][0]
            historyId = result['historyId']

            q.task_id = str(batchCode) + ',' + str(historyId) + ',' + str(queryDetailId)
            db_commit_helper(dbsession)

            for i in range(3000):
                time.sleep(3)
                res = rs.get(url="http://49.7.106.165:7001/datacenter/query/tasks?batchCode={}&historyId={}&queryDetailId={}".format(batchCode, historyId, queryDetailId), headers=header, proxies=proxies)
#                print(res.status_code)
#                print(res.text)
                if (not res.status_code in [200, 201]) or json.loads(res.text).get('result_msg')!='SUCCESS':
                    raise RuntimeError("查询kugou任务状态失败，desc: " + res.text)

                result = json.loads(res.text)['result']
                remote_task_state = result['status']
#                print(remote_task_state)
                if remote_task_state not in ('running'):
                    break

            # 不管结果怎样，先搞到日志
            res = rs.get(url="http://49.7.106.165:7001/datacenter/query/getTaskLogs/tasks?batchCode={}&historyId={}&queryDetailId={}".format(batchCode, historyId, queryDetailId), headers=header, proxies=proxies)
#            print(res.status_code)
#            print(res.text)
            if (not res.status_code in [200, 201]) or json.loads(res.text).get('result_msg')!='SUCCESS':
                raise RuntimeError("获取kugou日志失败, desc: " + res.text)
            log = json.loads(res.text)['result']

            if remote_task_state not in ('success'):
                raise RuntimeError("查询未成功。state={}, message={}".format(remote_task_state, log))

            # 成功了处理结果
            res = rs.get(url="http://49.7.106.165:7001/datacenter/query/result/tasks?batchCode={}&historyId={}&queryDetailId={}&start=0".format(batchCode, historyId, queryDetailId), headers=header, proxies=proxies)
#            print(res.status_code)
#            print(res.text)
            if (not res.status_code in [200, 201]):
                raise RuntimeError("获取kugou查询结果失败, desc: " + res.text)

            if res.text.startswith("Response code"):
                res_text = str(res.text).replace("Response code\n", "")
            result = convert_to_dataframe(res_text, '\t')
            if result.empty:
                result=[]
            else:
                cols = list(result.columns)
                import numpy as np
                result = np.array(result).tolist()
                result = [cols] + result

            result_path = "/data/k8s/kubeflow/pipeline/workspace/uthermai/online/idex_result/{}".format(str(qid) + '.csv')
            with open(result_path, "w") as f:
                for line in result:
                    f.write(deli.join([str(x) for x in line]) + '\n')

            q.stage = 'end'
            q.status = 'success'
            db_commit_helper(dbsession)

        except Exception as e:
            # 记录异常信息
            err_msg = traceback.format_exc()
            q.err_msg = err_msg
            q.status = 'failure'
            db_commit_helper(dbsession)
        finally:
            q.end_time = str(datetime.datetime.now())
            db_commit_helper(dbsession)

        return q.stage, q.status, q.err_msg

class Kugou_Impl():
    def submit_task_impl(self, qid, async=True):
        err_msg = ""
        result = ""
        try:
            if async:
                rtx = g.user.username
                async_task = handle_kugou_task.delay(qid, rtx)
            else:
                stage, status, _err_msg = handle_kugou_task(qid, rtx)
                if _err_msg != "":
                    raise RuntimeError(_err_msg)
                res = get_result(qid)
                if res['err_msg'] !="":
                    raise RuntimeError(res['err_msg'])
                result = res['result']
        except Exception as e:
            err_msg = traceback.format_exc()

        if async:
            return {"err_msg":err_msg, "task_id": qid}
        return {"err_msg":err_msg, "task_id": qid, "result":result}

    def check_task_impl(self, qid):
        stage = 'unknow'
        status = 'unknow'
        err_msg = ""
        result = []
        spark_ui_url = ""
        spark_log_url = ""
        deli = '<@>'
        try:
            with session_scope(nullpool=True) as dbsession:
                # 获取数据库记录
                q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(qid)).first()
                if not q:
                    raise RuntimeError("任务异常，数据库记录不存在")
                status = q.status
                stage = q.stage
                err_msg = q.err_msg
                spark_ui_url = q.spark_ui_url
                spark_log_url = q.spark_log_url
                # 返回结果
    #            if stage == 'end' and status =='success':
    #                url = 'http://%s/static%s'%(request.host, "/mnt/uthermai/online/idex_result/" + str(qid) + '.csv')
    #                res = requests.get(url).text
    #                res = res.split('\n')
    #                for line in res:
    #                    result.append(line.split(deli))
        except:
            status = 'failure'
            err_msg = traceback.format_exc()

        return {'stage':stage, 'state':status, 'err_msg':err_msg, "spark_log_url":spark_log_url, "spark_ui_url":spark_ui_url}

    def submit_and_check_task_until_done(self, req_data):
        res = self.submit_task_impl(req_data, False)
        return res

    def get_result(self, task_id):
        err_msg = ""
        qid = task_id
        deli = '<@>'
        try:
            url = 'http://%s/static%s'%(request.host, "/mnt/uthermai/online/idex_result/" + str(qid) + '.csv')
            url = 'http://%s/static%s'%("data.tme.woa.com", "/mnt/uthermai/online/idex_result/" + str(qid) + '.csv')
            res = requests.get(url)
            if res.status_code not in (200,201):
                raise RuntimeError("查询结果缓存异常, desc: " + res.text)
            res = res.text
            res = res.split('\n')
            res = [line.split(deli) for line in res]
        except:
            err_msg = traceback.format_exc()
            raise RuntimeError("下载失败，desc: " + err_msg)
        return {"err_msg": err_msg, "result": res}

    def download_url(self, task_id):
        err_msg = ""
        qid = task_id
        deli = '<@>'
        separator = request.args.get('separator')
        result_route = '/data/k8s/kubeflow/pipeline/workspace/uthermai/online/idex_result/'
        result_line_num = 0
        if separator == ',':
            if not os.path.exists(result_route + str(qid) + '.douhao.csv'):
                with open(result_route + str(qid) + '.csv', 'r') as f:
                    data = f.read()
                result_line_num = data.count('\n') -1
                with open(result_route + str(qid) + '.douhao.csv', 'w') as f:
                    f.write(data.replace(deli, ','))
            url = 'http://%s/static%s'%(request.host, "/mnt/uthermai/online/idex_result/" + str(qid) + '.douhao.csv')
        elif separator == '|':
            if not os.path.exists(result_route + str(qid) + '.shuxian.csv'):
                with open(result_route + str(qid) + '.csv', 'r') as f:
                    data = f.read()
                result_line_num = data.count('\n') -1
                with open(result_route+ str(qid) + '.shuxian.csv', 'w') as f:
                    f.write(data.replace(deli, '|'))
            url = 'http://%s/static%s'%(request.host, "/mnt/uthermai/online/idex_result/" + str(qid) + '.shuxian.csv')
        elif separator == 'TAB': 
            if not os.path.exists(result_route + str(qid) + '.tab.csv'):
                with open(result_route + str(qid) + '.csv', 'r') as f:
                    data = f.read()
                result_line_num = data.count('\n') -1
                with open(result_route+ str(qid) + '.tab.csv', 'w') as f:
                    f.write(data.replace(deli, '\t'))
            url = 'http://%s/static%s'%(request.host, "/mnt/uthermai/online/idex_result/" + str(qid) + '.tab.csv')
        else: 
            url = 'http://%s/static%s'%(request.host, "/mnt/uthermai/online/idex_result/" + str(qid) + '.csv')

        with session_scope(nullpool=True) as dbsession:
            # 获取数据库记录
            q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(qid)).first()
            if not q:
                raise RuntimeError("任务异常，数据库记录不存在")
            q.result_line_num = str(result_line_num)
            db_commit_helper(dbsession)

        return {"err_msg": "", "download_url": url}


    @expose('/stop/<task_id>',methods=(["GET"]))
    def stop(self, task_id):
        err_msg = ""
        try:
#            with session_scope(nullpool=True) as dbsession:
#                q = dbsession.query(Idex_Query).filter(Idex_Query.id==int(task_id)).first()
#                if not q:
#                    raise RuntimeError("找不到对应任务")
             if True:
                tdwTA = get_tdwTA()
                Authentication = tdwTA.getAuthentication()
                Authentication["Version"] ="2"
                res = requests.delete("http://api.idex.oa.com/tasks/" + str(task_id), headers=Authentication)
                if res.status_code not in (200, 201, 204):
                    raise RuntimeError("终止失败, desc: \n" + json.loads(res.text)['message'])
        except:
            err_msg = traceback.format_exc()
        return {"err_msg":err_msg}
