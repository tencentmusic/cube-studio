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

PLATFORM_RTX = "tmedataleap"
PLATFORM_TOKEN = "YmE0MGYxMjE0ZTVhMjg1YjhmMWVkZWM0OGUxMDg3Y2Y4OTg0ZDk2YjhlZmMxZGRl"

ENGINE={
# spark 失败不转mr
'thive': 'set `supersql.datasource.conf:hive.spark.failed.retry@thive` = `false`;set `supersql.execution.engine` = `native`; set `supersql.presto.apriori` = `false`;',
# 普通hive on spark
'auto': 'set `supersql.presto.apriori` = `false`;',
'mr': 'set `supersql.presto.apriori` = `false`;set `supersql.execution.engine` = `native`;set `supersql.datasource.conf:hive.execute.engine@thive` = `mapreduce`;set `supersql.datasource.conf:hive.spark.failed.retry@thive` = `false`;',
'presto': 'set `supersql.execution.engine` = `presto`'
}

def get_tdwTA(service='idex-openapi', rtx="", token="", bind_oa=""):
    # 鉴权
    from tdwTauthAuthentication import TdwTauthAuthentication
    if not rtx:
        rtx = g.user.username
        token = g.user.tdw_token
        bind_oa = g.user.bind_oa_rtx
#    rtx = 'uthermai'
#    token = 'NGI3OTUwOTU2OTNhZTQ1NzU5MTM5OTRlNGRlOWU5NDI0YzVjNDViYTM5NTczYjQx'
#    bind_oa = ""
    if (not token) or (token == 'None'): # 没有token就尝试代理
        if (not bind_oa) or (bind_oa == 'None'): # 缺少配置
            raise RuntimeError("首次使用星云，需要先找星云管理员uthermai注册星云账号。")
        tdwTA = TdwTauthAuthentication(PLATFORM_RTX, PLATFORM_TOKEN, service, 'pr_' + rtx)
    else:
        tdwTA = TdwTauthAuthentication(rtx, token, service)
    return tdwTA

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

# 提交idex，更新task状态
@celery_app.task(name="task.idex.handle_task", bind=False)
def handle_task(qid, rtx="", token="", bind_oa=""):

    logging.info("idex_task start, id:" + str(qid))
    with session_scope(nullpool=True) as dbsession:
        # 获取数据库记录
        q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(qid)).first()
        if not q:
            raise RuntimeError("任务异常，数据库记录不存在")

        try:
            if q.start_time != "":
                return
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

            cluster='tl'
            engine='thive'
            deli='<@>'
            group_id = q.group_id
            sql = q.qsql
            if not group_id:
                raise RuntimeError("应用组不合法")
            if not sql:
                raise RuntimeError("sql不合法")

            # 计算gaia_id
            q.stage = 'parse'
            db_commit_helper(dbsession)

            tdwTA = get_tdwTA('idex-openapi', rtx, token, bind_oa)
            # 查集群信息
            Authentication = tdwTA.getAuthentication()
            Authentication["Version"] ="2"
            res = rs.get("http://api.idex.oa.com/clusters/{cluster}".format(cluster=cluster), headers=Authentication, timeout=3)
            if res.status_code not in (200,201):
                raise RuntimeError("查询失败, desc: {}".format(res.text))
            pools_url = json.loads(res.text)['pools_url']
            #logger.info("pools_url: {}".format(str(pools_url)))

            # 查资源池
            Authentication = tdwTA.getAuthentication()
            Authentication["Version"] ="2"
            res = rs.get(pools_url, headers=Authentication, timeout=3)
            if res.status_code not in (200,201):
                raise RuntimeError("查询失败, desc: {}".format(res.text))
            gaias_url = json.loads(res.text)
            if not gaias_url:
                raise RuntimeError("当前没有可用集群资源")
            #logger.info("gaia_ids: {}".format(str(gaias_url)))

            # 选择内存剩余最多的资源池
            gaia_id = None
            mem = -99999999
            for gaia_url in gaias_url:
                if group_id not in gaia_url:
                    continue
                #print(gaia_url)
                # 查资源池信息
                Authentication = tdwTA.getAuthentication()
                Authentication["Version"] ="2"
                res = rs.get(gaia_url, headers=Authentication, timeout=3)
            #    print(res.text)
                if res.status_code not in (200,201):
                    raise RuntimeError("查询失败, desc: {}".format(res.text))

                gaia_mem = int(json.loads(res.text)['metrics']['min_memory']) - int(json.loads(res.text)['metrics']['used_memory'])
                if mem < gaia_mem:
                    mem = gaia_mem
                    gaia_id = json.loads(res.text)['gaia_id']

            if not gaia_id:
                raise RuntimeError('none gaia_id choiced，很有可能你没有{}应用组的权限。'.format(str(group_id)))

            # kuwo线上任务资源池与日常查询分开
            if group_id == "g_other_tme_infrastructure_tme_central_kuwo":
                gaia_id = "3527"
            q.gaia_id = gaia_id
            db_commit_helper(dbsession)

#            raise RuntimeError("idex后台压力较大，请稍等一会再重试")

            # 提交任务
            q.stage = 'execute'
            db_commit_helper(dbsession)

            tdwTA = get_tdwTA('idex-openapi', rtx, token, bind_oa)
            Authentication = tdwTA.getAuthentication()
            Authentication["Version"] ="2"
            body = {
                "cluster_id": cluster,
                "group_id": group_id, #应用组
                "gaia_id": gaia_id, #gaia ID 
                #"database": database, #数据库
                "type": 'sql',
                "statements": ENGINE[engine] + sql # SQL  加选择引擎参数
            }
            url = 'http://api.idex.oa.com/tasks'
            res = rs.post(url=url, headers=Authentication, json=body, timeout=3)
            #print(res.text)
            if res.status_code not in (200,201):
                raise RuntimeError("查询失败, desc: {}".format(res.text))

            task_url = json.loads(res.text)['task_url']
            log_url = json.loads(res.text)['log_url']

            if task_url:
                task_id = task_url.split('/')[-1]
            else:
                raise RuntimeError("task_id生成异常")
            q.task_id = task_id
            db_commit_helper(dbsession)

            # 轮询更新任务结果
            for i in range(5000):
                time.sleep(3)

                task_url = 'http://api.idex.oa.com/tasks/' + str(task_id)
                tdwTA = get_tdwTA('idex-openapi', rtx, token, bind_oa)
                Authentication = tdwTA.getAuthentication()
                Authentication["Version"] ="2"

                # 获取整个语句组的状态
                res = rs.get(task_url, headers=Authentication, timeout=3)
                if res.status_code not in (200, 201):
                    raise RuntimeError("查询失败, desc: {}\n".format(json.loads(res.text)['message']))
                state = json.loads(res.text)['state']
                statements_url = json.loads(res.text)['statements_url']

                if state not in ('running', 'success', 'failure', 'abortion', 'submitted'):
                    raise RuntimeError('未识别的state : ' + state)

                # 按理说不该走到这里，idex应该返回异常码。 
                if state in ('failure', 'abortion'):
                    raise RuntimeError("查询未成功。state={}, message={}".format(state, json.loads(res.text)['message']))

                # 如果运行中，获取最后一个语句的状态和job链接
                if state in ('running'):
                    Authentication = tdwTA.getAuthentication()
                    Authentication["Version"] ="2"
                    res = rs.get(statements_url, headers=Authentication, timeout=3)
                    if res.status_code not in (200,201):
                        raise RuntimeError("查询失败, desc: {}".format(res.text))

                    # 获取spark日志
                    if not q.spark_log_url:
                        if json.loads(res.text):
                            statement_url = json.loads(res.text)[-1]
                            Authentication = tdwTA.getAuthentication()
                            Authentication["Version"] ="2"
                            res = rs.get(statement_url, headers=Authentication, timeout=3)
                            if res.status_code in (200,201):
                                if json.loads(res.text)['state'] == 'running':
                                    jobs_url = json.loads(res.text).get('jobs_url')
                                    if jobs_url:
                                        jobs_url = jobs_url.strip()
                                        if jobs_url.endswith('/'):
                                            app_id = jobs_url.split('/')[-2]
                                        else:
                                            app_id = jobs_url.split('/')[-1]
                                        if app_id.startswith('application_'):
                                            spark_log_url = "http://tdw-proxy.tmeoa.com/cluster/app/{}".format(app_id)
                                            spark_ui_url = "http://tdw-proxy.tmeoa.com/proxy/{}".format(app_id)
                                            q.spark_log_url = spark_log_url
                                            q.spark_ui_url = spark_ui_url
                                            db_commit_helper(dbsession)
                else:
                    break

            # 按理说不该走到这里
            if state not in ('success'):
                raise RuntimeError("未知异常")

            # 获取结果，限制2w条。取最后一条语句的结果。
            Authentication = tdwTA.getAuthentication()
            Authentication["Version"] ="2"
            res = rs.get(statements_url, headers=Authentication, timeout=3)
            if res.status_code not in (200,201):
                raise RuntimeError("查询失败, desc: {}".format(res.text))
            result_url = json.loads(res.text)[-1] + "/result"

            Authentication = tdwTA.getAuthentication()
            Authentication["Version"] ="2"
            res = rs.get(result_url, headers=Authentication, timeout=3)
            if res.status_code not in (200,201):
                if not json.loads(res.text).get('message') and json.loads(res.text).get('message').startswith('找不到文件夹/文件'):
                    raise RuntimeError("查询失败, desc: {}".format(res.text))
            else:
                result=[]

            result = convert_to_dataframe(res.text)
            if result.empty:
                result=[]
            else:
                cols = list(result.columns)
                import numpy as np
                result = np.array(result).tolist()
                result = [cols] + result

            # 结果落文件
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

class Idex_Impl():
    def submit_task_impl(self, qid):
        err_msg = ""
        result = ""
        try:
            rtx = g.user.username
            token = g.user.tdw_token
            bind_oa = g.user.bind_oa_rtx
            async_task = handle_task.delay(qid, rtx, token, bind_oa)
        except Exception as e:
            err_msg = traceback.format_exc()
        return {"err_msg":err_msg, "task_id": qid}

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

    def submit_and_check_task_until_done(self, qid):
        err_msg = ""
        result = ""
        try:
            stage, status, _err_msg = handle_task(qid)
            logging.info("stage: " + stage + ',status: ' + status + ',err_msg: ' + _err_msg)
            if _err_msg != "": 
                raise RuntimeError(_err_msg)
            res = self.get_result(qid)
            if res['err_msg'] !="":
                raise RuntimeError(res['err_msg'])
            result = res['result']
        except Exception as e:
            err_msg = traceback.format_exc()

        return {"err_msg":err_msg, "task_id": qid, "result":result}

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
             with session_scope(nullpool=True) as dbsession:
                 q = dbsession.query(Sqllab_Query).filter(Sqllab_Query.id==int(task_id)).first()
                 if not q:
                     raise RuntimeError("找不到对应任务")
                 task_id = q.task_id
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
